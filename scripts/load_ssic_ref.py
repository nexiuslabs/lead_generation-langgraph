# scripts/load_ssic_ref.py
"""
Excel/CSV → ssic_ref loader with version + file hash upsert.

Usage:
  python scripts/load_ssic_ref.py /path/to/ssic.xlsx [--dsn postgresql://...]

- Default DSN is taken from src.settings.POSTGRES_DSN
- SSIC version can be overridden via env var SSIC_VERSION (default: "SSIC 2025A")
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2

# Ensure project root import path so `src.settings` can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.settings import POSTGRES_DSN  # type: ignore
except Exception as e:
    print("ERROR: Could not import POSTGRES_DSN from src/settings.py:", e)
    POSTGRES_DSN = None  # will validate later


SSIC_VERSION_DEFAULT = os.getenv("SSIC_VERSION", "SSIC 2025A")


def norm_code(x: object) -> Optional[str]:
    if x is None:
        return None
    s = re.sub(r"\D", "", str(x))
    if not s:
        return None
    return s.zfill(5)[:5]  # 5-digit canonical


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _excel_read_with_header_detection(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read Excel and detect the real header row when the first rows are titles."""
    sheet_arg = None
    if sheet is not None:
        try:
            sheet_arg = int(sheet) if str(sheet).isdigit() else sheet
        except Exception:
            sheet_arg = sheet

    # Read without header to inspect the first N rows
    raw = pd.read_excel(path, sheet_name=sheet_arg, header=None)

    def _scan_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        max_scan = min(30, len(df))
        for hdr in range(max_scan):
            row = df.iloc[hdr].tolist()
            vals = [str(v).strip().lower() for v in row]
            if not any(vals):
                continue
            code_cols = []
            title_cols = []
            def_cols = []
            for i, v in enumerate(vals):
                if not v or v == "nan":
                    continue
                # Code header cell: contains "code" (not postal) OR starts with/contains SSIC + year
                if ("code" in v and "postal" not in v) or re.search(r"\bssic\b", v):
                    code_cols.append(i)
                # Title header cell
                if "title" in v:
                    title_cols.append(i)
                # Definitions/description header cell
                if ("definition" in v) or ("detailed" in v) or ("description" in v) or ("detail" in v):
                    def_cols.append(i)

            # Require the headers to appear in at least two distinct columns to avoid matching a banner line
            distinct_hdr_cols = set(code_cols + title_cols + def_cols)
            has_code = len(code_cols) > 0
            has_title = len(title_cols) > 0
            has_def = len(def_cols) > 0
            if has_code and (has_title or has_def) and len(distinct_hdr_cols) >= 2:
                # Build columns from this row
                cols = []
                for i, v in enumerate(row):
                    name = str(v).strip()
                    cols.append(name if name and name.lower() != "nan" else f"col_{i}")
                df2 = df.iloc[hdr + 1 :].copy()
                df2.columns = cols
                return df2
        return None

    # If multiple sheets returned, scan each
    if isinstance(raw, dict):
        for _, df_sheet in raw.items():
            if isinstance(df_sheet, pd.DataFrame):
                scanned = _scan_df(df_sheet)
                if scanned is not None:
                    return scanned
        # Fallback to the first sheet with default header handling
        first_key = next(iter(raw.keys()))
        return pd.read_excel(path, sheet_name=first_key)

    # Single sheet case
    scanned = _scan_df(raw)
    if scanned is not None:
        return scanned

    # Fallback: let pandas guess header=0
    return pd.read_excel(path, sheet_name=sheet_arg)


def read_any(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".csv", ".tsv"}:
        sep = "," if suf == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    # default: Excel with header detection
    return _excel_read_with_header_detection(path, sheet)


def run(path: str, dsn: str, ssic_version: str = SSIC_VERSION_DEFAULT, sheet: Optional[str] = None) -> int:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    df = read_any(p, sheet)  # expect columns like: Code, Title, Description
    if df is None or df.empty:
        print("No rows found in input file.")
        return 0

    # Canonicalize headers to be resilient to spaces, punctuation, and casing
    orig_cols = [str(c) for c in df.columns]
    def _canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

    canon_map = {_canon(c): c for c in orig_cols}
    canon_cols = list(canon_map.keys())

    def _guess_code_col_by_values(df: pd.DataFrame) -> Optional[str]:
        best = None  # (ratio, valid_count, col_name)
        for col in df.columns:
            s = df[col].head(300)
            nonnull = s.dropna()
            if len(nonnull) == 0:
                continue
            valid = 0
            for v in nonnull:
                try:
                    if norm_code(v):
                        valid += 1
                except Exception:
                    continue
            ratio = valid / max(1, len(nonnull))
            tup = (ratio, valid, str(col))
            if best is None or tup > best:
                best = tup
        if best and (best[0] >= 0.15 and best[1] >= 10):
            return best[2]
        return None

    def _pick_code_col() -> Optional[str]:
        # Strong preference: the explicit SSIC code column (e.g., "SSIC 2025")
        for k in canon_cols:
            if "ssic" in k and all(s not in k for s in ("title", "group", "example", "cross", "reference")):
                return canon_map[k]
        # Fallback by values: choose the column whose cells look like SSIC codes,
        # but avoid columns that are clearly not code (groups/examples/cross refs/definitions/titles)
        candidates = []
        excluded_substr = ("group", "example", "cross", "reference", "defin", "detail", "desc", "title")
        for k in canon_cols:
            if any(s in k for s in excluded_substr):
                continue
            candidates.append(canon_map[k])
        if candidates:
            best = None
            for col in candidates:
                s = df[col].head(300).dropna()
                if len(s) == 0:
                    continue
                valid = 0
                for v in s:
                    try:
                        if norm_code(v):
                            # additionally, prefer cells that are short (avoid large text rows)
                            if len(str(v)) <= 10:
                                valid += 1
                    except Exception:
                        continue
                ratio = valid / max(1, len(s))
                tup = (ratio, valid, str(col))
                if best is None or tup > best:
                    best = tup
            if best and (best[0] >= 0.15 and best[1] >= 10):
                return best[2]
        # As a last resort, use the generic guesser
        return _guess_code_col_by_values(df)

    def _pick_title_col() -> Optional[str]:
        for k in ("ssic_title", "title", "activity_title"):
            if k in canon_map:
                return canon_map[k]
        title_like = [c for c in canon_cols if "title" in c]
        return canon_map[title_like[0]] if title_like else None

    def _pick_desc_col() -> Optional[str]:
        # Prefer the official "Detailed Definitions" column over generic names
        for k in ("detailed_definitions", "detailed_definition", "definition", "description", "details"):
            if k in canon_map:
                return canon_map[k]
        desc_like = [c for c in canon_cols if "detail" in c or "defin" in c or "desc" in c]
        return canon_map[desc_like[0]] if desc_like else None

    code_col = _pick_code_col()
    title_col = _pick_title_col()
    desc_col = _pick_desc_col()
    if not code_col or not title_col:
        print("ERROR: Missing expected columns in SSIC file (need code+title)")
        print(" - Available columns:", orig_cols)
        raise AssertionError("Missing expected columns in SSIC file (need code+title)")

    file_hash = sha256_file(p)

    rows = []
    for _, r in df.iterrows():
        code = norm_code(r[code_col])
        if not code:
            continue
        title = str(r[title_col]).strip()
        if not title or title.lower() == "nan":
            continue
        # Normalize description: use Detailed Definitions column where available,
        # and treat placeholders like "<Blank>"/NaN as empty.
        raw_desc = r.get(desc_col) if desc_col else None
        if raw_desc is None or pd.isna(raw_desc):
            desc = ""
        else:
            desc = str(raw_desc).strip()
            if desc.lower() in {"<blank>", "blank", "nan", "none"}:
                desc = ""
        rows.append((code, title, desc, ssic_version, file_hash))

    if not rows:
        print("No valid SSIC rows to upsert.")
        return 0

    conn = psycopg2.connect(dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                # Keep prior versions; only clear rows for same version+hash to prevent duping on reruns
                cur.execute(
                    "DELETE FROM ssic_ref WHERE version=%s AND source_file_hash=%s",
                    (ssic_version, file_hash),
                )
                cur.executemany(
                    """
                    INSERT INTO ssic_ref (code, title, description, version, source_file_hash)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (code) DO UPDATE
                       SET title=EXCLUDED.title,
                           description=EXCLUDED.description,
                           version=EXCLUDED.version,
                           source_file_hash=EXCLUDED.source_file_hash
                    """,
                    rows,
                )
        print(f"Upserted {len(rows)} SSIC rows. version={ssic_version} hash={file_hash[:12]}…")
        return len(rows)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load SSIC reference data into ssic_ref.")
    # Make path optional; fall back to SSIC_FILE env var or common defaults
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to SSIC Excel/CSV file (or set SSIC_FILE env var)",
    )
    parser.add_argument(
        "--dsn",
        default=POSTGRES_DSN,
        help="Postgres DSN (default from src.settings.POSTGRES_DSN)",
    )
    parser.add_argument(
        "--sheet",
        default=os.getenv("SSIC_SHEET"),
        help="Excel sheet name or index (optional; or set SSIC_SHEET env)",
    )
    parser.add_argument(
        "--version",
        default=SSIC_VERSION_DEFAULT,
        help=f"SSIC version label (default: {SSIC_VERSION_DEFAULT})",
    )
    args = parser.parse_args()

    if not args.dsn:
        print("ERROR: No DSN provided and POSTGRES_DSN not set in environment/.env")
        sys.exit(1)

    path_arg = args.path or os.getenv("SSIC_FILE")
    if not path_arg:
        # Try common defaults
        candidates = [
            ROOT / "data" / "ssic.xlsx",
            ROOT / "data" / "ssic.csv",
            Path.cwd() / "ssic.xlsx",
            Path.cwd() / "ssic.csv",
        ]
        for c in candidates:
            if c.exists():
                path_arg = str(c)
                break

    if not path_arg:
        print(
            "ERROR: No input file provided. Provide a path argument, set SSIC_FILE env var, or place 'ssic.xlsx/csv' in ./data or CWD."
        )
        parser.print_help()
        sys.exit(2)

    try:
        run(path_arg, args.dsn, args.version, args.sheet)
    except psycopg2.errors.UndefinedTable:
        print(
            "ERROR: ssic_ref table does not exist. Apply app migrations first: \n"
            "  1) Ensure POSTGRES_DSN is set in your .env (or environment)\n"
            "  2) From repo root, run: python scripts/run_app_migrations.py\n"
            "  3) Re-run: python scripts/load_ssic_ref.py [path|--sheet ...]"
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
