import os
import sys
import asyncio
import logging

# Ensure project root is on sys.path so `src` and other top-level modules import
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


async def main_async():
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from zoneinfo import ZoneInfo
    except Exception as e:
        logging.error(
            "APScheduler not installed or import failed: %s. Install 'apscheduler' to use the scheduler.",
            e,
        )
        return

    from run_nightly import run_all, list_active_tenants

    logging.basicConfig(level=logging.INFO)
    tz = ZoneInfo("Asia/Singapore")
    cron = os.getenv("SCHED_START_CRON", "0 1 * * *")
    try:
        minute, hour, dom, month, dow = cron.split()
    except Exception:
        minute, hour, dom, month, dow = "0", "1", "*", "*", "*"

    async def job():
        try:
            await run_all()
            # Refresh ICP patterns MV after nightly completes so suggestions stay current
            try:
                from src.icp_intake import refresh_icp_patterns as _refresh_icp_patterns
                await asyncio.to_thread(_refresh_icp_patterns)
                logging.getLogger("nightly").info("icp_patterns materialized view refreshed")
            except Exception as exc:
                logging.getLogger("nightly").warning("icp_patterns refresh failed: %s", exc)
            # After nightly completes, run acceptance checks per tenant and log results
            try:
                from scripts import acceptance_check as _acc
                import asyncpg
                dsn = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")
                if not dsn:
                    raise RuntimeError("POSTGRES_DSN not configured for acceptance checks")
                tenants = await asyncio.to_thread(list_active_tenants)
                # Load thresholds from env or use defaults from the script
                def _f(env_name: str, default: float) -> float:
                    try:
                        return float(os.getenv(env_name, str(default)))
                    except Exception:
                        return default
                thresholds = {
                    "min_domain_rate": _f("MIN_DOMAIN_RATE", 0.70),
                    "min_about_rate": _f("MIN_ABOUT_RATE", 0.60),
                    "min_email_rate": _f("MIN_EMAIL_RATE", 0.40),
                    "max_bucket_dominance": _f("MAX_BUCKET_DOMINANCE", 0.70),
                }
                async with asyncpg.create_pool(dsn=dsn, min_size=0, max_size=1) as pool:
                    async with pool.acquire() as conn:
                        for tid in tenants:
                            try:
                                await conn.execute("SELECT set_config('request.tenant_id', $1, true)", str(tid))
                            except Exception:
                                pass
                            metrics = await _acc.compute_metrics(conn)
                            passed, results = _acc.evaluate(metrics, thresholds)
                            logging.getLogger("nightly").info(
                                "acceptance-check tenant=%s passed=%s domain=%.2f about=%.2f email=%.2f dominance=%.2f",
                                tid,
                                passed,
                                (metrics.get("domain_rate") or 0.0),
                                (metrics.get("about_rate") or 0.0),
                                (metrics.get("email_rate") or 0.0),
                                results.get("bucket_dominance") or 0.0,
                            )
            except Exception as exc:
                logging.getLogger("nightly").warning("acceptance-check post-run failed: %s", exc)
        except Exception as exc:
            logging.exception("nightly run failed: %s", exc)

    sched = AsyncIOScheduler(timezone=tz)
    cron_job = sched.add_job(
        job,
        CronTrigger(minute=minute, hour=hour, day=dom, month=month, day_of_week=dow),
    )
    # Optional: schedule alerts checker (defaults to every 5 minutes)
    try:
        alerts_cron = os.getenv("ALERTS_CRON", "*/5 * * * *")
        aminute, ahour, adom, amonth, adow = alerts_cron.split()
        from scripts import alerts as _alerts
        async def alert_job():
            # run in thread to avoid blocking loop
            try:
                await asyncio.to_thread(_alerts.check_last_run_alerts)
            except Exception as exc:
                logging.getLogger("nightly").warning("alerts job failed: %s", exc)
        sched.add_job(alert_job, CronTrigger(minute=aminute, hour=ahour, day=adom, month=amonth, day_of_week=adow))
    except Exception as _e:
        logging.getLogger("nightly").warning("alerts schedule skipped: %s", _e)
    # Start the scheduler inside a running event loop
    sched.start()
    try:
        # After start(), next_run_time is available
        logging.info(
            "scheduler: cron=%s %s/%s next_run=%s",
            cron,
            tz.key if hasattr(tz, "key") else str(tz),
            tz,
            getattr(cron_job, "next_run_time", None),
        )
    except Exception:
        pass

    # Keep the loop alive forever
    stop = asyncio.Event()
    await stop.wait()


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
