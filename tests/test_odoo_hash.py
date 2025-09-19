import os
import sys

# ensure path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.onboarding import _odoo_admin_hash
from passlib.hash import pbkdf2_sha512


def test_odoo_admin_hash_format_and_verify():
    pwd = "secret-password"
    hashed = _odoo_admin_hash(pwd)
    assert "+" not in hashed and "=" not in hashed
    assert pbkdf2_sha512.verify(pwd, hashed)
