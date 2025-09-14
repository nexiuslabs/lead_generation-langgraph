import os
import sys
from unittest.mock import patch

# ensure path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.onboarding import _odoo_admin_user_create, _odoo_set_admin_credentials


def test_admin_user_create_and_login_before_after_credentials_update():
    server = "http://fake"
    db_name = "testdb"
    admin_login = "admin"
    admin_password = "adminpass"
    tenant_email = "tenant@example.com"
    new_password = "newpass"

    # environment so _odoo_set_admin_credentials targets the created user
    os.environ["ODOO_TEMPLATE_ADMIN_LOGIN"] = tenant_email

    users = {
        1: {"id": 1, "login": admin_login, "password": admin_password, "active": True},
    }
    next_id = [2]
    created = {}

    class FakeCommon:
        def authenticate(self, db, login, password, context):
            for u in users.values():
                if u["login"] == login and u["password"] == password and u.get("active", True):
                    return u["id"]
            return False

    class FakeModels:
        def execute_kw(self, db, uid, pwd, model, method, args, kwargs=None):
            nonlocal created
            kwargs = kwargs or {}
            if model == "res.users" and method == "search":
                domain = args[0]
                login = domain[0][2]
                return [u["id"] for u in users.values() if u["login"] == login]
            if model == "res.users" and method == "create":
                vals = args[0]
                uid = next_id[0]
                vals = vals.copy()
                vals["id"] = uid
                users[uid] = vals
                created = {"vals": args[0], "kwargs": kwargs}
                next_id[0] += 1
                return uid
            if model == "res.users" and method == "write":
                ids, vals = args
                for _id in ids:
                    users[_id].update(vals)
                return True
            if model == "res.users" and method == "read":
                return [{"partner_id": [1]}]
            if model == "res.partner" and method == "write":
                return True
            raise Exception("unsupported call")

    def fake_serverproxy(url):
        if url.endswith("/xmlrpc/2/common"):
            return FakeCommon()
        if url.endswith("/xmlrpc/2/object"):
            return FakeModels()
        raise AssertionError("unexpected url")

    with patch("xmlrpc.client.ServerProxy", side_effect=fake_serverproxy):
        _odoo_admin_user_create(server, db_name, admin_login, admin_password, tenant_email)
        assert created["vals"]["password"] == admin_password
        assert created["kwargs"]["context"]["no_reset_password"] is True

        common = fake_serverproxy(f"{server}/xmlrpc/2/common")
        # before updating credentials
        assert common.authenticate(db_name, tenant_email, admin_password, {})

        _odoo_set_admin_credentials(
            server,
            db_name,
            tenant_email,
            new_password,
            auth_login=admin_login,
            auth_password=admin_password,
        )
        # after updating credentials
        assert common.authenticate(db_name, tenant_email, new_password, {})

