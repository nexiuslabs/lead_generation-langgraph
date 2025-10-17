import pytest
from src.vendors import apify_linkedin as api


@pytest.mark.asyncio
async def test_company_url_from_domain(monkeypatch):
    monkeypatch.setenv("APIFY_TOKEN", "dummy")
    monkeypatch.setenv(
        "APIFY_COMPANY_FINDER_BY_DOMAIN_ACTOR_ID",
        "s-r~free-linkedin-company-finder---linkedin-address-from-any-site",
    )

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, params=None, json=None, headers=None):
            if "free-linkedin-company-finder" in url:
                return FakeResp({
                    "items": [
                        {"linkedinUrl": "https://www.linkedin.com/company/acme-asia/"}
                    ]
                })
            return FakeResp({"items": []})

    monkeypatch.setattr(api.httpx, "AsyncClient", FakeClient)

    out = await api.company_url_from_domain("https://ouch.com.sg")
    assert out and out.startswith("https://www.linkedin.com/company/")


@pytest.mark.asyncio
async def test_company_url_normalizes_regional_subdomain(monkeypatch):
    monkeypatch.setenv("APIFY_TOKEN", "dummy")

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, params=None, json=None, headers=None):
            return FakeResp([
                {"domain": "daisy.sg", "linkedin_url": "https://sg.linkedin.com/company/daisyaccountants"}
            ])

    monkeypatch.setattr(api.httpx, "AsyncClient", FakeClient)
    out = await api.company_url_from_domain("daisy.sg")
    assert out == "https://www.linkedin.com/company/daisyaccountants"


@pytest.mark.asyncio
async def test_contacts_via_domain_chain(monkeypatch):
    # Ensure default actor IDs are set and token present
    monkeypatch.setenv("APIFY_TOKEN", "dummy")
    monkeypatch.setenv(
        "APIFY_COMPANY_FINDER_BY_DOMAIN_ACTOR_ID",
        "s-r~free-linkedin-company-finder---linkedin-address-from-any-site",
    )
    monkeypatch.setenv("APIFY_EMPLOYEES_ACTOR_ID", "harvestapi~linkedin-company-employees")
    monkeypatch.setenv("APIFY_LINKEDIN_ACTOR_ID", "dev_fusion~linkedin-profile-scraper")

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, params=None, json=None, headers=None):
            # Company finder by domain
            if "free-linkedin-company-finder" in url:
                return FakeResp({
                    "items": [
                        {"linkedinUrl": "https://www.linkedin.com/company/acme-asia/"}
                    ]
                })
            # Employees actor
            if "linkedin-company-employees" in url:
                return FakeResp([
                    {"linkedinUrl": "https://www.linkedin.com/in/jane-smith-cofounder/"},
                    {"profileUrl": "https://www.linkedin.com/in/ravi-kumar-sales/"},
                    {"url": "https://www.linkedin.com/in/maria-lee-cto/"},
                ])
            # Profile scraper actor
            if "linkedin-profile-scraper" in url:
                return FakeResp([
                    {"fullName": "Jane Smith", "jobTitle": "Co-founder at Acme", "linkedinUrl": "https://www.linkedin.com/in/jane-smith-cofounder/", "locationName": "SG"},
                    {"fullName": "Ravi Kumar", "headline": "Head of Sales, APAC — Acme", "profileUrl": "https://www.linkedin.com/in/ravi-kumar-sales/", "location": "IN"},
                    {"fullName": "Maria Lee", "jobTitle": "CTO", "url": "https://www.linkedin.com/in/maria-lee-cto/", "locationName": "US"},
                ])
            return FakeResp({"items": []})

    monkeypatch.setattr(api.httpx, "AsyncClient", FakeClient)

    titles = ["founder", "head of sales"]
    out = await api.contacts_via_domain_chain("ouch.com.sg", titles, max_items=10)
    # Filter should keep only Jane and Ravi
    names = [c.get("full_name") for c in out]
    assert "Jane Smith" in names and "Ravi Kumar" in names
    assert all(c.get("linkedin_url", "").startswith("https://www.linkedin.com/") for c in out)
