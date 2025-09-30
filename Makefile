PY ?= python

.PHONY: acceptance-check
acceptance-check:
	$(PY) -m scripts.acceptance_check --json

.PHONY: acceptance-check-tenant
acceptance-check-tenant:
	@if [ -z "$(TID)" ]; then echo "Usage: make acceptance-check-tenant TID=123"; exit 2; fi
	$(PY) -m scripts.acceptance_check --tenant $(TID) --json

.PHONY: app-migrations
app-migrations:
	$(PY) scripts/run_app_migrations.py

.PHONY: nightly
nightly:
	$(PY) -m scripts.run_nightly
