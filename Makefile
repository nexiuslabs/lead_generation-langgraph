PY ?= python

.PHONY: acceptance-check
acceptance-check:
	$(PY) -m scripts.acceptance_check --json

.PHONY: acceptance-check-tenant
acceptance-check-tenant:
	@if [ -z "$(TID)" ]; then echo "Usage: make acceptance-check-tenant TID=123"; exit 2; fi
	$(PY) -m scripts.acceptance_check --tenant $(TID) --json

