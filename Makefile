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

.PHONY: run-orchestrator
run-orchestrator:
	$(PY) -m scripts.run_orchestrator --tenant-id $(TID) --input "$(INPUT)"

.PHONY: logs-tail
logs-tail:
	@LOG_DIR=$${TROUBLESHOOT_API_LOG_DIR:-.log_api}; \
	echo "Tailing $$LOG_DIR/api.log"; \
	tail -F "$$LOG_DIR"/api.log
