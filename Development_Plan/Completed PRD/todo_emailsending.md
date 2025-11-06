---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-03-24
---

# TODO — Email Delivery for Enrichment Results

- [ ] **Configure SendGrid adapter and settings** — Owner: Codex Agent – Frontend Generator (Due: 2025-03-28)
  - Implement helper in `src/notifications/sendgrid.py` and wire environment variables (`devplan/devplan_emailsending.md#architecture--design`, `#configuration--secrets`).
- [ ] **Persist recipient context through graph state** — Owner: Codex Agent – Frontend Generator (Due: 2025-03-28)
  - Update `_normalize` and `GraphState` to carry `notify_email` (`devplan/devplan_emailsending.md#architecture--design`, `#data--state-considerations`).
- [ ] **Trigger emails on scoring and REST paths** — Owner: Codex Agent – Frontend Generator (Due: 2025-03-29)
  - Modify chat score node and `/icp/enrich/top10` response to send emails and adjust acknowledgements (`devplan/devplan_emailsending.md#architecture--design`).
- [ ] **Extend background job announcements with email** — Owner: Codex Agent – Frontend Generator (Due: 2025-03-29)
  - Add deduplicated send logic in `_announce_completed_bg_jobs` (`devplan/devplan_emailsending.md#architecture--design`).
- [ ] **Testing & documentation updates** — Owner: Codex Agent – Frontend Generator (Due: 2025-03-30)
  - Add unit/integration coverage and refresh README/.env instructions (`devplan/devplan_emailsending.md#testing-strategy`, `#configuration--secrets`).

