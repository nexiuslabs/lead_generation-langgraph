---
owner: Codex Agent – Frontend Generator
status: draft
last_reviewed: 2025-03-21
---

# Feature PRD — Nexius Agent UI Enhancements

## Story
Nexius go-to-market teams need a production-ready interface that reflects the Nexius brand and unlocks tenant-specific flows. As a Nexius customer success lead, I want the LangGraph-based lead generation experience to present a cohesive Nexius Agent shell, so that my teams can onboard new tenants, manage candidate discovery, and monitor operational health without relying on generic developer tooling.

## Acceptance Criteria
- A Nexius-branded web client replaces the default LangGraph UI, featuring consistent typography, colors, and navigation for Chat, Candidates, Scores, Onboarding, and Operations sections.
- Authentication respects `/info` configuration, bootstraps Nexius SSO redirects when required, and persists session state for downstream API calls.
- Onboarding UX surfaces `/whoami`, `/session/odoo_info`, and `/onboarding/status` data with actionable prompts for `/onboarding/first_login`, `/onboarding/verify_odoo`, and `/onboarding/repair_admin` when remediation is needed.
- Chat experience streams responses from `/chat/stream/{session_id}` with clear loading, error, and retry affordances aligned to Nexius design language.
- Candidates and Scores pages render paginated datasets from `/candidates/latest` and `/scores/latest`, offering filtering, CSV export links, and responsive virtualized tables.
- Operations dashboard visualizes `/metrics`, `/metrics/ttfb`, and `/shortlist/status`, and includes a gated `/scheduler/run_now` trigger with audit logging.
- All primary views support responsive breakpoints, keyboard navigation, and meet WCAG 2.1 AA color contrast requirements.

## Dependencies
- FastAPI backend endpoints: `/info`, `/whoami`, `/session/odoo_info`, `/onboarding/*`, `/chat/stream/{session_id}`, `/candidates/latest`, `/scores/latest`, `/export/latest_scores.csv`, `/metrics`, `/metrics/ttfb`, `/shortlist/status`, `/scheduler/run_now`.
- Authentication tokens/cookies issued by the Nexius SSO integration as described in Section 6 of `AGENTS.md`.
- Shared design system assets (logo, typography, palette) to define Nexius-specific Tailwind tokens and Shadcn component themes.
- Virtualization library support (e.g., `@tanstack/react-virtual` or equivalent) to keep large tables performant.
- Accessibility tooling (Lighthouse, Pa11y) for validating WCAG 2.1 AA compliance.

## Success Metrics
- ≥90 Lighthouse accessibility score for desktop and mobile on Chat, Onboarding, Candidates, Scores, and Operations views.
- ≥4/5 internal UI aesthetics rating during Nexius design review.
- Onboarding completion time for new tenants reduced by 30% compared to current manual process.
- Support ticket volume related to UI navigation drops by 40% within one month post-launch.

## Risks & Mitigations
- **SSO integration drift:** Changes in Nexius identity provider may break bootstrap flow. *Mitigation:* centralize SSO config retrieval via `/info` and add health checks for token freshness.
- **API schema evolution:** Backend endpoint changes could desynchronize the UI. *Mitigation:* define typed API client wrappers and add regression tests covering critical flows.
- **Performance regressions:** Virtualized lists or streaming chat may stutter on low-end devices. *Mitigation:* budget performance testing time and introduce loading skeletons plus graceful fallbacks.

## Open Questions
1. What specific brand assets (logos, gradients, illustration styles) should anchor the Nexius theme beyond color and typography? *Owner:* Product Design (Jane D.) — **Due:** 2025-03-25.
2. Are there role-based navigation differences (e.g., ops vs. sales) that require conditional menu items or permissions? *Owner:* GTM Ops (Alex R.) — **Due:** 2025-03-26.
3. Should `/metrics/ttfb` instrumentation feed into the UI directly or rely on server-pushed events for accuracy? *Owner:* Platform Engineering (Priya S.) — **Due:** 2025-03-27.
