# PufferLab Progress Ledger

- **Active goal:** deliver Milestone 1 through the reviewed engineering loop: a real browser →
  FastAPI → turbopuffer comparison of BM25 and vector retrieval using an isolated tiny fixture.
- **Goal status:** active
- **Last updated:** 2026-08-22
- **Orchestrator:** root agent
- **Dedicated reviewer:** reviewer agent

This file is the source of truth for execution status. Update it at every handoff, review decision,
merge, verification result, or blocker. Process details live in
[`engineering-loop.md`](engineering-loop.md).

## Work queue

| ID | Deliverable | Owner | Branch / PR | Status | Evidence / next action |
|---|---|---|---|---|---|
| F0 | Product brief and implementation plan | foundation_fixes | `codex/pufferlab-plan` / PR #1 | verified | Reviewer approved after fixes and merged as `561a554`; plan is now on `main`. |
| F1 | Contract-first backend/frontend scaffold | foundation_fixes | `codex/t0-contract-scaffold` / PR #2 | verified | Reviewer reran contract gates, approved, and merged as `6acaf96`; GitHub Backend/Frontend CI passed. |
| LOOP | Agent policy, review runbook, and progress ledger | root | `codex/engineering-loop` / PR #3 | review_requested | Rebased on reviewed `main`; dedicated reviewer is auditing the policy and ledger. |
| M1-A | Narrow turbopuffer provider with fake and opt-in live tests | unassigned | `codex/t2-turbopuffer-provider` | planned | Depends on F1 and LOOP. |
| M1-B | Deterministic tiny fixture and idempotent ingestion command | unassigned | `codex/t3-tiny-ingestion` | planned | Depends on provider interface from M1-A; fixture preparation may start in parallel. |
| M1-C | BM25/vector retrieval orchestration and compare API | unassigned | `codex/t5-compare-api` | planned | Depends on M1-A and the T0 contracts. |
| M1-D | Playground query UI with side-by-side results | unassigned | `codex/t8-playground` | planned | Depends on compare API contract; UI shell work may start from a frozen response fixture. |
| M1-E | Live vertical-slice verification and setup documentation | unassigned | `codex/m1-live-verification` | planned | Requires M1-A through M1-D and a locally supplied API key. |

## Milestone 1 acceptance evidence

- [ ] One documented command ingests the tiny fixture into a uniquely owned turbopuffer namespace.
- [ ] Re-running ingestion is idempotent and readiness is verified.
- [ ] The browser submits one query through FastAPI and receives BM25 and vector result lists.
- [ ] Results show document identity, 1-based rank, typed score semantics, and client wall-clock time.
- [ ] The API key remains server-side and is absent from repository history and browser assets.
- [ ] Fake-provider tests run in normal CI; the opt-in live test creates and cleans its namespace.
- [ ] Backend, frontend, generated-contract, and GitHub default-branch checks pass.
- [ ] Every required PR records an independent reviewer decision and merge.

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-08-22 | Use one focused `codex/*` branch and GitHub PR per review unit. | Keeps diffs small, independently reversible, and easy to review. |
| 2026-08-22 | Keep a dedicated reviewer separate from implementation workers. | Prevents self-approval and creates a consistent quality gate. |
| 2026-08-22 | Track execution in this repository instead of chat-only state. | Survives agent handoffs and makes blockers and evidence auditable. |
| 2026-08-22 | Use fake provider tests in CI and opt-in isolated live tests. | Keeps CI deterministic while validating the real account path safely. |
| 2026-08-22 | Treat provider documentation as a contract-review input. | Independent review caught dense-vector and FTS constraints before provider implementation. |
