# PufferLab Progress Ledger

- **Active goal:** deliver Milestone 1 through the reviewed engineering loop: a real browser →
  FastAPI → turbopuffer comparison of BM25 and vector retrieval using an isolated tiny fixture.
- **Goal status:** active
- **Last updated:** 2026-08-22
- **Orchestrator:** root agent
- **Dedicated reviewer:** reviewer agent

This file is the versioned execution snapshot. Update it on an active task branch at each handoff,
review decision, or blocker. Between ledger updates, GitHub PR/check/merge history is canonical for
completed merges. The orchestrator batches that evidence into the next planned update and uses one
finalization PR at goal completion; the finalization PR's own merge stays canonical in GitHub.
Process details live in [`engineering-loop.md`](engineering-loop.md).

- **Main enforcement:** active GitHub ruleset
  [`21190317`](https://github.com/JapmeetBrar/pufferlab/rules/21190317) requires pull requests plus
  successful `Backend` and `Frontend` checks, blocks deletion/non-fast-forward updates, and has no
  bypass actors.
- **Identity limitation:** all agents share one GitHub identity, so the ruleset requires zero GitHub
  approvals, and extra approval for unattributed changes is disabled. Independent agent review and
  reviewer-only merge remain required and are recorded here or in the canonical PR history.

## Work queue

| ID | Deliverable | Owner | Branch / PR | Status | Evidence / next action |
|---|---|---|---|---|---|
| F0 | Product brief and implementation plan | foundation_fixes | `codex/pufferlab-plan` / PR #1 | verified | Reviewer approved after fixes and merged as `561a554`; plan is now on `main`. |
| F1 | Contract-first backend/frontend scaffold | foundation_fixes | `codex/t0-contract-scaffold` / PR #2 | verified | Reviewer reran contract gates, approved, and merged as `6acaf96`; GitHub Backend/Frontend CI passed. |
| LOOP | Agent policy, review runbook, and progress ledger | foundation_fixes | `codex/engineering-loop` / [PR #3](https://github.com/JapmeetBrar/pufferlab/pull/3) | verified | Independent reviewer approved the repaired workflow; PR #3 merged to protected `main` as `78d5ddf` with Backend and Frontend checks green. |
| M1-A | [Narrow turbopuffer provider with fake and opt-in live tests](implementation-plan.md#task-t2) | provider_worker | `codex/t2-turbopuffer-provider` / [PR #5](https://github.com/JapmeetBrar/pufferlab/pull/5) | implementing | First re-review verified detached provider errors and the corrected ledger, then requested changes because the provider-neutral FTS schema omitted frozen `k3`. Worker returned the task to `implementing` to align the typed boundary, fake exact-write coverage, and opt-in live schema. Multi-query, server RRF, recall/warm hints, and live hybrid parity remain deferred to M1-C/T5. |
| M1-B | [Deterministic tiny fixture and idempotent ingestion command](implementation-plan.md#task-t3) | fixture_worker | `codex/t3-tiny-ingestion` / [PR #4](https://github.com/JapmeetBrar/pufferlab/pull/4) | changes_requested | Reviewer requested fixes for strict JSON/canonicalization, provider schema compatibility, corpus-bound readiness, finite embeddings/error chains, and the stale ledger before re-review. |
| M1-C | [Retrieval orchestration](implementation-plan.md#task-t5) and [compare API](implementation-plan.md#task-t8) | unassigned | `codex/t5-compare-api` | planned | Depends on M1-A and the T0 contracts; acceptance criteria and tests are in task cards T5 and T8. |
| M1-D | [Playground query UI with side-by-side results](implementation-plan.md#task-t8) | unassigned | `codex/t8-playground` | planned | Depends on the compare API contract; UI acceptance criteria and tests are in task card T8. |
| M1-E | [Live vertical-slice verification](implementation-plan.md#task-t8) and [setup documentation](implementation-plan.md#task-t11) | unassigned | `codex/m1-live-verification` | planned | Requires M1-A through M1-D and a locally supplied API key; verification criteria are in task cards T8 and T11. |

## Review history

| Date | Task / PR | Transition | Evidence / next action |
|---|---|---|---|
| 2026-08-22 | LOOP / [PR #3](https://github.com/JapmeetBrar/pufferlab/pull/3) | `review_requested → changes_requested → implementing → review_requested → approved → merged → verified` | Dedicated review found and re-reviewed workflow blockers. The corrected policy and active ruleset `21190317` were approved; PR #3 merged as `78d5ddf` with required checks green. |
| 2026-08-22 | M1-A / [PR #5](https://github.com/JapmeetBrar/pufferlab/pull/5) | `review_requested → changes_requested → implementing → review_requested → changes_requested → implementing` | Review at `7ec8221` found retained secret-bearing exception context and a stale ledger; the worker fixed both at `2baa4a1`. First re-review verified those fixes, then found that `FullTextSearchSchema` omitted frozen, provider-supported `k3`. The worker is aligning the type and exact fake/live write shapes before requesting another review. |
| 2026-08-22 | M1-B / [PR #4](https://github.com/JapmeetBrar/pufferlab/pull/4) | `review_requested → changes_requested` | Independent review found five blocking data-integrity, provider-boundary, readiness, redaction, and process issues. The fixture worker must return the task to `implementing`, address every finding, and request re-review. |

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
| 2026-08-22 | Make GitHub merge/check history canonical between finite ledger updates. | Avoids recursively opening a PR solely to record the previous ledger PR's merge. |
| 2026-08-22 | Protect `main` with active ruleset `21190317` requiring PRs and Backend/Frontend checks. | Enforces review-unit and CI boundaries without bypasses; GitHub approvals remain zero because agents share one identity. |
