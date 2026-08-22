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
| M1-A | [Narrow turbopuffer provider with fake and opt-in live tests](implementation-plan.md#task-t2) | provider_worker | `codex/t2-turbopuffer-provider` / [PR #5](https://github.com/JapmeetBrar/pufferlab/pull/5) | verified | Dedicated review verified detached SDK errors, frozen `k3` schema support, exact fake/live write coverage, and the corrected ledger. PR #5 merged to protected `main` as `23ccd79`; post-merge Backend and Frontend checks passed. Multi-query, server RRF, recall/warm hints, and live hybrid parity remain deferred to M1-C/T5. |
| M1-B | [Deterministic tiny fixture and idempotent ingestion command](implementation-plan.md#task-t3) | fixture_worker | `codex/t3-tiny-ingestion` / [PR #4](https://github.com/JapmeetBrar/pufferlab/pull/4) | verified | Dedicated review approved the exact remote-ID/count readiness repair, real-SDK metadata normalization, observed ANN metric verification, and FTS byte-limit fix. PR #4 squash-merged to protected `main` as `7086d2b`; post-merge Backend and Frontend checks passed. Concrete sentence-transformers execution and the user-facing ingestion command remain later integration work. |
| M1-C | [Retrieval orchestration](implementation-plan.md#task-t5) and [compare API](implementation-plan.md#task-t8) | provider_worker | `codex/t5-compare-api` / [PR #6](https://github.com/JapmeetBrar/pufferlab/pull/6) | verified | Dedicated review approved the schema-bound filter validation and production ordering repair. PR #6 merged to protected `main` as `5ca296c`; post-merge Backend and Frontend checks passed. Hybrid multi-query/server RRF, warm hints, and eval execution remain deferred beyond the Milestone 1 comparison path. |
| M1-D | [Playground query UI with side-by-side results](implementation-plan.md#task-t8) | unassigned | `codex/t8-playground` | planned | Depends on the compare API contract; UI acceptance criteria and tests are in task card T8. |
| M1-E | [Tiny-fixture ingestion command](implementation-plan.md#task-t7) and [setup documentation](implementation-plan.md#task-t11) | fixture_worker | `codex/m1-ingest-cli` / [PR #7](https://github.com/JapmeetBrar/pufferlab/pull/7) | review_requested | The worker added a single manifest-pinned `pufferlab dataset ingest-tiny` command, safe random/explicit owned namespaces, shared normalized BGE query/passage runtime, pre-write plan and compact progress output, idempotent fake-runtime coverage, and fresh-clone setup guidance. Full backend, frontend, and generated-contract gates pass locally; independent review is required before merge. A real account credential has not been consumed, so live vertical-slice verification remains pending. |

## Review history

| Date | Task / PR | Transition | Evidence / next action |
|---|---|---|---|
| 2026-08-22 | LOOP / [PR #3](https://github.com/JapmeetBrar/pufferlab/pull/3) | `review_requested → changes_requested → implementing → review_requested → approved → merged → verified` | Dedicated review found and re-reviewed workflow blockers. The corrected policy and active ruleset `21190317` were approved; PR #3 merged as `78d5ddf` with required checks green. |
| 2026-08-22 | M1-A / [PR #5](https://github.com/JapmeetBrar/pufferlab/pull/5) | `review_requested → changes_requested → implementing → review_requested → changes_requested → implementing → review_requested → approved → merged → verified` | Review at `7ec8221` found retained secret-bearing exception context and a stale ledger; the worker fixed both at `2baa4a1`. First re-review verified those fixes, then found that `FullTextSearchSchema` omitted frozen, provider-supported `k3`. Head `6289037` aligned the typed boundary and exact fake/live write shapes; dedicated re-review approved it, and PR #5 merged as `23ccd79` with protected checks green. |
| 2026-08-22 | M1-B / [PR #4](https://github.com/JapmeetBrar/pufferlab/pull/4) | `review_requested → changes_requested → implementing → review_requested → changes_requested → implementing → review_requested → approved → merged → verified` | First review found strict JSON/canonicalization gaps, an imprecise schema boundary, count-only readiness, non-finite vectors, retained error details, and stale evidence; head `febee94` repaired those and integrated the merged typed provider. [Second re-review](https://github.com/JapmeetBrar/pufferlab/pull/4#issuecomment-5379439529) found that readiness still used local acknowledgements and approximate metadata, SDK optional noise was not normalized, the observed ANN metric was injected rather than read, and the FTS byte limit was one too high. Head `71bfc6a` added independently observed strong ordered IDs and exact Count aggregation, real-SDK metadata normalization, observed metric/schema drift checks, and the correct 254-byte FTS limit. Dedicated re-review approved it; PR #4 merged as `7086d2b` and protected post-merge checks passed. |
| 2026-08-22 | M1-C / [PR #6](https://github.com/JapmeetBrar/pufferlab/pull/6) | `implementing → review_requested → changes_requested → implementing → review_requested → changes_requested → implementing → review_requested → approved → merged → verified` | The worker normally merged M1-B from protected `main`, integrated the exact manifest and compiled schema hash without casts, and added the production-shaped BM25/ANN compare service and API. [Initial review](https://github.com/JapmeetBrar/pufferlab/pull/6#issuecomment-5379587091) found that filter overrides lacked schema-bound validation. The first repair added contract and service validation; [first re-review](https://github.com/JapmeetBrar/pufferlab/pull/6#issuecomment-5379634560) verified it but found that `RuntimeSearchBackend` checked missing credentials before constructing the validated service. The second repair validates with the same stored compiled write spec before `_get_service`; dedicated re-review approved it, PR #6 merged as `5ca296c`, and protected post-merge checks passed. No live credential was consumed; the isolated provider live test remains opt-in. |
| 2026-08-22 | M1-E / [PR #7](https://github.com/JapmeetBrar/pufferlab/pull/7) | `implementing → review_requested` | The worker wired the checked-in 20-document fixture to the exact pinned BGE passage embedder and merged ingestion/provider services behind one injectable CLI command. Tests cover help/preflight behavior, owned namespace generation and validation, exact manifest/model/revision, progress and success output, same-namespace idempotence, sanitized failures and exit codes, and the absence of deletion. Local gates pass with 151 backend tests, one expected live skip, and one frontend test; live credential execution remains pending. |

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
