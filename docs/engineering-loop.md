# PufferLab Engineering Loop

This runbook defines how PufferLab work moves from an objective to reviewed code on `main`.
The default is a small, evidence-backed pull request—not a long-lived implementation branch.

## Roles

- **Orchestrator:** owns the active goal, decomposes it into bounded tasks, assigns non-overlapping
  file ownership, tracks dependencies, and updates the progress ledger.
- **Worker:** implements one task on one branch, runs its acceptance checks, pushes the branch, opens
  a PR, and hands evidence to the reviewer. A worker does not merge its own PR.
- **Reviewer:** is independent from the worker. The reviewer reads the actual diff, checks scope,
  correctness, security, contracts, tests, and generated artifacts, then either requests precise
  fixes or merges the PR.

The reviewer role remains dedicated across the goal so review standards stay consistent. Workers
may change by task.

## Loop states

```text
planned -> assigned -> implementing -> review_requested
                                        |          |
                                        |          +-> approved -> merged -> verified
                                        |
                                        +-> changes_requested -> implementing -> review_requested
```

A task may move to `blocked` from any active state. The ledger must name the blocking condition and
the next action; “still working” is not a blocker.

Only the reviewer moves `review_requested` to `approved` or `changes_requested`. After requested
changes, the worker explicitly returns the task to `implementing`, fixes and validates the branch,
and creates a new `review_requested` handoff. Approval never follows `changes_requested` without
that implementation and re-review path.

## Finite progress ledger

`docs/progress.md` is a versioned execution snapshot, not a reason to create a new PR after every
merge. While a task PR is open, its branch records assignment, handoff, review decisions, blockers,
and the latest evidence. Once that PR merges, GitHub's immutable PR conversation, check runs, merge
event, and resulting `main` checks are canonical for that merge until the next planned task or
coordination PR brings the ledger snapshot forward.

The orchestrator batches outstanding merge and verification evidence into that next planned ledger
update. It must not create recursively self-referential “record the previous ledger merge” PRs. At
goal completion, the orchestrator opens exactly one finalization PR that records all preceding
delivery tasks as verified, adds sanitized final-verification evidence, and declares itself ready
for the goal-closing review. The finalization PR itself goes through the normal independent review
and protected merge. Its own reviewer verdict, merge event, and post-merge `main` checks remain
canonical in GitHub; no branch can predict those future events and no second finalization-ledger PR
is created.

## Branch and PR rules

1. Sync `main`, then create `codex/<task-slug>` from the required merged dependency.
2. Keep file ownership disjoint when workers run in parallel. If two tasks need the same contract,
   make the contract change a prerequisite PR.
3. Make commits explain behavior, not agent activity.
4. Push the branch and open a PR with summary, dependency/base, validation evidence, limitations,
   and rollback notes when relevant.
5. Use stacked PRs only when useful work cannot wait. State the review order in every stacked PR.
6. Never force-push after review starts unless the reviewer explicitly requests history repair.
7. Active GitHub repository ruleset `21190317` protects `main`: changes require a pull request,
   `Backend` and `Frontend` must succeed against the latest base, branch deletion and non-fast-forward
   updates are blocked, and there are no bypass actors.
8. All agents share one GitHub identity, so GitHub cannot distinguish worker from reviewer and the
   ruleset requires zero approving reviews. Extra approval for unattributed changes is explicitly
   disabled so local agent commit identities cannot introduce an unsatisfiable hidden approval.
   Independent agent review is recorded in the task handoff/ledger and enforced by role separation:
   a worker never merges its own PR.
9. Merge only through GitHub after required checks and the independent reviewer decision. Prefer a
   squash merge for a focused task; preserve merge commits only when the history itself is useful.

## Worker loop

1. Read the active goal and task card in `docs/progress.md`.
2. Inspect existing contracts and tests before editing.
3. Implement the smallest end-to-end behavior that satisfies the acceptance criteria.
4. Add or update tests in the same branch.
5. Run focused checks while iterating, then the full relevant gate before handoff.
6. Review the local diff for secrets, generated-file drift, unrelated edits, and stale documentation.
7. Update the progress ledger on the task branch and open the PR.
8. Respond to every reviewer finding with a fix or evidence-backed explanation.

## Reviewer loop

The reviewer does not rely on the worker summary alone.

1. Inspect the PR diff and its base dependency.
2. Check correctness, failure behavior, data/secret handling, API contract compatibility,
   concurrency/resource lifecycle, accessibility for UI changes, and scope discipline.
3. Rerun risk-relevant tests or add a reproducer for suspected defects.
4. Confirm generated OpenAPI and frontend types match their sources.
5. Request changes with file/line references for every blocking issue.
6. Re-review the updated head. Do not approve based only on a worker's claim that it is fixed.
7. Merge only when checks are green and no blocking finding remains.
8. Verify the default branch checks. Treat GitHub as canonical for the merge until the next planned
   ledger update or the single goal-finalization PR.

## Standard gates

Backend:

```bash
uv sync --locked
uv run ruff check backend scripts
uv run ruff format --check backend scripts
uv run mypy
uv run pytest
uv run python scripts/generate_openapi.py --check
```

Frontend:

```bash
cd web
pnpm install --frozen-lockfile
pnpm generate:api
git diff --exit-code -- src/api/schema.d.ts
pnpm lint
pnpm typecheck
pnpm test
pnpm build
```

Live turbopuffer tests are opt-in and use a test-owned namespace capability, not a caller-provided
cleanup string:

1. The test generates one immutable namespace ID internally as the reserved
   `pufferlab-live-test-` prefix plus a cryptographically random suffix, and retains that exact value
   for the full test. Namespace or cleanup targets must never come from arguments, user input, or
   environment variables; only credentials and region may come from the environment.
2. The test records creation success only after creating that exact generated namespace succeeds.
   Cleanup in `finally` may delete only the retained exact ID, only when creation success is true,
   and only after rechecking the reserved prefix. Cleanup helpers must not accept arbitrary targets.
3. Cleanup failure must be raised and visible to the test runner, including when the test body also
   failed; it must never be swallowed, converted to a warning, or hidden by a return from `finally`.

The API key must never be pasted into tracked files, logs, test output, screenshots, or PR text.

## Completion definition

The active goal is finished only when all of the following are true:

- The finalization PR records every preceding delivery task as `verified`, includes sanitized final
  verification evidence, and records its own state no later than `review_requested`.
- Every required PR was independently reviewed and merged.
- CI is green on `main`.
- The documented setup reproduces the intended behavior from a clean checkout.
- Remaining limitations are explicit and outside the stated goal.
- The finalization PR was independently reviewed and merged; its own verdict, merge, and post-merge
  checks are canonical in GitHub and do not require another ledger PR.
