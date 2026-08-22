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
                                      |             |
                                      |             v
                                      +<-- changes_requested
                                                    |
                                                    v
                                  approved -> merged -> verified
```

A task may move to `blocked` from any active state. The ledger must name the blocking condition and
the next action; “still working” is not a blocker.

## Branch and PR rules

1. Sync `main`, then create `codex/<task-slug>` from the required merged dependency.
2. Keep file ownership disjoint when workers run in parallel. If two tasks need the same contract,
   make the contract change a prerequisite PR.
3. Make commits explain behavior, not agent activity.
4. Push the branch and open a PR with summary, dependency/base, validation evidence, limitations,
   and rollback notes when relevant.
5. Use stacked PRs only when useful work cannot wait. State the review order in every stacked PR.
6. Never force-push after review starts unless the reviewer explicitly requests history repair.
7. Merge only through GitHub after required checks and reviewer approval. Prefer squash merge for a
   focused task; preserve merge commits only when the branch history itself is useful.

## Worker loop

1. Read the active goal and task card in `docs/progress.md`.
2. Inspect existing contracts and tests before editing.
3. Implement the smallest end-to-end behavior that satisfies the acceptance criteria.
4. Add or update tests in the same branch.
5. Run focused checks while iterating, then the full relevant gate before handoff.
6. Review the local diff for secrets, generated-file drift, unrelated edits, and stale documentation.
7. Update the progress ledger and open the PR.
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
8. Verify the default branch checks and update `docs/progress.md` with the merge evidence.

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

Live turbopuffer tests are opt-in and must use a uniquely owned namespace, redact credentials, and
clean up in `finally`. A live test may require the user to provide `TURBOPUFFER_API_KEY` locally; the
key must never be pasted into tracked files or command output.

## Completion definition

The active goal is finished only when all of the following are true:

- Every required task is `verified` in `docs/progress.md`.
- Every required PR was independently reviewed and merged.
- CI is green on `main`.
- The documented setup reproduces the intended behavior from a clean checkout.
- Remaining limitations are explicit and outside the stated goal.
