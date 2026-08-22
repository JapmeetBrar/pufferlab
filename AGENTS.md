# PufferLab Agent Workflow

Every agent working in this repository must read
[`docs/engineering-loop.md`](docs/engineering-loop.md) and
[`docs/progress.md`](docs/progress.md) before changing files.

## Non-negotiable rules

- Never commit directly to `main`. Use one focused `codex/<task>` branch per review unit.
- A worker never approves or merges its own pull request.
- Every contribution must pass an independent reviewer loop: inspect, test, request fixes if needed,
  re-review, then merge.
- Keep `docs/progress.md` current at each handoff, review decision, merge, or blocker.
- Keep secrets server-side. Never commit `.env`, turbopuffer API keys, raw vectors, or credentials in
  fixtures, logs, test output, screenshots, or PR descriptions.
- Prefer contract-first changes. Regenerate OpenAPI and TypeScript types whenever an API contract
  changes, and fail the PR if generated files drift.
- Run the checks listed in the task's acceptance criteria before requesting review. The reviewer
  reruns risk-relevant checks independently.
- A task is complete only when its PR is merged, the default branch is green, and the progress
  ledger records the evidence.

## Handoff format

Every worker handoff must include:

1. Branch and PR URL.
2. Files and behavior changed.
3. Commands run and their results.
4. Known limitations or untested paths.
5. Exact acceptance criteria satisfied.
