# PufferLab Agent Workflow

Every agent working in this repository must read
[`docs/engineering-loop.md`](docs/engineering-loop.md) and
[`docs/progress.md`](docs/progress.md) before changing files.

## Non-negotiable rules

- Never commit directly to `main`. Use one focused `codex/<task>` branch per review unit.
- A worker never approves or merges its own pull request.
- Every contribution must pass an independent reviewer loop: inspect, test, request fixes if needed,
  re-review, then merge.
- Keep `docs/progress.md` current on the task branch at each assignment, handoff, review decision, or
  blocker. After a PR merges, its GitHub PR, check, and merge history is canonical until the next
  planned ledger update; never open an unbounded chain of ledger-only PRs.
- Keep secrets server-side. Never commit `.env`, turbopuffer API keys, raw vectors, or credentials in
  fixtures, logs, test output, screenshots, or PR descriptions.
- Prefer contract-first changes. Regenerate OpenAPI and TypeScript types whenever an API contract
  changes, and fail the PR if generated files drift.
- Run the checks listed in the task's acceptance criteria before requesting review. The reviewer
  reruns risk-relevant checks independently.
- `main` is protected by active GitHub ruleset `21190317`: pull requests and successful `Backend`
  and `Frontend` checks are required, with no bypass actors. Because all agents use one GitHub
  identity, the required approval count is zero and extra approval for unattributed changes is
  disabled; independent agent review and the reviewer-only merge rule remain mandatory process
  controls.
- A delivery task is complete when its PR is independently reviewed and merged and the default
  branch is green. GitHub is the evidence source until the next ledger update. The goal closes with
  one reviewed finalization PR; its own merge record remains canonical in GitHub.

## Handoff format

Every worker handoff must include:

1. Branch and PR URL.
2. Files and behavior changed.
3. Commands run and their results.
4. Known limitations or untested paths.
5. Exact acceptance criteria satisfied.
