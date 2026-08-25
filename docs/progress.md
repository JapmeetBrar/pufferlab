# PufferLab Progress Ledger

- **Active goal:** polish the public repository with a concise reader-focused README, a bounded
  retained documentation set, scoped local-artifact ignores, and removal of stale merged branches.
- **Goal status:** review requested
- **Last updated:** 2026-08-25
- **Orchestrator:** root agent
- **Dedicated reviewer:** independent reviewer agent

This file is the current execution snapshot, not a complete project diary. GitHub pull requests,
checks, and merge events are canonical for completed work. Process details live in
[`engineering-loop.md`](engineering-loop.md).

## Active review unit

| ID | Deliverable | Owner | Branch / PR | Status | Evidence / next action |
|---|---|---|---|---|---|
| REPO-1 | Public README and repository hygiene | public repository polish worker | `codex/public-repo-polish` / [PR #46](https://github.com/JapmeetBrar/pufferlab/pull/46) | verified | [Independent exact-head review](https://github.com/JapmeetBrar/pufferlab/pull/46#issuecomment-5408342991) approved the delivery head after observing a 162-line README, `9/14/0` retained Markdown files/local links/fragments, artifact audit `267/968/19`, 68 focused tests, full `make check` with 1,539 backend tests plus one opt-in skip and 114 frontend tests/build, and all six provider-free browser journeys. Reviewer-only squash merge `229751db0bd70952a45573374e8bb4719eda23ee` established the public-repository polish; the [canonical protected-main record](https://github.com/JapmeetBrar/pufferlab/pull/46#issuecomment-5408388800) confirms run [`32832145650`](https://github.com/JapmeetBrar/pufferlab/actions/runs/32832145650) passed Backend and Frontend. No credential, provider/model network, namespace/live action, or user server was used. |
| REPO-F | Public-repository polish goal finalization | finalization worker + independent reviewer | `codex/public-repo-polish-finalization` | review requested | This single finite finalization unit changes only this ledger and records the independently verified delivery evidence above. `git diff --check`, all retained Markdown links, the dataset artifact/history boundary, and full `make check` pass on the finalization head. Independent exact-head final review, reviewer-only merge, deletion of this source branch, and protected-main verification remain open; their GitHub records will be canonical and no second finalization ledger PR will follow. No credential, provider/model network, namespace/live action, or user server was used. |

## Acceptance criteria

- [x] `README.md` is a concise public guide (no more than 240 lines) covering motivation, features,
  a copy/paste provider-free quickstart, dataset/workload and recorded results, architecture,
  secrets, live Unix evaluation, limitations, checks, and relevant links.
- [x] Completed planning, milestone, verification, observability, interview-study, and personal-note
  files are removed; Git history and merged PRs preserve their record.
- [x] `.gitignore` covers scoped editor, build, coverage, model, tabular-export, and personal-note
  artifacts without hiding tracked source, configuration, or public documentation.
- [x] Every retained tracked Markdown local link and fragment resolves, with no reference to a
  removed path.
- [x] Dataset artifact/history auditing still passes after the tracked note removal, and no secret,
  corpus text, vector, credential, or local runtime artifact enters the diff.
- [x] `git diff --check`, focused documentation/CLI checks, and full `make check` pass before handoff.
- [x] REPO-1 received independent exact-head review, reviewer-only merge, source-branch deletion, and
  green protected-main Backend and Frontend checks.

Only the finalization process remains open:

- [ ] An independent reviewer inspects the exact REPO-F head and reruns risk-relevant checks.
- [ ] That reviewer alone merges the approved finalization head after required PR checks pass.
- [ ] The finalization source branch is deleted after merge.
- [ ] The resulting protected-main Backend and Frontend checks pass and remain canonical in GitHub.

## Repository hygiene outcome

- The original 40 non-`main` remote branches were matched to merged pull requests and deleted before
  REPO-1. After PR #46 merged, its `codex/public-repo-polish` source branch was also deleted. Only
  `main` existed remotely before this finalization branch was published.
- The public documentation retained by this change is intentionally small: the README, API/evidence
  contracts, Unix dataset runbook and attribution, provider-free operator runbook, engineering
  loop, progress ledger, fixture note, and repository agent policy.
- Local corpus data, queries, qrels, model artifacts, SQLite databases, exports, coverage output,
  credentials, and personal notes remain outside Git.

## Prior history

Implementation history through [PR #45](https://github.com/JapmeetBrar/pufferlab/pull/45) is
canonical in GitHub. PR #45 was independently reviewed and merged to protected `main` as
`3f21161e9b4d9cf6822a4c7ecb1788afc3fce6f6`; required Backend and Frontend checks passed. Earlier
milestone branches and documents remain recoverable from their merged PRs and commits.

## Decision notes

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-25 | Keep detailed operating steps in `synthetic-demo.md` and the Unix dataset runbook. | The README should get a new reader to first success quickly without duplicating long port, lifecycle, download, or ingestion procedures. |
| 2026-08-25 | Remove completed milestone and interview documents instead of ignoring them. | `.gitignore` cannot hide tracked files; merged Git history already preserves them, and retaining stale plans makes the public surface harder to understand. |
| 2026-08-25 | Preserve the stored-versus-new-observation boundary in concise public documentation. | The UI no longer shows low-signal original-stage notices, but documentation must still avoid presenting a later live replay as evidence from the original run. |
| 2026-08-25 | Keep `docs/engineering-loop.md` and this finite ledger. | Repository policy requires branch-based implementation, independent review, and a current handoff record. |
