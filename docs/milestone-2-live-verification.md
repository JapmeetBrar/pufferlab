# Milestone 2 live evaluation verification

Use this runbook only on `codex/m2-live-finalization`, after M2-E is merged and protected `main`
is green. The objective is one fresh 47,382-document Unix namespace, one persisted 50-query run
across BM25, ANN, server RRF, and local reranking, an independent recomputation from SQLite, and
confirmed deletion of exactly the internally generated namespace.

## Safety boundary

- Keep `TURBOPUFFER_API_KEY` only in ignored `.env`; never print it, pass it on the command line,
  add it to a `VITE_*` variable, or copy it into logs, screenshots, Git, or a PR.
- Keep the archive, processed rows, checkpoints, SQLite database, exports, model caches, and live
  evidence beneath ignored `data/` paths. Do not stage any of them.
- [`m2_live_namespace_session.py`](../scripts/m2_live_namespace_session.py) owns one immutable
  `pufferlab-unix-live-<24 lowercase hex>` identity in mode-`0600`
  `data/m2-live-session.json`. Its suffix and record tag are HMAC-derived from an internally
  generated, fixed-path mode-`0600` `data/m2-live-owner.key`; start durably `fsync`s each file and
  its directory before returning. Production start/cleanup accept no path, token, key, or namespace
  injection. Cleanup authenticates the capability and removes the session record only after the
  provider confirms `NOT_FOUND`, closes cleanly, and the directory unlink is durable.
- The Unix ingestion command never deletes. Keep the coordinator shell and its cleanup traps alive
  from before the first remote write until explicit cleanup succeeds.
- Run the checked-in safety harness through an independent pre-live review before using the
  credential or creating a remote namespace. The same reviewer performs the final exact-head
  review and merge after sanitized evidence is recorded.

## 1. Prepare the isolated checkout

From the repository root:

```bash
git switch codex/m2-live-finalization
git check-ignore --quiet .env
! git ls-files --error-unmatch .env >/dev/null 2>&1
test "$(stat -f '%Lp' .env)" = 600
uv sync --locked --extra live-search
```

The San Francisco verification account uses `gcp-us-west1`. Confirm only key presence, region, and
resolved ignored data directory—never the key value:

```bash
uv run python - <<'PY'
from pufferlab.config import Settings

settings = Settings()
print(f"credential_configured={bool(settings.turbopuffer_api_key and settings.turbopuffer_api_key.get_secret_value())}")
print(f"region={settings.turbopuffer_region}")
print(f"data_dir={settings.pufferlab_data_dir.resolve()}")
PY
```

Require `credential_configured=True`, `region=gcp-us-west1`, and a data directory under the current
ignored checkout. If `data/m2-live-session.json` already exists, do not replace it, its owner key,
or either file's permissions; recover the retained capability with:

```bash
uv run --extra live-search python scripts/m2_live_namespace_session.py cleanup
```

## 2. Prepare and verify the pinned local pack

Resume the exact pinned download and prepare only the Unix subset:

```bash
mkdir -p data/cqadupstack-unix/source data/cqadupstack-unix/processed
curl -fL --retry 5 --retry-delay 2 -C - \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip \
  -o data/cqadupstack-unix/source/cqadupstack.zip
uv run python scripts/prepare_cqadupstack_unix.py \
  --archive data/cqadupstack-unix/source/cqadupstack.zip \
  --output-dir data/cqadupstack-unix/processed
uv run python scripts/audit_dataset_artifacts.py
```

Preparation must verify the 5,343,728,040-byte archive, published MD5
`4e41456d7df8ee7760a7f866133bda78`, and checked SHA-256 before row access. Require the exact
content-addressed directory
`cqadupstack-unix-6d54fb92c04b9f193d081a7c430d8804e24e71855d3cbaa2bb50cde838f181b8`,
47,382 documents, the checked curated 50 queries, and 83 judgments. No row content is live evidence.

## 3. Run reviewed preflight checks

After the namespace/verifier harness has passed its independent pre-live review, verify the real
pinned embedding runtime and the self-cleaning provider smoke test:

```bash
uv run --extra live-search python scripts/verify_real_embeddings.py
PUFFERLAB_RUN_LIVE=1 uv run --extra live-search pytest \
  backend/tests/live/test_turbopuffer_live.py -q
```

The smoke test owns a separate internally generated test namespace and must pass real write, BM25,
ANN, server-RRF parity, and exact `NOT_FOUND` cleanup. A missing credential after opt-in is a failure,
not a skip.

## 4. Start the recoverable namespace session

Use Bash for the trap behavior below. Keep this shell open:

```bash
bash
set -Eeuo pipefail
test ! -e data/m2-live-session.json

_cleanup_namespace() {
  uv run --extra live-search python scripts/m2_live_namespace_session.py cleanup
}
cleanup_live() {
  trap - EXIT INT TERM HUP
  _cleanup_namespace
}
_cleanup_on_exit() {
  local prior_status=$?
  trap - EXIT INT TERM HUP
  if ! _cleanup_namespace; then
    exit 1
  fi
  exit "$prior_status"
}
_cleanup_on_signal() {
  local signal_status="$1"
  trap - EXIT INT TERM HUP
  if ! _cleanup_namespace; then
    exit 1
  fi
  exit "$signal_status"
}

uv run python scripts/m2_live_namespace_session.py start
trap _cleanup_on_exit EXIT
trap '_cleanup_on_signal 130' INT
trap '_cleanup_on_signal 143' TERM
trap '_cleanup_on_signal 129' HUP
LIVE_NAMESPACE="$(
  uv run python scripts/m2_live_namespace_session.py show |
    awk -F= '/^PUFFERLAB_SEARCH_NAMESPACE=/{print $2}'
)"
readonly LIVE_NAMESPACE
[[ "$LIVE_NAMESPACE" =~ ^pufferlab-unix-live-[0-9a-f]{24}$ ]]
```

The short interval between local record creation and trap registration cannot create a remote
resource. File and directory `fsync` complete before `start` returns, so a later power loss retains
the authenticated ignored record and owner key for recovery. Never copy a record between checkouts:
the fixed local owner key deliberately makes a foreign otherwise-valid namespace fail closed.

## 5. Ingest the exact pack and persist the seed

Still in the coordinator shell:

```bash
readonly PROCESSED_PACK="data/cqadupstack-unix/processed/cqadupstack-unix-6d54fb92c04b9f193d081a7c430d8804e24e71855d3cbaa2bb50cde838f181b8"
uv run --extra live-search pufferlab dataset ingest-unix \
  --processed-pack "$PROCESSED_PACK" \
  --namespace "$LIVE_NAMESPACE"
```

Require the final remote readiness observation to bind the exact document count, IDs, schema,
distance metric, dataset revision, curated query set, and four immutable configuration IDs. Progress
may contain counts and identifiers only. If the command is interrupted, rerun the same command in
the same shell: checkpointed UUID upserts resume without a delete or a new namespace.

## 6. Execute, export, and independently verify the durable run

Capture only the CLI's contract-safe output in memory so the run ID can be reused without creating
a text-bearing evidence file:

```bash
EVAL_OUTPUT="$(uv run --extra live-search pufferlab eval run --seeded-defaults)"
printf '%s\n' "$EVAL_OUTPUT"
RUN_ID="$(printf '%s\n' "$EVAL_OUTPUT" | sed -n 's/^run_id=\([^ ]*\) status=completed.*/\1/p')"
readonly RUN_ID
unset EVAL_OUTPUT
[[ "$RUN_ID" =~ ^[0-9a-f-]{36}$ ]]

uv run pufferlab eval export "$RUN_ID" --output "exports/$RUN_ID.json"
uv run python scripts/verify_m2_evaluation.py "$RUN_ID"
```

Require CLI exit zero, terminal `completed`, 50/50 queries, four summaries, 200 typed successful
outcomes, and zero error rate for every configuration. The independent verifier accepts only the
run UUID; it authenticates the fixed live session, reloads the exact checked source/processed locks
and content-addressed pack, derives the full READY dataset, curated query/qrel set, and
manifest-bound four-config catalog, and requires contract equality with SQLite. It recomputes every
query metric from ranked IDs plus exact qrels before independently aggregating quality and latency,
compares the result with stored per-query metrics and summaries, validates the canonical export,
and prints no query/document text, vectors, or exception details.

## 7. Prove the artifact and secret boundaries

```bash
make check
uv run python scripts/audit_dataset_artifacts.py
uv run python scripts/verify_secret_boundaries.py
git diff --check
git status --short
```

The production frontend build from `make check` is required before the exact secret verifier. Inspect
`git status` path-by-path: only intended harness, tests, runbook, and ledger files may be tracked or
staged. The archive, processed pack, session record, checkpoints, database, export, caches, and any
runtime logs must remain ignored. Never add a sample export from the real run.

## 8. Confirm exact cleanup

After all durable verification succeeds, clean explicitly in the same coordinator shell:

```bash
SESSION_FINGERPRINT="$(
  printf '%s' "$LIVE_NAMESPACE" |
    shasum -a 256 |
    awk '{print substr($1,1,12)}'
)"
readonly SESSION_FINGERPRINT
cleanup_live
test ! -e data/m2-live-session.json
test -f data/m2-live-owner.key
printf 'namespace_fingerprint=%s cleanup=not_found_verified\n' "$SESSION_FINGERPRINT"
```

`cleanup_live` disarms every trap before deleting exactly the authenticated retained namespace. A
cleanup failure must leave the session record in place and fail the run; repair connectivity and
rerun the cleanup command. The ignored owner key remains for future locally owned sessions. Never
substitute another name, hand-author a record, copy one from another checkout, or delete either
ownership file manually during an active session.

## 9. Final evidence and review

Record only the following sanitized facts in [`progress.md`](progress.md): pinned archive/pack
hashes and counts; embedding and provider-smoke pass; region; a short namespace SHA-256 fingerprint;
dataset/query/config/run UUIDs; 50/200 coverage; recomputed metric aggregates and sample counts;
secret/artifact scan counts; exact `NOT_FOUND` cleanup; full local gates; and links to the pre-live
and final reviewer decisions. Do not record the namespace, key, source text, qrels, vectors, raw
provider bodies, database, export, or local filesystem evidence paths containing licensed data.

The finalization PR records its own state no later than `review_requested`. It must not predict its
approval, merge SHA, or post-merge checks. The dedicated reviewer inspects the exact immutable head,
merges it, and verifies protected `main`; GitHub remains canonical for those final events, so no
recursive ledger-only PR is opened.
