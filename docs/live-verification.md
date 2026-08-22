# Milestone 1 Live Verification

Use this runbook on the `codex/m1-live-verification` finalization branch to prove the complete
browser → FastAPI → turbopuffer → browser path against the checked-in 20-document fixture. Keep one
coordinator shell open from namespace creation through cleanup. A successful run includes the real
embedding model, isolated provider smoke test, ingestion and idempotent rerun, direct API check,
interactive desktop/mobile browser check, exact secret scan, confirmed namespace deletion, full
gates, and independent review of the final immutable PR head.

## Safety invariants

- Put `TURBOPUFFER_API_KEY` only in the ignored local `.env`. Never paste it into a command, log,
  screenshot, PR, or `VITE_*` variable.
- [`live_namespace_session.py`](../scripts/live_namespace_session.py) generates and records one
  immutable `pufferlab-tiny-<24 hex>` namespace in ignored `data/m1-live-session.json`. Its cleanup
  command accepts no namespace argument and refuses a malformed or substituted record.
- Create only one live session at a time. If a prior session record exists, recover it with the
  cleanup command before starting another run.
- Keep identifiers, counts, status, ranks, typed scores, and client timings. Do not retain API keys,
  request headers, query vectors, stored vectors, or `.env` contents.

## 1. Prepare the finalization checkout

From the repository root:

```bash
cp .env.example .env
chmod 600 .env
uv sync --locked --extra live-search
cd web && pnpm install --frozen-lockfile && cd ..
```

Add only the local key and account region to `.env`; leave the namespace blank because the live
session owns it:

```dotenv
TURBOPUFFER_API_KEY=<local-only>
TURBOPUFFER_REGION=gcp-us-central1
PUFFERLAB_SEARCH_NAMESPACE=
```

Confirm `.env` is ignored without printing it:

```bash
git check-ignore --quiet .env
! git ls-files --error-unmatch .env >/dev/null 2>&1
```

If `data/m1-live-session.json` already exists, do not remove or replace it manually. Run:

```bash
uv run --extra live-search python scripts/live_namespace_session.py cleanup
```

The retained record makes an interrupted run recoverable. Cleanup removes the record only after
turbopuffer returns the specific `NOT_FOUND` condition for that exact namespace.

## 2. Prove the exact real embedding model

Run the checked-in reproducer, which loads the manifest-pinned model revision through the same
query and passage embedding classes used by the server and ingestion command:

```bash
uv run --extra live-search python scripts/verify_real_embeddings.py
```

It embeds fixture query `query-002` plus passages `tiny-002` and `tiny-005`, rejects the wrong
model/revision/dimensions, non-finite values, or non-unit vectors, and prints no vector values. The
safe success output is:

```text
real_embedding_verification=passed
model=BAAI/bge-small-en-v1.5
revision=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
dimensions=384
query_norm=1.000000
passage_norms=1.000000,1.000000
```

This command—not fake-encoder unit tests—is the real-model evidence.

## 3. Run the isolated provider smoke test

```bash
PUFFERLAB_RUN_LIVE=1 uv run --extra live-search pytest \
  backend/tests/live/test_turbopuffer_live.py -q
```

The test loads credentials from the launching environment or ignored `.env`. Once explicitly
enabled, a missing key is a failure rather than a skipped-success result. The test generates its
own `pufferlab-live-test-*` namespace, proves real write/BM25/ANN score and timing behavior, deletes
the exact internally retained namespace in `finally`, polls metadata until `NOT_FOUND`, and always
closes the provider. Require `1 passed` and exit code zero.

## 4. Start a failure-safe browser namespace session

In the coordinator shell, enable exit-on-error before creating any resource:

```bash
set -Eeuo pipefail
uv run python scripts/live_namespace_session.py start
LIVE_NAMESPACE="$(
  uv run python scripts/live_namespace_session.py show |
    awk -F= '/^PUFFERLAB_SEARCH_NAMESPACE=/{print $2}'
)"
readonly LIVE_NAMESPACE
[[ "$LIVE_NAMESPACE" =~ ^pufferlab-tiny-[0-9a-f]{24}$ ]]

cleanup_live() {
  trap - EXIT INT TERM HUP
  uv run --extra live-search python scripts/live_namespace_session.py cleanup
}
trap cleanup_live EXIT INT TERM HUP
```

Keep this shell open. Any ordinary error, cancellation, or shell exit now invokes guarded cleanup.
A power loss cannot run a trap, so the ignored session record remains for the recovery command in
step 1. Never delete that record manually.

## 5. Ingest once and rerun idempotently

Still in the coordinator shell:

```bash
uv run --extra live-search pufferlab dataset ingest-tiny \
  --namespace "$LIVE_NAMESPACE" | tee data/m1-ingest-first.log
uv run --extra live-search pufferlab dataset ingest-tiny \
  --namespace "$LIVE_NAMESPACE" | tee data/m1-ingest-rerun.log
```

Each successful run prints a `verified` line derived from live remote inspection. Require:

- `remote_documents=20` and `exact_document_ids=true`;
- the same `observed_schema_hash` on both runs;
- `distance_metric=cosine_distance`;
- `metadata_ready=true` and `indexes_ready=true`;
- a final `ready ... documents=20` line for the same namespace.

The command does not print the UUID inventory or SDK metadata body. Internally, success requires
the independently queried strong-consistency ID set and exact Count aggregation to equal the 20
deterministic fixture UUIDs, and the observed schema—including ANN metric—to equal the compiled
fixture schema. The second stable-ID upsert proves idempotence when those checks remain identical.

## 6. Start FastAPI and the browser app

In a second terminal at the repository root, resolve the namespace from the immutable session and
start FastAPI:

```bash
LIVE_NAMESPACE="$(
  uv run python scripts/live_namespace_session.py show |
    awk -F= '/^PUFFERLAB_SEARCH_NAMESPACE=/{print $2}'
)"
PUFFERLAB_SEARCH_NAMESPACE="$LIVE_NAMESPACE" \
  uv run --extra live-search uvicorn pufferlab.main:app \
    --app-dir backend --host 127.0.0.1 --port 8000
```

In a third terminal:

```bash
cd web
pnpm dev --host 127.0.0.1 --port 5173
```

Back in the coordinator shell, execute the exact public API proof:

```bash
uv run python scripts/verify_live_api.py | tee data/m1-live-api.json
```

[`verify_live_api.py`](../scripts/verify_live_api.py) accepts only an uncredentialed loopback HTTP
origin. It checks health and configuration discovery, then posts fixture query `query-002`—“How
can I find the program listening on port 8080?”—with exactly contract version, query text, BM25 and
vector config IDs, and the debug flag. It requires non-empty ordered results, document IDs and
external IDs, contiguous 1-based ranks, typed score direction, provider/total wall-clock timings,
and no API-key, authorization, query-vector, or stored-vector response field.

## 7. Exercise the real browser path

Open `http://127.0.0.1:5173`, enter the exact `query-002` text from step 6, retain BM25 on the left
and vector on the right, and select **Compare results**. Verify all of the following against the
successful network response and rendered page:

1. The browser POST has only `contract_version`, `query_text`, `config_ids`, and
   `debug_provenance`; it contains no API key, authorization header, or vector.
2. Both result columns are non-empty. Every displayed hit includes identity, visible 1-based rank,
   score value/kind/direction, and client-wall-clock timing.
3. The page shows the observability notice without claiming unexposed provider internals.
4. The URL contains stable `q`, `left`, and `right` parameters; reloading restores the choices.
5. A desktop viewport uses two readable columns, while a 390-pixel viewport stacks them without
   clipped controls or horizontal page scrolling.

After inspecting the image for secrets, save desktop and mobile screenshots under ignored `data/`.
Do not capture browser request headers, `.env`, terminals containing credentials, or model vectors.

## 8. Prove the secret boundary and run all gates

Build the production browser assets before the exact secret scan:

```bash
cd web && pnpm build && cd ..
uv run python scripts/verify_secret_boundaries.py
```

The verifier reads the key through `SecretStr` without printing it. It requires `.env` to be ignored
and untracked, rejects the exact local key in the tracked worktree, all Git blob history, frontend
source, or production build, and rejects browser credential/query-vector field names. The API
verifier and browser network check separately prove the runtime request/response boundary.

Run the complete repository gates:

```bash
uv run ruff check backend scripts
uv run ruff format --check backend scripts
uv run mypy
uv run pytest
uv run python scripts/generate_openapi.py --check

cd web
pnpm generate:api
git diff --exit-code -- src/api/schema.d.ts
pnpm lint
pnpm typecheck
pnpm test
pnpm build
cd ..
git diff --check
```

## 9. Stop servers and confirm exact cleanup

Stop FastAPI and Vite with Ctrl-C. In the coordinator shell, explicitly run the same cleanup that
is already registered for error/cancellation:

```bash
cleanup_live
test ! -e data/m1-live-session.json
```

Require `status=not_found_verified`. The coordinator validates the exact recorded random name,
deletes only that namespace, polls its metadata for the specific provider `NOT_FOUND` code, closes
the provider in `finally`, and removes the session record only after confirmation. If cleanup fails,
the record remains: do not start a new session; repair connectivity and rerun the cleanup command.

## 10. Record evidence and complete the review loop

Update [`progress.md`](progress.md) with only sanitized evidence:

- the real-model verifier command/result and a PR comment link containing its safe output;
- provider smoke-test pass and confirmed test-namespace cleanup;
- a non-secret hash fingerprint of the browser namespace, both 20-document verified ingestion
  lines, and idempotent rerun result;
- the API verifier summary, exact browser assertions, and ignored screenshot paths;
- the secret-boundary result, full local gates, and protected PR checks;
- the early reviewer blocker link, repair head, and final review handoff.

Do not make this finalization PR predict or record its own future approval, merge commit, or
post-merge `main` run. GitHub's review, merge event, and protected-main checks are canonical for
those events. Mark the draft ready only after every live item above is complete, then require the
dedicated reviewer to inspect and test the exact immutable head, merge it, and verify `main`. Do not
open a recursive ledger PR solely to record that final merge.
