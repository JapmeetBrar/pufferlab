# PufferLab

PufferLab is a search evaluation and query-forensics workbench for turbopuffer. It compares lexical, vector, hybrid, and reranked retrieval; runs judged query sets; surfaces regressions; and opens failures in an evidence-based debugger.

The local product now includes a durable SQLite-backed run history, aggregate and per-query
regression analysis, a provider-free quality gate, guided live-search setup, stable query deep
links, an evidence-honest forensic drawer, and explicit live actions. Stored-run pages are
provider-free; only clearly labeled actions can start provider work. See:

- [Complete study and presentation guide](docs/pufferlab-study-guide.md)
- [Project decision and implementation brief](docs/project-decision-and-implementation-brief.md)
- [Shared contracts](docs/contracts.md)
- [Implementation plan](docs/implementation-plan.md)
- [Milestone 3 execution plan](docs/milestone-3-execution.md)
- [Provider-free operator runbook](docs/synthetic-demo.md)
- [Observability and demo runbook](docs/observability.md)
- [Historical Milestone 1 live-verification record](docs/live-verification.md)
- [CQADupStack Unix local-pack runbook](docs/datasets/cqadupstack-unix.md)

## Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) with the repository's locked environment
- Node.js 22 or newer
- pnpm 11 (the repository pins pnpm 11.19.0)

If `pnpm` is not installed, install the repository's pinned major before continuing:

```bash
npm install --global pnpm@11
```

Confirm the tools before continuing:

```bash
python3 --version
uv --version
node --version
pnpm --version
```

## Five-minute provider-free workflow

Install the locked Python and browser dependencies from the repository root:

```bash
uv sync --locked
cd web
pnpm install --frozen-lockfile
cd ..
```

Choose one ignored local data directory. Keep this exact value in every shell that reads the demo:

```bash
export PUFFERLAB_DATA_DIR="$PWD/data/demo-m4"
export TURBOPUFFER_API_KEY=
export TURBOPUFFER_REGION=
export PUFFERLAB_SEARCH_NAMESPACE=
SEED_OUTPUT="$(uv run pufferlab demo seed)"
printf '%s\n' "$SEED_OUTPUT"

RUN_ID="$(printf '%s\n' "$SEED_OUTPUT" | awk -F'[ =]' '/^run_id=/{print $2}')"
CONFIG_IDS="$(printf '%s\n' "$SEED_OUTPUT" | awk -F'config_ids=' '/^run_id=/{print $2}')"
RERANKER_ID="$(printf '%s\n' "$CONFIG_IDS" | cut -d, -f4)"
test -n "$RUN_ID"
test -n "$RERANKER_ID"

uv run pufferlab doctor --mode demo
```

The seed command creates the directory when needed and writes one complete 50-query, four-config,
200-outcome run. It requires no `.env`, API key, model download, provider, or network access.
`doctor` is also provider-free unless `--live` is explicitly supplied.

Evaluate the same authenticated durable evidence twice. The first command uses text output and
passes after allowing the one synthetic no-positive-qrel exclusion. The second uses JSON and proves
that policy failure has a distinct exit:

```bash
PASS_EXIT=0
uv run pufferlab eval gate "$RUN_ID" \
  --candidate "$RERANKER_ID" \
  --min-paired-queries 49 \
  --format text || PASS_EXIT=$?
test "$PASS_EXIT" -eq 0

POLICY_EXIT=0
uv run pufferlab eval gate "$RUN_ID" \
  --candidate "$RERANKER_ID" \
  --min-delta 1 \
  --min-paired-queries 49 \
  --format json || POLICY_EXIT=$?
test "$POLICY_EXIT" -eq 4
```

`eval gate` exits `0` for a passing policy, `4` for valid evidence that fails policy, `2` for an
invalid policy or evidence/binding, and `1` for an internal failure. It accepts only a completed
canonical 50-query/four-config run and never starts a provider, model, migration, recovery, or
database write. Latency is deliberately not a gate input. Persisted ranked document UUIDs are
trusted local evidence, not cryptographically authenticated protection against a malicious full-
database rewrite.

Allocate two distinct currently free loopback ports, excluding `8000` and `5173`, and save only
their validated numbers in a mode-0600 ignored data file:

```bash
(
set -eu
export PUFFERLAB_PORT_FILE="$PWD/data/pufferlab-m4-ports.txt"
uv run python - <<'PY'
import os
import socket
import stat
from pathlib import Path

blocked = {8000, 5173}
sockets = []
ports = []
try:
    while len(ports) < 2:
        candidate = socket.socket()
        candidate.bind(("127.0.0.1", 0))
        port = int(candidate.getsockname()[1])
        if port in blocked or port in ports:
            candidate.close()
            continue
        sockets.append(candidate)
        ports.append(port)
    target = Path(os.environ["PUFFERLAB_PORT_FILE"])
    flags = os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise RuntimeError("unsafe port file")
        os.fchmod(descriptor, 0o600)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{ports[0]} {ports[1]}\n".encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(f"api_port={ports[0]} web_port={ports[1]}")
finally:
    for held in sockets:
        held.close()
PY
PUFFERLAB_EXTRA_PORT=
IFS=' ' read -r PUFFERLAB_API_PORT PUFFERLAB_WEB_PORT PUFFERLAB_EXTRA_PORT \
  < "$PUFFERLAB_PORT_FILE"
test -n "$PUFFERLAB_API_PORT"
test -n "$PUFFERLAB_WEB_PORT"
test -z "$PUFFERLAB_EXTRA_PORT"
case "$PUFFERLAB_API_PORT" in *[!0-9]*) exit 1 ;; esac
case "$PUFFERLAB_WEB_PORT" in *[!0-9]*) exit 1 ;; esac
test "$PUFFERLAB_API_PORT" -ge 1
test "$PUFFERLAB_API_PORT" -le 65535
test "$PUFFERLAB_WEB_PORT" -ge 1
test "$PUFFERLAB_WEB_PORT" -le 65535
test "$PUFFERLAB_API_PORT" != "$PUFFERLAB_WEB_PORT"
test "$PUFFERLAB_API_PORT" != 8000
test "$PUFFERLAB_API_PORT" != 5173
test "$PUFFERLAB_WEB_PORT" != 8000
test "$PUFFERLAB_WEB_PORT" != 5173
)
```

In terminal 1, use the same data directory, blank live settings again, load those ports, and start
the installed one-worker server. A bind collision fails rather than silently choosing another port:

```bash
(
set -eu
export PUFFERLAB_DATA_DIR="$PWD/data/demo-m4"
export TURBOPUFFER_API_KEY=
export TURBOPUFFER_REGION=
export PUFFERLAB_SEARCH_NAMESPACE=
export PUFFERLAB_PORT_FILE="$PWD/data/pufferlab-m4-ports.txt"
PUFFERLAB_EXTRA_PORT=
IFS=' ' read -r PUFFERLAB_API_PORT PUFFERLAB_WEB_PORT PUFFERLAB_EXTRA_PORT \
  < "$PUFFERLAB_PORT_FILE"
test -n "$PUFFERLAB_API_PORT"
test -n "$PUFFERLAB_WEB_PORT"
test -z "$PUFFERLAB_EXTRA_PORT"
case "$PUFFERLAB_API_PORT" in *[!0-9]*) exit 1 ;; esac
case "$PUFFERLAB_WEB_PORT" in *[!0-9]*) exit 1 ;; esac
test "$PUFFERLAB_API_PORT" -ge 1
test "$PUFFERLAB_API_PORT" -le 65535
test "$PUFFERLAB_WEB_PORT" -ge 1
test "$PUFFERLAB_WEB_PORT" -le 65535
test "$PUFFERLAB_API_PORT" != "$PUFFERLAB_WEB_PORT"
test "$PUFFERLAB_API_PORT" != 8000
test "$PUFFERLAB_API_PORT" != 5173
test "$PUFFERLAB_WEB_PORT" != 8000
test "$PUFFERLAB_WEB_PORT" != 5173
export PUFFERLAB_API_PORT PUFFERLAB_WEB_PORT
export PUFFERLAB_CORS_ORIGINS="http://127.0.0.1:$PUFFERLAB_WEB_PORT"
exec uv run pufferlab serve --host 127.0.0.1 --port "$PUFFERLAB_API_PORT"
)
```

In terminal 2, return to the repository root, load the same ports, build against that exact API
origin, and run a strict preview:

```bash
(
set -eu
export PUFFERLAB_DATA_DIR="$PWD/data/demo-m4"
export PUFFERLAB_PORT_FILE="$PWD/data/pufferlab-m4-ports.txt"
PUFFERLAB_EXTRA_PORT=
IFS=' ' read -r PUFFERLAB_API_PORT PUFFERLAB_WEB_PORT PUFFERLAB_EXTRA_PORT \
  < "$PUFFERLAB_PORT_FILE"
test -n "$PUFFERLAB_API_PORT"
test -n "$PUFFERLAB_WEB_PORT"
test -z "$PUFFERLAB_EXTRA_PORT"
case "$PUFFERLAB_API_PORT" in *[!0-9]*) exit 1 ;; esac
case "$PUFFERLAB_WEB_PORT" in *[!0-9]*) exit 1 ;; esac
test "$PUFFERLAB_API_PORT" -ge 1
test "$PUFFERLAB_API_PORT" -le 65535
test "$PUFFERLAB_WEB_PORT" -ge 1
test "$PUFFERLAB_WEB_PORT" -le 65535
test "$PUFFERLAB_API_PORT" != "$PUFFERLAB_WEB_PORT"
test "$PUFFERLAB_API_PORT" != 8000
test "$PUFFERLAB_API_PORT" != 5173
test "$PUFFERLAB_WEB_PORT" != 8000
test "$PUFFERLAB_WEB_PORT" != 5173
export PUFFERLAB_API_PORT PUFFERLAB_WEB_PORT
cd web
VITE_API_BASE_URL="http://127.0.0.1:$PUFFERLAB_API_PORT" pnpm build
exec pnpm exec vite preview --host 127.0.0.1 --port "$PUFFERLAB_WEB_PORT" --strictPort
)
```

Open `http://127.0.0.1:<web-port>/runs` with the printed web port, then use this interview flow:

1. Open the run labeled **Synthetic demo**. Its durable metrics and all provider-free reads come
   from `data/demo-m4/pufferlab.sqlite3`.
2. In **Regressions and gains**, change candidate/order/row controls. The run URL records those
   choices as `candidate`, `order`, and `limit` query parameters.
3. Choose **Inspect recorded query** on a regression. The server-issued URL is
   `/playground?run=<uuid>&query=<uuid>&left=<uuid>&right=<uuid>`; it contains identities, not query
   text.
4. Choose **Inspect document** to open the forensic drawer. Its URL adds only a `document=<uuid>`.
   Refresh, use Back to close it, then Forward to restore it.
5. Confirm that original stage evidence is `NOT_OBSERVABLE`, synthetic timing is unavailable, and
   live replay is disabled for this read-only origin.
6. Open the Playground. Its guided capability state is `action_required` without live settings,
   the compare control stays disabled, and the browser sends no comparison request.

The equivalent recorded-query route is
`/runs/<run-uuid>/queries/<query-uuid>?left=<uuid>&right=<uuid>[&document=<uuid>]`. Merely opening,
refreshing, or navigating either form performs GET-only durable reads and never starts provider
work. See [the demo runbook](docs/synthetic-demo.md) for idempotence and cleanup boundaries.

## Ingest the tiny fixture

Prepare the ignored server-only settings file without replacing an existing credential file:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
test ! -L .env
test -e .env || cp .env.example .env
test -f .env
git check-ignore --quiet .env
if git ls-files --error-unmatch .env >/dev/null 2>&1; then exit 1; else test "$?" -eq 1; fi
chmod 600 .env
if grep -Eq '^PUFFERLAB_SEARCH_NAMESPACE=.+$' .env; then exit 1; else test "$?" -eq 1; fi
)
```

Edit `.env`, add the API key, and keep or replace the region for the account. Leave
`PUFFERLAB_SEARCH_NAMESPACE` blank, then run:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
test ! -L .env
test -f .env
git check-ignore --quiet .env
if git ls-files --error-unmatch .env >/dev/null 2>&1; then exit 1; else test "$?" -eq 1; fi
if grep -Eq '^PUFFERLAB_SEARCH_NAMESPACE=.+$' .env; then exit 1; else test "$?" -eq 1; fi
uv sync --locked --extra live-search
uv run pufferlab doctor --mode live-tiny
uv run pufferlab dataset ingest-tiny
uv run pufferlab namespace show-tiny
)
```

The first default doctor is local and provider-free. With no prior per-user receipt it reports
`action_required`; an existing receipt may instead be resumable or report a credential/region
mismatch. The ingestion command explains the creating region, generated namespace, schema hash,
pinned embedding model, and 20-document write before constructing the model or provider. It durably
records one generated target for this OS user before the first provider action. `show-tiny` reads
that authenticated local receipt without contacting turbopuffer and prints the exact server-side
assignments to copy into `.env`, for example:

```dotenv
TURBOPUFFER_REGION=gcp-us-west1
PUFFERLAB_SEARCH_NAMESPACE=pufferlab-tiny-0123456789abcdef01234567
```

After copying both assignments, confirm the guided local state with the provider-free default:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
test ! -L .env
test -f .env
git check-ignore --quiet .env
if git ls-files --error-unmatch .env >/dev/null 2>&1; then exit 1; else test "$?" -eq 1; fi
uv run pufferlab doctor --mode live-tiny
)
```

Only when a potentially billable live check is explicitly intended, run it separately:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
test ! -L .env
test -f .env
git check-ignore --quiet .env
if git ls-files --error-unmatch .env >/dev/null 2>&1; then exit 1; else test "$?" -eq 1; fi
uv run pufferlab doctor --mode live-tiny --live
)
```

The default command is provider-free. `--live` makes one bounded, potentially billable metadata
request; success still does not prove schema, exact corpus identity, or working BM25/ANN retrieval.

Generated ownership exists only when `ingest-tiny` omits `--namespace`. Re-run that same argument-
free command to resume the single authenticated receipt idempotently; do not pass the printed name
back as an argument. The receipt is bound to the exact credential value and creating region, lives
at PufferLab's fixed per-user state location rather than `PUFFERLAB_DATA_DIR`, and must not be edited
or removed manually.

An explicit target is a separate caller-managed workflow:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
uv run pufferlab dataset ingest-tiny --namespace pufferlab-tiny-0123456789abcdef01234567
)
```

Explicit targets must be safe `pufferlab-*` names, are never recorded as PufferLab-owned, and can
never be deleted by `cleanup-tiny`, even if their spelling resembles a generated name.

When the generated namespace is no longer needed, keep the exact creating credential available and
run the target-free cleanup command:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
uv run pufferlab namespace cleanup-tiny
)
```

Cleanup acts only on the authenticated fixed receipt, durably records `cleanup_requested` before
delete, and verifies not-found before removing the receipt. Cleanup contacts the provider, and its
bounded metadata verification may be billed as zero-row queries. Before the terminal fixed move, a
nonzero exit retains recoverable state: do not delete the receipt or owner key, restore the creating
credential if it was rotated, and rerun the same command. A retained `not_found_verified` receipt
reruns without a provider and finishes local removal. After the fixed move, remote absence is
already committed and retry remains provider-free, but a wipe/fsync crash may leave only an ignored
quarantine and a rerun may report a missing receipt; never restore that quarantine as authority.
The owner key remains for future generated ownership.

After successful cleanup, remove the stale `PUFFERLAB_SEARCH_NAMESPACE` line from `.env`, clear any
shell override, and restart the API before using the Playground:

```bash
unset PUFFERLAB_SEARCH_NAMESPACE
```

The ingestion and cleanup commands never print the API key or embedding vectors. Run
`uv run pufferlab dataset ingest-tiny --help` for batching and bounded readiness options.

## Run the curated Unix evaluation

First prepare the ignored CQADupStack Unix pack using the
[dataset runbook](docs/datasets/cqadupstack-unix.md). Then ingest its exact content-addressed
directory and persist the READY dataset, curated 50-query set, and four immutable configurations:

```bash
uv sync --locked --extra live-search
uv run pufferlab dataset ingest-unix \
  --processed-pack data/cqadupstack-unix/processed/cqadupstack-unix-6d54fb92c04b9f193d081a7c430d8804e24e71855d3cbaa2bb50cde838f181b8
```

The command generates a caller-managed `pufferlab-unix-*` namespace unless `--namespace` names an
existing caller-managed target for an idempotent resume. Unix targets never enter the generated-
tiny receipt and are not eligible for `namespace cleanup-tiny`. The command verifies the ignored
pack, checkpoints stable-ID writes, waits for exact remote readiness, and prints only safe
revision/configuration identities.

Run the persisted 50-query suite across BM25, ANN, server RRF, and server RRF plus the pinned local
reranker with one command:

```bash
uv run pufferlab eval run --seeded-defaults
```

Progress appears only after outcomes commit to `data/pufferlab.sqlite3`. Exit status is `0` only
when all 200 config/query attempts succeed, `3` when the run completes with coverage failures, and
nonzero for cancellation or a systemic failure. `config seed` is an idempotent way to recreate the
canonical config revisions for one persisted dataset:

```bash
uv run pufferlab config seed
```

Export completed or partial durable state beneath the ignored data directory, using the `run_id`
printed by `eval run`:

```bash
printf 'Run ID from eval run: '
read -r RUN_ID
test -n "$RUN_ID"
uv run pufferlab eval export "$RUN_ID" --output "exports/$RUN_ID.json"
```

Exports contain typed ranks, metrics, timings, warnings, and redacted failures—never query/document
text, credentials, request bodies, or vectors.

## Serve persisted live runs

For the tiny-fixture Playground, copy `show-tiny`'s assignments into `.env`. For persisted
evaluation runs, point `PUFFERLAB_DATA_DIR` at the same SQLite directory used by the CLI. Then start
the API using ports allocated by the provider-free block above:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
export PUFFERLAB_PORT_FILE="$PWD/data/pufferlab-m4-ports.txt"
PUFFERLAB_EXTRA_PORT=
IFS=' ' read -r PUFFERLAB_API_PORT PUFFERLAB_WEB_PORT PUFFERLAB_EXTRA_PORT \
  < "$PUFFERLAB_PORT_FILE"
test -n "$PUFFERLAB_API_PORT"
test -n "$PUFFERLAB_WEB_PORT"
test -z "$PUFFERLAB_EXTRA_PORT"
case "$PUFFERLAB_API_PORT" in *[!0-9]*) exit 1 ;; esac
case "$PUFFERLAB_WEB_PORT" in *[!0-9]*) exit 1 ;; esac
test "$PUFFERLAB_API_PORT" -ge 1
test "$PUFFERLAB_API_PORT" -le 65535
test "$PUFFERLAB_WEB_PORT" -ge 1
test "$PUFFERLAB_WEB_PORT" -le 65535
test "$PUFFERLAB_API_PORT" != "$PUFFERLAB_WEB_PORT"
test "$PUFFERLAB_API_PORT" != 8000
test "$PUFFERLAB_API_PORT" != 5173
test "$PUFFERLAB_WEB_PORT" != 8000
test "$PUFFERLAB_WEB_PORT" != 5173
export PUFFERLAB_API_PORT PUFFERLAB_WEB_PORT
export PUFFERLAB_CORS_ORIGINS="http://127.0.0.1:$PUFFERLAB_WEB_PORT"
exec uv run pufferlab serve --host 127.0.0.1 --port "$PUFFERLAB_API_PORT"
)
```

Health is available at `GET http://127.0.0.1:<api-port>/api/v1/health`, capabilities at
`GET http://127.0.0.1:<api-port>/api/v1/capabilities`, and interactive API documentation at
`/docs`. Health means only that the process is alive. Capability state `locally_configured` means
local requirements are present; it does not assert credential validity, remote namespace
existence, index readiness, schema, or corpus identity. `action_required` provides allowlisted
setup guidance and prevents the Playground from sending a compare request.

PufferLab's local evaluation controller deliberately supports exactly one server worker. Startup
holds an exclusive guard beside the configured `pufferlab.sqlite3`, migrates the database, marks
orphaned running jobs interrupted, and reclaims valid queued jobs oldest-first. A second API worker
fails startup instead of executing the same durable run twice. `serve` accepts only loopback hosts;
it exits `0` after graceful shutdown, `2` for invalid host/port input, and `1` for startup or
unexpected runtime failure. SIGINT/SIGTERM starts bounded shutdown; a hard ten-second watchdog may
skip Python cleanup, so the next startup's durable migration/recovery path performs reconciliation.

Run history, run detail, regressions, query detail, and export are SQLite reads and remain usable
without a provider. The live BM25-versus-vector Playground and explicit query replay use the
optional local embedding runtime and the exact persisted provider namespace. The backend loads the
fixture's pinned `BAAI/bge-small-en-v1.5` revision lazily on the first vector comparison.
`TURBOPUFFER_API_KEY` and query vectors stay inside the backend process.

Live replay is deliberately different from opening a stored run. It is available only for an exact
stored live run/query/config binding and only after the user presses **Run live replay
(cost-bearing)**. The server authenticates the complete persisted 50-query suite against the
checked source anchor before it constructs credential, embedding, reranking, or provider-capable
objects. It derives query text, graded judgments, configs, and namespace server-side; the browser
cannot supply them.

> **Cost and credential warning:** live replay can incur embedding and turbopuffer usage. Confirm
> that `.env` contains the intended server-only key and region and that the run's original namespace
> is still ready. Selecting **Include separate counterfactual provenance probes** makes additional
> provider requests. Never paste the key, licensed query text, qrels, namespace, provider bodies, or
> raw vectors into logs, screenshots, issues, or pull requests.

## Frontend

```bash
(
set -eu
export PUFFERLAB_PORT_FILE="$PWD/data/pufferlab-m4-ports.txt"
PUFFERLAB_EXTRA_PORT=
IFS=' ' read -r PUFFERLAB_API_PORT PUFFERLAB_WEB_PORT PUFFERLAB_EXTRA_PORT \
  < "$PUFFERLAB_PORT_FILE"
test -n "$PUFFERLAB_API_PORT"
test -n "$PUFFERLAB_WEB_PORT"
test -z "$PUFFERLAB_EXTRA_PORT"
case "$PUFFERLAB_API_PORT" in *[!0-9]*) exit 1 ;; esac
case "$PUFFERLAB_WEB_PORT" in *[!0-9]*) exit 1 ;; esac
test "$PUFFERLAB_API_PORT" -ge 1
test "$PUFFERLAB_API_PORT" -le 65535
test "$PUFFERLAB_WEB_PORT" -ge 1
test "$PUFFERLAB_WEB_PORT" -le 65535
test "$PUFFERLAB_API_PORT" != "$PUFFERLAB_WEB_PORT"
test "$PUFFERLAB_API_PORT" != 8000
test "$PUFFERLAB_API_PORT" != 5173
test "$PUFFERLAB_WEB_PORT" != 8000
test "$PUFFERLAB_WEB_PORT" != 5173
export PUFFERLAB_API_PORT PUFFERLAB_WEB_PORT
cd web
pnpm install --frozen-lockfile
pnpm generate:api
VITE_API_BASE_URL="http://127.0.0.1:$PUFFERLAB_API_PORT" pnpm build
exec pnpm exec vite preview --host 127.0.0.1 --port "$PUFFERLAB_WEB_PORT" --strictPort
)
```

The built dashboard uses the API origin embedded at build time. Never put `TURBOPUFFER_API_KEY` or
another secret in a `VITE_*` variable.

## Checks

```bash
uv sync --locked
uv run ruff check backend scripts
uv run ruff format --check backend scripts
uv run mypy
uv run pytest
uv run python scripts/generate_openapi.py --check
uv run python scripts/audit_dataset_artifacts.py

cd web
pnpm lint
pnpm typecheck
pnpm test
pnpm build
```

## Credentials

`TURBOPUFFER_API_KEY` is read only by the backend. Never put it in a `VITE_*` variable, commit it,
print it in logs, or include it in exported run artifacts.
