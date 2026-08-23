# PufferLab operator runbook

This is the current interview and local-operator path. The main workflow is provider-free: it seeds
authenticated synthetic evidence, evaluates a quality policy, and serves the built dashboard from
one local SQLite database. The optional final section describes the separately authorized,
cost-bearing generated tiny-namespace lifecycle.

## 1. Install the pinned tools

Use Python 3.12 or 3.13, `uv`, Node.js 22 or newer, and pnpm 11. The repository pins pnpm 11.19.0.
Confirm the executables from the repository root:

```bash
python3 --version
uv --version
node --version
pnpm --version
```

If pnpm is missing, install its pinned major:

```bash
npm install --global pnpm@11
```

Install the locked Python and browser dependencies:

```bash
uv sync --locked
cd web
pnpm install --frozen-lockfile
cd ..
```

## 2. Seed one provider-free run and capture its IDs

Choose one ignored data directory and explicitly blank every live setting. Environment variables
override any existing `.env`, so this keeps the workflow provider-free even in a previously used
checkout:

```bash
export PUFFERLAB_DATA_DIR="$PWD/data/demo-m4"
export TURBOPUFFER_API_KEY=
export TURBOPUFFER_REGION=
export PUFFERLAB_SEARCH_NAMESPACE=

SEED_OUTPUT="$(uv run pufferlab demo seed)"
printf '%s\n' "$SEED_OUTPUT"

RUN_ID="$(printf '%s\n' "$SEED_OUTPUT" | awk -F'[ =]' '/^run_id=/{print $2}')"
DATASET_VERSION_ID="$(printf '%s\n' "$SEED_OUTPUT" | awk -F'[ =]' '/^dataset_id=/{print $2}')"
CONFIG_IDS="$(printf '%s\n' "$SEED_OUTPUT" | awk -F'config_ids=' '/^run_id=/{print $2}')"
RERANKER_ID="$(printf '%s\n' "$CONFIG_IDS" | cut -d, -f4)"
test -n "$RUN_ID"
test -n "$DATASET_VERSION_ID"
test -n "$RERANKER_ID"
printf 'captured_run_id=%s\ncaptured_candidate_id=%s\n' "$RUN_ID" "$RERANKER_ID"
```

`demo seed` creates one deterministic completed run with 50 queries, four configs, and 200
outcomes. Its output contains only safe durable UUIDs, origin/timing labels, and counts. It contains
no credential, namespace, vector, licensed query, qrel, provider response, or document text.
Repeating the command with the same `PUFFERLAB_DATA_DIR` validates and reuses the same identities;
it does not create another run.

Run the default provider-free diagnostic:

```bash
uv run pufferlab doctor --mode demo
```

Expected exit is `0`, with `state=ready`, `queries=50`, `configs=4`, and `completed_runs=1`.

## 3. Exercise gate pass and policy-fail exits

The first gate prints text and passes after allowing the synthetic query whose qrels contain no
positive judgment. The second prints JSON and intentionally fails only the aggregate-delta policy:

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

The process contract is:

- `0`: the policy passed;
- `4`: valid evidence was evaluated and failed policy;
- `2`: policy, evidence, or run/candidate binding is invalid;
- `1`: an internal failure.

The gate opens an existing SQLite catalog read-only and constructs no provider, search runtime,
embedding model, reranker, migration, or recovery path. It accepts only a completed canonical
50-query/four-config run, authenticates the exact source/qrels/config structure, validates all 200
outcomes, and recomputes quality metrics and summaries. Typed failed outcomes count toward error
rate and reduce paired coverage. Latency is not a gate input.

The qrels and structural hashes are authenticated, but the persisted ranked document UUIDs are
explicitly trusted local evidence, not cryptographically authenticated evidence. A malicious full
database rewrite is outside this local gate's threat model and would require a separate signing and
key-management design.

## 4. Allocate ports and serve the built dashboard

Allocate two distinct currently free loopback ports. The script rejects the user's common `8000`
and `5173` ports and writes only two validated numbers into a mode-0600 ignored data file. It
refuses symbolic links, non-regular files, foreign owners, and hard links before truncation:

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
sockets: list[socket.socket] = []
ports: list[int] = []
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

The exact bind remains authoritative; if another process wins a port after allocation, the strict
server command fails instead of switching ports.

In terminal 1, return to the repository root, export the same data directory, explicitly blank live
settings again, load the allocated ports, and use the installed one-worker server command:

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

`serve` accepts only loopback hosts and fixes one worker. Unlike doctor and gate, server startup
mutates local SQLite as needed: it migrates the configured catalog, marks orphaned jobs interrupted,
and reclaims valid queued work. Invalid host/port input exits `2`, startup/runtime failure exits `1`,
and graceful shutdown exits `0`. SIGINT/SIGTERM starts bounded shutdown; a hard ten-second watchdog
may skip Python cleanup, so the next startup performs durable migration/recovery reconciliation.

In terminal 2, return to the repository root, load the same allocated ports, build the dashboard
with the exact API origin embedded, and start a strict preview:

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

Open `http://127.0.0.1:<web-port>/runs`, replacing `<web-port>` with the printed value. The API
health endpoint is `http://127.0.0.1:<api-port>/api/v1/health`; health means only that the process
is alive.

### Manual dashboard checks

1. Open **Synthetic demo · read-only** and verify four config summaries and 50-of-50 durable query
   groups.
2. Change candidate, regressions/gains order, and row limit. Confirm `candidate`, `order`, and
   `limit` remain in the URL after refresh.
3. Follow **Inspect recorded query**. The deep link contains only run/query/config UUIDs, never
   query text.
4. Open **Inspect document**, refresh, press Back, then Forward. The drawer closes and restores from
   only the document UUID.
5. Confirm original stage evidence is `NOT_OBSERVABLE`, timing is unavailable, and synthetic live
   replay is disabled.
6. Open the Playground. With the blank live settings it presents actionable setup guidance and
   keeps **Compare results** disabled. No compare POST should be sent.
7. Repeat at a narrow viewport and confirm the tables scroll internally without page overflow.

Opening, refreshing, or navigating stored synthetic pages performs provider-free GETs. Stop both
servers with Ctrl-C when finished. The ignored database and port file can remain for the
idempotence check.

## 5. Understand capability and doctor states

`GET /api/v1/capabilities` and the default doctor inspect local configuration only:

- `action_required` means one or more allowlisted local requirements are missing or invalid. The
  Playground explains the next action and does not send a comparison POST.
- `locally_configured` means the local requirements are present. It does not prove credential
  validity, remote namespace existence, index readiness, schema, exact corpus identity, or working
  retrieval.

Available diagnostics are:

```bash
uv run pufferlab doctor --mode demo
uv run pufferlab doctor --mode live-tiny
test -n "${DATASET_VERSION_ID:-}"
uv run pufferlab doctor --mode evaluation --dataset-version "$DATASET_VERSION_ID"
uv run pufferlab doctor --mode all --dataset-version "$DATASET_VERSION_ID"
```

Omit `--dataset-version` in evaluation/all mode only when one persisted dataset is unambiguous.
Default doctor commands are read-only and provider-free. Adding `--live` is an explicit,
potentially billable metadata-only check with at most one bounded outbound attempt in total.
Doctor exits `0` when every requested check is ready, `2` for missing/invalid local
configuration/evidence, `3` for failed/not-found/updating live metadata, `1` for an internal
failure, and `130` when interrupted.

Provider-capable work begins only through an explicitly labeled action: generated/explicit/live
dataset ingestion, `doctor --live`, a live evaluation run, pressing the cost-bearing Playground
compare/replay action, or authenticated generated cleanup. Seed, gate, default doctor, stored
dashboard reads, and `namespace show-tiny` are provider-free.

## 6. Optional generated tiny namespace lifecycle

This section can create billable remote state. Use it only when a live rehearsal is explicitly
authorized. Keep `TURBOPUFFER_API_KEY` only in ignored `.env`; never put it in a command, log,
screenshot, issue, PR, export, or `VITE_*` variable. First prepare the settings file without
replacing an existing credential file:

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

Edit `.env`, add the account's supported region and key, and leave
`PUFFERLAB_SEARCH_NAMESPACE` blank. Then run the generated ingestion:

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

The first default doctor makes no provider call. With no prior per-user receipt it reports
`action_required`; an existing receipt may instead be resumable or report a credential/region
mismatch. `ingest-tiny` with no `--namespace` durably creates or resumes the one authenticated
generated-tiny receipt for this OS user. It binds the random target to the exact credential value
and creating region before provider access. Its fixed state is independent of cwd, `HOME`, XDG
variables, and `PUFFERLAB_DATA_DIR`; do not move, edit, or remove the receipt, owner key, or lock.

`show-tiny` is provider-free and prints exact `TURBOPUFFER_REGION` and
`PUFFERLAB_SEARCH_NAMESPACE` assignments from the authenticated receipt. Copy those assignments
into `.env`, then run the second default doctor:

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

That check can report `locally_configured`, which still does not mean the remote namespace is
healthy. Restart `pufferlab serve` after configuration.

An optional live diagnostic is explicit:

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

It performs one bounded, potentially billable metadata request. A successful metadata check still
does not prove schema, exact corpus identity, authentication for other operations, or working
BM25/ANN retrieval.

### Start the credential-aware Playground

Use the ports allocated in section 4; rerun that provider-free allocator first if the port file is
absent or either strict bind is no longer available. In terminal 1, return to the repository root,
unset the blank shell overrides so the verified `.env` can load, and start the installed server:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
export PUFFERLAB_DATA_DIR="$PWD/data/live-tiny"
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

In terminal 2, return to the repository root and serve a strict build that contains only the API
origin, never a credential:

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
VITE_API_BASE_URL="http://127.0.0.1:$PUFFERLAB_API_PORT" pnpm build
exec pnpm exec vite preview --host 127.0.0.1 --port "$PUFFERLAB_WEB_PORT" --strictPort
)
```

Open `http://127.0.0.1:<web-port>/playground` with the printed web port. Confirm the capability is
`locally_configured`; that state is local guidance, not remote-health evidence. Enter a tiny-fixture
query and press **Compare results** only when the cost-bearing provider request is explicitly
intended. Verify BM25/vector results or an honest redacted runtime failure, and confirm no key,
vector, provider body, or namespace appears in the URL or browser output.

Stop both the API and preview with Ctrl-C before cleanup. Do not rely on process exit, browser
navigation, or API startup to remove remote state; cleanup is always a separate explicit command.

Re-run the argument-free ingestion command to resume the generated receipt:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
uv run pufferlab dataset ingest-tiny
)
```

Do not pass the printed generated namespace back through `--namespace`. Any explicit target is
caller-managed, is never recorded as PufferLab-owned, and can never be removed by `cleanup-tiny`,
even if its spelling resembles a generated target.

### Clean only the authenticated generated target

Stop the API or live work first. Keep the same creating credential available and run:

```bash
(
set -eu
unset TURBOPUFFER_API_KEY TURBOPUFFER_REGION PUFFERLAB_SEARCH_NAMESPACE
uv run pufferlab namespace cleanup-tiny
)
```

Cleanup accepts no namespace, path, token, or ownership argument. It verifies the fixed receipt,
uses its exact creating region, commits `cleanup_requested` before delete, and performs bounded
not-found metadata verification before terminal receipt removal. Cleanup contacts the provider,
and its metadata verification may be billed as zero-row queries.

If cleanup exits nonzero before the terminal fixed receipt move, preserve all state and rerun the
same command. The exact creating credential is required; if it was rotated, restore that value
rather than editing the receipt. A retained `cleanup_requested` receipt safely repeats
delete/already-absent and verifies again. A retained `not_found_verified` receipt finishes local
removal without constructing a provider or requiring the key.

Once the canonical receipt has been durably moved away, remote absence is already committed and a
retry remains provider-free. A crash during the subsequent held-inode wipe can leave an ignored
quarantine and the command may report that no receipt is available; never rename a quarantine back
into authority or authorize another delete from it.

After successful cleanup, remove the stale namespace assignment from `.env`, clear any shell
override, and restart the API:

```bash
unset PUFFERLAB_SEARCH_NAMESPACE
```

The fixed owner key remains so future generated receipts use the same local ownership root.

## Troubleshooting

- **`pnpm: command not found`:** run `npm install --global pnpm@11`, then reinstall from `web/`.
- **API reports another process owns the database:** stop the older PufferLab server. One database
  deliberately supports one server process.
- **Run history is empty:** make `PUFFERLAB_DATA_DIR` identical in seed and server shells, then
  restart with `pufferlab serve`.
- **Preview calls the wrong API:** rebuild with the exact `VITE_API_BASE_URL`; changing it after
  `pnpm build` does not rewrite the built assets.
- **Latency is unavailable:** this is required for synthetic evidence; zero milliseconds would be
  a fabricated observation.
- **Live compare is disabled:** read the guided `action_required` steps. Do not bypass them with a
  direct POST.
- **Cleanup reports a key or region mismatch:** preserve the receipt and restore its creating
  credential/configuration. Never substitute a namespace or manually delete receipt state.
