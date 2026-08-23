# CQADupStack Unix dataset pack

PufferLab does not ship CQADupStack text. This runbook downloads the one pinned BEIR archive into
the ignored `data/` directory, verifies the complete archive against both BEIR's published MD5 and
PufferLab's locally recorded SHA-256, then prepares only the Unix subset as a content-addressed
local pack.

## Prepare the local pack

Prerequisites are a clean checkout, `uv`, enough disk for the 5,343,728,040-byte archive and local
processed output, and network access to the pinned TU Darmstadt endpoint.

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

The download command is resumable. The preparation command refuses a partial archive, checks the
whole-file MD5 before the local SHA-256, checks the ZIP member size/CRC inventory, and never calls
ZIP extraction APIs on unverified bytes. Its successful output names the content-addressed pack and
source-lock hash without printing corpus/query text, qrels, metadata, or vectors.

The script also recomputes the curated 50-query selection from the ignored local queries and qrels.
It fails if `curated-50.json` is not the exact deterministic selection.

## Pinned acquisition chain

The machine-readable source of truth is
[`datasets/cqadupstack-unix/source-lock.json`](../../datasets/cqadupstack-unix/source-lock.json).
It pins:

- BEIR commit `ef83d29307061c65d04b035b4f4e7c18bd8374af`, its Apache-2.0 license,
  `examples/dataset/md5.csv`, and the official archive URL;
- archive size `5,343,728,040`, published MD5 `4e41456d7df8ee7760a7f866133bda78`,
  the locally computed completed-download SHA-256, HTTP source metadata, and the three exact Unix
  member paths/sizes/CRCs;
- CQADupStack commit `f73fc5b2cc708c61d33bc76a3de93de0bf5bf584`, its Apache-2.0 tooling
  license, paper DOI `10.1145/2838931.2838934`, and source dump date `2014-09-26`;
- the transformation specification hash, expected 47,382 documents, 1,072 queries and 1,693 qrels,
  plus text-free hashes used by the repository-history exposure scan.

Any URL, revision, published checksum, member inventory, transformation hash, or locally computed
archive hash drift fails closed. Updating the lock is a new dataset revision and requires a fresh
license/provenance review.

[`processed-pack-lock.json`](../../datasets/cqadupstack-unix/processed-pack-lock.json) is the
separate independently reviewed commitment to the real deterministic output. Keeping it outside
the source-lock hash avoids a circular content address. Before yielding any row, public loaders
bind it to the checked source-lock hash, archive SHA-256, current preprocessing hash, expected
record counts, and reviewed content SHA-256; recompute actual row-file hashes/counts and the
canonical content address; require the exact content-address directory name; and reject missing,
extra, non-regular, or symlinked pack entries. A changed row remains invalid even if its local
manifest hashes and directory are self-updated.

## Transformation and retained records

The adapter reads only `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv` after whole-archive
verification. It normalizes Unicode to NFC and line endings to LF, preserves other whitespace,
sorts by numeric source ID with a lexical fallback, and writes canonical JSONL. It discards the
large nested upstream `metadata` payload rather than carrying source bodies or personal data through
an opaque field.

Every processed document/query retains:

| Field | Value |
|---|---|
| `source_dataset` / `source_subset` | `CQADupStack` / `unix` |
| `original_post_id` | Exact upstream source ID |
| `canonical_post_url` | `https://unix.stackexchange.com/questions/<post-id>` |
| `source_site` / `source_dump_date` | Unix & Linux Stack Exchange / `2014-09-26` |
| `transformation_version` / `content_hash` | Pinned adapter version / canonical SHA-256 |
| `content_license` | `CC-BY-SA-2.5 OR CC-BY-SA-3.0` |
| author, contribution timestamp, revision timestamp | Explicit `null` with `unavailable_in_pinned_archive` status |

The pinned BEIR rows expose no author, contribution timestamp, or revision timestamp. Without a
timestamp, choosing CC BY-SA 2.5 versus 3.0 per post would be invented metadata, so the adapter links
both possible licenses and records the limitation. See [`NOTICE-DATASETS.md`](../../NOTICE-DATASETS.md).

Official qrels are accepted only when both their query and document IDs were retained. Duplicate or
dangling query-document pairs fail preparation. The local pack directory name hashes the source
lock, preprocessing specification and canonical hashes/counts of documents, queries and qrels.
The ingestion-oriented `FixtureQuery` view still exposes positive expected document IDs for
compatibility, while `load_curated_unix_local_pack` separately retains every official integer
relevance grade so evaluation materialization is lossless.

## Curated 50-query suite

[`curated-50.json`](../../datasets/cqadupstack-unix/curated-50.json) contains source query IDs plus
PufferLab-authored tags and reasons only. It contains no query/post text, qrel value, title, body,
snippet, author or upstream metadata.

The `pufferlab-curated-50-v1` selector assigns observable tags from ignored local query text and
judgment counts, deterministically permutes each tag pool with SHA-256 of selection version, stratum
and source ID, then round-robins 13 exact-token, 13 semantic, 12 hybrid and 12 reranker primary cases
without duplicates. The checked manifest records all applicable tags. This provides coverage for
lexical anchors, natural-language intent, fusion candidates, and multi-judgment reranking cases
without committing the licensed inputs used to derive them.

## Evaluation and ingestion application boundary

M2-E should call `UnixDatasetApplicationService.from_paths(...).ingest` with its provider-neutral
`IngestionService`, absolute-data-directory `IngestionCheckpointStore`, verified processed-pack
path, checked source/processed-pack locks, and manifest paths. The application service owns local
materialization, fail-closed provenance/content verification, checkpoint lookup/save, stable-ID
resume, readiness verification, and seed construction. It never deletes a namespace.

The successful `UnixIngestionResult` exposes an `IngestionReport` plus one deterministic
`UnixEvaluationSeed` containing:

- a READY contract-native `DatasetVersion` bound to the exact caller namespace, corpus hash, and
  compiled schema hash;
- a contract-native `QuerySet` and ordered `JudgedQuery` values whose `Qrel` objects preserve the
  official integer relevance grades and stable document UUIDs;
- a parallel `CuratedJudgedQuerySeed` for every query with the checked primary tag, all authored
  tags, and PufferLab-authored reason, because the current `JudgedQuery` contract has no reason
  field.

IDs and content hashes are UUIDv5/SHA-256 derivations of immutable inputs. The logical revision
timestamp is the pinned upstream dump date rather than local wall-clock time. M2-E may persist
`dataset_version`, `query_set`, and `judged_queries` directly; it can retain the parallel curation
metadata in application memory for analysis without parsing the ID-only manifest again.

## Resume and namespace safety

Corpus upserts use UUIDv5 identities derived from immutable dataset version plus source document ID.
After every successful batch, `IngestionCheckpointStore` atomically writes the canonical completed
ID set under the caller's absolute `PUFFERLAB_DATA_DIR`. A restart verifies namespace, corpus hash,
schema hash and every completed ID before skipping work. Replayed or uncheckpointed batches are safe
stable-ID upserts.

The checkpoint and ingestion interfaces expose no delete operation. A caller-supplied namespace is
bound to the checkpoint and used only through a fixed-length filename hash; it is never interpreted
as a cleanup target. Only the separate live-test ownership workflow may delete an internally
generated, pattern-validated test namespace.

## Git boundary and audit

Tracked files are limited to adapter/audit code, synthetic tests, citations/notices, source/index
locks, and the ID-only curated manifest. The following remain ignored and forbidden from all Git
history: archives/partials, extracted corpus/query/qrel data, upstream metadata or personal fields,
processed rows/attribution sidecars, embeddings/vectors/model caches, SQLite databases, exports,
logs, screenshots and live evidence.

`scripts/audit_dataset_artifacts.py` checks current candidates and every historical Git blob. It
rejects forbidden paths/suffixes, ZIP signatures and hashed token windows from known upstream
samples, and proves the documented raw/processed/cache/database/export/log paths with
`git check-ignore --no-index`. It refuses to run in a shallow repository because that cannot prove
whole-history coverage; the Backend CI checkout therefore uses `fetch-depth: 0`. Run it before
every dataset PR handoff.

To remove a local pack, delete only the explicit ignored directory you created under
`data/cqadupstack-unix/`. The archive and processed pack are reproducible from the source lock; Git
cannot restore local copies because they were never tracked.
