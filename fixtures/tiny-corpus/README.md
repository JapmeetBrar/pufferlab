# Tiny Unix fixture

This directory contains 20 original synthetic troubleshooting documents and five query examples.
It is safe to use offline and does not contain source-dataset text, credentials, or stored vectors.

`manifest.json` pins the intended BGE model and spells out every vector and FTS setting.
`documents.jsonl` and `queries.jsonl` are strict JSONL: one object per line, no blank lines, no
unknown fields, and unique external IDs. Document UUIDs and query expectations are derived at load
time, so generated identities cannot drift in the fixture.

Concrete sentence-transformers/BGE execution is deliberately outside this pack. The ingestion
service consumes an embedding protocol; the integration branch will supply the pinned model adapter.
