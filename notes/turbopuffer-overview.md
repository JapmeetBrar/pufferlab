# Turbopuffer — Interview Prep Overview

*Compiled July 2026 from turbopuffer's public docs, engineering blog, and press coverage.*

---

## 1. Elevator pitch

Turbopuffer is a **serverless search engine (vector + full-text) built directly on object storage (S3/GCS/Azure Blob)**, instead of on local SSDs like every other vector database (Pinecone, Qdrant, Weaviate, Milvus). Because object storage is ~10-20x cheaper than replicated SSD storage, turbopuffer can be **10-100x cheaper** than incumbents for the "cold-ish, bursty, multi-tenant" data patterns that dominate real AI products (RAG over a codebase, a user's docs, a support inbox) — at the cost of higher latency on the first ("cold") query to a dataset.

Mission, in their words: **"make every byte searchable."**

---

## 2. The story

- **Founders**: Simon Hørup Eskildsen (CEO) and Justine Li (CTO). Both spent ~8 years at Shopify (2013–2021) on core infrastructure, scaling it from ~1,000 to 1M+ requests/sec. Simon was a Principal Engineer; Justine a Senior Staff Engineer.
- **The spark (late 2022)**: Simon was consulting for Readwise and found that building a vector-search recommendation engine would cost **~$20,000/month** — more than 4x what Readwise paid for its *entire* relational database. That absurd cost-to-value ratio was the founding insight: existing vector DBs were priced/architected for a world where storage is always-hot SSD, but most real workloads are spiky and multi-tenant (most data is cold most of the time).
- **Build**: ~4 months of focused work, **three full rewrites** to land on the right architecture, launched publicly October 4, 2023.
- **Funding — deliberately unusual**: Turbopuffer took almost no venture money for a long time. First outside capital was an early-2024 angel check from Lachy Groom, structured as a "prove product-market-fit or shut down by end of 2024" bet. They hit that bar. Thrive Capital led a seed round in December 2025 (Lachy Groom doubled down too). Total primary capital raised remained under $1M for a long time — very lean by AI-infra standards. **Not YC-backed.**
- **Traction (as of ~2026)**: ~$100M annualized run-rate by March 2026, 10x sales growth and 5x headcount growth in 2025. Handles **2.5+ trillion vectors/documents** and **10M+ writes/sec** in production. Notable customers: **Cursor** (largest known customer — 1T+ code chunks across 80M+ namespaces, cut storage/retrieval costs ~95%), **Notion** (10B+ vectors, migrated from Pinecone, cut search costs ~60%, removed a per-user AI usage charge as a direct result), **Anthropic, Atlassian, Linear, Ramp, Grammarly, Superhuman, Suno** (radio feature).
- **Team**: recruited heavily from Shopify, plus people from CockroachDB, Materialize, and maintainers of projects like Lucene, Pebble, and Rust verification tooling — i.e., people who've built serious distributed databases/storage engines before.
- **Stated values** (useful to know for interview culture-fit signaling): *overstep > understep*, *correctness > simplicity > performance*, *customer traces > first principles > hunches*, *show > tell*.

---

## 3. The core architectural idea

Every other vector DB treats it as a **compute problem**: keep an in-memory or SSD-resident graph (HNSW) or index per node, replicate it 3x for durability, and pay for that hot capacity 24/7 whether or not anyone queries it.

Turbopuffer treats it as a **storage problem**: object storage is the *source of truth* and is nearly free (~$0.02/GB vs. ~$0.60/GB for 3x-replicated SSD vs. ~$5/GB for RAM). Everything else — NVMe, RAM — is just a **cache** that's rebuilt on demand. If a namespace hasn't been queried in a while, it costs almost nothing to store. The moment it's queried, data "inflates" up through the cache tiers — they describe this as a **pufferfish effect** (also the mascot/name pun): cold and flat until poked, then it puffs up into fast cache tiers.

This single idea explains almost every other design decision below.

### Storage tiers
| Tier | Latency | Relative cost/GB | Role |
|---|---|---|---|
| Object storage (S3/GCS/Azure) | ~200–500ms | ~$0.02 | Source of truth, durable |
| NVMe SSD | ~10–20ms | ~$0.60 (3x replicated) | Warm cache |
| RAM | <10ms | ~$5 | Hot cache |

### Write path
- Writes are appended as new files to a **write-ahead log (WAL)** living in the namespace's own S3 prefix (each namespace = its own isolated prefix + indexes).
- Concurrent writes are batched via **group commit** — roughly one WAL entry per second per namespace — trading a bit of write latency (p50 ≈ 165ms for a 500KB write) for much higher aggregate throughput.
- **Durability guarantee**: a write only returns success once it's durably in object storage — no "eventually persisted" ambiguity.
- New/unindexed writes are still immediately searchable via exhaustive scan of the not-yet-indexed WAL tail, so there's no visibility gap while background indexing catches up.
- **Consistency**: strong by default (read-your-writes). An eventually-consistent read mode is available for latency-sensitive paths, trading up to ~1 hour of staleness (worst case) for sub-10ms reads.

### Indexing: centroid-based (SPFresh-derived), *not* HNSW
This is the most interview-relevant technical decision. Turbopuffer explicitly avoided graph-based ANN indexes (HNSW, DiskANN) — the industry-standard approach used by Pinecone/Qdrant/Weaviate/pgvector — because **graph traversal requires many small, sequentially-dependent round trips** (HNSW can need ~10-20 dependent hops to find a nearest neighbor). That's fine when your index lives in RAM/SSD with microsecond access, but catastrophic against object storage with 200-500ms per round trip.

Instead, turbopuffer uses a **clustered/centroid index** (based on ideas from the SPFresh paper):
1. Fetch **all cluster centroids** in one round trip (small — centroids summarize the space).
2. Identify the nearest centroids, then fetch **only those clusters' data** in a second (parallel) round trip.

That's it — **exactly 2 sequential dependent steps**, regardless of dataset size, with everything else parallelized. This is *why* object-storage-native vector search is viable at all: the access pattern is shaped to match S3's strengths (large parallel reads) and avoid its weakness (round-trip latency).

### Query execution & the "JIT compiler" mental model
- **Cold query** (namespace not cached anywhere): 3–4 round trips to S3 → ~200-500ms.
- **Warm query** (namespace cached on a node's NVMe/RAM): ~8-20ms, dominated by local compute, not network.
- The system is explicitly compared to a **JIT compiler**: the more a namespace is queried, the more of it gets promoted into fast caches, so it gets faster over time — optimizing for realistic, skewed (power-law) access patterns rather than uniform worst-case access.
- Any query node can theoretically serve any namespace, but the router prefers sending repeat traffic to the *same* node for cache locality.
- Apps can send a **cache-warming "pre-flight" hint** before a latency-sensitive moment (e.g., Notion warms cache the instant a user opens its Q&A dialog, so the *real* query a second later is already warm).

### Query capabilities (a single query can combine all of these)
- **Vector search** — ANN via the centroid index (~90-95% recall@10 typical).
- **BM25 full-text search** — for exact-match cases embeddings miss (SKUs, IDs, rare tokens).
- **Regex search** — compiled to trigram indexes (useful for code search — relevant to why Cursor uses it).
- **Attribute filters** — inverted/bitmap indexes, auto-inferred schema (with manual overrides for things like `uuid`/`datetime`).

### Multi-tenancy as a first-class primitive
- A **namespace** = an isolated container (its own S3 prefix + indexes) for a set of documents you'd query together — e.g., one namespace per user, per workspace, per repo. Millions of namespaces are normal (Cursor has 80M+).
- This is a deliberate fit for real AI products, which are almost always multi-tenant with wildly uneven activity (most users/workspaces idle most of the time) — the exact shape object storage economics reward.
- Enterprise customers can opt into **dedicated, single-tenant clusters** if they need hard isolation instead of the shared multi-tenant pool.
- Fun infra detail: the entire cluster **coordination layer is reportedly just a single JSON file on S3** — an extreme "operational simplicity" bet consistent with their stated values.

---

## 4. Why it's good (the pitch, distilled)

1. **Cost**: 10-100x cheaper for realistic (spiky, multi-tenant, mostly-cold) workloads, because you stop paying to keep 100% of data hot 100% of the time.
2. **Scale-to-zero economics without a scale-to-zero *product* problem**: no cold-start "spin up a server" step — cold namespaces are just slower for one query, not unavailable.
3. **No infra to manage**: no shards, no replica counts, no capacity planning — it's genuinely serverless (bring data, get queries).
4. **Unifies vector + full-text + filters** in one query/one system, instead of stitching together a vector DB + Elasticsearch + a relational DB filter layer.
5. **Built by people who've run this class of system in anger** (Shopify-scale infra, CockroachDB/Materialize alumni) — the "correctness > simplicity > performance" ordering in their values is a direct rebuke of move-fast-and-lose-data infra culture.
6. **Proven at extreme scale** with logo-quality customers (Cursor, Notion, Anthropic) as public reference points, not just benchmarks.

---

## 5. Tradeoffs & honest limitations

Be ready to discuss these — an interviewer will respect you more for naming them than for reciting the marketing pitch.

- **Cold-query latency**: 200-500ms (sometimes cited up to 800ms) on a first touch is *real* and not always acceptable — for a latency-critical, always-hot, small dataset (e.g., <10M vectors, constantly queried), a traditional in-memory/SSD-resident index (Qdrant, or pgvector if you're already on Postgres) will simply be faster with less design effort.
- **Write latency**: because writes go to object storage with WAL group-commit batching, single-write latency (~100-200ms) is higher than an in-memory system doing an immediate in-process index insert. Great throughput, mediocre single-write tail latency.
- **Not the fastest **raw** query engine**: for pure "lowest possible p50 at massive constant QPS on a static hot dataset," a dedicated in-memory system (Qdrant, or even a well-tuned HNSW-in-RAM setup) will out-perform it. Turbopuffer optimizes total-cost-of-ownership and elasticity, not absolute peak speed.
- **Consistency vs. latency knob** requires the caller to think about it — strong consistency by default is the safe/correct choice, but the eventual-consistency escape hatch (up to ~1hr staleness worst case) is a footgun if used naively.
- **ANN recall isn't 100%** (typically 90-95% recall@10) — same caveat as any approximate index; not the right tool if you need exact nearest neighbor guarantees.
- **Ecosystem/maturity**: younger and smaller than Pinecone; fewer integrations, smaller community, less "enterprise checkbox" tooling (though this is closing fast given the growth numbers).
- **It's genuinely a different index family (SPFresh-style centroid vs. HNSW)**, so intuitions/benchmarks from HNSW-based systems don't transfer directly — recall/latency tuning knobs are different.

---

## 6. How it stacks up (quick comparison)

| | Turbopuffer | Pinecone | Qdrant | Weaviate | pgvector |
|---|---|---|---|---|---|
| Storage model | Object storage native (compute/storage fully separated) | Managed, proprietary (serverless tier also object-storage-based) | Self-host or managed, SSD/RAM resident | Self-host or managed, SSD/RAM resident | Extension on Postgres, SSD/RAM |
| Index | Centroid/cluster (SPFresh-derived) | Proprietary | HNSW | HNSW | HNSW / IVFFlat |
| Best at | Huge, multi-tenant, spiky-access data; cost efficiency at scale | Zero-ops, "fastest path from OpenAI embedding to prod" | Lowest raw latency (<5ms), self-hosted control | Native hybrid search + GraphQL, multi-modal | Simplicity if already on Postgres, <50-100M vectors |
| Weak at | Cold-query latency, brand-new/tiny workloads | Cost at scale (storage ~$0.33/GB vs turbopuffer's ~$0.02/GB) | Cost/ops burden of always-hot infra at huge scale | Ops burden at huge scale | Scale ceiling, no native serverless elasticity |
| Real example | Notion: −60% cost migrating off Pinecone. Cursor: −95% cost. | Baseline / easiest onboarding | Sub-5ms latency use cases | Rich filtering + multi-modal search | <10M vectors on a $30/mo Postgres box |

---

## 7. Pricing model (as of 2026, for context)

- Usage-based: pay for **writes**, **queries** (scan rate ~$1/PB after a Feb 2026 5x price cut from $5/PB), and **GB-month storage** (~$0.02/GB).
- Tiers: Launch (min ~$16/mo after a cut from $64), Scale, Enterprise (min $4,096/mo).
- Newer feature: **namespace pinning** (Apr 2026) — pay GB-hours instead of per-query for namespaces you want to force-keep hot (predictable latency for a known-important dataset), min 64GB / 10 minutes.
- The pricing mechanics *are* the architecture made visible: you're billed close to raw storage + actual access, not for provisioned-but-idle capacity.

---

## 8. Likely interview angles & talking points

**System design / architecture questions to expect:**
- "Design a vector search system that needs to serve millions of tenants cheaply." → lead with the compute/storage separation + object storage economics insight, not just "use HNSW."
- "Why not HNSW?" → round-trip count argument (2 dependent steps vs. 10-20), and that graph traversal is inherently latency-sensitive/sequential in a way clustering isn't.
- "How do you keep writes durable and reads consistent with an S3-backed WAL?" → group commit, per-namespace S3 prefix, durable-before-ack semantics, strong-by-default consistency with an opt-in eventual mode.
- Trade-off questions ("when would you *not* use this?") → know the cold-latency and small-hot-dataset caveats cold (see §5) — this is where naming the honest tradeoffs (rather than only pitching strengths) will stand out.

**Good questions to ask them** (shows you understand the architecture, not just the pitch):
- "How does the centroid index get rebuilt/rebalanced as a namespace grows — what's the write-amplification story there?"
- "How do you decide what to evict from NVMe/RAM cache across millions of namespaces — LRU, or something access-pattern-aware?"
- "With writes batched via per-namespace group commit, how do very-high-write-rate single namespaces (e.g. a single huge Cursor repo) avoid becoming a bottleneck?"
- "How much of the 95%+ cost savings customers report is object storage economics vs. the multi-tenant namespace design itself?"

**Company/culture questions worth having a POV on:**
- Their explicit value ordering (correctness > simplicity > performance) — be ready to talk about a time you made that same trade.
- They stayed capital-light (<$1M raised) for a long time by choice, chasing profitability/PMF signal over headline valuation — worth understanding *why* founders coming from Shopify (a company also known for engineering discipline) might default to that.

---

## 9. Sources
- [turbopuffer.com/docs/architecture](https://turbopuffer.com/docs/architecture)
- [turbopuffer.com/docs/concepts](https://turbopuffer.com/docs/concepts)
- [turbopuffer.com/about](https://turbopuffer.com/about)
- [turbopuffer.com/pricing](https://turbopuffer.com/pricing) and [pricing changelog](https://turbopuffer.com/docs/pricing-log)
- [turbopuffer.com/customers/cursor](https://turbopuffer.com/customers/cursor), [turbopuffer.com/customers/notion](https://turbopuffer.com/customers/notion)
- [Jason Liu — "TurboPuffer: Object Storage-First Vector Database Architecture"](https://jxnl.co/writing/2025/09/11/turbopuffer-object-storage-first-vector-database-architecture/)
- [Ajay Edupuganti — "How Turbopuffer Serves 2.5 Trillion Vectors on S3"](https://ajay-edupuganti.medium.com/how-turbopuffer-serves-2-5-trillion-vectors-on-s3-7d7ab7f9a7fa)
- [pmf.show — "How Simon Eskildsen Built TurboPuffer"](https://www.pmf.show/blog/how-simon-eskildsen-built-turbopuffer-the-vector-db-powering-cursor-and-notion/)
- [BetaKit — "Ex-Shopify engineers raise fresh financing"](https://betakit.com/ex-shopify-engineers-raise-fresh-financing-to-scale-turbopuffers-ai-search/)
- [Software Engineering Daily — Turbopuffer episode](https://softwareengineeringdaily.com/2025/09/30/turbopuffer-with-simon-horup-eskildsen/)
- [Amplify Partners interview with Simon Eskildsen](https://www.amplifypartners.com/barrchives/how-turbopuffer-is-building-the-future-of-vector-databases-with-ceo-simon-eskildsen)
