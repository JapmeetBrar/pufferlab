# Dataset notices

## PufferLab Synthetic Unix Troubleshooting Corpus

The files under `fixtures/tiny-corpus/` were authored specifically for PufferLab as synthetic test
data. They do not reproduce CQADupStack posts. They are provided under CC0-1.0 and retain their
fixture source URL and external identity in every document.

The BAAI BGE model named in the fixture manifest is not redistributed by this repository. Its own
license and terms apply when an integration downloads and runs it.

## CQADupStack Unix local dataset pack

PufferLab can prepare the Unix subset from the BEIR-distributed CQADupStack archive. The archive,
extracted data, processed text, qrels, attribution sidecars, embeddings, vectors, caches, databases,
and evaluation exports are not redistributed by this repository. They remain ignored local
artifacts.

BEIR acquisition software and the original CQADupStack repository tooling are licensed under
Apache License 2.0. Their software license does not relicense the underlying Stack Exchange posts.
PufferLab's adapter was implemented independently and retains the exact repository revisions,
license links, citations, archive checksum registry, and paper identifiers in
`datasets/cqadupstack-unix/source-lock.json`.

The underlying questions and post content originated on Unix & Linux Stack Exchange. Content in
the pinned 2014-09-26 source dump may be subject to CC BY-SA 2.5 (contributed before 2011-04-08 UTC)
or CC BY-SA 3.0 (contributed from that date through the dump). The pinned BEIR members do not expose
author, contribution timestamp, or revision timestamp fields, so PufferLab records the conservative
expression `CC-BY-SA-2.5 OR CC-BY-SA-3.0`, the original post ID, canonical source URL, source site,
dump date, transformation version, and content hash. It marks author/date/revision metadata
`unavailable_in_pinned_archive` instead of inventing it.

- Source site: <https://unix.stackexchange.com/>
- Stack Exchange license chronology: <https://stackoverflow.com/help/licensing>
- CC BY-SA 2.5: <https://creativecommons.org/licenses/by-sa/2.5/>
- CC BY-SA 3.0: <https://creativecommons.org/licenses/by-sa/3.0/>
- CQADupStack paper: <https://doi.org/10.1145/2838931.2838934>
- BEIR paper: <https://arxiv.org/abs/2104.08663>

Any application surface that displays or exports locally prepared source text must carry the
canonical post link and this dataset notice. PufferLab-authored selection tags and reasons in the
checked-in curated manifest do not reproduce upstream post text.
