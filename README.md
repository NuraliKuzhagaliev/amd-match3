# AMD AI League: Match 3

This repository contains the Match 3 RAG source for our shared lablab.ai entry. [Match 2 VisionForge OCR source](https://github.com/NuraliKuzhagaliev/match-2) is maintained separately. The two Docker image references are provided only in the organizer's private submission fields.

Offline RAG implementation for the supplied competition contract. It reads PDF,
Word tables, all Excel sheets, CSV, text/log/Python source, and images. It returns
short values with file citations, or an explicit empty response when evidence is
insufficient. Source code is read as text and is never executed.

## Architecture

- **One resident GPU worker**: Qwen3-VL-8B-Instruct for vision and extraction,
  Qwen3-Embedding-0.6B for semantic retrieval. Both use BF16 on ROCm with SDPA.
- **One indexing pass**: isolated parsers, per-file timeout, image transcription,
  bounded parallel parsing (four subprocesses), overlapping chunks, persisted vectors and provenance.
  Word/Excel rows retain title and column context; mixed PDF text/image pages
  receive a vision pass, and large images use overlapping supplemental crops.
  Bad files are recorded
  individually; unknown types and encrypted PDFs are skipped.
- **Hybrid retrieval**: BM25 + dense cosine similarity + shared identifier links
  for multi-document chains, with source diversity and withdrawn-revision hints.
  Exact query identifiers anchor retrieval; distinctive numbered bridge IDs
  expand it. Generic signal/column names cannot fan out into unrelated products.
  Missing numbered identifiers or explicit signal names cause an abstention
  instead of substituting a similar product or signal.
- **Evidence validation**: the model selects necessary chunk IDs; code verifies
  that they exist and the complete answer value occurs in the selected evidence, then
  deduplicates file paths and records original source excerpts automatically.
  These checks cannot prove logical entailment or source necessity; the
  independent benchmark evaluates those decisions.
  Selected evidence must cover the explicit question identifiers and preserve
  a connected chain of numbered IDs to the answer. An audit cannot trim away
  the only product/ticket bridge or, for identifier-free questions, erase the
  question's sole lexical link to the evidence.
- **Focused semantic audit**: a second bounded model call compares the requested
  fact with the fact stated in selected evidence, rejects unsupported candidates,
  and removes redundant citations. This is a model judgment, not a formal proof.
- **Thin CLI**: every question starts a new `app.py` process, while models stay in
  the worker. Requests use a local Unix socket, not a public network port.
- **Offline evaluation**: model revisions are pinned and downloaded during image
  build. Runtime sets offline flags and uses `local_files_only=True`.

This implementation is intentionally extractive: computed values not literally
present in the evidence are refused. It targets the brief's value extraction
tasks, rather than arbitrary mathematical analysis over documents.

## Build and run

Use a Linux host with an AMD GPU supported by the mandated ROCm base,
1–48 GiB GPU memory, and sufficient disk for the 46.68 GiB uncompressed image.
The build downloads pinned model weights; the evaluation runtime runs offline.

```bash
docker build -t amd-match3:local .
docker run -d --name amd-match3 --network none --cap-drop DAC_OVERRIDE \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc=host --shm-size 8g amd-match3:local
docker cp /path/to/corpus/. amd-match3:/app/corpus/
docker exec amd-match3 python3 /app/app.py --index /app/corpus
docker exec amd-match3 python3 /app/app.py --corpus /app/corpus \
  --query-id query_01 --query 'What is the maximum junction temperature of the TQ-40?'
docker exec amd-match3 cat /app/output/query_01_output.json
```

The container remains running. Each query is a separate CLI process that talks
to the resident worker over a Unix socket. No network port is required.
Model revisions are recorded in `models.lock.json`; `--no-deps` installation
preserves the ROCm torch provided by the mandated base. Model and dependency
notices are in `THIRD_PARTY_NOTICES.md`.

## Verification and limits

The frozen application passed 287/287 development questions on an AMD Radeon
Pro W7900D, checking answers and exact citation sets. This covers the official
10-question sample and four additional development suites (10, 120, 120, 27).
The 27-question transfer suite was used to correct scalar-unit formatting, so
its final pass is development evidence, not a blind evaluation. Synthetic
fixtures do not establish the official hidden score.

The slowest measured question took 7.32 seconds; the longest model-load-plus-
indexing phase took 105.28 seconds. Peak process PyTorch reserved memory was
19.87 GiB. System-wide GPU counters also included neighbouring jobs on the
shared host and are not an isolated process-memory measurement.

GPU tests executed the source directly in the free AMD Jupyter environment.
A full Docker launch on an AMD GPU remains unverified: that notebook had no
container runtime, and the local computer had no AMD GPU. Archive checks
verified every compressed layer digest, every uncompressed diff ID, frozen
application hashes, and offline imports of the actual code layer. These do
not replace the missing full container/GPU integration test.

Run the 47 Linux unit/integration tests (parsers, retrieval, evidence, output
contract, packaging, and permission handling) with:

```bash
docker build -f Dockerfile.test -t amd-match3-tests .
docker run --rm --network none --cap-drop DAC_OVERRIDE amd-match3-tests
```

On a compatible AMD Docker host, `scripts/check_container_gpu.py` provides
the remaining end-to-end container check; consult its `--help` first.
Development metadata and frozen runtime hashes are in `validation-summary.json`.
Weights, competition starter documents, raw traces, account details and the
competition submission registry reference are excluded from this repository.
