# F-1 Waypoint: Tech Stack

The stack is boring on purpose. The interesting parts are the eval and the retrieval decisions. Where a choice is settled by measurement, the eval numbers are published in the README.

## Summary

| Layer | Choice |
|---|---|
| Language | Python 3.12, managed with uv |
| Lint, format, test | ruff, pytest, pre-commit |
| Backend | FastAPI |
| Store | SQLite (documents, chunks, hashes) |
| Vector search | sqlite-vec, with FAISS as the alternative |
| Keyword search | BM25 through a library, with a tokenizer that keeps form numbers intact |
| Embeddings, reranker | Chosen by eval, ablation table in the README |
| LLM | Low-cost models through OpenRouter, two roles, pinned, chosen by eval |
| Ingestion | httpx, trafilatura or BeautifulSoup, PyMuPDF |
| Eval | Own harness, YAML questions validated with pydantic |
| Frontend | React, Vite, TypeScript |
| Hosting | Cloudflare Pages (frontend), Railway (API) |
| CI | GitHub Actions: tests on push, scheduled freshness job |

## Reasoning

**FastAPI.** Python is where the retrieval and eval ecosystem lives. FastAPI's typed models make the answer contract (answer, citations, refusal reason) explicit in the schema.

**One SQLite file.** The corpus is small, likely hundreds to low thousands of chunks, so a managed vector database adds cost and operations with no benefit. Documents, chunks and vectors sit together, so removing a chunk removes its vector, and delete-sync is a plain `DELETE` in a transaction. That keeps the freshness story easy to explain and test.

**sqlite-vec over FAISS.** Vectors live next to chunks in one file. FAISS is better known but lives outside the database, so two stores must be kept in sync. At this size, brute-force search is fast enough, and the index type is not worth agonizing over.

**BM25 through a library.** Form numbers such as I-983 and I-765 and citations such as 8 CFR 214.2(f)(10) are keyword queries, not semantic ones. A library keeps tokenization under my control. One unit test proves these terms survive tokenization as searchable terms, and that test is the clearest explanation of why hybrid retrieval exists. SQLite FTS5 is the fallback.

**Embeddings and reranker by eval.** Each is compared against at least one alternative on Recall@5, and the results are published as an ablation table. Local pinned models are reproducible and need no API in CI. Hosted models are easy but can drift and silently change scores, so any hosted model is pinned and recorded with every run.

**Models through OpenRouter.** A public demo makes premium models too expensive. Two roles, a classifier that labels questions and a generator that writes grounded answers, each map to a pinned model and provider in one config file with fallback routing off, so a silent switch cannot change eval scores. Structured output (answer, claims, chunk IDs) is required so the citation checker can verify every claim in code. Changing a model fails CI until a new eval run is accepted. The product names its model provider in its privacy line.

**No framework for retrieval or generation.** No LangChain or LlamaIndex, no managed vector database, no agent framework. Writing the retrieval and refusal code directly is the learning point, and fewer abstractions means every line can be explained.

**React, Vite, TypeScript.** The answer card, citation chips, source drawer and tabs are natural components. No state library and no UI kit: CSS variables from the design system and a typed API client.

**Hosting.** The frontend is a static build, so Cloudflare Pages fits. The API runs on Railway with the corpus database baked into the image.

## Ingestion

- HTML pages are fetched with httpx and extracted, with manual handling for tables such as the 90 and 150 day table. Raw HTML stays on disk next to the extracted text.
- PDFs (Form I-983 and its instructions) are read with PyMuPDF and spot-checked by eye.
- eCFR and Federal Register content comes from their official structured endpoints where available.
- Chunking splits on heading structure, one section per chunk, with document ID, section title, URL, tier and content hash on every chunk.

## Eval harness

- Hand-written questions in YAML with gold passages (document ID plus section), not gold answers.
- Scorers: Recall@5, deadline accuracy, citation correctness, refusal accuracy, refusal reason accuracy and fallback rate.
- Output is a table for the README and a JSON file of per-question results for failure reading.
- Every run records the git SHA, corpus snapshot hash, embedding model, reranker and the model and provider for each LLM role.

## Repository layout

```
f1-waypoint/
  README.md
  llm.yaml
  core/            ingestion, retrieval, generation, evals (shared)
  modules/
    opt_stem/      module.yaml, corpus, evals, prompts
  app/
    api/           FastAPI
    web/           React, Vite, TypeScript
  .github/workflows/
```
