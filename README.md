# F-1 Waypoint

Cited answers to F-1 OPT and STEM OPT questions, drawn only from official sources. When the sources do not cover a question, it says so and refuses.

F-1 Waypoint is an independent project. It is not affiliated with any government agency, and it is not legal advice. For anything specific to your case, talk to your school's DSO or an immigration attorney.

**Status:** in development. The corpus snapshot, eval set, chunking and vector search are built. Hybrid retrieval, the refusal gate and answer generation are next. See [Progress](#progress).

## Why

The same OPT questions get asked again and again on Reddit, forums and university pages, and the answers sit in a small set of official documents. People still get them wrong, and a missed deadline can mean SEVIS termination. Employers have a parallel problem: many assume hiring an OPT student requires sponsorship or cost, and screen out qualified candidates.

## What it does

- **Ask.** A free-text question in, a short answer out. Every claim carries a citation to the exact source passage, and every number, duration and form ID in the answer appears word for word in a cited passage.
- **Application form help.** Scripts for the two work-authorization questions on job applications, with a cited reason why "authorized to work: Yes" and "requires sponsorship: Yes" are both correct.
- **Employer brief.** A one-page cited summary of what hiring an OPT student involves, with a verified-on date.

## What it refuses

Questions about one person's case, predictions such as EAD arrival or approval odds, legal strategy, and topics outside OPT and STEM OPT. A refusal names the reason and points to the right next step. It is a designed state, not an error page.

## How it works

A retrieval-augmented generation system built without an orchestration framework.

- Hybrid retrieval: BM25 plus vector search, then a reranker. The BM25 tokenizer keeps form numbers like I-983 and citations like 8 CFR 214.2(f)(10) intact.
- A refusal gate runs before generation: input screen, retrieval score threshold, then a question classifier.
- A citation check runs after generation. If it fails, the user sees the source passages instead of an unverified answer.
- A scheduled job hashes the official pages, detects changes, re-runs the eval and syncs deletions.
- An eval set of 67 questions with gold passages (not gold answers), drafted with AI and reviewed by hand, scored on Recall@5, deadline accuracy, citation correctness, refusal accuracy, refusal reason accuracy and fallback rate.

## Progress

**Retrieval baseline (Phase 3):** Recall@5 is **0.89** on the 55 eval questions that have gold passages. That is vector search only (BGE-small, cosine similarity over about 1,300 chunks), with structure-aware chunks of up to 150 words, copies across pages removed and split sections joined. A gold passage may list other official passages that state the same rule; finding any of them counts.

Chunking comparison, same embedding model and retriever for every row, before copy removal:

| Chunking | Chunks | Recall@5 |
|---|---|---|
| Structure-aware, 150 words | 1,331 | 0.87 |
| Structure-aware, 200 words | 1,005 | 0.76 |
| Structure-aware, 300 words | 670 | 0.73 |
| Fixed-size, 150 words | 1,323 | 0.73 |
| Fixed-size, 300 words | 664 | 0.71 |
| Fixed-size, 200 words | 992 | 0.60 |

Structure-aware chunks split at section headings and never inside a sentence. Fixed-size chunks cut every N words with a 20 percent overlap. The other-passage alternatives were found by reviewing the structure-aware 150 run's misses, so the gap to the other rows may be somewhat overstated.

**Hybrid retrieval (Phase 4):** Recall@5 is **0.96**, up from 0.89. Each question is searched two ways, by vector search and by keyword search (BM25). The two result lists are merged by reciprocal rank fusion, and a cross-encoder reranker picks the top 5. Every piece was kept only if the eval showed it helped:

| Method | Recall@5 | Right passage among candidates | Notes |
|---|---|---|---|
| BM25 alone | 0.62 | 0.69 | Own tokenizer keeps I-983 and 8 CFR 214.2(f)(10) whole; dropping common words raised it from 0.51 |
| Vector alone (BGE-small) | 0.89 | 0.89 | Phase 3 baseline |
| Hybrid, no reranker | 0.85 | 0.96 | Fusion alone lowers the top 5 but finds the right passage more often |
| Vector + MiniLM-L-12 reranker | 0.89 | 0.89 | A reranker cannot help without BM25's extra candidates |
| Hybrid + MiniLM-L-6 reranker | 0.95 | 0.96 | |
| **Hybrid + MiniLM-L-12 reranker** | **0.96** | 0.96 | Chosen; about 0.6 s per question on a laptop CPU |
| Same chain with BGE-base embeddings | 0.93 | 0.95 | The bigger embedding model was worse |

BM25 did not win on form numbers, which vector search also handled. It won on exact domain phrases such as "unemployment clock" and "volunteer", where vector search ranked the right passage far down. The two remaining misses are vague or differently worded questions, such as "how long does OPT last", where the retrieved passages state other durations. Phase 4 also removed website menus and banners that sat inside the pages' main content; the vector baseline stayed at 0.89.

## Stack

Python 3.12 with FastAPI, SQLite for chunks and vectors, BGE-small embeddings, BM25 through bm25s and a MiniLM-L-12 reranker, the models through fastembed (ONNX Runtime, no PyTorch), React with Vite and TypeScript, hosted on Cloudflare Pages and Railway. Details and reasoning are in [Tech Stack](docs/core/Tech%20Stack.md).

## Docs

- [PRD](docs/core/PRD.md): problem, users, scope, answer contract, refusal policy, success metrics
- [Architecture](docs/core/Architecture.md): modules, pipelines, data model, failure handling
- [Design System](docs/core/Design%20System.md): principles and visual rules
- [Tech Stack](docs/core/Tech%20Stack.md): choices and reasoning
- [Glossary](GLOSSARY.md): shared terms

## Privacy

No accounts. Raw question text is not logged. Questions are sent to a model provider. The input screen rejects A-numbers and receipt numbers.

## Sources

Answers come from public pages and forms of the U.S. Citizenship and Immigration Services (USCIS), U.S. Immigration and Customs Enforcement (ICE) and the Department of Homeland Security (DHS), plus the Code of Federal Regulations via eCFR. The snapshot in `modules/opt_stem/corpus/` holds copies saved on the dates in `registry.json`, with the original URL for each in `modules/opt_stem/sources.yaml`. The live page is always the authority. This project is independent and not affiliated with any of these agencies.
