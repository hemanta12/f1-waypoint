# F-1 Waypoint

Cited answers to F-1 OPT and STEM OPT questions, drawn only from official sources. When the sources do not cover a question, it says so and refuses.

F-1 Waypoint is an independent project. It is not affiliated with any government agency, and it is not legal advice. For anything specific to your case, talk to your school's DSO or an immigration attorney.

**Status:** design phase. The specs are written and the implementation has not started. No eval results exist yet. They will be published here after the first baseline run.

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
- A hand-written eval set of about 65 questions with gold passages (not gold answers), scored on Recall@5, deadline accuracy, citation correctness, refusal accuracy, refusal reason accuracy and fallback rate.

## Stack

Python 3.12 with FastAPI, SQLite with sqlite-vec, React with Vite and TypeScript, hosted on Cloudflare Pages and Railway. Details and reasoning are in [Tech Stack](docs/core/Tech%20Stack.md).

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
