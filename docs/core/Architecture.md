# F-1 Waypoint: Architecture

Diagrams use Mermaid.

## Modules

The product is a student platform with one module in v1. A module is a self-contained topic with its own source registry, eval set and eval results, plus a small `module.yaml` with four fields: `id`, `covered_topics`, `retrieval_threshold` and `referrals`. Shared code (ingestion, retrieval, generation, refusal gate, citation checker, eval harness, model wrapper) lives in `core/` and takes a module ID. v1 ships `opt_stem`. Adding a topic later means adding a folder, a registry, an eval set and a boundary, not changing the pipeline.

The model wrapper has two roles, a classifier and a generator. Each is mapped to a model and provider in one config file, pinned with no silent fallback. Changing a model requires a new accepted eval run.

## System overview

```mermaid
flowchart LR
  SRC["Official sources"] --> ING["Ingestion<br/>fetch, extract, chunk, index"]
  FRESH["Freshness job<br/>scheduled re-crawl, hash diff"] -.->|"triggers"| ING
  ING --> CAND["Candidate corpus"]
  CAND --> EVAL{"Eval harness<br/>regression vs last accepted run?"}
  EVAL -->|"no regression"| LIVE[("Live corpus<br/>vector index, BM25 index, registry")]
  EVAL -->|"regression"| HOLD["Held for human review"]

  STU(["Student"]) --> UI["Web UI"]
  UI --> API["API"]
  API --> PIPE["Answer pipeline<br/>screen, retrieve, gate, generate, check"]
  LIVE --> PIPE
  PIPE --> API
  API --> UI
```

Read it as two loops that meet at the live corpus. The top loop builds and vets the corpus. The bottom loop answers questions from it. They share only the registry and the indexes, and a changed source reaches students only after the eval passes.

### Offline pipeline

```mermaid
flowchart LR
  SRC["Official sources"] --> FETCH["Fetcher"]
  FETCH --> RAW["Raw snapshots"]
  FETCH --> REG[("Registry<br/>ID, URL, tier, hash, verified-on")]
  RAW --> EXTRACT["Text extraction"]
  EXTRACT --> CHUNK["Heading-based chunker"]
  CHUNK --> EMB["Embedder"]
  EMB --> VEC[("Vector index")]
  CHUNK --> BM[("BM25 index")]
```

### Online request path

```mermaid
flowchart TD
  UI["Web UI"] --> API["API"]
  API --> SCR{"Screen<br/>length cap, ID reject, prefilter"}
  SCR -->|"rejected"| REJ["Rejection, no model call"]
  SCR -->|"passed"| RET["Hybrid retrieval<br/>vector + BM25 fusion"]
  RET --> RR["Reranker"]
  RR --> GATE{"Refusal gate<br/>(detail below)"}
  GATE -->|"refuse"| REF["Refusal response"]
  GATE -->|"answerable or case-specific"| GEN["Grounded generation"]
  GEN --> CHK{"Citation checker"}
  CHK -->|"pass"| ANS["Answer or cited refusal"]
  CHK -->|"fail"| FB["Source passages only"]
  REJ --> OUT["Response to API"]
  REF --> OUT
  ANS --> OUT
  FB --> OUT
  VEC[("Vector index")] -.-> RET
  BM[("BM25 index")] -.-> RET
  REG[("Registry")] -.->|"verified-on, tier"| OUT
```

## Query sequence (Ask view)

```mermaid
sequenceDiagram
  actor U as Student
  participant W as Web UI
  participant A as API
  participant R as Retrieval
  participant K as Classifier model
  participant G as Generator model
  participant C as Citation checker

  U->>W: Types question
  W->>A: POST /ask (question, module)
  A->>A: Screen: length cap, ID reject, prefilter
  alt Screen rejects
    A-->>W: Rejection, no model call
  else Screen passes
    A->>R: Hybrid search + rerank
    R-->>A: Top passages with scores
    alt Top score below module threshold
      A-->>W: Refusal: weak_retrieval
    else Above threshold
      A->>K: Classify (question, passages)
      K-->>A: Label
      alt out_of_scope, prediction or legal_strategy
        A-->>W: Refusal with reason and referral
      else answerable or case_specific
        A->>G: Passages only, answer contract
        G-->>A: Claims with chunk IDs
        A->>C: Verify each citation
        alt Citations pass
          C-->>A: OK
          A-->>W: answerable: answer, citations, verified-on<br/>case_specific: refusal with cited general rule
        else Citation fails
          C-->>A: Fail
          A-->>W: answerable: source passages, no synthesized answer<br/>case_specific: plain refusal with referral
        end
      end
    end
  end
  W-->>U: Answer card or refusal card
```

For an answerable question, a failed citation check degrades to showing the source passages instead of a synthesized answer. For a case-specific question it degrades to a plain refusal with the referral.

## Refusal gate

```mermaid
flowchart TD
  Q["Question that passed the screen<br/>+ reranked passages"] --> T{"Top rerank score<br/>above module threshold?"}
  T -->|"no"| R4["Refuse: weak_retrieval<br/>corpus does not cover this"]
  T -->|"yes"| C{"Classifier label"}
  C -->|"out_of_scope"| R1["Refuse: out_of_scope<br/>name what the module covers"]
  C -->|"prediction"| R2["Refuse: prediction<br/>point to official case status"]
  C -->|"legal_strategy"| R5["Refuse: legal_strategy<br/>point to DSO or attorney"]
  C -->|"case_specific"| G3["Generate cited general rule"]
  C -->|"answerable"| G0["Generate grounded answer"]
  G3 --> K3{"Citation check"}
  K3 -->|"pass"| R3["Refuse the verdict, show cited rule<br/>point to DSO"]
  K3 -->|"fail"| R3b["Plain refusal, point to DSO"]
  G0 --> K0{"Citation check"}
  K0 -->|"pass"| A0["Answer with citations"]
  K0 -->|"fail"| A1["Source passages only"]
```

The threshold is checked before the classifier, so a case-specific question with weak retrieval is refused as weak retrieval. The threshold, the prefilter patterns and the classifier prompt are tuned on the eval set, and the chosen values and the failures behind them are recorded.

## Data model

```mermaid
erDiagram
  DOCUMENT ||--o{ CHUNK : contains
  DOCUMENT ||--o{ GOLD_PASSAGE : "is cited by"
  EVAL_QUESTION ||--o{ GOLD_PASSAGE : expects
  DOCUMENT {
    string doc_id PK
    string module_id
    string title
    string source_url
    string source_tier
    date retrieved_at
    date verified_on
    string content_hash
    string status
  }
  CHUNK {
    string chunk_id PK "doc_id plus section"
    string doc_id FK
    string section_title
    int ordinal
    text body
    string chunk_hash
  }
  EVAL_QUESTION {
    string q_id PK
    string module_id
    string category
    text question
    bool should_refuse
    string expected_reason "empty when answerable"
    json expected_facts
  }
  GOLD_PASSAGE {
    string q_id FK
    string doc_id FK
    string section_title "matches a chunk by doc_id plus section"
  }
```

Document IDs never change when a page is re-fetched. A chunk ID derives from the document ID plus the section, so gold passages survive a content edit unless the section itself disappears, which should break the eval on purpose.

## API sketch

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Ask view. Takes `question` and `module`, single-turn. Returns the answer, claims with chunk IDs, or a refusal |
| `GET /scripts` | Application question scripts with cited reasons |
| `GET /employer-brief` | One-page brief with citations and verified-on date |
| `GET /sources/{chunk_id}` | Source drawer content: passage, document, section, official URL, tier |
| `GET /health` | Liveness and corpus version |
| `POST /feedback` | Report a wrong answer, only after the user confirms |

```json
{
  "status": "answered | refused",
  "answer": "string",
  "claims": [{ "text": "string", "chunk_ids": ["string"] }],
  "refusal": { "reason": "weak_retrieval | prediction | case_specific | legal_strategy | out_of_scope", "referral": "DSO | SEVP | USCIS" },
  "verified_on": "YYYY-MM-DD"
}
```

Claims are allowed on a refusal only for the case-specific reason, and they go through the same citation checker.

## Freshness

A scheduled job re-crawls every source, compares content hashes, and lists changed and removed pages. A corpus update goes live only after the eval re-runs with no regression against the last accepted run. Removed pages are deleted from the index, with no leftover chunks. A page confirmed by a successful re-crawl gets a new verified-on date, and the UI marks sources as possibly out of date after 14 days.

## Failure handling

| Failure | Handling |
|---|---|
| Cited passage does not exist or lacks the claimed number | Citation check fails, passages shown with no synthesized answer |
| Near-duplicate chunks | Deduplicate by document and section before generation |
| Source page removed or restructured | Freshness job flags it, eval fails on the affected gold passages |
| Table extraction mangled | Manual spot-check, table-aware extraction, registry validation |
| Model API down or over budget | Passages only, with a notice |
| Abuse of the public demo | Layered rate limits and spend caps |
| Question about a topic with no module | Scoped refusal naming what the product covers today |
