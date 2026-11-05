# F-1 Waypoint

A question-answering product for F-1 students that answers only from official sources, cites every claim, and refuses what it cannot ground.

## Language

**Module**:
A self-contained topic with its own corpus, eval set and refusal boundary. v1 has one: `opt_stem`.
_Avoid_: Mode, plugin, topic area

**Audience**:
Who is asking: student or employer. It has no switch of its own: the employer reaches their view by a link. Never changes the retrieval or the rules.
_Avoid_: Mode, role, persona

**View**:
One of the three screens a user can use: Ask, Application form help, Employer brief.
_Avoid_: Mode, Mode 1, Mode 2

**Corpus**:
The frozen set of official documents a Module answers from.
_Avoid_: Knowledge base, dataset

**Refusal**:
A deliberate, cited-where-possible non-answer with a reason and a referral to an official channel. Not an error.
_Avoid_: Failure, error, fallback

**Refusal reason**:
Exactly one of `weak_retrieval`, `prediction`, `case_specific`, `legal_strategy`, `out_of_scope`.

**Case-specific question**:
A question whose answer depends on one person's facts. The product states the general rule from the Corpus and refuses only the verdict.
_Avoid_: Personal question

**Gold passage**:
The document and section that proves the correct answer to an eval question. Eval targets passages, not answers.
_Avoid_: Gold answer, ground truth

**Verified-on date**:
The date of the last successful re-crawl that confirmed a source page, whether or not its content changed. Older than 14 days marks sources as possibly out of date.
_Avoid_: Last updated

**Source tier**:
Whether a document is `official` (government publication) or `university` (a school's guide). Deadlines and numbers come from `official` only, and every citation shows its tier.
_Avoid_: Source quality, trust level
