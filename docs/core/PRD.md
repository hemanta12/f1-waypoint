# F-1 Waypoint: Product Requirements

## Summary

F-1 Waypoint answers questions about OPT and STEM OPT reporting from official documents. Every answer cites the exact source passage. The tool refuses what it cannot ground in its sources, including questions about one person's case and predictions about approvals or arrival dates. Two further views explain OPT to employers and give students scripts for the two work-authorization application questions.

The product is built as a platform for F-1 students. Version 1 ships one topic module, OPT and STEM OPT. Each future topic is its own module with its own sources, eval set and refusal boundary. Version 1 does not list or tease other topics.

It is also a portfolio project: a retrieval-augmented generation system with a hand-written eval set, published metrics, hybrid retrieval with an ablation table, and a pipeline that notices when official pages change.

## Problem

The same closed set of questions gets asked again and again on Reddit, immigration forums and university pages. The answers live in a small set of official documents, and people still get them wrong. A wrong answer can mean SEVIS termination and loss of status. The usual failure is a forgotten or misunderstood deadline. That is an information problem, not a legal problem, which is why a grounded retrieval tool fits.

Employers have a parallel problem. Many assume hiring an OPT student needs sponsorship filings or cost, so qualified candidates get screened out.

## Users

- **Student on initial OPT.** Worried about the 90-day unemployment clock, the 10-day reporting window and what counts as employment. Mobile-first, anxious, wants a fast and checkable answer.
- **Student on STEM OPT.** Needs the 150-day aggregate limit, the 6-month validation, I-983 evaluations, material changes, and the split between the SEVP Portal and the school's DSO.
- **Student job seeker.** Filling out application forms. Needs to know why "authorized to work: Yes" and "requires sponsorship: Yes" are both correct.
- **Employer or recruiter.** Wants to know what hiring an OPT student requires: no sponsorship filing, no employer cost, E-Verify rules, I-983 duties.

## Goals

1. Answer recurring OPT and STEM OPT reporting and deadline questions correctly, with a cited source passage for every claim.
2. Refuse safely and visibly when a question is out of scope.
3. Publish real eval numbers.
4. Survive source updates through content hashing, re-evaluation and delete-sync.
5. Serve the employer views from the same retrieval core, not a separate build.

## Non-goals (v1)

- Legal advice, case evaluation, or anything that depends on one person's facts.
- Predictions of any kind, such as EAD arrival or approval odds.
- H-1B fees and lottery, and nonresident tax content.
- Accounts, saved profiles, payments, or personalized deadline calculators.
- Other topic modules, and visa categories other than F-1 (J-1 and others).
- Multi-turn conversation. Each question stands alone in v1. Follow-ups are the first planned addition after launch.

## Scope: three views

- **Ask.** A free-text question in, a cited answer or an explicit refusal out.
- **Application form help.** Scripts for the two application questions, each with a cited reason the answers differ.
- **Employer brief.** A one-page, cited summary for employers: no sponsorship filing, no employer cost, E-Verify rules, I-983 duties. On screen only, with a verified-on date.

Ask and Application form help are the two tabs on the home screen. The employer brief is reached by a link.

## Answer contract

1. A short, direct answer first, in plain language.
2. The rule it comes from, with the number or window stated.
3. Inline citation markers tied to source chips.
4. One line on what the answer does not cover, when relevant.
5. A not-legal-advice line.

Hard rules: no claim without a cited passage. Every number, duration and form ID in a claim must appear word for word in a cited passage. Deadlines and numbers must cite an official source. No softening of a refusal to make a demo smoother.

## Refusal policy

| Reason | Example | Behavior |
|---|---|---|
| Weak retrieval | An obscure question with no good source passage | Refuse, say the sources do not cover it |
| Prediction | "When will my EAD arrive?" | Refuse, no estimate, point to official case status |
| Case-specific | "I reported late, is my status gone?" | State the general rule with a citation, refuse the verdict, point to the DSO |
| Legal strategy | "Should I switch visa types?" | Refuse, point to the DSO or an attorney |
| Out of scope | H-1B lottery, tax forms, J-1 | Scoped refusal naming what the product covers |

Order of checks: input screen, then retrieval, then the score threshold, then a classifier that labels the question. A case-specific question with weak retrieval is refused as weak retrieval. A refusal is a designed, calm state, not an error.

## Success metrics

Targets are starting points and get reset after the first baseline run.

| Metric | What it measures | Target |
|---|---|---|
| Recall@5 | The right source passage is in the top 5 results | 0.85 or higher |
| Deadline accuracy | The stated number and reporting channel match the gold answer | 95 percent or higher |
| Citation correctness | Cited passages exist, and every number and form ID appears in them | 100 percent |
| Refusal accuracy | Refusal cases are refused, answerable cases are answered | 100 percent on refusal cases, 90 percent or higher on answerable |
| Refusal reason accuracy | A refusal has the right reason and the right referral | 100 percent on the eval set |
| Fallback rate | Answerable questions where the citation check fails and only passages are shown | Tracked, reviewed if above about 5 percent |
| Freshness | Time to detect a changed official page | Within one scheduled run |

The eval set is about 65 hand-written questions with gold passages (not gold answers): deadline and clock, reporting, I-983, employer, near-miss answerable questions that sound risky, and refusal cases covering every refusal reason with paraphrases. A hand-graded sample checks that cited passages actually support each claim.

## Sources and privacy

- **Official sources only.** USCIS, DHS Study in the States, ICE and eCFR documents are the only basis for answers. No university guides are used; when a question depends on a school's own process, the answer points the student to their school's international office. Every citation shows its source in plain words.
- No accounts. Raw question text is not logged. Questions are sent to a model provider, which the product names. The input screen rejects A-numbers and receipt numbers.
- A "Report a wrong answer" action stores a question only after the user confirms.
- The product never implies government affiliation.

## Risks

| Risk | Mitigation |
|---|---|
| Legal advice drift | Refusal gate, disclaimer on every answer, refusal cases in the eval |
| Stale rules | Hash-based freshness job, eval gate on updates, visible verified-on date |
| Inflated scores | Hand-written eval, gold passages rather than gold answers |
| Table extraction errors | Manual spot-check, table-aware chunking |
| Cost abuse on a public demo | Layered rate limits and spend caps |
| Scope creep | Non-goals list, module boundary, home screen states exactly what is covered |
| Looking like an official site | Visual rules in the design system |
