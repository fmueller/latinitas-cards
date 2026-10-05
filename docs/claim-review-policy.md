# Claim review and calibration policy (v0.2.0)

## Current decision: automatic acceptance disabled

No numeric automatic-acceptance threshold is justified for any analyzer/category.
`assess_claim` therefore withholds every unreviewed label, segmentation and
explanation. This includes deterministic rules, CLTK/Stanza, lexical data and
LLM proposals. Self-confidence and successful extraction are not calibration.
The small synthetic results below establish the gate contract, not population
accuracy. Assigning a percentage threshold from them would be invented policy.

Before enabling automation, obtain independently adjudicated Latin-domain claims
for each versioned analyzer and category, separate development and held-out
evaluation, and publish accepted correct/errors, coverage and withheld strata.
Project review must justify a numeric precision threshold and sample adequacy
from those results (including error uncertainty), not from extraction counts or
principal-part performance transferred to future contextual token parsing.
Until then the applicable threshold is **not established: require explicit review**.

## Offline review path

The focused Python API is `latinitas_cards.claim_review`. Construct a `Claim` for
one proposal about an exact source/candidate ID and text, with `kind` (`label`, `segmentation`,
`explanation`), value, ordered alternatives, versioned analyzer identity, and
versioned `Evidence` containing reference and text. Supply the confirmed
`DeckProfile` and its review/version identifier; include the `SourceExtraction`
handoff unchanged when present. Evidence is provenance, not an approval signal.

Call `review_claim(claim, status="accepted" | "withheld", reviewer=..., reason=...)`
after inspecting that individual claim. Record the returned `ReviewDecision`;
`assess_claim(claim, decision)` exposes status, reason, claim evidence and decision.
Choosing an alternative requires a new claim/decision, not accepting the whole
analysis. Unsupported proposals cannot be accepted: replace them with supported
evidence and renew review. Use `asserted_claims` at the knowledge boundary; it
rechecks the binding so stale assessments cannot leak unsupported facts.

The SHA-256 binding includes exact candidate ID, kind/value, alternatives,
analyzer, all evidence text/versions, raw extraction, actual profile configuration
and profile version. Any change invalidates both acceptance and withholding
decisions. Versions identify the caller's evidence/configuration snapshots;
callers must supply the actual current profile/evidence, not retained stale copies.
This is an in-process domain contract, not authenticated reviewer identity or a
persistent review UI. No network call is made; future optional LLM acquisition
must be explicitly enabled as network use. No review decision contains destination
write approval. Setup's proposed fourth-role label is not explicit PPP/supine
adjudication. Existing recipe keys and source-extraction expectations are unchanged;
T-052 owns subsequent supported principal-part analysis and rendering integration.

## Reviewed evaluation material and overlap disclosure

`tests/fixtures/claim-review/evaluation.jsonl` contains 18 purposively authored
synthetic claims: six each for labels, segmentation and explanations. Expectations
were written from stipulated grammatical cases, not obtained from parser/analyzer
output. Detailed/coarse v-perfects, irregular comparison, omitted roles, PPP versus
supine and deponent boundaries are included. The independent workflow reviewers
adjudicate these expectations and policy separately from the implementer; their
results are recorded in T-051's task notes. Agent review is **not Latin-expert
approval** or evidence of real-domain analyzer precision.

Development: the unit-test proposals in `claim_review_test.py`. Evaluation: the
JSONL sample replayed in `claim_evaluation_test.py`. These intentionally overlap
in amāvī, PPP/supine and irregular examples; **this is not held-out evaluation**.
The fixture's `reason` stipulates the evidence/context of each hypothetical case;
it is not an analyzer's observed performance or independent external corpus.
The manual replay follows the authored adjudications, measuring review-contract
behavior only; it must not be advertised as measured human reviewer accuracy.

## Reproducible results and denominators

Run `uv run pytest tests/unit/claim_evaluation_test.py -v`. `summarize_evaluation`
counts accepted errors even when a mistaken reviewer approves a rejected claim.
Precision is accepted correct / all accepted; with no acceptances it is undefined,
not 100%. Coverage is all accepted / all evaluated claims, not generated cards.

All rows below concern `synthetic-proposals/v1`; no production analyzer was run.

| Mode / category | n | Accepted correct | Accepted errors | Precision | Coverage | Withheld correct / rejected / ambiguous / unsupported |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Automatic / label | 6 | 0 | 0 | undefined | 0/6 | 2 / 2 / 1 / 1 |
| Automatic / segmentation | 6 | 0 | 0 | undefined | 0/6 | 2 / 1 / 1 / 2 |
| Automatic / explanation | 6 | 0 | 0 | undefined | 0/6 | 2 / 2 / 1 / 1 |
| Explicit review replay / label | 6 | 2 | 0 | 2/2 | 2/6 | 0 / 2 / 1 / 1 |
| Explicit review replay / segmentation | 6 | 2 | 0 | 2/2 | 2/6 | 0 / 1 / 1 / 2 |
| Explicit review replay / explanation | 6 | 2 | 0 | 2/2 | 2/6 | 0 / 2 / 1 / 1 |

Each threshold decision has only six synthetic proposals, two accepted replay
claims and no held-out observations. Consequently **no numeric threshold is
approved** in any category. CLTK, Stanza, lexical-rule and LLM categories each
have zero evaluated observations and remain withheld. Sample counts are claims,
not unique source entries. No private-deck coverage or universal Latin correctness
is claimed. Extraction fixtures and T-050's retained alternatives/hints are
independent mechanical regressions; their counts do not enter this evaluation.
