# Extraction content review (T-032)

This note records the stratified extraction-content review executed for T-032. It
adjudicates representative content against the currently documented parser and
normalization contract in [principal-part-parsing.md](principal-part-parsing.md).
It is a review record, not a release claim: the reviewed counts below are counts
of the committed synthetic and sanitized populations, not a claim of universal or
private-deck correctness.

## Scope, method, and limits

Two committed populations were reviewed; no other content was accessed. No
private deck, collection, or export data was used, and none of the reviewed
values derive from a private source:

- Population A: `tests/fixtures/extraction-review-corpus.csv`, 25 synthetic entries
  written for this review against a comma-separated four-role profile
  (`present_infinitive, present_1s, perfect_1s, perfect_passive_participle`,
  the fourth role being `Partizip Perfekt Passiv (PPP)`).
- Population B: the committed sanitized fixture
  `tests/fixtures/representative-university-latin.apkg`, 5 sanitized entries,
  reviewed in full under its approved profile.

Selection is reproducible: every entry carries a hand-derived recorded outcome
(status plus skip code, derived from the documented matrix and safe-HTML
contract, not from running the pipeline); entries are grouped by that recorded
outcome, sorted by stable source ID, and the first two of each stratum are
deep-reviewed. The committed test `tests/unit/extraction_content_review_test.py`
verifies that the pipeline reproduces every recorded outcome, that the selection
is deterministic, and that this note states the same counts.

Population totals and per-category counts (population A):

Population A totals: 25 entries — generated: 10, incomplete: 8, unsupported: 3,
ambiguous: 4.

| Category | Count |
| --- | --- |
| generated | 10 |
| incomplete | 8 |
| unsupported | 3 |
| ambiguous | 4 |

Population B: generated 3, incomplete 1, unsupported 1, ambiguous 0.

These are the historical extraction strata, not today's disjoint generation
outcomes. With linguistic review gates, population A generates 13/25 entries
(the ten full entries plus three optional omissions), wholly skips 12/25, and
has 13 generated entries with warnings. All thirteen require individual fourth-role
linguistic review; two also contain unresolved source evidence. Population B
generates 3/5, wholly skips 2/5, and has 3 generated entries with warnings: two
need fourth-role linguistic review and one has unresolved alternatives in all roles.
Warning memberships overlap generated outcomes; multiple warnings on one entry
count once. Clean extraction is not PPP/supine claim approval.

The review deep-reviewed 15 population-A entries (two per stratum, covering
every distinct outcome class) plus all 5 sanitized entries; the unreviewed
remainder: 10 population-A entries (`rev-003` through `rev-010`, `rev-017`,
`rev-021`) share a recorded classification check but did not receive
value-level adjudication. A structural parse or separator success alone does not
establish semantic eligibility, and these synthetic counts are not private-deck
accuracy figures.

## Adjudicated causes

| Cause | Reviewed findings |
| --- | --- |
| Mapping error | None found in the reviewed sample: the confirmed profile fields mapped the intended source fields for every sampled entry. |
| Extraction/normalization defect | One demonstrated, reproduced, and fixed in-contract defect (below). No additional in-contract defect was found in the reviewed sample. |
| Genuinely missing data | `rev-011` (blank principal-parts field), `rev-002` and sanitized `fixture-guid-005` (blank optional German gloss). Missing optional data stays a generated note with an empty meaning; missing required data stays a structured incomplete skip. |
| Unsupported layout | `rev-019` (em dash), `rev-020` (semicolons), `rev-021` (slashes) against the comma profile; sanitized `fixture-guid-004`. Correct structured `unsupported/separator_mismatch` outcomes, no fallback parsing. |
| Unresolved role | `rev-023`/`rev-025` (unmarked or mixed-hint short layouts), `rev-024` (multi-object lemma), `rev-009` and sanitized `fixture-guid-002` (pipe alternatives kept verbatim as display text). No role is shifted, inferred, or silently selected; these remain review items for v0.2.0. |

## Demonstrated defect and correction

The reproduced in-contract defect: raw nonempty markup-only values (for example
the historical synthetic `amo — amare — <b></b> — amatum`) passed parsing as
non-omitted parts, generated notes with no skips, emitted completion and
recognition cards for the markup-only role, rendered a blank perfect answer, and
recognized an empty string; a markup-only lexical entry additionally crashed
note construction. The correction normalizes semantic display text exactly once
in the parser (source markup, comments, and encoded whitespace that carry no
readable text are omitted roles exactly like an explicit blank slot), preserves
the raw source segment separately as provenance (`PrincipalPartValue.raw`,
`ParsedPrincipalParts.raw_lexical_entry`), and escapes already-normalized text at
rendering without decoding entities a second time. Leading required roles that
normalize to nothing are structured `incomplete/required_role_omitted` outcomes;
non-leading ones keep the coherent note, report one
`incomplete/omitted_principal_part` skip, and emit no card for the omitted role.
No PPP/supine inference, omitted-role shift, or silent alternative selection was
added; broader alternatives, omissions, mixed hints, and calibrated morphology
remain v0.2.0 scope, and the T-026 safe-HTML rendering contract is unchanged.

## Terminology boundary

The historically approved German role labels (`Infinitiv`, `Präsens, 1. Person
Singular`, `Perfekt, 1. Person Singular`, `Partizip Perfekt Passiv (PPP)`,
`Supinum`) are reused unchanged. This review and the correction introduce no new
user-visible German wording; the German glosses inside the synthetic corpus are
test data, not interface terminology.
