# Reviewed source-extraction fixtures (v0.2.0)

This bounded contract prepares regression inputs for subsequent extraction work;
it does **not** broaden today's parser. Machine-readable, hand-authored expectations
are in `tests/fixtures/source-extraction/reviewed.jsonl`. No parser output was used
to author them. Independent workflow review of these expectations is recorded in
T-049's task notes. That review is structural/content adjudication by agents, not
a Latin expert's linguistic approval or calibrated accuracy measurement.

## Available evidence and selection

The [v0.1 review](extraction-content-review.md) recorded 25 synthetic CSV entries
and five sanitized APKG entries. Population A reported 10 generated, eight
incomplete, three unsupported and four ambiguous review strata. Those recorded
strata partition the 25 entries, but “generated” there excludes rev-015/016/017,
which were classified by their omitted-role warning despite generating a note.
Actual entry-level generation membership is 13/25 generated (including three
with omitted-role warnings) and 12/25 wholly skipped (five incomplete, three
unsupported, four ambiguous). The eight incomplete-warning memberships overlap
the 13 generated memberships by three; they are **not** a disjoint outcome
partition. These figures follow the recorded outcomes/tests, not a new private
population measurement. Its value-level
review covered 15 CSV and five APKG entries; ten CSV entries lacked deep review.
In particular rev-008, rev-009 and rev-021 below now receive explicit candidate
expectations rather than inheriting a claim of earlier deep adjudication.

Evidence selects em dash (rev-019), semicolon (rev-020, sanitized fixture-guid-004),
slash (rev-021), pipe alternatives (rev-009, fixture-guid-002), and the embedded
`poet.` hint (rev-008). Original **sanitized/synthetic** CSV field text is preserved
verbatim in `raw`; APKG references identify analogous boundaries, not identical
text or a claim to original private provenance. Additional role-boundary and
counterexample replays explicitly label their synthetic origin. No private deck,
private counts, source filename, or identifying metadata is needed or included.

Missing evidence remains missing: no reviewed general hint grammar, automatic
delimiter detection, unmarked omission recovery, arbitrary nested alternatives,
newline-separated roles, or morphological correctness corpus. Those require
future examples/review, not inferred support here.

## Profile-confirmation contract and matrix

For every fixture, explicitly map `Lemma` to lexical entry, `Forms` to source forms,
and `German gloss` to optional meaning in the synthetic CSV. For the representative
APKG, explicitly map `Entry`, `Construction hints`, and `German gloss` respectively,
as in [representative validation](representative-deck-validation.md). Inspect a
representative raw value and confirm roles/delimiters before extraction. A changed
delimiter profile is a new confirmation, not automatic fallback.

| Profile ID | Ordered roles (positions start at 1) | Literal delimiter / extra confirmation |
| --- | --- | --- |
| `ppp-comma` | present_infinitive, present_1s, perfect_1s, perfect_passive_participle | `,` |
| `supine-comma` | present_infinitive, present_1s, perfect_1s, supine | `,`; explicitly confirm supine instead of PPP |
| `ppp-dash` | same as ppp-comma | ` — ` |
| `ppp-semicolon` | same as ppp-comma | `;` |
| `ppp-slash` | same as ppp-comma | ` / `; slash separates roles |
| `ppp-comma-pipe` | same as ppp-comma | `,`; `|` separates alternatives within a role |
| `ppp-comma-poet` | same as ppp-comma | `,`; only trailing line-boundary `poet.` in slot 3 is a confirmed hint |
| `short-dash` | present_1s, present_infinitive, perfect_1s | ` — `; confirm three roles for this source, not a deponent detector |
| `supine-dash` | present_1s, present_infinitive, perfect_1s, supine | ` — ` |

| Selected layout | Current support / intended contract | Evidence and withholding boundary |
| --- | --- | --- |
| Literal four-slot layout, HTML text, single entity decoding | Existing promised-layout regressions; retain candidates and raw text | comma, markup, single-decode; double decoding is forbidden |
| Explicit blank/markup-only non-leading slot | Existing fixed v0.1 regression; keep later positions | omitted-perfect, exceptional-omission; withhold only omitted-role cards |
| Confirmed alternate role delimiter | Already mechanically supported by explicit profiles; additional reviewed source layouts, not new autodetection | dash, semicolon, slash; comma-profile failures remain correct |
| Pipe alternatives | New candidate-list contract; current parser keeps pipe text verbatim | pipe-alternatives; no selection or deduplication |
| Embedded hint | New bounded hint-extraction contract; current parser retains mixed form/hint display | embedded-hint; only specified hint/position, no linguistic assertion |
| Fourth-role and shorter-profile boundaries | Existing profile semantics, newly explicit expectations | ppp-fourth, supine-fourth, deponent-confirmed; endings never assign roles |
| Near-counterexamples | Withheld, not promised support | unmarked-short, extra-slot, mixed-delimiters, conflicting-hint; exact reasons in JSONL |

Candidate lists are ordered by confirmed role position; each inner list contains
alternatives, a singleton contains one candidate, and `[]` is an explicit omission.
`null` withholds the entire role assignment; it must never be interpreted as four
omissions. Hints have 1-based positions, raw evidence and retained text. `rules`
records only the applied/expected bounded transformations. Status is the reviewed
**extraction target**, not an assertion of current parser output, card eligibility,
or linguistic truth. Preserve case/macrons and multiword candidates. PPP and supine
stay distinct even when surface text is identical. The shorter `sequor` example
retains the confirmed `perfect_1s` label as a positional contract, not a claim that
`secutus sum` proves that analysis. No new German UI wording is approved here.

## Sample and denominator

All 17 JSONL records are reviewed expectation cases, selected purposively, not
randomly; repeats such as the PPP/supine pair are separate profile-conditioned
cases, **not unique source entries**. Disjoint extraction-target counts: 13
supported, three ambiguous, one unsupported = 17. Ambiguous/unsupported cases
withhold the entire assignment (four of 17); two supported cases have an omitted
role (two of 13 supported). Two supported cases carry unresolved evidence: pipe
alternatives and the `poet.` hint (two of 13 supported). These warning categories
are not a partition of generated notes or mutually exclusive in general.

No generation run or card count is claimed for this future-contract sample.
“Supported” means extractable under the stated confirmation, not “generated.”
In reports, count source entries with no generated note separately from generated
entries with omitted/withheld-role warnings; count cards separately from entries.
Historical Population A's generated-with-omission cases rev-015/016/017 are not
wholly skipped entries. This sample retains rev-015 explicitly. The original
optional-gloss case rev-002 also generates, but is not an extraction omission.

Regression consumers should assert raw provenance, profile role order, exact
candidate lists, hint positions, omissions, rules and withholding reasons against
this file. Do not regenerate expectations from the parser or automatically bless
differences. The later implementation must test both new targets and unchanged
comma-profile rejection before claiming support. These cases cannot establish
universal coverage, calibrated morphology, or correctness of private sources.
