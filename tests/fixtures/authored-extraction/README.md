# Synthetic extraction exercise

These two invented texts contain no private notes or corpus dependency. This is an
agent exercise, not a binary Markdown parser test. The checked-in JSONL snapshots
are expected results; the unit test validates their contracts, not extraction ability.

## Executed 2026-10-04

The implementing agent read `first.md`, the import guide, and the skill, and wrote
`first.actual.jsonl` in a scratch directory before expected JSONL existed. It then
read that actual output plus `revised.md` and wrote `reextracted.actual.jsonl` and
`changes.actual.json` separately. Only after both actual candidates were validated
and previewed were they published as the expected fixtures here. QA keys are
explicitly reviewed in the synthetic input. Vocabulary and occurrence labels were
reviewed against the source rows by the implementing agent.

| Source item | Established key | Re-extraction result |
|---|---|---|
| Harbour / Words / navis row | `vocab/navis/ship` | Unchanged; existing skip retained despite no decision column in revision |
| Harbour / In the passage / sailor | `form/harbour/sailor` | Unchanged; renamed heading and table → bullet do not change identity |
| Harbour / Check yourself / cargo | `qa/harbour/cargo` | Changed answer: “A basket of apples.” → “A basket of roses.” from separate Answer key |
| Garden / Lexicon / rosa bullet | `vocab/rosa/rose` | Unchanged; bullet → table |
| Garden / Forms / speaker | `form/garden/speaker` | Changed reference: “scene A” → “scene B”; key unchanged |
| Garden / Questions / path + Solutions / path | `qa/garden/path` | Missing from revision; retained unchanged pending review |
| Garden / Lexicon / porta row (revision only) | `vocab/porta/gate` | New, include |

The heading equivalences are explicit in the revised synthetic notes; established
section labels are retained for this exercise. In general, heading/provenance
corrections may update labels without renaming keys. The two occurrences of `amat`
keep distinct keys and distinct IDs. All six old keys and statuses survive, including
the missing included question. **That question remains preview-eligible**; decide
whether to skip it before a later export. Nothing here retires a live Anki note.

Commands executed from the repository root against both actual scratch files:

```bash
uv run latinitas-cards authored validate /tmp/t025-exercise/first.actual.jsonl --namespace synthetic-course
uv run latinitas-cards authored preview /tmp/t025-exercise/first.actual.jsonl --namespace synthetic-course
uv run latinitas-cards authored validate /tmp/t025-exercise/reextracted.actual.jsonl --namespace synthetic-course
uv run latinitas-cards authored preview /tmp/t025-exercise/reextracted.actual.jsonl --namespace synthetic-course
```

All four exited 0, with `Merged duplicates: 0`. First: form 2, qa 2, vocab 2;
include 5, skip 1; `Valid. Effective selection: 5`. Re-extracted: form 2, qa 2,
vocab 3; include 6, skip 1; `Valid. Effective selection: 6`. Representative preview
showed the same garden form ID with its corrected citation, the retained path QA,
and the newly added gate vocabulary. Additional section-filtered preview showed
the corrected cargo answer. No export command or Anki write was executed.

The actual output comparisons and identity/status checks passed: one new key,
two changed keys (answer and reference respectively), one missing retained key,
three unchanged matched keys. The checked-in `changes.expected.json` lists these
categories; missing is determined from revised notes, not JSONL set subtraction.

## Repeat the exercise

Read the notes first, without looking at expected JSONL. Write a fresh first candidate
and reconcile revised notes into a separate second candidate using the first as the
existing input. Compare to the expected files only afterward. Run validation and
preview on the actual candidates, inspect the new/changed/missing ledger, and verify
every old key/status/ID before reporting success. Do not claim this exercise was
performed merely because the expected files passed validation.
