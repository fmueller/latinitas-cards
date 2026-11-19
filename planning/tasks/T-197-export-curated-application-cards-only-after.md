---
id: T-197-export-curated-application-cards-only-after
title: Export curated application cards only after rendered approval
status: completed
priority: medium
spec_ref: specs/v0.5.0.md#curated-grammar-and-mnemonics
dependencies:
    - T-196-validate-provenance-bound-curated-rule-and
    - T-195-track-export-receipts-and-installed-card
updated_at: "2026-11-19T06:41:01Z"
---

# T-197-export-curated-application-cards-only-after Export curated application cards only after rendered approval

## Description

P50-6 curated preview/recipe/export. Own deterministic application, mnemonic-recall,
formation-rule recall and component-explanation recipes, approval bindings and withdrawal
integration. Share provenance/export safety, not automatic policy. Follow approved
contracts and spec execution/validation requirements.

## Acceptance

- Default emphasizes rule application to Latin/context; mnemonic recall is opt-in.
  Each recipe has one objective. Preview prompt, answer, mnemonic, explanation,
  caveats and provenance; save effective recipes/selection for reruns.
- Require formation-rule recall without a mnemonic cue and component explanation as
  distinct opt-in recipes in v0.5.0, independently selectable through profiles/CLI
  in public preview/export. An explanation only on a production card's answer does
  not satisfy these recipes. Link related drills via shared source/rule provenance
  without automatically multiplying cards.
- Recall supported verbal/nominal formation rules with applicability and exceptions;
  explain stems/bases, formation markers, endings and supported adjustments. Cover
  the spec's first-conjugation imperfect rule, amābās component roles and nominal
  puellārum contrast using cited, human-approved answers and legitimate alternative
  segmentations. Do not require historical/etymological explanations or exhaustive
  v0.5.5 morphology coverage. Unsupported/ambiguous analyses remain withheld.
- Own integration of the saved opt-in grammar-study preset with T-191's production
  recipes. Enable production, formation-rule recall, component explanation and mnemonic
  recall together; persist preset/effective selection in the profile for preview/export
  reruns and allow individual overrides under approved precedence rules. Preview/reports
  show effective choices. Do not change unselected-profile defaults, implicitly enable
  clozes, enumerate every form, invent missing content or bypass either approval model.
- Support German and English prompts, reference answers, terminology, explanations,
  caveats and reports for all four preset types; persist the output-language choice.
  The spec's bilingual noun/adjective designs are illustrative, not approved content.
  Mnemonic alternatives require separate language-specific review; missing approved
  content is reported/withheld for that recipe without blocking other eligible recipes.
- Export requires explicit rule/explanation AND actual rendered-card approval, plus
  language-specific mnemonic/association approval whenever a mnemonic is displayed on
  either side of any recipe. Approved rule-only items support non-mnemonic recipes;
  missing mnemonic content/approval does not block those cards. Unchanged approvals
  persist; wording/caveat/rendered-content changes require affected renewal, including
  theme/presentation changes under the approved policy. Mnemonic-only edits do not
  invalidate unchanged approved rule-only cards.
- Preserve item identity across corrections and distinct recipe/semantic exercise
  keys, CSV ownership and personal notes. Do not duplicate free-form qa intake.
- Withdrawal/invalidation uses receipts and supported managed/manual handoffs.
  Automatic linguistic success cannot bypass any required human approval.

## Verification Notes

Exercise neither/rule-only/rendered-only/both required approval cases for rule-only cards;
add missing/unapproved/approved mnemonic cases for every recipe that displays one.
Test unchanged rerun, content/theme edits, recall opt-in, withdrawal and personal-field
preservation through public preview/export. Run the mandatory validation chain.

For formation-rule recall/component explanation, exercise each recipe independently,
default exclusion, no mnemonic cue/answer leakage, no implicit production/recall sibling
cards, reference-backed answers, unsupported/ambiguous analyses and approval invalidation
after rule/explanation changes. Automatic form acceptance must not bypass curated approval.

For grammar-study, save/reload the profile and preview/export all four eligible types;
contrast one recipe override, no-preset defaults and missing content/evidence/approval.
Verify truthful partial output, no implicit clozes and stable nonduplicated reruns.
Run these journeys in German and English, retaining the selected language on reload;
contrast a separately approved mnemonic alternative with missing language-specific content.
With an approved rule/explanation and rendered cards but no mnemonic at all, export
rule recall/component explanation and report mnemonic recall unavailable. Repeat with
approved German mnemonic content but no English mnemonic: approved English rule-only
cards still export. Editing/withdrawing the German mnemonic invalidates affected cards
without invalidating unchanged rule-only cards or bypassing installed-content handoff.

## Implementation Notes

Implemented finite application, formation-rule recall, component explanation and
opt-in mnemonic recall through public curated preview/review/export. Saved shared
profiles retain effective language, preset, recipe overrides and selection;
grammar-study composes independent production acceptance without changing defaults
or generating missing content. Exact rule/rendered and every displayed mnemonic's
language-specific approval are required. Curated CSV uses an explicit separate
note type, stable identities and unowned personal fields/tags.

Receipt-backed withdrawal compares all historical exports, preserves stale sibling
handoffs and supports empty current intake. Seen decisions install atomically before
replay; old histories cannot erase withdrawals. Separate production publication
failures report recovery-required uncertainty, not zero export or rolled-back
curated output. German/English safety guidance and static rendered review are
available; no Anki writes or native preservation claims are made.

TDD and required simplification were followed by six independent scoped review
lanes, boundary adversarial verification and an overall verifier. Two correction
rounds resolved empty-source handoffs, replayed rejection retention, interrupted
history writes, truthful partial publication, localization/prose and discriminating
CSV/ownership checks. The new status-label finding received one targeted follow-up;
the broad review campaign was not restarted. No review defects remain open.

Final reviewed source patch SHA-256:
4edb55486cbb9fba15f896612f2a8dc06ec12c7ea1fc4254ee7cb20506f43b98.
The exact final Ruff -> mypy -> pytest chain passed: 3762 passed, 7 skipped,
one warning; strict mypy checked 234 source files. Formatting, schema drift and
diff checks passed. Public manual acceptance used 42 actual CLI subprocesses on
original synthetic verbal/nominal material in German/English; seven static
render captures were inspected, including both card themes and a narrow viewport.
Synthetic reviewer identities establish engineering behavior, not real linguistic
approval. No mutation adapter was installed; targeted deliberate regressions
demonstrated that strengthened CSV/sibling assertions fail on wrong behavior.

Implementation completion is not independent outer acceptance. T-200 inference,
studies and evaluation remain paused pending concrete-plan approval. T-201 still
owns real-Desktop Anki preservation/resolution acceptance; offline receipts do not
establish it. Outer final adversarial QA/native German-card video and maintainer
publication remain separate gates. No successor task was taken.
- 2026-11-19T06:40:51Z: verification pass
- 2026-11-19T06:41:01Z: Implementation complete following pass verification at 2026-11-19T06:40:51Z; final reviewed source and Ruff/mypy/pytest evidence preserved. Synthetic engineering approval tests only; independent outer acceptance, T200 inference approval and T201 native Anki preservation remain separate. No successor taken.
