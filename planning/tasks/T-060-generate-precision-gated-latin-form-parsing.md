---
id: T-060-generate-precision-gated-latin-form-parsing
title: Generate precision-gated Latin form parsing exercises
status: completed
priority: medium
spec_ref: specs/v0.2.0.md#calibrated-form-parsing
dependencies:
    - T-052-generate-supported-principal-part-relationships
    - T-058-expose-deterministic-managed-plans-and-operation
updated_at: "2026-10-06T19:58:53Z"
---

# T-060-generate-precision-gated-latin-form-parsing Generate precision-gated Latin form parsing exercises

## Description

Add the distinct later form-parsing exercise step using the same evaluated claim policy, without treating principal-part comparison as proof of full token analysis.

## Acceptance

- Map encountered forms to lemma and only applicable accepted features such as person/number/tense/mood/voice/case/gender; use Latin-first prompts and explicit required-role eligibility.
- Specify stable object/task/template bindings before generation. Reuse coherent-object identity where appropriate; do not merge unrelated contextual objects or repurpose existing principal-part template slots. Document the supported preview/export and manual reference-setup route for new bindings; refuse incompatible managed application rather than silently provisioning templates.
- Ambiguous forms expose alternatives and review reasons rather than unconditional assertions; unsupported features and withheld claims cannot appear as facts or generate misleading cards.
- Automatic generation requires domain-relevant measured precision and approved per-claim policy. Optional analyzers/LLMs do not make default installation or core operation network-required.
- Preview the accepted claims and skips. Structural card additions under the initial managed CSV scope remain blocked for application; generation does not authorize destination changes.
- Tests include asymmetric ambiguous forms, absent features, and accepted versus rejected claims; principal-part completion/recognition stay unchanged.
- Repeat native update/presentation checks affected by parsing changes before claiming that scope verified or release-ready; earlier existing-recipe evidence cannot certify new behavior. These retests are part of this task's completion, not a reason to delay the initial native gates.

## Verification

Use red/green tests and the mandatory ruff, mypy, pytest -v chain. Record reviewed/evaluated linguistic evidence and affected native retest results for the supported exercise scope.

## Implementation evidence — 2026-10-05

Implementation delivered; lifecycle acceptance remains partial. The authorized
native-device deferral does not waive this task's completion criteria. Do not
claim native compatibility or release readiness. T-062 remains independently
blocked and T-038 publishing remains outside this resumed scope.

The offline `form-parsing preview`/`export` path uses individually reviewed
contextual claims, not principal-part comparison as token proof. No automatic
acceptance policy or calibration threshold was invented. Required lemma plus
declared applicable features use fresh per-claim review. Alternatives, provenance,
withholding and conflicting acceptance reasons remain inspectable. The sanitized
puellae fixture distinguishes plural nominative subject from singular genitive/
dative alternatives; its optional unreviewed gender is never asserted. These
explicit fixture judgments are not measured analyzer precision.

The v1 contextual object namespace, dedicated note type, semantic retrieval key
and frozen ordinal-0 template are documented in
`docs/contextual-form-parsing.md`. The principal-part registry/schema is unchanged.
Personal Notes is absent from the CSV. Managed contextual application/provisioning,
structural additions, retirement/reactivation and migration remain unsupported;
the existing managed schema guard refuses the contextual schema.

### Workflow-v3 evidence

| Step | Executed evidence | Outcome |
|---|---|---|
| 1 | Fetched origin/main, confirmed accepted baseline containment; Taskrail validate and next JSON | Valid, exactly T-060 on specs/v0.2.0.md |
| 2 | Focused pytest initially failed collection: missing form_parsing module | RED; minimal domain/command implementation then six focused tests GREEN |
| 2 | Personal Notes omission regression failed against exported CSV | RED; omission fix GREEN; conflicts/escaping/schema refusal covered |
| 3 | Exact Ruff/mypy/pytest chain initially found command-group test regression | Fixed test's new subgroup traversal; full chain restarted, 834 passed |
| 4 | Dedicated Task loaded code-simplifier | No changes; 56 focused tests, Ruff and mypy passed |
| 5 | Parallel separate Tasks loaded code-reviewer for General, Python, Security, Database | General/Database no findings; Python F-1 and Security SEC-1 below |
| 5 | Fresh candidate-validation Task loaded code-reviewer | Both independently reproduced/validated; distinct, none rejected |
| 6 | New whitespace/bidi CLI tests failed three cases, then passed three | Both findings fixed with strict RED/GREEN |
| 7 | Fresh exact uv run ruff check; uv run mypy; uv run pytest -v | All checks passed; 87 source files; 837 tests passed |
| 7 | Fresh disposition-verification Task loaded code-reviewer | Both RESOLVED, no new issues; independently repeated full gate, 837 passed |
| 8 | Verification records implementation checks but required native client evidence incomplete | Fail/blocked rather than falsely completed |

Selected lanes cover correctness, Python JSON/typing/error behavior, untrusted
source/terminal/HTML and file ownership, and native database/import evidence.
Three specialists fit the default budget. Framework and ML/RAG lanes omitted:
no web framework, analyzer, LLM or calibration policy was introduced. Database
companions inspected the SQLite/native harness rather than asserting PostgreSQL
applicability. General and Database returned: "No concrete task-relevant findings."

### Verbatim validated review findings and dispositions

- **F-1 (Python, low):** "Reject whitespace-only proposal values and analyzer names
  during input validation, or translate claim-construction validation errors into
  a user-facing CLI error." Evidence: proposal min_length=1 admitted blanks;
  claim construction outside the preview load error boundary raised ValueError.
  **Fixed:** nonblank proposal model validation; value/analyzer CLI regressions
  failed exit 1 versus required exit 2, then both passed. Fresh reviewer: RESOLVED.
- **SEC-1 (Security, low):** "Escape bidi control characters in user-authored preview
  values before emitting them to the terminal; otherwise a crafted JSON input can
  visually reorder or disguise preview content." Evidence: ensure_ascii=False
  emitted literal U+202E from context/evidence.
  **Fixed:** standard ASCII JSON escapes preserve machine-readable original data;
  regression failed raw U+202E then passed visible escape and lossless parsed data.
  Fresh reviewer: RESOLVED. No review findings were deferred.

### Native/backend and presentation retests

Fresh `uv run --with anki==26.9.3 python scripts/check-managed-anki.py` passed all
16 existing top-level scenarios. Native report SHA-256:
`eedbdb757a884e943f46f8009474dbbb827b0bcc000c0eb88e3716402a7dcc3b`.
This rechecks actual managed content/tag safety and structural refusal; it does
not authorize contextual managed updates or certify parsing presentation.

Fresh `uv run --with anki==26.9.3 python scripts/check-form-parsing-anki.py`
passed one explicitly reviewed contextual note/card, exact imported/native
question/answer content, ordinal 0, unmapped Personal Notes and identical no-op
notes/cards/revlog/notetypes/fields/templates/decks/deck_config tables. The new
disposable collection has no existing review history; this is not a scheduling-
preserving migration proof. Native report SHA-256:
`509202296c133fa01464555c3e39486d5f9699eccf329778b3f463b677475323`.
Backend version 26.9.3, Python 3.14.2, Linux x86_64. Test collections were isolated;
no user collection was opened. Harness provisioning is fixture-only.

Chromium 2x rendered/inspected muted-light and monochrome-dark revealed answers
and changed-context withholding. The final screenshot is legible, shows only
accepted lemma/case/number, no optional gender/tense and no changed-context card.
This is browser evidence only. AnkiMobile iPhone/iPad presentation and affected
Desktop GUI manual setup/import/reveal light/dark checks remain **open/deferred**.
AnkiMobile is unavailable in this Linux orb. The earlier T-062 partial native
Desktop evidence covers existing recipes, not these new bindings. Leave this
task blocked for those native retests; hand implementation to the parent's
approved synthetic adversarial E2E phase without marking acceptance complete.

Taskrail verification at **2026-10-05T07:08:19Z** recorded `fail` for the missing
native acceptance, not failing implementation checks. `taskrail block` recorded
the specific open retests; subsequent `taskrail validate` reported `state valid`
and `taskrail next --json` reported `no eligible task`. No second task was started
and no duplicate follow-up was created: the outstanding native work remains
owned by this task and T-062.

### Native retests — 2026-10-06

- Unblocked and started through Taskrail after T-062's native session. The same
  maintainer-run kit included two contextual parsing decks (`G` muted/light, `H`
  monochrome/dark) exported by the `form-parsing export` CLI. The carrier package
  delivered the parsing note type per the maintainer's 2026-10-06 waiver of hand-typed
  setup.
- Fresh import through the native Anki Desktop 26.09.2 dialog created 2 notes/cards per
  deck. On Desktop and on AnkiMobile 25.09 (iPhone 16 Pro Max, iOS 26; iPad Air M4,
  iPadOS 26) the maintainer confirmed: Latin-first prompt, reveal, only accepted lemma,
  case and number for *puellae* (the unreviewed optional gender is absent), all six
  accepted features for *amāvit*, no card for the withheld *Rosam puellae dat.*, and
  readable light/dark plus client dark mode.
- The maintainer accepted Latin terminology for now. Configurable Latin/user-language
  terminology is specified as v0.2.1 T-065.
- Limitations: no screenshots; synthetic stipulated cases; only the tested clients. See
  `docs/contextual-form-parsing.md` and `docs/morphology-native-verification.md`.
- Chain: `uv run ruff check` passed, `uv run mypy` passed (87 files), `uv run pytest -v`
  passed (838).

## Implementation Notes

- 2026-10-05T07:08:19Z: verification fail
- 2026-10-05T07:08:19Z: Implementation delivered and reviewed, 837 tests and native backend retests pass. Required new contextual bindings AnkiMobile iPhone/iPad presentation and affected Desktop GUI manual setup/import/reveal/light-dark retests remain open under authorized device deferral. Browser/backend evidence is not native presentation proof; no lifecycle completion or release readiness. Hand off to approved parent synthetic adversarial E2E phase.
- 2026-10-06T19:58:53Z: verification pass
