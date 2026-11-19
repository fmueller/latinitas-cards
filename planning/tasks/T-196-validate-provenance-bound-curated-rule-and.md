---
id: T-196-validate-provenance-bound-curated-rule-and
title: Validate provenance-bound curated rule and mnemonic drafts
status: completed
priority: medium
spec_ref: specs/v0.5.0.md#curated-grammar-and-mnemonics
dependencies:
    - T-187-implement-versioned-automatic-decisions-and-reuse
updated_at: "2026-11-19T04:56:48Z"
---

# T-196-validate-provenance-bound-curated-rule-and Validate provenance-bound curated rule and mnemonic drafts

## Description

P50-6 versioned curated intake using applicable authored conventions and focused
domain/command modules. Own item schema, validation and rule/optional-mnemonic review;
do not silently recast qa notes. Follow approved contracts and spec execution/validation
requirements.

## Acceptance

- Validate stable author keys, precise rule, optional original mnemonic wording/language,
  examples, caveats/exceptions, tags and course/lecture provenance. Separate rules
  from memory aids, and quoted course wording from suggestions. Accept rule-only items
  without placeholder mnemonics; record rule/explanation approval independently of
  approval for a language-specific mnemonic and its association with the rule.
- Document/test agent extraction from permitted lecture notes into a candidate file.
  Prefer course aids; optional generated aids stay marked candidates. No mandatory
  hosted processing or automatic authoritative import.
- Preview candidates and record explicit rule and optional-mnemonic approval/rejection.
  Preserve German wordplay; English alternatives need separate review. Changed rule/
  explanation/mnemonic/caveat invalidates affected approval, not unrelated rule-only
  cards on a mnemonic-only edit. Any displayed mnemonic requires approval, not only
  mnemonics used by the recall recipe.
- Source rights and extraction/provider permission are explicit; a later local model
  does not make earlier hosted extraction local. Automatic form acceptance cannot
  approve curated rules or mnemonics.

## Verification Notes

Check malformed/missing provenance, quoted/suggested drafts, approved/rejected variants,
rule-only intake, stable keys after edits and unchanged approval reuse. Contrast approved
English rule/explanation content with no English mnemonic and an approved German mnemonic.
Export needs rule AND rendered approval, plus mnemonic approval whenever one is shown;
this task cannot authorize it alone. Run the mandatory validation chain.

## Implementation Notes

Fetched origin/main base: a5199d9afea7f40078b45a4520cbcd140e73332b. Before supported
start, Taskrail reported state valid, T-196 as sole eligible candidate, off_spec=false,
and no active owner. This worker owns only T-196, not successor selection.
Pinned spec SHA256 fbbea23eabd998d0dee79c1582a76834da1716c145a4c223334272eef51a1f6a
and approved contract SHA256 c216cc31283eb1a3cfbcd90809f3166b68bdf564f4a081e498a391176d010f08
remain unchanged from T-186's exact adoption record.

Implemented focused curated domain/command modules and strict v1 candidate,
approval and companion permission schemas. Validate/preview/review are offline;
candidate inspection reports exact wording, evidence, caveats, provenance,
fingerprints and independent statuses. Rule-only items have empty mnemonic arrays,
not placeholders. Existing authored QA intake is unchanged. A new history output
retains prior decisions; explicit supersession resolves conflicts, never timestamp
precedence. Item keys persist across edits. Rule fingerprints include example
explanations; mnemonic-only edits preserve rule and other-language approvals.

Permission records explicitly attest original source hashes, rights, extraction
permission and historical local/hosted provider processing/permission. They cannot
authenticate legal rights, historical truth or human reviewer identity and do not
authorize future provider execution. Documentation and the original synthetic
lecture fixture demonstrate permitted extraction, source quotations, preserved
German wordplay, a rule without an aid, and exclusion of an existing QA note.
No private lecture material or provider extraction run was used.

TDD: initial missing-module RED preceded implementation, then 33 cases passed.
A diagnosed CLI supersession defect failed the prior-rejection regression before
decision loading was corrected to validate combined retained history; 34 cases
then passed. Extended initial coverage reached 49 curated cases. Dedicated
simplification changed only NoReturn error handling and unreachable returns.

Independent high-risk conventional review covered general correctness, security,
Python/data/public schemas, tests and prose/claims. Because coordinator subagents
cannot spawn agents, the parent dispatched the five independent lanes. No runtime
defect was found. One broad adversarial round used separate boundary and overall
verifiers; initial overall verdict was insufficient for four evidence gaps.

Correction 1 closed all four: separate mnemonic-source permission deny/allow;
actual CLI German-approved/English-rejected mnemonic recording/reload with rule
unreviewed; earlier rejection/later approval conflicts in both orders and after
partial supersession; byte preservation on success/stale/refused overwrite and
old-history retention. Renamed the hosted-processing test to describe declared
attestation preservation, not authentication. Four deliberate temporary runtime
regressions each failed its intended assertion (four failed, 47 deselected), then
were fully restored. Domain SHA256 f6c168587112b0e46b1756fb4d3a2d733140e5d5db64eb995ceecf789af25bf1
matched the pre-probe bytes. All 51 curated and 241 curated+authored tests passed.
The same simplifier found no correction changes; scoped independent recheck
resolved C1-C5 and returned HOLDS. Coordinator final review-only verdict: APPROVE,
no remaining required finding. One broad round and one targeted follow-up used;
no budget reset, broad restart or unchanged-claim review-shopping.

Optional cross-fingerprint reversion and retained rendered-history scenarios were
traced correct by independent reviewers, not claimed as full downstream recipe
acceptance. Network tripwires/additional format vectors remain optional. Mutation
locate found no adapter (only partial mutmut config), so no mutation setup/run.
Final reviewed implementation diff SHA256 e39721e6115473a84a7e9b67b5630b4e1a4c433b857a36bd186fc026af4be858
includes untracked implementation files and excludes planning recording; the only
post-correction delta was formatter-only documentation example spacing, read back
by the coordinator. Its sorted per-file Git/no-index diff identity is distinct
from a Git commit or linguistic approval.

Final-gate correction 2 (F1): the first full suite found the existing explicit
command-module enumeration did not unpack the new curated group. A targeted RED
confirmed its callback=None assertion failure. Four test lines now assert exactly
validate/preview/review and inspect those leaf callbacks, following adjacent group
patterns without weakening the final module/callback assertions. CLI+curated
checks passed (102); dedicated simplifier independently repeated them with no
edits, and the same coordinator verified F1 resolved with no new review blocker.
This bounded correction does not restart the broad challenge campaign. Final
reviewed diff SHA256 is now
1d3df4170de14e074b37c2b4c722dc9172a72a826dca36987de8e5edfbe5bff8
(65,565 bytes, same method). The previous full gate was not green; final validation
must restart from Ruff on the corrected state. No runtime/schema/fixture change.

Final-gate correction 3 (F2): first full pytest finished 2 failed, 3717 passed,
7 skipped, one existing serializer warning in 1057.53 seconds; it was not green.
The other failure was the schema directory's no-unregistered-files guard. Targeted
RED confirmed missing curated entries in the existing json_schemas registry.
Added its three model registrations and regenerated only the new schemas through
scripts/export-json-schemas.py, preserving model constraints and standard exporter
bytes. Removed the redundant equality loop from the new extraction test because
the unchanged registry-wide drift/exporter tests now cover all three models.
All 119 CLI+curated+schema tests, affected Ruff and mypy passed. Dedicated
simplifier independently repeated 119 tests/whole mypy/affected Ruff; no edits.
Coordinator verified F2 resolved, exact model/exporter bytes and coverage transfer;
review-only APPROVE remains, no new blocker. Final implementation diff SHA256:
19a8660d5455b65f74569ed8056cc3908db3f68bc3fcadb85f94e3fc3aaf490e
(65,934 bytes, same method). Five initial lanes, one broad challenge, one targeted
follow-up and bounded F1/F2 coordinator reviews are cumulative, not reset budgets.
Exact full Ruff/mypy/pytest chain restarted from the beginning after this fix.

Manual verification ran ten real CLI subprocess probes against the permitted
synthetic fixture: validation/preview, synthetic rule-only decision/reload,
existing-output refusal with byte preservation, missing export command, missing
permissions, stale-review refusal with no output, stale-preview invalidation and
unpermitted mnemonic-document refusal. All passed; no provider or destination
write, real human content approval, hosted/private processing or native test.

T-197 retains actual rendered-card approval/recipes/export. Every exported card
must require rule/explanation AND rendered approval, plus language-specific
mnemonic/association approval for any aid displayed on either side of any recipe.
Candidate preview is not a learning card or export authorization. T-200 judge,
study, reference/development feasibility runs remain paused. No automatic policy
is enabled; v050-disabled revision 1 empty scope and missing_token_bound refusal
remain unchanged. T-195's bounded offline evidence (51 public CLI cases/142 selected
tests and independently confirmed 15 green exact-head CI checks) is retained as
prior evidence, not rerun or strengthened here. Native Anki resolution/history and
scheduling preservation remain unproven, owned by T-201. Outer adversarial QA,
German-card video, release and publication gates are not satisfied by this task.

Final gate on the corrected reviewed executable state passed on 2026-11-19:
uv run ruff check; uv run mypy (231 source files); uv run pytest -v
(3722 passed, 7 intentional parameter-combination skips, one existing Pydantic
serializer warning, 1005.19 seconds). No executable inputs changed afterward.
The skips are five derived output-lock spellings and two absent-inode hardlink
cases, not unexecuted curated tests. v0.4 characterization and missing_token_bound
refusal passed in that chain. Format (562 files), diff/Taskrail checks and
author/CI-scope/commit/push guard suites passed. Bootstrap revision 1 empty scope
and exact digest 9958ecce67297078d192aff579e352a07ee2c020d9045b60c44c3b29faffd538
were independently checked unchanged. Before lifecycle recording, refreshed
origin/main still matched the base and selector returned only active T-196,
off_spec=false, with state valid. Supported verify/complete record task-local
implementation evidence only; independent integrated acceptance remains due.
- 2026-11-19T04:56:39Z: verification pass
- 2026-11-19T04:56:48Z: Task-local curated intake implementation complete; verification pass 2026-11-19T04:56:39Z after final Ruff/mypy/pytest chain (3722 passed, 7 intentional skips), 10 manual CLI probes and independent scoped review. C1-C5/F1/F2 resolved; no successor, export authority, provider execution, native/empirical acceptance or policy enablement. Delivery/independent integrated acceptance remain separate.
