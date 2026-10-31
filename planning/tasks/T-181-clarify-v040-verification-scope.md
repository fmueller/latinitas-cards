---
id: T-181-clarify-v040-verification-scope
title: Clarify v0.4.0 verification scope and record remaining release checks
status: completed
priority: medium
spec_ref: specs/v0.4.0.md#native-acceptance
dependencies: []
updated_at: "2026-10-31T13:58:38Z"
---

# T-181-clarify-v040-verification-scope Clarify v0.4.0 verification scope and record remaining release checks

## Description

Clarify the authorized v0.4.0 Native Acceptance scope: AnkiMobile and real
scheduled collections/history are not release-readiness blockers. Preserve
truthful evidence limits, run the remaining real-source tests, and audit future
release-policy inheritance without changing future product contracts.

## Acceptance

- Spec, native kit and readiness docs explicitly exclude Mobile and real scheduled
  collections/history from v0.4.0 readiness without asserting compatibility.
- Candidate identity and delta from the native-tested revision are recorded.
- Caesar BG and Aeneid normalize/validate with pinned source digests and independent
  passage-count checks; scoped translation licence/provenance review is recorded.
- Candidate-bound screenshots and narrated decision evidence distinguish reused
  native captures from new smoke results, with independent review and media QA.
- Future gate adoption/gaps are reported; broader policy changes await approval.
- T-173 stays blocked and byte-unchanged; no tag, publication, commit or push.

## Verification Notes

- Full restarted mandatory chain passed: ruff; mypy (198 source files); pytest
  (3,212 passed, 7 skipped, 1 warning in 590.34 seconds). Four targeted tests,
  formatting, local Markdown links, Taskrail validation, distribution inspection
  and isolated wheel/sdist installation also passed.
- Initial full-suite failures were two obsolete all-pending documentation
  assertions, now updated to retain the truthful evidence/approval boundary.
  Independent final diff review and general challenge found no substantive issue.
- Narrated decision evidence is 201.247 seconds, SHA256
  `7deccc1fe2ed1360291b7f5b1e2951d50dc7d73a3e94fe4f6ead0d684724894d`.
  Final-hash independent decode/audio-binding/frame/script QA passed within its
  stated limits. Full movie analysis was unavailable upstream; local extracted
  audio ASR and sampled-frame checks do not claim human listening.
- This task supplies evidence and local scope clarification, not whole-release
  acceptance, future-policy implementation or publication authorization.

## Implementation Notes

- Candidate `041989c54f32ed3a0ebc0481fd81776f2c368396` equals fetched
  `origin/main` and the prior native-tested revision. Only docs, documentation
  contract assertions and tracked-work records change locally; no runtime delta
  or fresh native rerun.
- Independent XML extraction matched Caesar's 2,150 and Aeneid's 9,843 passages;
  Aeneid has 54 editorially deleted lines retained in notes. Both source gaps
  warn without dropping their passages. See versioned readiness documentation
  for upstream revision/digests and evidence boundaries.
- Future release audit found partial requirements but no complete inherited
  candidate/media/QA/human-approval process. No future Fixed Contracts or skills
  were rewritten. A separately approved shared policy is recommended.
- 2026-10-31T13:58:38Z: verification pass
- 2026-10-31T13:58:38Z: Local docs/contracts clarification and remaining test evidence supplied; future gate audit remains a proposal. No T173 lifecycle or publication action.
