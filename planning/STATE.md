---
schema_version: 1
updated_at: "2026-10-29T12:44:18Z"
active_spec_version: v0.4.0
active_spec_path: specs/v0.4.0.md
current_task: ""
current_task_title: ""
status_summary: blocked
blockers:
    - 'T-173-publish-v0-4-0-maintainer-release-gate: maintainer release gate'
    - 'T-174-decide-whether-generate-recipe-overrides-should: maintainer decision: whether a generate --recipe override should be excluded from claim fingerprints, which changes the unchanged v0.2.1 fingerprint contract'
    - 'T-175-decide-whether-token-glosses-should-be-phrase: maintainer decision: whether translation annotations anchored to one word token (lexical glosses) should be excluded from phrase translation candidates, which the Fixed Contracts Recipes candidate rule currently reports as invalid_span'
    - 'T-176-decide-whether-cref-pattern-example-should-warn: maintainer decision: whether the Aeneid line, Book cRefPattern example should warn, which the Fixed Contracts TEI subset deepest-first case-insensitive citation_mismatch rule currently treats as matching'
next_action: Resolve blocker on T-176-decide-whether-cref-pattern-example-should-warn
last_verification_result: pass for T-158-normalize-perseus-epidoc-tei-structure-citations at 2026-10-29T12:44:18Z
relevant_artifacts: []
continuation_notes:
    - Retrofitted onto an existing repo before its first release; specs/v0.1.0.md is the active spec and records both the pipeline built so far and the work left to cut 0.1.0.
---

# STATE

## Active Spec

- `specs/v0.4.0.md`

## Current Focus

- Task: none

## Status

- blocked

## Blockers

- T-173-publish-v0-4-0-maintainer-release-gate: maintainer release gate
- T-174-decide-whether-generate-recipe-overrides-should: maintainer decision: whether a generate --recipe override should be excluded from claim fingerprints, which changes the unchanged v0.2.1 fingerprint contract
- T-175-decide-whether-token-glosses-should-be-phrase: maintainer decision: whether translation annotations anchored to one word token (lexical glosses) should be excluded from phrase translation candidates, which the Fixed Contracts Recipes candidate rule currently reports as invalid_span
- T-176-decide-whether-cref-pattern-example-should-warn: maintainer decision: whether the Aeneid line, Book cRefPattern example should warn, which the Fixed Contracts TEI subset deepest-first case-insensitive citation_mismatch rule currently treats as matching

## Last Verification

- pass for T-158-normalize-perseus-epidoc-tei-structure-citations at 2026-10-29T12:44:18Z

## Next Action

- Resolve blocker on T-176-decide-whether-cref-pattern-example-should-warn

## Relevant Artifacts

- None

## Notes

- Retrofitted onto an existing repo before its first release; specs/v0.1.0.md is the active spec and records both the pipeline built so far and the work left to cut 0.1.0.

## Task Counts

- todo: 11
- in_progress: 0
- completed: 161
- blocked: 4
- cancelled: 0
