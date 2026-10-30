---
schema_version: 1
updated_at: "2026-10-30T00:36:08Z"
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
    - 'T-177-decide-where-allow-unknown-keeps-tei-elements: maintainer decision: where --allow-unknown keeps unknown TEI elements without a canonical-text position (between textparts or lines, inside flattened notes, headings and variants, inside app or choice, or in omitted passages), which the Fixed Contracts TEI subset says are kept in source_extra but T-159 fails closed'
    - 'T-178-decide-whether-import-translations-rejects-a-non: maintainer decision: reject a non-Latin CORPUS in import-translations'
next_action: 'Start task T-171-prepare-the-v0-4-0-native-acceptance-kit: Prepare the v0.4.0 native acceptance kit for Linux Anki Desktop'
last_verification_result: pass for T-170-document-corpus-first-tei-and-english-workflows at 2026-10-30T00:36:08Z
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
- T-177-decide-where-allow-unknown-keeps-tei-elements: maintainer decision: where --allow-unknown keeps unknown TEI elements without a canonical-text position (between textparts or lines, inside flattened notes, headings and variants, inside app or choice, or in omitted passages), which the Fixed Contracts TEI subset says are kept in source_extra but T-159 fails closed
- T-178-decide-whether-import-translations-rejects-a-non: maintainer decision: reject a non-Latin CORPUS in import-translations

## Last Verification

- pass for T-170-document-corpus-first-tei-and-english-workflows at 2026-10-30T00:36:08Z

## Next Action

- Start task T-171-prepare-the-v0-4-0-native-acceptance-kit: Prepare the v0.4.0 native acceptance kit for Linux Anki Desktop

## Relevant Artifacts

- None

## Notes

- Retrofitted onto an existing repo before its first release; specs/v0.1.0.md is the active spec and records both the pipeline built so far and the work left to cut 0.1.0.

## Task Counts

- todo: 3
- in_progress: 0
- completed: 169
- blocked: 6
- cancelled: 0
