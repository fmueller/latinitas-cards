---
id: T-077-preserve-morphology-across-setup-reconfigure
title: Preserve hand-edited morphology settings across setup reconfiguration
status: completed
priority: medium
spec_ref: specs/v0.2.0.md#morphology-themes
dependencies: []
updated_at: "2026-10-08T00:34:53Z"
---

# T-077-preserve-morphology-across-setup-reconfigure Preserve hand-edited morphology settings across setup reconfiguration

## Description

Found by the T-069 new-user walkthrough. `setup --reconfigure --confirm` silently resets a
hand-edited profile `morphology` section to muted/light/static, and a partial morphology object
without `version` is accepted silently. Themes are only selectable by editing profile JSON.
Not release-blocking.

## Acceptance

- Reconfiguration preserves or explicitly reports changes to existing morphology settings.
- Morphology objects without an explicit version fail clearly or are documented as defaulted.
- Regression tests cover both cases.

## Verification Notes

- Strict RED: focused regression command failed with 5 failures and 2 passes:
  confirmed reconfiguration reset monochrome/dark/disclosure and explicit
  versionless objects were accepted by setup and profile loading.
- GREEN: `uv run pytest tests/unit/profile_setup_test.py
  tests/unit/morphology_theme_test.py -q` passed all 57 tests.
- Final chain: `uv run ruff check` passed; `uv run mypy` reported no issues
  in 89 source files; `uv run pytest -v` passed all 894 tests.
- Installed CLI smoke preserved exact version-1 monochrome/dark/disclosure
  settings through `setup --reconfigure --confirm`, persisted a target-deck
  correction, and passed reload/roundtrip, cancellation and source-immutability
  checks. A versionless partial object exited 2 with actionable JSON on stderr
  and left the saved profile unchanged. The smoke harness initially read the
  wrong stream; correcting it to the existing stderr contract passed.
- Dedicated code-simplifier Task loaded its skill, made no changes, and passed
  the 57 focused tests. Separate read-only General, Python and Security Tasks
  loaded code-reviewer and their routed reviewer/companion references. Each
  concluded: "No concrete task-relevant findings." Fresh candidate validation
  upheld the empty candidate set; fresh disposition verification found no new
  or unresolved issues. No findings required fixes or deferral (one cycle).
- Lane rationale: General for acceptance/architecture; Python for Pydantic
  and compatibility; Security for input validation and saved-state data-loss
  risk. Database omitted: no SQL, migrations or database clients changed.
  UI/framework lanes omitted: no render code or UI appearance changed, and
  existing native-client evidence is not expanded by this task.

## Implementation Notes

- Pinned to `specs/v0.2.0.md#morphology-themes`; accepted remote baseline
  contained the completed T-076 work, with no file transfers needed.
- Reconfiguration carries saved morphology through the existing proposal
  override path before applying explicit setup corrections. Profile loading
  requires an explicit morphology version when the section is present;
  absent-section defaults, schema-1 upgrades, and partial runtime overrides
  retain their existing contracts. No new theme flags or task API added.
- Claim/presentation fingerprint separation, stable note/card identities and
  semantic eligibility are unchanged and covered by the existing theme tests.
- README and changelog document preservation and version requirements.
- No follow-up work discovered within this task; T-078 through T-081 and the
  subsequent high round remain the orchestrator's scope, not this cycle.
- 2026-10-08T00:34:53Z: verification pass
