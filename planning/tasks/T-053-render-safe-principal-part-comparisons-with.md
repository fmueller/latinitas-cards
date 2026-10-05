---
id: T-053-render-safe-principal-part-comparisons-with
title: Render safe principal-part comparisons with versioned morphology themes
status: completed
priority: medium
spec_ref: specs/v0.2.0.md#morphology-themes
dependencies:
    - T-052-generate-supported-principal-part-relationships
    - T-048-define-destination-snapshots-and-file-transport
updated_at: "2026-10-05T04:03:00Z"
---

# T-053-render-safe-principal-part-comparisons-with Render safe principal-part comparisons with versioned morphology themes

## Description

Extend the human-readable deck profile and existing answer rendering with restrained semantic presentation, preserving the stable template contract.

## Acceptance

- Explicitly version theme/profile settings and report effective overrides; incompatible versions fail clearly. Ship named muted and monochrome options, each with light/dark presentation. Test profile serialization round-trip, override precedence/reporting, and unknown theme/version rejection. Explicitly support or reject older profiles without silently reinterpreting them.
- Specify the styling delivery contract: managed answer markup versus manually installed versioned reference CSS/setup, compatibility checks for existing note types, and the same effective theme in preview/imported notes. CSV does not install styling. Block incompatible setup changes pending separately reviewed manual setup; do not automate provisioning or rebind scheduled templates.
- Escape source strings and use controlled markup/classes for lemma, stem, formation marker, and ending; source HTML never passes through. Labels, separators, and typography remain useful without color. Test hostile strings in explanations, alternatives, labels, and absent/withheld output, including tags/event attributes, quotes, encoded entities, and line breaks; assert literal expected text, controlled markup, and no active source HTML or double decoding. Personal Notes remain outside generated-field sanitization and writes.
- Show the focused German explanation and compact comparison on answer reveal. Additional four-role comparison uses Stammformen vergleichen and includes accepted further explanation; withheld explanations and absent roles are not rendered as facts.
- Provide native details/summary disclosure and a fully readable static fallback; keep the static fallback as the unverified-client default until native evidence supports disclosure. No untested JavaScript dependency is added. Static fallback still requires native readability/reveal checks.
- Compare independently expected claims, withholding status, eligibility, note/card identities, recipe keys, and template bindings across muted/monochrome, light/dark, disclosure/static, and preview/export paths. If the schema cannot carry content safely, expose the migration block instead of rewriting scheduled templates. Presentation changes affecting the approved payload invalidate prior approval.
- Retain T-026 safe HTML and T-027 Personal Notes regressions. Render representative accepted, withheld, and missing-role answers in both themes and light/dark modes; inspect captures. Browser checks are not native Anki compatibility claims.

## Verification

Use red/green tests and the mandatory ruff, mypy, pytest -v chain. Record inspected visual artifacts and defer native compatibility claims to the native presentation gate.

### Workflow evidence

- Source guard: accepted main commit contained after fetch; clean checkout,
  valid Taskrail state, exactly this eligible task and pinned spec before writes.
- RED: morphology tests initially reported 9 failures (schema still 1 and no
  presentation settings). The claim-binding matrix then failed on changed
  presentation fingerprints; preview parity failed on missing answer output.
  The manual setup diagnostic test failed on the old generic layout message.
  GREEN: versioned settings, presentation-only claim exclusion, shared rendering,
  preview reporting and manual setup diagnostics resolved these failures.
- Initial full chain caught one obsolete div-wrapper assertion (791 passed,
  1 failed); updated it to the versioned section without weakening prompt guards.
  Restarted the full chain: ruff passed, mypy passed (79 files), pytest 794 passed.
- Dedicated code-simplifier loaded its skill and moved settings resolution outside
  the card loop; focused 90 tests, ruff and mypy passed. No other simplification.
- Separate read-only General, Security and Python lanes loaded code-reviewer and
  their dedicated guidance (Security security-review; Python python-patterns).
  Each returned verbatim: "No concrete task-relevant findings." Each ran 90
  focused tests. General/Security/Python cover acceptance, trust boundaries and
  JSON persistence; no SQL/framework/network/ML behavior changed.
- Fresh candidate-validation found no concrete task-local issue, with empty
  validated/rejected candidate IDs. No findings required fix or deferral.
  Fresh disposition-verification returned "No concrete task-relevant findings."
  One review cycle; full reviewed chain passed with 794 tests.
- Inspected 2x Chromium reference captures: both themes and light/dark in static,
  disclosure closed/open, and 390px narrow static; accepted perfect analysis,
  absent infinitive and withheld fourth role visible as appropriate. DOM reported
  16 role rows, 18px/400 body typography, no narrow horizontal overflow; summary
  click opened disclosure while focused/compact sections remained visible.
  Visual artifacts are linked in the task execution report, not committed paths.
- Delivery: managed answer markup v1 and manual reference CSS v2 only. Schema 1
  without presentation upgrades explicitly to schema 2 defaults; older reviews
  require renewal. CSS digest mismatch blocks setup, and the existing journal
  rejects changed answer fields rather than expanding scheduled application.
  No provisioning, scheduled template rebind, automatic claim acceptance or JS.
- Remaining release gate: actual AnkiMobile/Desktop reveal/readability and native
  disclosure evidence. Browser captures are not native proof. T-058 is not
  implemented in this cycle; no new follow-up beyond the existing gate/task.

## Implementation Notes

- 2026-10-05T04:03:00Z: verification pass
