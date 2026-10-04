---
id: T-053-render-safe-principal-part-comparisons-with
title: Render safe principal-part comparisons with versioned morphology themes
status: todo
priority: medium
spec_ref: specs/v0.2.0.md#morphology-themes
dependencies:
    - T-052-generate-supported-principal-part-relationships
    - T-048-define-destination-snapshots-and-file-transport
updated_at: "2026-10-04T23:05:46Z"
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
