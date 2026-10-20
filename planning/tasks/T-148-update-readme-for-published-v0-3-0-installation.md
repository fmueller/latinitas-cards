---
id: T-148-update-readme-for-published-v0-3-0-installation
title: Update README for published v0.3.0 installation
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies:
    - T-038-publish-releases-on-pypi
updated_at: "2026-10-20T22:19:00Z"
---

# T-148-update-readme-for-published-v0-3-0-installation Update README for published v0.3.0 installation

## Description

Update the README after verified v0.3.0 publication under the existing PyPI
publishing contract. Correct stale candidate, installation and tag-availability
claims without changing runtime behavior, published artifacts or release gates.

## Acceptance

- The current-release banner links to the public v0.3.0 release and PyPI version.
- Installation and tagged-checkout instructions no longer describe publication
  or the v0.3.0 tag as future work; the release-evidence link is current.
- Existing isolated install/upgrade commands, platform limits, annotation-extra
  caveats, CPU-index instructions and bundled-corpus exclusions remain intact.
- Installation documentation tests reject the stale pre-release claims while
  preserving coverage of the practical installation and safety instructions.

## Verification Notes

- Record targeted RED/GREEN, full validation, independent reviews and delivery
  evidence without committing ignored artifact paths.

## Implementation Notes

- README audit found the release banner, PyPI heading/introduction, tagged
  checkout note, extras preamble and release-evidence link still described the
  already published release as pending. The installation test required two
  obsolete phrases. Updated that existing test before the README: RED failed
  because the installation section lacked the published PyPI version link.
- GREEN: the installation/runbook and release-metadata tests passed (4 tests).
  The dedicated code-simplifier corrected a residual future-publication phrase
  in the GPU-extra warning and extended the existing regression. Parent review
  strengthened its line-wrap-sensitive forbidden literal to "After publication";
  deliberately restoring the original GPU wording failed with
  "AssertionError: After publication", then restoring the correction passed
  all four targeted tests. Install commands and safety warnings are unchanged.
- General was the sole review lane: this is documentation and documentation
  assertions, not Python runtime behavior or a new security, persistence,
  framework or network boundary. General reported "No concrete task-relevant
  findings." Fresh candidate validation confirmed the empty set. No validated
  findings or deferrals remained. Initial full checks passed: Ruff, mypy,
  pytest (2637 passed, 7 skipped, one existing warning, 439.01 seconds).
  A fresh exact chain is run after the simplifier's final changes and reviews.
- Final exact Ruff, mypy, pytest chain passed after all changes and reviews
  (2637 passed, 7 skipped, one existing warning, 432.91 seconds). Diff checking
  and Taskrail validation passed. Fresh disposition verification reported
  "No concrete task-relevant findings." One review cycle, no validated
  findings or deferrals. Taskrail verification/completion follows these checks;
  no release artifacts, tags, spec contents or runtime code changed.
- 2026-10-20T22:19:00Z: verification pass
- 2026-10-20T22:19:00Z: README links public v0.3.0 release/PyPI, removes stale future publication/tag wording, preserves installation commands and practical limits; full checks and independent review passed.
