---
id: T-076-verify-managed-csv-desktop-import-dialog
title: Verify managed CSV updates through the Anki Desktop import dialog
status: todo
priority: medium
spec_ref: specs/v0.2.0.md#safe-update-application
dependencies: []
updated_at: "2026-10-06T20:17:05Z"
---

# T-076-verify-managed-csv-desktop-import-dialog Verify managed CSV updates through the Anki Desktop import dialog

## Description

Found by the T-069 review. T-061 verified managed CSV only through Anki's backend import API;
the documented user path is the Desktop import dialog, which earlier skipped tag-only rows.
Disclosed as a v0.2.0 limitation. Not release-blocking.

## Acceptance

- Native Desktop dialog runs on a sanitized fixture record client/version, settings, content, tag-only and no-op outcomes with full card/review-log comparisons.
- Skipped tag-only effects remain pending/unresolved, and docs state the verified scope.

## Verification Notes

- Pending.

## Implementation Notes


