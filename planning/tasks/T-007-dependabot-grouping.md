---
id: T-007-dependabot-grouping
title: Group minor Dependabot security updates
status: completed
priority: medium
spec_ref: specs/v0.1.0.md#uv-toolchain
dependencies: []
updated_at: "2026-09-15T11:29:43Z"
---

# T-007-dependabot-grouping Group minor Dependabot security updates

## Description

After the move to uv, Dependabot opened one pull request per package for
gitpython, urllib3, torch, stanza, pytest, pygments, and idna, even though the
uv entry already groups minor and patch updates. Those were security updates,
which ignore groups unless a group sets `applies-to: security-updates`. See
`specs/v0.1.0.md#uv-toolchain`.

## Acceptance

- The uv entry in `.github/dependabot.yaml` has a minor/patch group for
  version updates and one for security updates.
- The configuration validates against the Dependabot schema.
- The obsolete pip-ecosystem pull requests (#42-#49) are closed.

## Verification Notes

- `check-jsonschema --builtin-schema vendor.dependabot` passed on the config.
- #42-#49 closed with a comment and their branches deleted.

## Implementation Notes

- 2026-09-15T11:29:43Z: verification pass
