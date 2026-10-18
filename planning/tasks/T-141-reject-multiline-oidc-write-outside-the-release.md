---
id: T-141-reject-multiline-oidc-write-outside-the-release
title: Reject multiline OIDC write outside the release publish job
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-18T13:00:21Z"
---

# T-141-reject-multiline-oidc-write-outside-the-release Reject multiline OIDC write outside the release publish job

## Description

Round 7 synthetic E2E at product revision
4005fd74797fbc53991ba075e3b11f469ef5e640 confirmed a medium-severity local
guard false negative under the PyPI Publishing contract. In release.yml,
multiline id-token write permission outside the publish job passes the scanner,
while its single-line equivalent rejects. No workflow step, publication command,
OIDC request or hosted workflow was executed. This is a local scope-check gap,
not demonstrated publication or a Trusted Publisher/environment bypass.

Dedupe covered all 140 preexisting tasks, including 138 completed. Completed
T-122 owns the outside-publish permission guarantee and fixed quoted keys and
job scope. Completed T-139 fixes multiline values only in non-release files;
those retests pass. This is one residual release-branch instance of that family,
not multiple findings for each YAML representation. Neither completed task is
reopened. This standalone task tracks the still-unimplemented case, with no
dependencies or gate changes. T-038 remains maintainer-only and blocked.

## Acceptance

- Reject multiline id-token write permissions at release.yml root and in jobs
  other than publish, with nonzero exit naming the workflow and scope error.
  Cover plain/quoted continued scalars, comments/blank lines, and folded values
  equivalent to write. Single-line controls must continue to reject.
- Preserve allowed multiline write in publish, harmless read permissions,
  unchanged repository workflows, strict tag-trigger checks and T-139's
  non-release scalar regression coverage.
- Keep verification local and fixtures inert. Do not execute any fixture step,
  publish, trigger hosted workflows, or change release/environment permissions.
  Add no default runtime YAML dependency; do not claim arbitrary YAML or shell
  semantic analysis.
- Run focused guard tests and the mandatory ruff/mypy/pytest chain after the
  eventual implementation. Filing this finding authorizes no implementation.

## Verification Notes

Exact safe reproduction from repository root; only the scanner executes:

```bash
probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
cp .github/workflows/*.yml "$probe/"
awk '
  {print}
  $0 == "  build:" {
    print "    permissions:"
    print "      id-token:"
    print "        # YAML comment does not alter the value"
    print ""
    print "        write"
  }
' .github/workflows/release.yml > "$probe/release.yml"
bash scripts/check-publish-guard.sh "$probe"
# Actual: exit 0, publish guard passed. Expected: exit 1, scope diagnostic.
awk '
  {print}
  $0 == "  build:" {
    print "    permissions:"
    print "      id-token: write"
  }
' .github/workflows/release.yml > "$probe/release.yml"
bash scripts/check-publish-guard.sh "$probe"
# Actual and expected: exit 1, id-token: write outside publish.
```

Independent isolated PyYAML 6.0.3 parsing confirmed the continued build/root
values equal exactly write. Five failed representations share the same raw-line
scope matcher. Direct build/root controls reject; read permission and multiline
write in publish pass. A separate parent-owned reproduction with key/value
comments and a blank line repeated exit 0 versus direct exit 1 and read exit 0.
The actual repository workflows pass. No parser dependency was added to the
product and no fixture publication commands were run.

## Implementation Notes

Findings-only filing. The non-release branch normalizes continued/folded scalar
values, but the release branch in scripts/check-publish-guard.sh still matches
id-token and write on one raw line. Preserve job/root scope when addressing
this residual. Plan hashes remain consistency checks, not authentication;
this finding grants no release or security clearance.
