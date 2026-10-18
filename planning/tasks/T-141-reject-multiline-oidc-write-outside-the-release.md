---
id: T-141-reject-multiline-oidc-write-outside-the-release
title: Reject multiline OIDC write outside the release publish job
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-18T13:33:33Z"
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

The release scanner now remembers the scope and indentation of a continued or
folded id-token permission and checks its value across comments and blank lines.
Dedenting ends that pending permission. The existing direct-value check, trigger
allowlist and non-release scanner are unchanged. No workflow or dependency changed.

Strict TDD: bash scripts/check-publish-guard-test.sh first failed with
"FAIL: accepted unsafe release.yml: root multiline OIDC permission (write)".
After the minimal scanner change, the same suite passed, including 28 new inert
scope/value cases: root, build, publish and after-publish; plain/single/double
quoted values, folded >/>-/>+ scalars, comments, blank lines and harmless read.
Only local scanner processes inspect those copied workflow fixtures.

Initial mandatory chain passed: uv run ruff check; uv run mypy (169 source files);
uv run pytest -v (2633 passed, 7 skipped, one warning). A dedicated Task loaded
code-simplifier and recommended no edits; focused suite, bash -n and diff checks
passed afterward. Independent parallel Tasks loaded code-reviewer for General
(ECC code-reviewer) and Security (ECC security-reviewer plus security-review
reference). Each concluded verbatim: "No concrete task-relevant findings."
Fresh candidate-validation Task loaded code-reviewer and confirmed zero candidates
to validate, reject or deduplicate, with no new task-local finding. No dispositions
or deferrals were needed. General covers shell/control-flow correctness; Security
covers OIDC scope. Python/framework/database lanes were omitted because their
code and contracts are unchanged.

Final exact mandatory chain passed again (2633 passed, 7 skipped, one warning),
followed sequentially by mise run check, including its full pytest run and all
shell policy/guard suites. Fresh dedicated code-reviewer disposition-verification
Task concluded verbatim: "No concrete task-relevant findings." It confirmed the
empty disposition set, no deferrals and no newly introduced task-local issues.
One review cycle; no post-review product fixes were needed.

This is a bounded lexical guard, not arbitrary YAML or shell semantic analysis.
No publication, OIDC request, release/tag creation or manual hosted trigger ran.
Plan hashes remain consistency checks, not authentication. Round 7 remains
COMPLETE but NOT CLEAN; round 6 remains NOT CLEAN/INCOMPLETE. Advisory 144 remains
uninvestigated/unresolved. T-038 and off-spec T-128 are untouched; this change
grants no publication authorization or broad security clearance.
- 2026-10-18T13:33:33Z: verification pass
- 2026-10-18T13:33:33Z: Reject continued/folded release OIDC write outside publish; reviewed bounded lexical guard, unchanged workflows and release gates. All final local checks pass.
