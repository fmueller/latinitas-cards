---
id: T-142-reject-literal-block-oidc-write-in-workflow-guard
title: Reject literal-block OIDC write in workflow guard
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-18T14:19:53Z"
---

# T-142-reject-literal-block-oidc-write-in-workflow-guard Reject literal-block OIDC write in workflow guard

## Description

Round 8 (final authorized round) synthetic E2E at product revision
9796cd47084bdeab319a62ab7d89fda92c764d9f confirmed a medium-severity local
no-publish guard false negative under the PyPI Publishing contract. The
ordinary literal-block scalar `id-token: |-` followed by indented `write`
has exactly the YAML value `write`, but the scanner accepts it in release
root permissions, the release build job, and a non-release workflow. The
equivalent direct values reject. This is one residual scalar-family finding,
not three separate defects or a claim about arbitrary YAML/shell semantics.

No fixture step, publication command, OIDC request or hosted workflow ran.
The finding demonstrates a local guard gap, not actual publication or a
Trusted Publisher/environment authorization bypass.

Dedupe inventoried all 141 existing tasks, including 139 completed. Completed
T-122 owns the no-publish and outside-publish guarantee; T-139 added continued
and folded non-release scalar checks; T-141 added continued/folded release
scope checks. Their original regressions pass on this revision. Literal-block
permissions remain unhandled in both scanner branches and are not tracked by
an open task. This standalone task does not reopen completed tasks, change
gates or dependencies, or authorize another implementation/testing round.
T-038 remains blocked and maintainer-only; T-128 remains future v0.8.0 work.

## Acceptance

- Reject the literal-block permission fixtures below with nonzero exit naming
  the workflow and appropriate forbidden capability/scope diagnostic. Cover
  release root and non-publish jobs, plus non-release workflow permissions.
- Preserve literal read permission, literal write inside the permitted publish
  job, direct-value controls, strict tag checks, and T-139/T-141 continued and
  folded regressions. Keep fixtures inert and verification local.
- Add no default runtime YAML dependency and make no universal YAML/shell
  security claim. A constrained syntax refusal is an available design choice;
  do not silently weaken the documented guard guarantee.
- Run the focused guard suite and mandatory ruff/mypy/pytest chain after any
  separately authorized implementation. Filing this task authorizes no fix,
  task start, release, publication, gate change or hosted workflow trigger.

## Verification Notes

Exact executable offline reproduction from the repository root. Only the
local scanner executes; all copied workflow contents remain inert:

```bash
probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
cp .github/workflows/*.yml "$probe/"
awk '
  { print }
  $0 == "  build:" {
    print "    permissions:"
    print "      id-token: |-"
    print "        write"
  }
' .github/workflows/release.yml > "$probe/release.yml"
bash scripts/check-publish-guard.sh "$probe"
# Actual: exit 0, publish guard passed. Expected: exit 1, scope error.

cp .github/workflows/release.yml "$probe/release.yml"
cat > "$probe/extra.yaml" <<'YAML'
name: Inert literal permission fixture
on: push
jobs:
  probe:
    runs-on: ubuntu-latest
    permissions:
      id-token: |-
        write
    steps:
      - run: echo harmless
YAML
bash scripts/check-publish-guard.sh "$probe"
# Actual: exit 0. Expected: exit 1, publishing capability outside release.yml.

sed -i '/      id-token: |-/c\      id-token: write' "$probe/extra.yaml"
sed -i '/^        write$/d' "$probe/extra.yaml"
bash scripts/check-publish-guard.sh "$probe"
# Actual and expected: exit 1, naming extra.yaml.
```

Fresh independent PyYAML 6.0.3 parsing asserted the permission string equals
exactly `write` (without a trailing newline). A 24-fixture scope/value/style
table found three literal-block false negatives; folded explicit-indent and
quoted continued controls reject outside publish, and all read/allowed-publish
controls pass. A separately authored awk/sed reproduction using unquoted keys
confirmed release-build and non-release literal forms exit 0, equivalent
direct forms exit 1, and read/allowed-publish forms exit 0. Thus the result
does not depend on key quoting or a generated-fixture parser assumption.

The scanner's pending-scalar recognition handles `>` and bare continued
values, not `|`. This observation explains both manifestations without
requiring any workflow execution or adding a parser to product dependencies.

## Implementation Notes

Findings-only filing from final authorized round 8. No implementation, task
start, further round, release, tag, publication or deployment was performed.
