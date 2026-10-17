---
id: T-139-reject-multiline-publishing-capabilities-in
title: Reject multiline publishing capabilities in workflow guard
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-17T13:08:31Z"
---

# T-139-reject-multiline-publishing-capabilities-in Reject multiline publishing capabilities in workflow guard

## Description

Round 6 synthetic E2E at revision 0e6afbcf9d9f33493d7426b461a4e6473f0cd9f9
confirmed a medium-severity local no-publish guard false negative. The
PyPI Publishing contract requires the guard to reject publishing capability
outside release.yml. Ordinary folded command scalars and multiline permission
values pass the line-oriented scan despite having the same YAML values as
rejected single-line controls. No workflow or publication was executed, and
no bypass of Trusted Publisher or environment authorization was demonstrated.

Dedupe: all 138 existing tasks, including completed tasks, were inspected.
Completed T-122 owns this guard and fixed quoted-key/job-scope cases; the
remaining multiline-scalar variants below are not that already-fixed syntax.
T-138 dependency/action updates do not establish semantic guard completeness.
T-038 remains the blocked maintainer-only release gate.

## Acceptance

- Reject the folded-command and multiline-permission fixtures below with a
  nonzero exit naming the offending workflow; equivalent direct controls also
  reject, while the unchanged repository workflows pass.
- Cover these ordinary YAML representations without adding a default runtime
  YAML dependency or claiming arbitrary shell-command semantic analysis.
- Preserve tag-only release checks and existing guard regression coverage.
- Run the mandatory ruff, mypy and pytest chain and local guard tests. Do not
  trigger hosted workflows or publish as verification.

## Verification Notes

Executable offline reproduction from repository root (only local guard runs):

```bash
probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
cp .github/workflows/*.yml "$probe/"
cat > "$probe/extra.yaml" <<'YAML'
name: Synthetic fixture, never executed
on: push
jobs:
  probe:
    runs-on: ubuntu-latest
    steps:
      - run: >-
          uv
          publish
YAML
bash scripts/check-publish-guard.sh "$probe"
# Actual: exit 0, "publish guard passed". Expected: reject, exit 1.
cat > "$probe/extra.yaml" <<'YAML'
name: Synthetic fixture, never executed
on: push
jobs:
  probe:
    runs-on: ubuntu-latest
    permissions:
      id-token:
        write
    steps:
      - run: echo harmless
YAML
bash scripts/check-publish-guard.sh "$probe"
# Actual: exit 0. Expected: reject, exit 1.
printf 'jobs:\n  probe:\n    steps:\n      - run: uv publish\n' > "$probe/extra.yaml"
if bash scripts/check-publish-guard.sh "$probe"; then
  echo 'ERROR: direct positive rejection control failed' >&2
  exit 1
fi
```

An isolated PyYAML 6.0.3 parser independently asserted the folded `run` equals
exactly `uv publish` and the permission equals exactly `write`. It was used
only for evidence, not installed into product requirements. Fresh actual
guard invocations confirmed both false negatives and direct rejection.

## Implementation Notes

Findings-only filing: no fix, task start, dependency change, gate transition,
release or publication is authorized by this task's creation. A constrained
syntax refusal is an available approach if semantic parsing is inappropriate;
do not weaken the documented guarantee silently.
