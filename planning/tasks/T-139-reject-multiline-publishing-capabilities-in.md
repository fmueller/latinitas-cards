---
id: T-139-reject-multiline-publishing-capabilities-in
title: Reject multiline publishing capabilities in workflow guard
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-18T11:33:26Z"
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

### Implementation cycle (2026-10-18)

- Revalidated `taskrail validate` and `taskrail next --json` before start:
  T-139 selected, `off_spec: false`, no other task active. Accepted remote
  main base: `9a2633c3179e4c62f4eee9c1a4cda3d409d74973`.
- Spec pin remained v0.3.0, SHA256
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
- Added bounded scalar normalization before the existing non-release grep.
  Release trigger and job checks are unchanged; no dependencies added.
  This remains lexical scanning, not general YAML or shell analysis.
- RED: `bash scripts/check-publish-guard-test.sh` exited 1 with
  `FAIL: accepted unsafe extra.yaml: folded uv publish (>)`.
  Independent pre-fix multiline permission reproduction exited 0 with
  `publish guard passed`, confirming the second false negative.
- GREEN: the same suite rejected `>`, `>-`, `>+`, plain/single/double-quoted
  multiline write values, named `extra.yaml`, preserved direct rejection,
  accepted harmless folded commands/read permission and repository workflows,
  and passed all existing release guard fixtures.
- Initial exact chain: `uv run ruff check` passed;
  `uv run mypy` passed (169 source files);
  `uv run pytest -v`: 2625 passed, 7 skipped, 1 warning in 365.00s.
- Dedicated Task loaded `code-simplifier`: no changes recommended; focused
  guard suite passed. No suggestions rejected.
- Separate parallel read-only Task review lanes loaded `code-reviewer`:
  General (`ecc/agents/code-reviewer.md`, general review/testing rules) and
  Security (`security-reviewer.md`, `security-review/REFERENCE.md`,
  `cloud-infrastructure-security.md`). General concluded verbatim:
  "No concrete task-relevant findings."
  Security chosen for publishing/OIDC boundary; no base-language lane
  applies to shell/AWK, and no framework/database/ML/network domain changed.
  Two lanes are within the default specialist budget.

#### Review finding and disposition

Security reviewer candidate, retained verbatim:

> FINDING SEC-1 — security
> Severity: medium
> Evidence: `scripts/check-publish-guard.sh:15-22`. The accumulator appends every indented line, including YAML comment lines, to `pending`; the publishing-pattern check then misses a valid multiline permission value.
> Finding: The guard accepts an `id-token: write` permission outside `release.yml` when a YAML comment appears between the key and its value.
> Failure/impact: In a copied workflow directory, this fixture makes the guard exit 0 with `publish guard passed`, although the permission value is `write`:

```yaml
jobs:
  probe:
    permissions:
      id-token:
        # document the OIDC permission
        write
    steps:
      - run: echo harmless
```

> The comment is ignored by YAML parsing, so it does not change the permission value. This leaves a task-relevant false negative in the no-publish guard.
> Recommended direction: Ignore comment-only lines while accumulating continued values, and add this fixture to the guard regression tests; keep the check lexical and dependency-free.

- Fresh read-only candidate-validator Task loaded `code-reviewer`, reproduced
  exit 0 and returned `VALIDATED — SEC-1`. No rejected or duplicate candidates.
- SEC-1 disposition: fixed. Added comments/blank-line regression (including
  inline key comments). RED: suite exited 1 with
  `FAIL: accepted unsafe extra.yaml: multiline id-token with comments and blank line`.
  Minimal fix skips YAML comments/blank lines only for continued plain
  permission/environment values, not folded command content. GREEN: suite
  passed all fixtures. No deferred findings.
- All fixture checks are offline text scans. No publication, hosted workflow
  trigger, release, tag, deployment, or gate manipulation was performed.
  Round6 remains NOT CLEAN/INCOMPLETE; T-038 remains maintainer-blocked.
- Final exact chain after SEC-1: `uv run ruff check` passed;
  `uv run mypy` passed (169 source files); `uv run pytest -v` passed:
  2625 passed, 7 skipped, 1 warning in 326.94s.
- Sequential `mise run check` exited 0: Ruff, format (396 files), mypy,
  pytest (2625 passed, 7 skipped, 1 warning in 333.72s), commit/push/author,
  mutation result/configuration, distribution inspection (54 cases), and
  publishing guard suites passed. `git diff --check` and `bash -n` passed.
- Fresh read-only Task loaded `code-reviewer` in disposition-verification
  mode, with General/Security references. Conclusion verbatim:
  "SEC-1 — RESOLVED." and "New task-local findings: None."
  It reran the focused guard suite, diff check and shell syntax check.
  One review/fix/recheck cycle; no unresolved or deferred findings.
- 2026-10-18T11:33:26Z: verification pass
