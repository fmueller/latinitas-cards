---
id: T-143-reject-equivalent-pypi-environment-forms-outside
title: Reject equivalent PyPI environment forms outside release workflow
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-18T21:19:04Z"
---

# T-143-reject-equivalent-pypi-environment-forms-outside Reject equivalent PyPI environment forms outside release workflow

## Description

Round 9 synthetic E2E at accepted product revision
218600700adc2092a6f4d517071b38d4b7321be2 confirms a medium-severity local
no-publish guard false negative under specs/v0.3.0.md#pypi-publishing.
Non-release jobs selecting the exact environment name `pypi` through an
ordinary literal scalar or environment object pass the guard. The equivalent
direct string rejects. This is one environment-representation finding with
five observed manifestations, not five separate defects.

GitHub workflow syntax supports both a string environment name and an object
with `name` (and optional `url`). Independent PyYAML 6.0.3 parsing confirms
the literal strip forms yield exactly `pypi`, without a newline, and the
objects contain exactly that name. No fixture steps, workflow, publication,
deployment, OIDC request or hosted workflow were executed. This demonstrates
a local lexical guard gap, not a Trusted Publisher or approval bypass.

Dedupe inventoried all 142 preexisting tasks, including 140 completed.
Completed T-122 owns non-release `environment: pypi` rejection. T-139 handles
continued/folded non-release values (including environment continuation),
T-141 handles continued/folded release OIDC scope, and T-142 handles literal
OIDC values. All relevant old controls pass at this pin; none tracks literal
environment values or supported environment objects. T-038 stays blocked
and maintainer-only; T-128 stays excluded future v0.8.0. No gate, priority,
dependency or spec change is requested.

## Acceptance

- Reject non-release jobs selecting `pypi` through literal strip scalars
  (`|-`, `|2-`, `|-2`) or block/flow environment objects with `name: pypi`.
  Return nonzero and name the offending workflow/capability.
- Preserve rejection of direct, continued and folded `pypi` environment
  values; preserve benign `staging` controls, allowed publish-job environment
  representations, strict tags, and existing OIDC scope/scalar controls.
- Keep fixtures inert and checks local. A bounded syntax refusal is an
  available design choice; no runtime YAML dependency or universal YAML/shell
  guarantee is required. Do not weaken the documented guard boundary.
- After separately authorized implementation, run focused guard checks and
  the mandatory ruff/mypy/pytest chain. Filing authorizes no implementation,
  task start, release, publication, deployment or hosted workflow trigger.

## Verification Notes

Self-contained offline reproduction from repository root. Only the scanner
executes; the copied workflow contents remain inert:

```bash
probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
cp .github/workflows/*.yml "$probe/"
cat > "$probe/extra.yaml" <<'YAML'
name: Inert environment fixture
on: push
jobs:
  probe:
    runs-on: ubuntu-latest
    environment: |-
      pypi
    steps:
      - run: echo inert
YAML
bash scripts/check-publish-guard.sh "$probe"
# Actual: exit 0, publish guard passed. Expected: nonzero, naming extra.yaml.

sed -i '/    environment: |-/c\    environment:' "$probe/extra.yaml"
sed -i 's/^      pypi$/      name: pypi/' "$probe/extra.yaml"
bash scripts/check-publish-guard.sh "$probe"
# Actual: exit 0. Expected: nonzero for the supported environment object.

sed -i '/    environment:/c\    environment: pypi' "$probe/extra.yaml"
sed -i '/^      name: pypi$/d' "$probe/extra.yaml"
bash scripts/check-publish-guard.sh "$probe"
# Actual and expected: exit 1, publishing capability outside release.yml.
```

Fresh semantic/scope table: 72 local scanner invocations, 165 assertions;
five false negatives, all environment forms above. All 48 OIDC combinations
(root/build/publish/non-release, six scalar forms, write/none) pass, as do
direct/continued/folded environment rejection, benign staging and allowed
publish environment controls. Separate shell construction confirms literal
and block-object false negatives, direct rejection, and all three benign
staging controls. This independent confirmation does not reuse generated YAML.

The non-release scanner normalizes folded scalars and bare environment
continuations, but literal-header recognition is restricted to id-token;
neither its normalized nor raw grep recognizes environment objects.
This explains the observations without executing workflow contents.

Reference: GitHub Actions workflow syntax, jobs.<job_id>.environment:
https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idenvironment

## Implementation Notes

Implemented under the separately authorized single-task lease on cumulative
base 53dd4a59c72d36412ae2869a625cb41d3cf6f974, containing accepted T-142.
The v0.3.0 spec SHA256 remains
fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704.

The existing non-release scanner now normalizes literal environment scalars
and block-object names and recognizes inline/multiline flow-object names.
The regression matrix covers 11 representations, two names (pypi/staging),
and two scopes (non-release/publish). Existing OIDC and tag controls remain.
No workflow, gate, spec, dependency, or runtime YAML parser changed.

Strict TDD: the expanded guard suite first failed with
`FAIL: accepted unsafe extra.yaml: |- pypi environment`; a separate
multiline-flow case then failed with
`FAIL: accepted unsafe extra.yaml: flow-multiline pypi environment`.
Both passed after minimal source changes. The final inert suite passes
179 scanner cases, including 44 new matrix cases. No fixture steps execute.

Dedicated code-simplifier Task loaded its skill and made no changes.
Separate parallel General and Security code-reviewer Tasks loaded their
mapped reviewers; both returned "No concrete task-relevant findings."
General loaded common review/testing/security guidance; Security loaded
security-review and cloud-infrastructure-security guidance. No language or
framework lane is triggered by this shell-only change; database, network,
ML, and other specialist domains are unaffected. A fresh candidate validator
confirmed "Candidate outcome: empty set validated." There are no findings
to fix, reject, deduplicate, or defer. A fresh final disposition verifier
confirmed no unresolved or newly introduced task-relevant issues after one
review cycle.

Initial exact ruff/mypy/pytest chain passed (2633 passed, 7 skipped,
390.47 seconds). Final exact `uv run ruff check`, `uv run mypy`,
`uv run pytest -v` passed (169 source files; 2633 passed, 7 skipped,
386.15 seconds). Sequential `mise run check` passed, including its full
pytest run (2633 passed, 7 skipped, 381.64 seconds), packaging and shell
guard suites. Final direct guard, synthetic guard suite, and
`git diff --check` passed; the final verifier reran the focused suite.

This remains a bounded local lexical guard, not universal YAML/shell
analysis or evidence of a Trusted Publisher/approval bypass. No release,
publication, deployment, manual hosted workflow trigger, next task, or
round-10 investigation was performed. T-038 remains maintainer-blocked;
T-128 and advisory 144 are outside this task. Hosted exact-head acceptance
and remote delivery are reported separately after push.
- 2026-10-18T21:19:04Z: verification pass
- 2026-10-18T21:19:04Z: Implemented bounded environment representation normalization after one reviewed cycle; final ruff/mypy/pytest and mise check passed. Exact-head hosted CI acceptance follows authorized push.
