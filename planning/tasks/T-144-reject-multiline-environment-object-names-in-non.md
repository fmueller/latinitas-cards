---
id: T-144-reject-multiline-environment-object-names-in-non
title: Reject multiline environment object names in non-release guard
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-19T18:11:51Z"
---

# T-144-reject-multiline-environment-object-names-in-non Reject multiline environment object names in non-release guard

## Description

Final authorized round 10 synthetic E2E at accepted product revision
80a05ac6488bd500d5c9da2ecdf85a2d4373b818 confirms a medium-severity local
no-publish guard false negative under the PyPI Publishing contract. A
non-release job's environment object can select the exact name `pypi`
through a continued, folded or literal `name` scalar, but the scanner
accepts it. The equivalent direct `name: pypi` rejects. This is one nested
scalar-normalization finding, not ten distinct defects.

GitHub workflow syntax supports an environment object with `name` and
`url`; ordinary YAML scalar rules apply to the string name. Independent
PyYAML 6.0.3 checks confirm the exact value `pypi` in every suspect, with
no trailing newline. The fixtures remain inert: only the local scanner
executes, never workflow steps, publication, OIDC or deployment. This is
not evidence of bypassing Trusted Publisher or environment approval.

Dedupe examined all 143 preexisting tasks, including 141 completed.
Completed T-122 owns the guard; T-139 handles continued/folded top-level
values, T-141 release OIDC scope, and T-142 literal OIDC values. T-143
handles literal environment strings and block/flow objects with direct
names. Its controls pass at this pin, but its block-object name
normalization emits `environment: |-` (or an empty value) without joining
that child's scalar content. No open task covers this remaining
composition of supported object and scalar forms. T-038 remains blocked
and maintainer-only; T-128 remains future v0.8.0.

The interrupted original round-10 tester recovered this same unpushed
finding. The replacement tester independently reproduced it before
recreating this task through `taskrail task new`, which assigned the
expected T-144 id. No original task file or state was copied into tracked
planning files. This is the same round 10, not a new testing budget.

## Acceptance

- Reject non-release environment objects whose exact `name` is `pypi`
  through continued, folded-strip or literal-strip scalars, including
  explicit indentation/chomping forms and either name/url key order.
  Return nonzero and name the offending workflow/capability.
- Preserve direct object-name rejection, benign `staging` names, allowed
  publish-job names and all accepted T-139/T-141/T-142/T-143 controls.
- Keep fixtures inert and checks local. A bounded syntax refusal is an
  acceptable design choice; no runtime YAML dependency or universal
  YAML/shell guarantee is required. Do not weaken the guard contract.
- Filing authorizes no implementation, task start, release, publication,
  hosted workflow trigger, gate/dependency/spec change or round 11.

## Verification Notes

Self-contained offline reproduction from the repository root:

```bash
probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
cp .github/workflows/*.yml "$probe/"
cat > "$probe/extra.yaml" <<'YAML'
name: Inert nested scalar confirmation
on: push
jobs:
  probe:
    runs-on: ubuntu-latest
    environment:
      name: |-
        pypi
      url: https://example.invalid/inert
    steps:
      - run: echo inert
YAML
bash scripts/check-publish-guard.sh "$probe"
# Actual: exit 0, publish guard passed. Expected: nonzero naming extra.yaml.

sed -i 's/name: |-/name: >-/' "$probe/extra.yaml"
bash scripts/check-publish-guard.sh "$probe"
# Same false negative; folded-strip name is exactly pypi.

sed -i 's/name: >-/name: # target/' "$probe/extra.yaml"
bash scripts/check-publish-guard.sh "$probe"
# Same false negative; continued plain name is exactly pypi.

sed -i '/name: # target/c\      name: pypi' "$probe/extra.yaml"
sed -i '/^        pypi$/d' "$probe/extra.yaml"
bash scripts/check-publish-guard.sh "$probe"
# Direct control: exit 1, publishing capability outside release.yml.
```

Replacement independent object-name matrix: 48 scanner calls / 146
assertions; ten false-negative exits (five multiline forms, both name/url
orders). All 96 exact parsed name/url assertions pass. Direct names reject;
all benign staging and permitted release publish-job controls pass.
Separate hand-written fixtures independently confirm continued/folded/
literal false negatives, direct rejection and literal staging acceptance,
with five parser subprocesses and five scanner calls. No fixture steps run.
The existing inert 179-case guard suite also passes at the tested pin.

Reference: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idenvironment

## Implementation Notes

Finding only; unstarted and unfixed. Final authorized adversarial round
10 is NOT CLEAN. Further implementation or another round requires a new
user decision through the coordinator; round 11 is not authorized.
