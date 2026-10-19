---
id: T-144-reject-multiline-environment-object-names-in-non
title: Reject multiline environment object names in non-release guard
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-19T21:35:32Z"
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

The coordinator subsequently authorized this one-task implementation and
focused delivery to main. Round 10 remains COMPLETE NOT CLEAN; this fix
does not reset its result or execute round 11. The coordinator retains
ownership of the separately authorized round 11 after accepted delivery.

### Workflow-v3 evidence

1. Understand: fetched base `4db2b0230708be810128efb231e41fcbae05d96d`
   contains accepted T143 `80a05ac6488bd500d5c9da2ecdf85a2d4373b818`.
   Taskrail validation passed and its sole candidate was T144 before start.
   The active spec remains v0.3.0, SHA-256
   `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
   The nonrelease scanner owns this bounded lexical contract. Nested name
   headers were emitted but not passed through its scalar normalization.
2. RED: `bash scripts/check-publish-guard-test.sh` exited 1 before the
   scanner change: `FAIL: accepted unsafe extra.yaml:
   object-continued-first pypi environment`.
   GREEN: the same command passed after routing object name headers into
   the existing scalar normalization path. The 40 added cases cover five
   scalar forms, both key orders, pypi/staging and nonrelease/publish scope.
   The suite now has 219 guard cases; all prior controls remain unchanged.
3. Initial checks: `uv run ruff check` passed; `uv run mypy` reported
   `Success: no issues found in 169 source files`; `uv run pytest -v`
   reported `2633 passed, 7 skipped, 1 warning in 480.30s`.
4. A dedicated Task loaded `code-simplifier`, inspected the scripts and
   returned: `No changes made.` It independently reran the inert guard
   suite successfully. No simplification suggestions were rejected.
5. Separate parallel read-only Tasks loaded `code-reviewer` for General
   and Security. General used the ECC code reviewer; Security used the
   ECC security reviewer, security-review reference, cloud-infrastructure
   companion and common security rules. Both returned verbatim:
   `No concrete task-relevant findings.` General covers the shell/AWK
   correctness and regression contract; Security covers its input-scanning
   trust boundary. No Python/framework/database/network runtime changed,
   so no additional materially triggered specialist lane was identified.
   A fresh candidate-validation Task loaded `code-reviewer`, inspected the
   diff, reran the inert guard suite and returned verbatim:
   `The candidate list was empty and remains empty.`
6. Dispositions: no candidates, validated findings, rejected candidates,
   deferrals or fixes required. The finding list is empty, not omitted.
7. Fresh final checks: the exact `uv run ruff check` -> `uv run mypy` ->
   `uv run pytest -v` chain passed (2633 passed, 7 skipped, 1 warning in
   453.72s). Then sequential `mise run check` passed, including formatting,
   mypy, pytest (2633 passed, 7 skipped, 1 warning in 439.60s) and all
   configured shell guard suites. Separate final synthetic guard and
   actual-workflow guard runs passed. No full suites overlapped.
   A fresh read-only Task loaded `code-reviewer` in disposition-verification
   mode and returned verbatim: `Approve — no unresolved or newly introduced
   task-relevant issues.` It reran both guards and `git diff --check`, all
   exit 0. One review cycle; no fixes required.
8. Taskrail verification passed at `2026-10-19T21:35:32Z`; completion at
   the same timestamp followed all review and local checks. Before these
   lifecycle writes, validation, the pinned spec hash and unchanged fetched
   origin/main base were rechecked. Hosted exact-head CI acceptance remains
   a delivery handoff condition, not a claim made by local verification.

### Scope and limits

Only the local scanner and inert text-fixture tests change. Workflow,
release gate, dependency, spec and actual publishing behavior are untouched.
No fixture steps, OIDC, publication, release, tag, deployment or manually
triggered hosted workflow execute. This is no broad security clearance,
universal YAML/shell semantic guarantee or Trusted Publisher bypass claim.
T038 stays maintainer-blocked; T128 and advisory 144 remain outside scope.
- 2026-10-19T21:35:32Z: verification pass
- 2026-10-19T21:35:32Z: Reviewed and locally verified T144 only; focused main delivery authorized, exact-head hosted CI acceptance remains pending.
