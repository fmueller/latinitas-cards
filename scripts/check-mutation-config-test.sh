#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

grep -Eq '"mutmut>=3\.8\.0,<4"' "$root/pyproject.toml" || fail 'mutmut development dependency is not pinned'
grep -F '[tool.mutmut]' "$root/pyproject.toml" >/dev/null || fail 'mutmut configuration is missing'
grep -F 'source_paths = ["src/latinitas_cards/"]' "$root/pyproject.toml" >/dev/null || fail 'mutation source path is wrong'
grep -F 'testpaths = ["tests"]' "$root/pyproject.toml" >/dev/null || fail 'pytest collection is not bounded'
grep -F 'name = "mutmut"' "$root/uv.lock" >/dev/null || fail 'mutmut is absent from the lock'

grep -F '[tasks."test:mutate"]' "$root/mise.toml" >/dev/null || fail 'differential mutation task is missing'
grep -F '[tasks."test:mutate:gate"]' "$root/mise.toml" >/dev/null || fail 'full mutation task is missing'
grep -F 'run = "bash scripts/mutate-all.sh"' "$root/mise.toml" >/dev/null || fail 'full mutation task bypasses the fail-closed wrapper'
grep -F 'bash scripts/check-mutation-results-test.sh' "$root/mise.toml" >/dev/null || fail 'result checks are absent from the ordinary gate'
grep -F 'bash scripts/check-mutation-config-test.sh' "$root/mise.toml" >/dev/null || fail 'configuration checks are absent from the ordinary gate'

workflow="$root/.github/workflows/mutation.yml"
[ -f "$workflow" ] || fail 'mutation workflow is missing'
grep -F 'schedule:' "$workflow" >/dev/null || fail 'weekly trigger is missing'
grep -F 'workflow_dispatch:' "$workflow" >/dev/null || fail 'manual trigger is missing'
if grep -Eq '^  (push|pull_request):' "$workflow"; then
  fail 'expensive mutation workflow must not run for pushes or pull requests'
fi
grep -F 'uv run mutmut run' "$workflow" >/dev/null || fail 'workflow does not run mutmut'
grep -F 'uv run mutmut export-cicd-stats' "$workflow" >/dev/null || fail 'workflow does not export machine-readable stats'
grep -F 'bash scripts/check-mutation-results.sh' "$workflow" >/dev/null || fail 'workflow does not validate complete results'
grep -F 'actions/upload-artifact@' "$workflow" >/dev/null || fail 'workflow does not preserve mutation artifacts'

build_workflow="$root/.github/workflows/build.yml"
grep -F 'bash scripts/check-mutation-results-test.sh' "$build_workflow" >/dev/null || fail 'result checks are absent from CI'
grep -F 'bash scripts/check-mutation-config-test.sh' "$build_workflow" >/dev/null || fail 'configuration checks are absent from CI'

policy="$root/docs/mutation-testing.md"
[ -f "$policy" ] || fail 'mutation policy documentation is missing'
grep -Fi 'no efficacy floor' "$policy" >/dev/null || fail 'report-only policy is not documented'
grep -Fi 'clean baseline' "$policy" >/dev/null || fail 'baseline requirement is not documented'

printf 'mutation configuration checks passed\n'
