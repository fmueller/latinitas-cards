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
grep -F '"data/",' "$root/pyproject.toml" >/dev/null || fail 'mutation sandbox omits the bundled corpus'
grep -F 'testpaths = ["tests"]' "$root/pyproject.toml" >/dev/null || fail 'pytest collection is not bounded'
grep -F 'name = "mutmut"' "$root/uv.lock" >/dev/null || fail 'mutmut is absent from the lock'

grep -F '[tasks."test:mutate"]' "$root/mise.toml" >/dev/null || fail 'differential mutation task is missing'
grep -F '[tasks."test:mutate:gate"]' "$root/mise.toml" >/dev/null || fail 'full mutation task is missing'
grep -F 'run = "bash scripts/mutate-all.sh"' "$root/mise.toml" >/dev/null || fail 'full mutation task bypasses the fail-closed wrapper'
grep -F 'bash scripts/check-mutation-results-test.sh' "$root/mise.toml" >/dev/null || fail 'result checks are absent from the ordinary gate'
grep -F 'bash scripts/check-mutation-config-test.sh' "$root/mise.toml" >/dev/null || fail 'configuration checks are absent from the ordinary gate'

workflow="$root/.github/workflows/mutation.yml"
[ -f "$workflow" ] || fail 'mutation workflow is missing'
grep -Eq '^    timeout-minutes: 120$' "$workflow" || fail 'mutation workflow must allow 120 minutes'
grep -F 'shard: [0, 1, 2, 3, 4, 5, 6, 7]' "$workflow" >/dev/null || fail 'mutation workflow must cover all eight shards'
grep -F 'fail-fast: false' "$workflow" >/dev/null || fail 'a failed shard must not cancel other results'
grep -F 'scripts/configure-mutation-shard.py "$SHARD" 8' "$workflow" >/dev/null || fail 'shard configuration is missing'
grep -F 'name: mutation-results-${{ matrix.shard }}' "$workflow" >/dev/null || fail 'shard artifact names must be unique'
grep -F 'mutation-scope.json' "$workflow" >/dev/null || fail 'shard scope is not preserved'
grep -F 'schedule:' "$workflow" >/dev/null || fail 'weekly trigger is missing'
grep -F 'workflow_dispatch:' "$workflow" >/dev/null || fail 'manual trigger is missing'
if grep -Eq '^  (push|pull_request):' "$workflow"; then
  fail 'expensive mutation workflow must not run for pushes or pull requests'
fi
grep -F 'uv run mutmut run' "$workflow" >/dev/null || fail 'workflow does not run mutmut'
grep -F 'uv run mutmut export-cicd-stats' "$workflow" >/dev/null || fail 'workflow does not export machine-readable stats'
grep -F 'bash scripts/check-mutation-results.sh' "$workflow" >/dev/null || fail 'workflow does not validate complete results'
grep -F 'actions/upload-artifact@' "$workflow" >/dev/null || fail 'workflow does not preserve mutation artifacts'

overview="$(awk '/^  mutation-overview:/ {capture=1; next} capture && /^  [^ ]/ {exit} capture {print}' "$workflow")"
grep -Fxq '    needs: mutation-tests' <<< "$overview" || fail 'overview must wait for all shards'
grep -Fxq '    if: always()' <<< "$overview" || fail 'overview must run after shard failure'
grep -Fq 'merge-multiple: false' <<< "$overview" || fail 'overview must keep shard files separate'
grep -Fq 'SHARD_RESULT: ${{ needs.mutation-tests.result }}' <<< "$overview" || fail 'overview must check shard outcomes'
grep -Fq 'python scripts/summarize-mutations.py mutation-artifacts' <<< "$overview" || fail 'overview is missing'
grep -Fq -- '--shards 8 --shard-result "$SHARD_RESULT" --output mutation-summary' <<< "$overview" || fail 'overview scope is wrong'
grep -Fq 'cat mutation-summary/summary.md >> "$GITHUB_STEP_SUMMARY"' <<< "$overview" || fail 'overview is not published'
for step in 'Aggregate results' 'Publish overview' 'Upload overview'; do
  awk -v name="$step" '
    $0 == "      - name: " name {found=1; next}
    found && /^      - / {exit}
    found && $0 == "        if: always()" {unconditional=1}
    END {exit !unconditional}
  ' <<< "$overview" || fail "$step must run after failure"
done
grep -Fxq '          name: mutation-overview' <<< "$overview" || fail 'overview artifact name is wrong'
grep -Fxq '          path: mutation-summary/' <<< "$overview" || fail 'overview artifact path is wrong'
grep -Fxq '          retention-days: 30' <<< "$overview" || fail 'overview must retain reports for 30 days'

build_workflow="$root/.github/workflows/build.yml"
grep -F 'bash scripts/check-mutation-results-test.sh' "$build_workflow" >/dev/null || fail 'result checks are absent from CI'
grep -F 'bash scripts/check-mutation-config-test.sh' "$build_workflow" >/dev/null || fail 'configuration checks are absent from CI'

policy="$root/docs/mutation-testing.md"
[ -f "$policy" ] || fail 'mutation policy documentation is missing'
grep -Fi 'no efficacy floor' "$policy" >/dev/null || fail 'report-only policy is not documented'
grep -Fi 'clean baseline' "$policy" >/dev/null || fail 'baseline requirement is not documented'

printf 'mutation configuration checks passed\n'
