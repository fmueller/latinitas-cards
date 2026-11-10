#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checker="$script_dir/ci-scope.sh"
workflow="$script_dir/../.github/workflows/build.yml"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" -c user.name=Test -c user.email=test@example.com commit -qm base --allow-empty
base="$(git -C "$tmp" rev-parse HEAD)"

assert_scope() {
  local expected="$1" before="$2" after="$3" output
  output="$(cd "$tmp"; bash "$checker" "$before" "$after")"
  if [[ "$output" != "$expected" ]]; then
    printf 'FAIL: expected %q, got %q\n' "$expected" "$output" >&2
    exit 1
  fi
}

commit_file() {
  mkdir -p "$(dirname "$tmp/$1")"
  printf 'content\n' >> "$tmp/$1"
  git -C "$tmp" add -- "$1"
  git -C "$tmp" -c user.name=Test -c user.email=test@example.com commit -qm change
}

docs=$'full=false\nplanning=false'
planning=$'full=false\nplanning=true'
full=$'full=true\nplanning=true'

commit_file 'docs/guide with spaces.md'
assert_scope "$docs" "$base" HEAD
commit_file 'planning/tasks/T-999.md'
assert_scope "$planning" "$base" HEAD
commit_file 'src/example.py'
code="$(git -C "$tmp" rev-parse HEAD)"
commit_file 'CHANGELOG.md'
# A docs-only tip must not hide application changes earlier in a push.
assert_scope "$full" "$base" HEAD
assert_scope "$docs" "$code" HEAD
assert_scope "$full" '' HEAD
assert_scope "$full" '0000000000000000000000000000000000000000' HEAD
assert_scope "$full" 'unavailable-revision' HEAD
assert_scope "$full" HEAD HEAD

for path in README.md LICENSE pyproject.toml uv.lock .github/workflows/build.yml \
  lefthook.yml scripts/example.sh tests/example_test.py docs/schemas/example.json \
  docs/example.py .agents/skills/example/SKILL.md unknown.txt $'docs/newline\nname.py'; do
  before="$(git -C "$tmp" rev-parse HEAD)"
  commit_file "$path"
  assert_scope "$full" "$before" HEAD
done

for path in specs/v9.md .taskrail/config.yaml planning/STATE.md; do
  before="$(git -C "$tmp" rev-parse HEAD)"
  commit_file "$path"
  assert_scope "$planning" "$before" HEAD
done

before="$(git -C "$tmp" rev-parse HEAD)"
git -C "$tmp" mv src/example.py docs/renamed.md
git -C "$tmp" -c user.name=Test -c user.email=test@example.com commit -qm rename
assert_scope "$full" "$before" HEAD
before="$(git -C "$tmp" rev-parse HEAD)"
git -C "$tmp" rm -q docs/renamed.md
git -C "$tmp" -c user.name=Test -c user.email=test@example.com commit -qm delete
assert_scope "$docs" "$before" HEAD

# Execute the actual workflow steps, not copies of their selection/gate logic.
workflow_script() {
  awk -v name="$1" '
    $0 == "      - name: " name { found=1 }
    found && /^        run: \|$/ { run=1; next }
    run && /^          / { sub(/^          /, ""); print; next }
    run { exit }
  ' "$workflow"
}
workflow_script 'Classify changed paths' > "$tmp/select.sh"
workflow_script 'Require selected checks to pass' > "$tmp/result.sh"
test -s "$tmp/select.sh"
test -s "$tmp/result.sh"
mkdir -p "$tmp/scripts"
cp "$checker" "$tmp/scripts/ci-scope.sh"

assert_event_scope() {
  local event="$1" expected="$2"
  : > "$tmp/output"
  (cd "$tmp"; EVENT="$event" BEFORE="$base" PR_BASE="$code" PR_HEAD=HEAD \
    GITHUB_SHA=HEAD GITHUB_OUTPUT="$tmp/output" bash -e select.sh)
  if [[ "$(cat "$tmp/output")" != "$expected" ]]; then
    printf 'FAIL: wrong scope for %s\n' "$event" >&2
    exit 1
  fi
}
assert_event_scope push "$full"
assert_event_scope workflow_dispatch "$full"
# Main has application changes absent from this docs-only PR branch.
git -C "$tmp" switch -qc docs-branch "$base"
commit_file docs/pr.md
assert_event_scope pull_request "$docs"
commit_file src/pr.py
commit_file docs/pr.md
assert_event_scope pull_request "$full"

assert_result() {
  local expected="$1" results="$2" status=0
  RESULTS="$results" bash -e "$tmp/result.sh" > /dev/null || status=$?
  if [[ "$expected" == pass && "$status" != 0 || "$expected" == fail && "$status" == 0 ]]; then
    printf 'FAIL: unexpected Build Result: %s\n' "$results" >&2
    exit 1
  fi
}
for scope in docs planning full; do
  results="$(jq -n --arg scope "$scope" '{
    changes: {result: "success", outputs: {
      full: (if $scope == "full" then "true" else "false" end),
      planning: (if $scope == "docs" then "false" else "true" end)}},
    planning: {result: (if $scope == "docs" then "skipped" else "success" end)}
  } + (["lint-and-typecheck", "guards", "unit-tests", "package"] | map({
    key: ., value: {result: (if $scope == "full" then "success" else "skipped" end)}
  }) | from_entries)')"
  assert_result pass "$results"
  for job in changes planning lint-and-typecheck guards unit-tests package; do
    for status in failure cancelled; do
      assert_result fail "$(jq --arg job "$job" --arg status "$status" '.[$job].result=$status' <<< "$results")"
    done
    if [[ "$(jq -r --arg job "$job" '.[$job].result' <<< "$results")" == success ]]; then
      assert_result fail "$(jq --arg job "$job" '.[$job].result="skipped"' <<< "$results")"
    else
      assert_result fail "$(jq --arg job "$job" '.[$job].result="success"' <<< "$results")"
    fi
  done
  assert_result fail "$(jq 'del(.changes.outputs.full)' <<< "$results")"
  assert_result fail "$(jq 'del(.changes.outputs.planning)' <<< "$results")"
done
printf 'CI scope checks passed\n'
