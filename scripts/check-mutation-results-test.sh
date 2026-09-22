#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checker="$script_dir/check-mutation-results.sh"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_accepts() {
  local name="$1"
  local results="$2"
  shift 2
  local output
  if ! output="$(printf '%s\n' "$results" | bash "$checker" "$@" 2>&1)"; then
    fail "$name was rejected: $output"
  fi
}

assert_rejects() {
  local name="$1"
  local results="$2"
  local expected="$3"
  shift 3
  local output
  if output="$(printf '%s\n' "$results" | bash "$checker" "$@" 2>&1)"; then
    fail "$name was accepted"
  fi
  if [[ "$output" != *"$expected"* ]]; then
    fail "$name did not report '$expected': $output"
  fi
}

assert_reports() {
  local name="$1"
  local results="$2"
  local expected="$3"
  shift 3
  local output
  output="$(printf '%s\n' "$results" | bash "$checker" "$@" 2>&1)" || true
  if [[ "$output" != *"$expected"* ]]; then
    fail "$name did not report '$expected': $output"
  fi
}

weak=""
for i in 1 2 3; do
  weak+="    latinitas_cards.cli.x__normalize__mutmut_$i: killed"$'\n'
done
for i in 4 5 6 7 8 9 10; do
  weak+="    latinitas_cards.cli.x__normalize__mutmut_$i: survived"$'\n'
done

class_methods=""
for i in 1 2 3 4; do
  class_methods+="    latinitas_cards.deck.xǁDeckǁ_load__mutmut_$i: killed"$'\n'
done
class_methods+="    latinitas_cards.deck.xǁDeckǁ_load__mutmut_5: timeout"$'\n'
class_methods+="    latinitas_cards.deck.xǁDeckǁ_load__mutmut_6: survived"$'\n'

# Baseline mode reports weak modules but deliberately has no efficacy floor.
assert_accepts report-only-weak-module "$weak"
assert_reports raw-counts "$weak" "3/10"
assert_reports raw-score "$weak" "30.0%"
assert_reports survivors-visible "$weak" "survived=7"
assert_accepts class-methods "$class_methods"
assert_reports class-method-timeout-count "$class_methods" "5/6"
assert_reports class-method-statuses "$class_methods" "timeout=1"

# Exact differential scope ignores unrelated cached outcomes.
unrelated="${weak//latinitas_cards.cli/latinitas_cards.cloze}"
unexecuted="${unrelated/survived/not checked}"
assert_accepts selected-only "$weak"$'\n'"$unexecuted" --module latinitas_cards.cli
assert_rejects missing-selected "$weak" "missing results: latinitas_cards.absent" --module latinitas_cards.absent
assert_rejects selected-unexecuted "$unexecuted" "incomplete mutants" --module latinitas_cards.cloze

# A full report is incomplete when any discovered mutant was not executed.
assert_rejects full-unexecuted "$weak"$'\n'"$unexecuted" "incomplete mutants"
assert_rejects interrupted "$weak"$'\n'"    latinitas_cards.cli.x__run__mutmut_11: check was interrupted by user" "incomplete mutants"
assert_rejects unknown-status "$weak"$'\n'"    latinitas_cards.cli.x__run__mutmut_11: future status" "unknown statuses"

# `no tests` is a completed coverage outcome, not interrupted execution. The
# report-only baseline keeps it visible without turning it into an implicit
# 100% mutant-coverage gate.
no_tests="$weak"$'\n'"    latinitas_cards.cli.x__run__mutmut_11: no tests"
assert_accepts no-tests-is-reportable "$no_tests"
assert_reports no-tests-is-visible "$no_tests" "no_tests=1"
assert_rejects no-results "" "no mutation results were found"

# Output order must not depend on awk hash ordering.
first="$(printf '%s\n' "$unrelated"$'\n'"$weak" | bash "$checker")"
second="$(printf '%s\n' "$unrelated"$'\n'"$weak" | bash "$checker")"
[[ "$first" == "$second" ]] || fail "output was not stable across runs"

if printf '%s\n' "$weak" | bash "$checker" --module 'not-a-module!' >/dev/null 2>&1; then
  fail "an invalid module name was accepted"
fi
if printf '%s\n' "$weak" | bash "$checker" --floor 70 >/dev/null 2>&1; then
  fail "an unsupported efficacy floor was accepted"
fi

bash "$script_dir/mutate-diff-test.sh"
bash "$script_dir/mutate-all-test.sh"
printf 'mutation result checks passed\n'
