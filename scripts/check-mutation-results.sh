#!/usr/bin/env bash
# Validate and summarize the raw statuses from a completed mutmut run.
#
# This initial policy is deliberately report-only for efficacy. It fails when
# results are absent or incomplete, but it does not copy a numerical floor from
# another project before Latinitas Cards has a clean baseline of its own.
#
# Reads `mutmut results --all true` on stdin, or runs it when given no input.
# Repeat --module with exact dotted module names for a differential run.
set -euo pipefail

selected=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --module)
      if [[ ! "${2:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$ ]]; then
        echo "check-mutation-results: --module requires a dotted module name" >&2
        exit 2
      fi
      selected+="$2 "
      shift 2
      ;;
    *)
      echo "check-mutation-results: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -t 0 ]; then
  results="$(mutmut results --all true)"
else
  results="$(cat)"
fi

printf '%s\n' "$results" | awk -v selected="$selected" '
  BEGIN {
    count = split(selected, names, " ")
    for (i = 1; i <= count; i++) {
      wanted[names[i]] = 1
      total[names[i]] = 0
    }
  }
  match($0, /^[[:space:]]*[A-Za-z_][A-Za-z0-9_.ǁ]*__mutmut_[0-9]+:[[:space:]]*[a-z_ ]+$/) {
    split($0, parts, ":")
    name = parts[1]
    status = parts[2]
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", status)
    sub(/\.[^.]*__mutmut_[0-9]+$/, "", name)
    module = name

    if (count && !(module in wanted)) {
      next
    }
    parsed++
    total[module]++
    if (status == "killed") {
      killed[module]++
    } else if (status == "timeout") {
      timeout[module]++
    } else if (status == "survived") {
      survived[module]++
    } else if (status == "no tests") {
      no_tests[module]++
    } else if (status == "skipped") {
      skipped[module]++
    } else if (status == "suspicious") {
      suspicious[module]++
    } else if (status == "caught by type check") {
      typecheck[module]++
    } else if (status == "segfault") {
      segfault[module]++
    } else if (status == "not checked" || status == "check was interrupted by user") {
      incomplete[module]++
    } else {
      unknown[module]++
    }
  }
  END {
    if (!count && parsed == 0) {
      print "check-mutation-results: no mutation results were found" > "/dev/stderr"
      exit 2
    }

    n = 0
    for (module in total) {
      modules[++n] = module
    }
    for (i = 1; i < n; i++) {
      for (j = i + 1; j <= n; j++) {
        if (modules[j] < modules[i]) {
          swap = modules[i]; modules[i] = modules[j]; modules[j] = swap
        }
      }
    }

    failed = 0
    for (i = 1; i <= n; i++) {
      module = modules[i]
      t = total[module] + 0
      if (t == 0) {
        printf "check-mutation-results: missing results: %s\n", module > "/dev/stderr"
        failed = 1
        continue
      }
      detected = killed[module] + timeout[module]
      efficacy = 100 * detected / t
      printf "  %-36s %3d/%-3d  %5.1f%%  killed=%d timeout=%d survived=%d no_tests=%d skipped=%d suspicious=%d typecheck=%d segfault=%d", \
        module, detected, t, efficacy, killed[module] + 0, timeout[module] + 0, \
        survived[module] + 0, no_tests[module] + 0, skipped[module] + 0, \
        suspicious[module] + 0, typecheck[module] + 0, segfault[module] + 0
      if (incomplete[module]) {
        printf "  incomplete mutants (%d)", incomplete[module]
        failed = 1
      } else if (unknown[module]) {
        printf "  unknown statuses (%d)", unknown[module]
        failed = 1
      } else {
        printf "  report only"
      }
      printf "\n"
    }

    if (failed) {
      print "check-mutation-results: missing, incomplete, or unknown mutation results" > "/dev/stderr"
      exit 1
    }
  }
'
