#!/usr/bin/env bash
# Intentionally line-oriented: release trigger structure is a strict allowlist.
set -euo pipefail
directory="${1:-.github/workflows}"
shopt -s nullglob
files=("$directory"/*.yml "$directory"/*.yaml)
for file in "${files[@]}"; do
  if [[ $(basename "$file") != release.yml ]]; then
    # Also scan folded scalars, literal permissions and continued values.
    # This is a bounded lexical check, not YAML or shell semantic analysis.
    if awk '
      {
        print
        if (pending && ($0 ~ /^[[:space:]]*$/ || (continued && $0 ~ /^[[:space:]]*#/))) next
        indent = match($0, /[^ ]/) - 1
        if (pending && indent <= base) {
          print pending
          pending = ""
        }
        if (pending) {
          value = $0
          sub(/^[[:space:]]+/, "", value)
          pending = pending " " value
          next
        }
        if ($0 ~ /:[[:space:]]*>[-+1-9]*([[:space:]]+#.*)?[[:space:]]*$/ ||
            $0 ~ /id-token["\047]?[[:space:]]*:[[:space:]]*\|[-+1-9]*([[:space:]]+#.*)?[[:space:]]*$/ ||
            $0 ~ /(id-token|environment)["\047]?[[:space:]]*:[[:space:]]*(#.*)?$/) {
          pending = $0
          continued = ($0 !~ /:[[:space:]]*[>|]/)
          if (continued) sub(/[[:space:]]*#.*$/, "", pending)
          sub(/[>|][+1-9-]*([[:space:]]+#.*)?[[:space:]]*$/, "", pending)
          base = indent
        }
      }
      END { if (pending) print pending }
    ' "$file" | grep -En "id-token[\"']?[[:space:]]*:[[:space:]]*[\"']?write|pypa/gh-action-pypi-publish|twine[[:space:]]+upload|uv[[:space:]]+publish|environment[\"']?[[:space:]]*:[[:space:]]*[\"']?pypi"; then
      printf 'publish guard failed: %s: publishing capability outside release.yml\n' "$file" >&2
      exit 1
    fi
  else
    awk '
      function fail(message) {
        print "publish guard failed: " FILENAME ": " message > "/dev/stderr"
        bad = 1
      }
      /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
      {
        indent = match($0, /[^ ]/) - 1
        if (pending && indent <= base) pending = 0
        if (pending && $0 ~ /^[[:space:]]*["\047]?write/) {
          if (!allowed) fail("id-token: write outside publish")
        }
        if ($0 ~ /id-token["\047]?[[:space:]]*:[[:space:]]*([>|][-+1-9]*)?([[:space:]]+#.*)?[[:space:]]*$/) {
          pending = 1
          base = indent
          allowed = (jobs && job == "  publish:")
        }
      }
      /^[^ ]/ {
        trigger = ($0 == "on:")
        jobs = ($0 == "jobs:")
        job = ""
        if (trigger) seen++
        if ($0 ~ /^["\047]?on["\047]?[[:space:]]*:/ && !trigger) fail("non-tag trigger syntax")
      }
      trigger && !/^on:/ {
        if ($0 == "  push:") push++
        else if ($0 == "    tags:") tags++
        else if ($0 == "      - '\''v[0-9]+.[0-9]+.[0-9]+'\''") pattern++
        else fail("non-tag trigger: " $0)
      }
      jobs && /^  [^ ]/ { job = $0 }
      /id-token["\047]?[[:space:]]*:[[:space:]]*["\047]?write/ {
        if (!jobs || job != "  publish:") fail("id-token: write outside publish")
      }
      END {
        if (seen != 1 || push != 1 || tags != 1 || pattern != 1) fail("expected only the release tag trigger")
        exit bad
      }
    ' "$file"
  fi
done
printf 'publish guard passed\n'
