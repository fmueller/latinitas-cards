#!/usr/bin/env bash
set -euo pipefail

# Missing history (including new branches) must never turn into a docs-only run.
base="${1:-}"
head="${2:-HEAD}"
paths="$(mktemp)"
trap 'rm -f "$paths"' EXIT
full=false
planning=false
if [[ -z "$base" ]] || ! git diff --no-renames --name-only -z "$base" "$head" -- > "$paths"; then
  full=true
elif [[ ! -s "$paths" ]]; then
  full=true
else
  while IFS= read -r -d '' path; do
    case "$path" in
      planning/*|specs/*|.taskrail/*) planning=true ;;
      docs/*.md|AGENTS.md|CLAUDE.md|CHANGELOG.md) ;;
      *) full=true ;;
    esac
  done < "$paths"
fi
if [[ "$full" == true ]]; then
  planning=true
fi
printf 'full=%s\nplanning=%s\n' "$full" "$planning"
