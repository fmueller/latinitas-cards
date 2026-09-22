#!/usr/bin/env bash
# Run mutation tests only for Python modules changed from a configurable base.
# Override the comparison point with BASE, for example `BASE=HEAD~3`.
set -euo pipefail

base="${BASE:-main}"
if ! git rev-parse --verify --quiet "$base" >/dev/null; then
  echo "mutate-diff: unknown base ref: $base" >&2
  exit 2
fi

merge_base="$(git merge-base "$base" HEAD 2>/dev/null || echo "$base")"
changed="$(git diff --name-only --diff-filter=d "$merge_base" -- src/latinitas_cards/ | grep '\.py$' || true)"

if [ -z "$changed" ]; then
  echo "mutate-diff: no source modules changed since $base; nothing to mutate"
  exit 0
fi

patterns=()
scope=()
while IFS= read -r path; do
  [ -z "$path" ] && continue
  module="${path#src/}"
  module="${module%.py}"
  module="${module//\//.}"
  if [[ "$module" == *.__init__ ]]; then
    module="${module%.__init__}"
    patterns+=("$module.x__*")
  else
    patterns+=("$module.*")
  fi
  scope+=(--module "$module")
done <<<"$changed"

echo "mutate-diff: mutating ${#patterns[@]} module(s) changed since $base"
printf '  %s\n' "${patterns[@]}"

# Rebuild mutmut test-to-mutant coverage so existing tests are considered for
# newly added functions. Keep other cached mutant state for faster iteration.
rm -f mutants/mutmut-stats.json
uv run mutmut run "${patterns[@]}"
uv run mutmut results --all true | bash "$(dirname "${BASH_SOURCE[0]}")/check-mutation-results.sh" "${scope[@]}"
