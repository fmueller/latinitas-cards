#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/bin"
export MUTATION_CALLS="$tmp/calls"
cat >"$tmp/bin/uv" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$MUTATION_CALLS"
case "$*" in
  'run mutmut run') exit "${RUN_STATUS:-0}" ;;
  'run mutmut results --all true')
    printf '%s\n' 'latinitas_cards.cli.x__run__mutmut_1: killed'
    exit "${RESULT_STATUS:-0}"
    ;;
  *) exit 2 ;;
esac
SH
chmod +x "$tmp/bin/uv"
export PATH="$tmp/bin:$PATH"

if ! output="$(bash "$script_dir/mutate-all.sh" 2>&1)"; then
  echo "FAIL: full mutation wrapper rejected valid output: $output" >&2
  exit 1
fi
[[ "$output" == *'1/1'* ]] || { echo 'FAIL: full mutation report missing'; exit 1; }

for variable in RUN_STATUS RESULT_STATUS; do
  if env "$variable=7" bash "$script_dir/mutate-all.sh" >"$tmp/output" 2>&1; then
    echo "FAIL: $variable failure was hidden" >&2
    exit 1
  fi
done

printf 'full mutation checks passed\n'
