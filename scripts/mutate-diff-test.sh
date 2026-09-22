#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/bin"
export MUTATION_FIXTURE="$tmp/results" MUTATION_CALLS="$tmp/calls"
export EXPECTED_RUN='run mutmut run latinitas_cards.cli.*'
cat >"$tmp/bin/git" <<'SH'
#!/usr/bin/env bash
case "$1" in
  rev-parse|merge-base) echo main ;;
  diff) printf '%s\n' "${CHANGED_PATH:-src/latinitas_cards/cli.py}" ;;
  *) exit 2 ;;
esac
SH
cat >"$tmp/bin/uv" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$MUTATION_CALLS"
case "$*" in
  "$EXPECTED_RUN")
    [ ! -e mutants/mutmut-stats.json ] || echo 'stale stats' >>"$MUTATION_CALLS"
    exit "${RUN_STATUS:-0}" ;;
  'run mutmut results --all true') cat "$MUTATION_FIXTURE"; exit "${RESULT_STATUS:-0}" ;;
  *) exit 2 ;;
esac
SH
chmod +x "$tmp/bin/"*
export PATH="$tmp/bin:$PATH"
mkdir -p "$tmp/project/mutants/src"
cd "$tmp/project"

printf '{}\n' >mutants/mutmut-stats.json
touch mutants/src/kept
for i in $(seq 1 10); do
  status=survived
  [ "$i" -le 3 ] && status=killed
  printf 'latinitas_cards.cli.x__run__mutmut_%s: %s\n' "$i" "$status"
  printf 'latinitas_cards.cloze.x__run__mutmut_%s: not checked\n' "$i"
done >"$MUTATION_FIXTURE"

if ! output="$(bash "$script_dir/mutate-diff.sh" 2>&1)"; then
  echo "FAIL: cli-only differential run: $output" >&2
  exit 1
fi
[[ "$output" == *'3/10'* ]] || { echo 'FAIL: selected raw counts missing'; exit 1; }
[[ "$output" != *'latinitas_cards.cloze'* ]] || { echo 'FAIL: unrelated verdict'; exit 1; }
grep -Fx 'run mutmut results --all true' "$MUTATION_CALLS" >/dev/null
if grep -Fx 'stale stats' "$MUTATION_CALLS" >/dev/null || [ -e mutants/mutmut-stats.json ]; then
  echo 'FAIL: differential run reused a stale mutmut stats cache' >&2
  exit 1
fi
[ -e mutants/src/kept ] || { echo 'FAIL: removed more than the stats cache'; exit 1; }

for variable in RUN_STATUS RESULT_STATUS; do
  if env "$variable=7" bash "$script_dir/mutate-diff.sh" >"$tmp/output" 2>&1; then
    echo "FAIL: $variable failure was hidden" >&2
    exit 1
  fi
done

printf 'latinitas_cards.cloze.x__run__mutmut_1: killed\n' >"$MUTATION_FIXTURE"
if bash "$script_dir/mutate-diff.sh" >"$tmp/output" 2>&1; then
  echo 'FAIL: missing selected results passed' >&2
  exit 1
fi
grep -F 'missing results: latinitas_cards.cli' "$tmp/output" >/dev/null

: >"$MUTATION_CALLS"
if ! CHANGED_PATH=docs/usage.md bash "$script_dir/mutate-diff.sh" >"$tmp/output" 2>&1; then
  echo 'FAIL: documentation-only change failed' >&2
  exit 1
fi
grep -F 'nothing to mutate' "$tmp/output" >/dev/null
[ ! -s "$MUTATION_CALLS" ] || { echo 'FAIL: documentation-only change invoked uv'; exit 1; }

# Package initializers select only functions defined in that initializer. A
# glob ending in `.*` would also select every descendant module.
for package_case in root nested; do
  case "$package_case" in
    root)
      changed_path=src/latinitas_cards/__init__.py
      module=latinitas_cards
      unrelated=latinitas_cards.cli
      ;;
    nested)
      changed_path=src/latinitas_cards/commands/__init__.py
      module=latinitas_cards.commands
      unrelated=latinitas_cards.commands.cloze
      ;;
  esac
  export EXPECTED_RUN="run mutmut run $module.x__*"
  {
    printf '%s\n' "    $module.x__version__mutmut_1: killed"
    printf '%s\n' "    $unrelated.x__run__mutmut_1: not checked"
  } >"$MUTATION_FIXTURE"
  : >"$MUTATION_CALLS"
  if ! output="$(CHANGED_PATH="$changed_path" bash "$script_dir/mutate-diff.sh" 2>&1)"; then
    echo "FAIL: $package_case initializer differential run: $output" >&2
    exit 1
  fi
  [[ "$output" == *"$module"* ]] || { echo "FAIL: $package_case initializer result missing"; exit 1; }
  [[ "$output" != *"$unrelated"* ]] || { echo "FAIL: $package_case initializer selected descendants"; exit 1; }
  grep -Fx "$EXPECTED_RUN" "$MUTATION_CALLS" >/dev/null
done

printf 'differential mutation checks passed\n'
