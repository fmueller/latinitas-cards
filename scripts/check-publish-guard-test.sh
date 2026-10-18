#!/usr/bin/env bash
# Only inspect temporary workflow text: no actions or publishing tools run.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}
fresh() {
  rm -rf "$temporary/workflows"
  mkdir "$temporary/workflows"
  cp "$root"/.github/workflows/*.yml "$temporary/workflows/"
}
reject() {
  local file="$1"
  if bash "$root/scripts/check-publish-guard.sh" "$temporary/workflows" > "$temporary/output" 2>&1; then
    fail "accepted unsafe $file: $2"
  fi
  grep -F "$file" "$temporary/output" >/dev/null || fail "did not name $file"
  printf 'PASS: rejected %s: %s\n' "$file" "$2"
}
fresh
bash "$root/scripts/check-publish-guard.sh" "$temporary/workflows"
for entry in 'build.yml|id-token: write' 'mutation.yml|pypa/gh-action-pypi-publish@bad' \
  'extra.yml|twine upload dist/*' 'extra.yaml|uv publish' 'extra.yaml|environment: pypi' \
  'build.yml|permissions: {"id-token": "write"}' "extra.yaml|permissions: {'id-token': 'write'}"; do
  fresh
  file="${entry%%|*}"
  printf '\n%s\n' "${entry#*|}" >> "$temporary/workflows/$file"
  reject "$file" "${entry#*|}"
done
for scalar in '>' '>-' '>+'; do
  fresh
  printf 'jobs:\n  probe:\n    steps:\n      - run: %s\n          uv\n          publish\n' "$scalar" > "$temporary/workflows/extra.yaml"
  reject extra.yaml "folded uv publish ($scalar)"
done
for value in write '"write"' "'write'"; do
  fresh
  printf 'jobs:\n  probe:\n    permissions:\n      id-token:\n        %s\n    steps:\n      - run: echo harmless\n' "$value" > "$temporary/workflows/extra.yaml"
  reject extra.yaml "multiline id-token $value"
done
for comment in '' ' # OIDC permission'; do
  fresh
  printf 'jobs:\n  probe:\n    permissions:\n      id-token:%s\n\n        # document the OIDC permission\n        write\n    steps:\n      - run: echo harmless\n' "$comment" > "$temporary/workflows/extra.yaml"
  reject extra.yaml 'multiline id-token with comments and blank line'
done
fresh
printf 'jobs:\n  probe:\n    permissions:\n      id-token:\n        read\n    steps:\n      - run: >-\n          echo\n          harmless\n' > "$temporary/workflows/extra.yaml"
bash "$root/scripts/check-publish-guard.sh" "$temporary/workflows"
for trigger in workflow_dispatch schedule release pull_request branches; do
  fresh
  sed -i "/^on:/a\  $trigger:" "$temporary/workflows/release.yml"
  reject release.yml "$trigger trigger"
done
fresh
sed -i '/^  build:/a\    permissions:\n      id-token: write' "$temporary/workflows/release.yml"
reject release.yml 'build OIDC permission'
for scope in root build publish after-publish; do
  for value in write '"write"' "'write'" '>\n        write' '>- # folded\n        write' '>+\n        write' read; do
    fresh
    case "$scope" in
      root) anchor='^permissions:$'; indent='  ' ;;
      build) anchor='^  build:$'; indent='      ' ;;
      publish) anchor='^      id-token: write$'; indent='      ' ;;
      after-publish) anchor='^  verify-pypi:$'; indent='      ' ;;
    esac
    # Insert text only; never execute a workflow or any of its steps.
    permission="${indent}id-token: # permission comment\n\n${indent}  # value comment\n${indent}  ${value}"
    if [[ "$scope" != root && "$scope" != publish ]]; then
      permission="    permissions:\n${permission}"
    fi
    if [[ "$scope" == publish ]]; then
      sed -i "/$anchor/c\\${permission}" "$temporary/workflows/release.yml"
    else
      sed -i "/$anchor/a\\${permission}" "$temporary/workflows/release.yml"
    fi
    if [[ "$scope" == publish || "$value" == read ]]; then
      bash "$root/scripts/check-publish-guard.sh" "$temporary/workflows"
    else
      reject release.yml "$scope multiline OIDC permission ($value)"
      grep -F 'id-token: write outside publish' "$temporary/output" >/dev/null || fail 'missing scope diagnostic'
    fi
  done
done
fresh
sed -i 's/^permissions:$/permissions: {id-token: write}/; /^  contents: read$/d' "$temporary/workflows/release.yml"
reject release.yml 'inline top-level OIDC permission'
fresh
sed -i '/^  build:/a\    permissions: {"id-token": "write"}' "$temporary/workflows/release.yml"
reject release.yml 'quoted build OIDC permission'
fresh
sed -i 's/^  verify-pypi:$/  "verify-pypi":\n    permissions: {"id-token": "write"}/' "$temporary/workflows/release.yml"
reject release.yml 'quoted job after publish must not inherit its guard scope'
fresh
printf '\n"on": workflow_dispatch\n' >> "$temporary/workflows/release.yml"
reject release.yml 'quoted non-tag trigger'
printf 'publish guard synthetic checks passed\n'
