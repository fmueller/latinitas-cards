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
for trigger in workflow_dispatch schedule release pull_request branches; do
  fresh
  sed -i "/^on:/a\  $trigger:" "$temporary/workflows/release.yml"
  reject release.yml "$trigger trigger"
done
fresh
sed -i '/^  build:/a\    permissions:\n      id-token: write' "$temporary/workflows/release.yml"
reject release.yml 'build OIDC permission'
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
