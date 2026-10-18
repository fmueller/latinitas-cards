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
for representation in direct continued folded '|-' '|2-' '|-2' block block-bare flow flow-bare flow-multiline; do
  for name in pypi staging; do
    case "$representation" in
      direct) environment="    environment: $name" ;;
      continued) environment="    environment:\n      $name" ;;
      folded) environment="    environment: >-\n      $name" ;;
      '|'*) environment="    environment: $representation\n      $name" ;;
      block) environment="    environment: # deployment target\n      url: https://example.invalid\n      'name': '$name'" ;;
      block-bare) environment="    environment:\n      name: $name" ;;
      flow) environment="    environment: {url: https://example.invalid, \"name\": \"$name\"}" ;;
      flow-bare) environment="    environment: {name: $name}" ;;
      flow-multiline) environment="    environment: {url: https://example.invalid,\n      name: $name}" ;;
    esac
    for scope in nonrelease publish; do
      fresh
      if [[ "$scope" == nonrelease ]]; then
        printf 'name: Inert environment fixture\non: push\njobs:\n  probe:\n    runs-on: ubuntu-latest\n%b\n    steps:\n      - run: echo inert\n' "$environment" > "$temporary/workflows/extra.yaml"
      else
        sed -i '/^    environment: pypi$/d' "$temporary/workflows/release.yml"
        sed -i "/^  publish:$/a\\$environment" "$temporary/workflows/release.yml"
      fi
      if [[ "$scope" == nonrelease && "$name" == pypi ]]; then
        reject extra.yaml "$representation pypi environment"
        grep -F 'publishing capability outside release.yml' "$temporary/output" >/dev/null || fail 'missing capability diagnostic'
      else
        bash "$root/scripts/check-publish-guard.sh" "$temporary/workflows"
      fi
    done
  done
done
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
for scalar in '|' '|-' '|+' '|2' '|2-' '|-2' '|2+' '|+2'; do
  for scope in root build publish after-publish nonrelease; do
    for value in write read; do
      fresh
      file=release.yml
      case "$scope" in
        root) anchor='^permissions:$'; indent='  ' ;;
        build) anchor='^  build:$'; indent='      ' ;;
        publish) anchor='^      id-token: write$'; indent='      ' ;;
        after-publish) anchor='^  verify-pypi:$'; indent='      ' ;;
        nonrelease) file=extra.yaml; indent='      ' ;;
      esac
      permission="${indent}id-token: $scalar # literal permission\n${indent}  $value"
      if [[ "$scope" == nonrelease ]]; then
        printf 'jobs:\n  probe:\n    permissions:\n%b\n    steps:\n      - run: echo harmless\n' "$permission" > "$temporary/workflows/$file"
      elif [[ "$scope" == publish ]]; then
        sed -i "/$anchor/c\\${permission}" "$temporary/workflows/$file"
      else
        [[ "$scope" == root ]] || permission="    permissions:\n${permission}"
        sed -i "/$anchor/a\\${permission}" "$temporary/workflows/$file"
      fi
      if [[ "$scope" == publish || "$value" == read ]]; then
        bash "$root/scripts/check-publish-guard.sh" "$temporary/workflows"
      else
        reject "$file" "$scope literal OIDC permission ($scalar)"
        diagnostic='id-token: write outside publish'
        [[ "$scope" != nonrelease ]] || diagnostic='publishing capability outside release.yml'
        grep -F "$diagnostic" "$temporary/output" >/dev/null || fail 'missing literal scope diagnostic'
      fi
    done
  done
done
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
