#!/usr/bin/env bash
# Exercise both distributions, never an editable checkout or optional NLP extras.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# != 1 ]]; then
  echo 'usage: check-install.sh PYTHON' >&2
  exit 2
fi
python="$1"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
version="$(python3 -I -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["version"])' "$root/pyproject.toml")"
shopt -s nullglob
wheels=("$root/dist/latinitas_cards-$version-"*.whl)
sdists=("$root/dist/latinitas_cards-$version.tar.gz")
if [[ ${#wheels[@]} != 1 || ${#sdists[@]} != 1 || ! -f ${sdists[0]} ]]; then
  echo 'expected one wheel and one sdist in dist/; run check-dist.sh first' >&2
  exit 1
fi
# All installation and execution happens outside the source checkout.
cd "$temporary"
for artifact in "${wheels[0]}" "${sdists[0]}"; do
  environment="$temporary/$(basename "$artifact").venv"
  uv venv --python "$python" "$environment"
  uv pip install --no-config --python "$environment/bin/python" "$artifact"
  "$environment/bin/python" -I - "$artifact" <<'PY'
import hashlib
import importlib.metadata
from pathlib import Path
import sys

for name in ("cltk", "stanza", "torch"):
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        continue
    sys.exit(f"optional NLP distribution installed: {name}")
artifact = Path(sys.argv[1])
print(f"installing {artifact.name}: SHA256 {hashlib.sha256(artifact.read_bytes()).hexdigest()}")
print(f"interpreter: {sys.version}; optional NLP distributions absent")
PY
  cli="$environment/bin/latinitas-cards"
  env -u PYTHONPATH -u PYTHONHOME "$cli" --help > "$temporary/help.txt"
  actual="$(env -u PYTHONPATH -u PYTHONHOME "$cli" --version)"
  [[ "$actual" == "latinitas-cards $version" ]] || { echo "unexpected version: $actual" >&2; exit 1; }
  output="$environment/corpus.jsonl"
  env -u PYTHONPATH -u PYTHONHOME "$cli" corpus normalize \
    "$root/tests/fixtures/corpus/usfx/gen-1-1-5.usfx.xml" --adapter usfx \
    --corpus-id packaging-smoke --output "$output" --json > "$temporary/normalize.json"
  env -u PYTHONPATH -u PYTHONHOME "$cli" corpus validate "$output" --json > "$temporary/validate.json"
  "$environment/bin/python" -I - "$temporary/validate.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    result = json.load(stream)
if result.get("ok") is not True:
    sys.exit(f"corpus validation failed: {result}")
PY
  echo "PASS: $(basename "$artifact"): --help, $actual, corpus normalize, corpus validate (ok=true)"
done
echo 'distribution installation checks passed'
