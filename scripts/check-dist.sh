#!/usr/bin/env bash
# Build in isolation, rebuild from the sdist, and inspect without importing the app.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# == 0 ]]; then
  temporary="$(mktemp -d)"
  trap 'rm -rf "$temporary"' EXIT
  uv build "$root" --out-dir "$temporary/dist"
  python3 -I - "$temporary" <<'PY'
from pathlib import Path
import sys
import tarfile

directory = Path(sys.argv[1])
with tarfile.open(next((directory / "dist").glob("*.tar.gz"))) as archive:
    archive.extractall(directory / "source", filter="data")
PY
  uv build --wheel "$temporary"/source/* --out-dir "$temporary/dist/rebuilt"
  bash "$root/scripts/check-dist.sh" --inspect "$temporary/dist"
  # Only inspected artifacts replace this version's outputs; do not remove other files.
  mkdir -p "$root/dist"
  cp "$temporary"/dist/*.whl "$temporary"/dist/*.tar.gz "$root/dist/"
elif [[ $# == 2 && $1 == --inspect ]]; then
  python3 -I - "$root" "$2" <<'PY'
import configparser
from email.parser import BytesParser
import hashlib
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tomllib
import zipfile

root, directory = map(Path, sys.argv[1:])
project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
runtime = {
    str(path.relative_to(root / "src"))
    for path in (root / "src/latinitas_cards").rglob("*")
    if path.is_file() and "__pycache__" not in path.parts
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(path, sdist=False):
    if sdist:
        with tarfile.open(path) as archive:
            member_names = archive.getnames()
            files = {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}
        prefixes = {name.split("/")[0] for name in files}
        require(len(prefixes) == 1, f"{path.name}: sdist root")
        files = {name.split("/", 1)[1]: data for name, data in files.items()}
        metadata_key = "PKG-INFO"
        entry_key = "src/latinitas_cards.egg-info/entry_points.txt"
        license_prefix = ""
    else:
        with zipfile.ZipFile(path) as archive:
            member_names = archive.namelist()
            files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
        metadata_keys = [name for name in files if name.endswith(".dist-info/METADATA")]
        require(len(metadata_keys) == 1, f"{path.name}: METADATA")
        metadata_key = metadata_keys[0]
        info = metadata_key.rsplit("/", 1)[0]
        entry_key = f"{info}/entry_points.txt"
        license_prefix = f"{info}/licenses/"
    for name in member_names:
        require(not {"tests", "data", "planning", ".amp"}.intersection(PurePosixPath(name).parts),
                f"{path.name}: forbidden member {name}")
    require(metadata_key in files, f"{path.name}: missing {metadata_key}")
    metadata = BytesParser().parsebytes(files[metadata_key])
    for field, expected in (
        ("Name", project["name"]),
        ("Version", project["version"]),
        ("License-Expression", "GPL-3.0-or-later"),
    ):
        require(metadata.get_all(field) == [expected], f"{path.name}: {field} must equal {expected}")
    require(len(metadata.get_all("Requires-Python", [])) == 1 and
            set(metadata["Requires-Python"].replace(" ", "").split(",")) == set(project["requires-python"].split(",")),
            f"{path.name}: Requires-Python must equal {project['requires-python']}")
    require(set(metadata.get_all("License-File", [])) == {"LICENSE", "THIRD_PARTY_NOTICES.md"},
            f"{path.name}: License-File must include LICENSE and THIRD_PARTY_NOTICES.md")
    require(set(metadata.get_all("Provides-Extra", [])) == {"annotate", "annotate-gpu"},
            f"{path.name}: Provides-Extra must include annotate and annotate-gpu")
    for license_file in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        require(license_prefix + license_file in files, f"{path.name}: missing {license_file}")
    require(entry_key in files, f"{path.name}: missing entry point")
    entry = configparser.ConfigParser()
    entry.read_string(files[entry_key].decode())
    require(entry.get("console_scripts", "latinitas-cards", fallback=None) == "latinitas_cards.cli:app",
            f"{path.name}: invalid entry point")
    for resource in sorted(runtime):
        require(("src/" if sdist else "") + resource in files, f"{path.name}: missing runtime file {resource}")
    print(f"inspected {path.name}: {len(files)} files; SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return sorted(member_names)


try:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    rebuilt = sorted((directory / "rebuilt").glob("*.whl"))
    require(len(wheels) == len(sdists) == len(rebuilt) == 1, "expected one wheel, one sdist and one rebuilt wheel")
    first = inspect(wheels[0])
    second = inspect(rebuilt[0])
    inspect(sdists[0], sdist=True)
    require(first == second, "wheel member lists differ")
except (ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile, configparser.Error) as error:
    sys.exit(f"distribution inspection failed: {error}")
print("distribution inspection passed")
PY
else
  echo 'usage: check-dist.sh [--inspect DIR]' >&2
  exit 2
fi
