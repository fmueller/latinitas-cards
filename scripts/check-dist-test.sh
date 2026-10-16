#!/usr/bin/env bash
# Offline synthetic artifact tests: no build backend or package installation.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -I - "$root" <<'PY'
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

root = Path(sys.argv[1])
project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
version = project["version"]
metadata = (
    f"Metadata-Version: 2.4\nName: latinitas-cards\nVersion: {version}\n"
    "License-Expression: GPL-3.0-or-later\n"
    "License-File: LICENSE\nLicense-File: THIRD_PARTY_NOTICES.md\n"
    "Requires-Python: <3.15,>=3.13\nProvides-Extra: annotate\nProvides-Extra: annotate-gpu\n\n"
).encode()
entry = b"[console_scripts]\nlatinitas-cards = latinitas_cards.cli:app\n"
runtime = {
    str(path.relative_to(root / "src")): b"synthetic runtime resource"
    for path in (root / "src/latinitas_cards").rglob("*")
    if path.is_file() and "__pycache__" not in path.parts
}
info = f"latinitas_cards-{version}.dist-info"
wheel = runtime | {
    f"{info}/METADATA": metadata,
    f"{info}/entry_points.txt": entry,
    f"{info}/licenses/LICENSE": b"license",
    f"{info}/licenses/THIRD_PARTY_NOTICES.md": b"notices",
}
sdist = {f"src/{name}": data for name, data in runtime.items()} | {
    "PKG-INFO": metadata,
    "src/latinitas_cards.egg-info/entry_points.txt": entry,
    "LICENSE": b"license",
    "THIRD_PARTY_NOTICES.md": b"notices",
}


def check(label, target=None, change=None, expected=None):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for kind, original in (("wheel", wheel), ("rebuilt", wheel), ("sdist", sdist)):
            members = original.copy()
            if kind == target:
                change(members)
            if kind == "sdist":
                with tarfile.open(directory / f"latinitas_cards-{version}.tar.gz", "w:gz") as archive:
                    for name, data in members.items():
                        member = tarfile.TarInfo(f"latinitas_cards-{version}/{name}")
                        if name.endswith("/"):
                            member.type = tarfile.DIRTYPE
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))
            else:
                destination = directory if kind == "wheel" else directory / "rebuilt"
                destination.mkdir(exist_ok=True)
                with zipfile.ZipFile(destination / f"latinitas_cards-{version}-py3-none-any.whl", "w") as archive:
                    for name, data in members.items():
                        archive.writestr(name, data)
        result = subprocess.run(
            ["bash", str(root / "scripts/check-dist.sh"), "--inspect", str(directory)],
            capture_output=True, text=True,
        )
        output = result.stdout + result.stderr
        if expected is None:
            assert result.returncode == 0, (label, output)
        else:
            assert result.returncode != 0 and expected in output, (label, expected, output)
        print(f"PASS: {label}")


check("conforming wheel, rebuilt wheel and sdist")
for target in ("wheel", "rebuilt", "sdist"):
    key = "PKG-INFO" if target == "sdist" else f"{info}/METADATA"
    for field, before, after in (
        ("Version", f"Version: {version}\n", "Version: 99.0.0\n"),
        ("License-Expression", "License-Expression: GPL-3.0-or-later\n", ""),
        ("License-File", "License-File: THIRD_PARTY_NOTICES.md\n", ""),
        ("Name", "Name: latinitas-cards\n", "Name: other\n"),
        ("Requires-Python", "Requires-Python: <3.15,>=3.13\n", "Requires-Python: >=3.14\n"),
        ("Provides-Extra", "Provides-Extra: annotate-gpu\n", ""),
    ):
        check(f"{target}: defective {field}", target,
              lambda members: members.update({key: metadata.replace(before.encode(), after.encode())}), field)
    entry_key = "src/latinitas_cards.egg-info/entry_points.txt" if target == "sdist" else f"{info}/entry_points.txt"
    check(f"{target}: missing entry point", target, lambda members: members.pop(entry_key), "entry point")
    for forbidden in ("tests/leak.py", "data/lat-clementine.usfx.xml", "planning/task.md", ".amp/leak"):
        check(f"{target}: forbidden {forbidden}", target,
              lambda members: members.update({forbidden: b"leak"}), forbidden)
    for forbidden in ("tests/", "data/", "planning/", ".amp/"):
        check(f"{target}: forbidden directory {forbidden}", target,
              lambda members: members.update({forbidden: b""}), forbidden.rstrip("/"))
    notice = "THIRD_PARTY_NOTICES.md" if target == "sdist" else f"{info}/licenses/THIRD_PARTY_NOTICES.md"
    check(f"{target}: absent notices file", target, lambda members: members.pop(notice), "THIRD_PARTY_NOTICES.md")
    resource = next(iter(runtime))
    resource_key = f"src/{resource}" if target == "sdist" else resource
    check(f"{target}: missing runtime file", target, lambda members: members.pop(resource_key), resource)
check("unequal wheel member lists", "rebuilt", lambda members: members.update({"extra.txt": b"extra"}), "member lists")
check("unequal wheel directory lists", "rebuilt", lambda members: members.update({"extra/": b""}), "member lists")
print("distribution inspection checks passed (54 cases)")
PY
