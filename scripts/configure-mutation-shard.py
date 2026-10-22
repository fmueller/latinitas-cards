"""Restrict a disposable CI checkout to one disjoint mutation source shard."""

import argparse
import json
import tomllib
from pathlib import Path


def configure_shard(root: Path, shard: int, shards: int) -> None:
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("shard must be between zero and shards minus one")
    project = root / "pyproject.toml"
    text = project.read_text(encoding="utf-8")
    settings = tomllib.loads(text)["tool"]["mutmut"]
    if "only_mutate" in settings:
        raise ValueError("only_mutate is already configured")
    files = sorted(
        (root / "src/latinitas_cards").rglob("*.py"),
        key=lambda path: (-path.stat().st_size, path.as_posix()),
    )
    # Deal larger files first so the largest modules land in different jobs.
    selected = [path.relative_to(root).as_posix() for path in files[shard::shards]]
    if not selected:
        raise ValueError("shard has no source files")
    marker = "[tool.mutmut]\n"
    if text.count(marker) != 1:
        raise ValueError("expected one tool.mutmut section")
    project.write_text(text.replace(marker, marker + f"only_mutate = {json.dumps(selected)}\n"), encoding="utf-8")
    scope = {"shard": shard, "shards": shards, "files": selected}
    (root / "mutation-scope.json").write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scope, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=int)
    parser.add_argument("shards", type=int)
    args = parser.parse_args()
    configure_shard(Path(__file__).resolve().parents[1], args.shard, args.shards)
