"""Exercise CI shard configuration through its command-line interface."""

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/configure-mutation-shard.py"


@pytest.mark.parametrize("shards", [4, 8])
def test_shards_partition_all_source_files_without_changing_other_settings(tmp_path: Path, shards: int) -> None:
    paths = [
        "large.py",
        "nested/medium.py",
        "small.py",
        "extra.py",
        "fifth.py",
        "sixth.py",
        "seventh.py",
        "eighth.py",
        "__init__.py",
        "nested/__init__.py",
    ]
    groups: list[list[str]] = []
    for shard in range(shards):
        root = tmp_path / str(shard)
        (root / "scripts").mkdir(parents=True)
        script = root / "scripts" / SCRIPT.name
        shutil.copyfile(SCRIPT, script)
        source = root / "src/latinitas_cards"
        for index, name in enumerate(paths):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#" * (200 - index * 10))
        (root / "pyproject.toml").write_text(
            '[project]\nname = "example"\n[tool.mutmut]\n'
            'source_paths = ["src/latinitas_cards/"]\nmutate_only_covered_lines = true\n'
            'also_copy = ["docs/"]\n[tool.ruff]\nline-length = 120\n'
        )
        subprocess.run([sys.executable, str(script), str(shard), str(shards)], check=True, capture_output=True)
        config = tomllib.loads((root / "pyproject.toml").read_text())
        scope = json.loads((root / "mutation-scope.json").read_text())
        assigned = config["tool"]["mutmut"].pop("only_mutate")
        assert assigned == [f"src/latinitas_cards/{name}" for name in paths[shard::shards]]
        assert scope == {"shard": shard, "shards": shards, "files": assigned}
        assert config["tool"]["mutmut"] == {
            "source_paths": ["src/latinitas_cards/"],
            "mutate_only_covered_lines": True,
            "also_copy": ["docs/"],
        }
        assert config["project"] == {"name": "example"}
        assert config["tool"]["ruff"] == {"line-length": 120}
        groups.append(assigned)
    flattened = [path for group in groups for path in group]
    assert len(flattened) == len(set(flattened)) == len(paths)


@pytest.mark.parametrize("shard,shards", [(0, 0), (-1, 4), (4, 4), (1, 4)])
def test_invalid_or_empty_shard_does_not_rewrite_config(tmp_path: Path, shard: int, shards: int) -> None:
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / SCRIPT.name
    shutil.copyfile(SCRIPT, script)
    source = tmp_path / "src/latinitas_cards"
    source.mkdir(parents=True)
    (source / "only.py").write_text("pass\n")
    config = tmp_path / "pyproject.toml"
    original = '[tool.mutmut]\nsource_paths = ["src/latinitas_cards/"]\n'
    config.write_text(original)
    result = subprocess.run([sys.executable, str(script), str(shard), str(shards)], capture_output=True)
    assert result.returncode != 0
    assert config.read_text() == original
    assert not (tmp_path / "mutation-scope.json").exists()
