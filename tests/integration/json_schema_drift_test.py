import shutil
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel

from latinitas_cards.json_schemas import SCHEMAS, render_schema

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs/schemas"


def schema_version(model: type[BaseModel]) -> int:
    # A JSONL record union carries its v1 version in the header branch.
    return int(get_args(model.model_fields["version"].annotation)[0]) if "version" in model.model_fields else 1


@pytest.mark.parametrize("format_name", SCHEMAS)
def test_schema_matches_model(format_name: str) -> None:
    model = SCHEMAS[format_name]
    version = schema_version(model)
    assert (SCHEMA_DIR / f"{format_name}.v{version}.schema.json").read_text() == render_schema(model)


def test_schema_directory_has_no_unregistered_files() -> None:
    expected = {f"{name}.v{schema_version(model)}.schema.json" for name, model in SCHEMAS.items()}
    assert {path.name for path in SCHEMA_DIR.iterdir() if path.is_file()} == expected


def test_exporter_regenerates_complete_registry(tmp_path: Path) -> None:
    script = SCHEMA_DIR.parents[1] / "scripts/export-json-schemas.py"
    isolated_script = tmp_path / "scripts" / script.name
    isolated_script.parent.mkdir()
    shutil.copyfile(script, isolated_script)

    result = subprocess.run([sys.executable, str(isolated_script)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    expected_names = {f"{name}.v{schema_version(model)}.schema.json" for name, model in SCHEMAS.items()}
    exported = tmp_path / "docs/schemas"
    assert {path.name for path in exported.iterdir()} == expected_names
    for name in expected_names:
        assert (exported / name).read_bytes() == (SCHEMA_DIR / name).read_bytes(), name
