from pathlib import Path

import pytest

from latinitas_cards.json_schemas import SCHEMAS, render_schema

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs/schemas"


@pytest.mark.parametrize("format_name", SCHEMAS)
def test_schema_matches_model(format_name: str) -> None:
    model = SCHEMAS[format_name]
    version = model.model_fields["version"].default
    assert (SCHEMA_DIR / f"{format_name}.v{version}.schema.json").read_text() == render_schema(model)


def test_schema_directory_has_no_unregistered_files() -> None:
    expected = {f"{name}.v{model.model_fields['version'].default}.schema.json" for name, model in SCHEMAS.items()}
    assert {path.name for path in SCHEMA_DIR.iterdir() if path.is_file()} == expected
