"""Explicitly regenerate committed schemas from the complete registry."""

from pathlib import Path
from typing import get_args

from latinitas_cards.json_schemas import SCHEMAS, render_schema


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "docs/schemas"
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        # A JSONL record union carries its v1 version in the header branch.
        version = int(get_args(model.model_fields["version"].annotation)[0]) if "version" in model.model_fields else 1
        (directory / f"{name}.v{version}.schema.json").write_text(render_schema(model), encoding="utf-8")


if __name__ == "__main__":
    main()
