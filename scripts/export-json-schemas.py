"""Explicitly regenerate committed schemas; tests never invoke this script."""

from pathlib import Path

from latinitas_cards.json_schemas import SCHEMAS, render_schema


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "docs/schemas"
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        version = model.model_fields["version"].default
        (directory / f"{name}.v{version}.schema.json").write_text(render_schema(model), encoding="utf-8")


if __name__ == "__main__":
    main()
