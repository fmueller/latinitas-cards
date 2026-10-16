"""Validate sidecar-write policies without mutating any input."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from ..cli_contract import Diagnostic, run_contract
from ..policy import load_policy

app = typer.Typer(help="Validate versioned sidecar-write policies.")


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(help="Versioned policy JSON file.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Check a policy's schema and authorization restrictions."""

    def body() -> tuple[JsonValue, list[Diagnostic]]:
        policy = load_policy(file)
        return {"rules": len(policy.rules), "receipt_actor": policy.receipt_actor}, []

    run_contract("policy validate", json_output, body)
