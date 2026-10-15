"""Normalized corpus inspection commands."""

from typing import Annotated

import typer
from pydantic import JsonValue

from ..cli_contract import CliResult, Diagnostic, run_contract
from ..corpus_validation import validate_corpus
from ..profile_setup import encode_unsafe_controls

app = typer.Typer(help="Inspect normalized corpora without modifying them.")


def _render_text(result: CliResult) -> None:
    errors = sum(d.severity == "error" for d in result.diagnostics)
    warnings = sum(d.severity == "warning" for d in result.diagnostics)
    typer.echo(f"{errors} errors, {warnings} warnings")
    for diagnostic in result.diagnostics:
        typer.echo(
            encode_unsafe_controls(
                f"{diagnostic.severity} {diagnostic.code} {diagnostic.ref or '-'}: {diagnostic.message}",
                preserve_line_breaks=False,
            ),
            err=True,
        )


@app.command()
def validate(
    corpus: Annotated[str, typer.Argument(help="Normalized corpus JSONL file.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit the versioned result envelope.")] = False,
) -> None:
    """Validate record schemas, ordering, identity and canonical passage content."""

    def body() -> tuple[JsonValue, list[Diagnostic]]:
        counts, diagnostics = validate_corpus(corpus)
        return {"counts": dict(counts)}, diagnostics

    run_contract("corpus validate", json_output, body, render_text=_render_text)
