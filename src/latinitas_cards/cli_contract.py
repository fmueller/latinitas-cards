"""Closed diagnostic vocabulary and result contract for the v0.3 CLI groups."""

import io
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Self

import typer
from pydantic import BaseModel, ConfigDict, JsonValue, field_validator, model_validator

from .canonical import canonical_json
from .profile_setup import encode_unsafe_controls

type ExitCode = Literal[0, 1, 2, 3, 4, 5, 70]


class Code(StrEnum):
    EXISTS = "exists"
    BINDING_DRIFT = "binding_drift"
    BUSY = "busy"
    DENIED = "denied"
    ACTOR_REFUSED = "actor_refused"
    PROMPT_REQUIRED = "prompt_required"
    TIMESTAMP_REQUIRED = "timestamp_required"
    UNSUPPORTED_VERSION = "unsupported_version"
    STALE = "stale"
    ORPHANED = "orphaned"
    UNMATCHED = "unmatched"
    INTERNAL = "internal"
    UNREADABLE = "unreadable"
    INVALID_JSON = "invalid_json"
    INVALID_POLICY = "invalid_policy"
    MISSING_HEADER = "missing_header"
    UNSUPPORTED_FORMAT = "unsupported_format"
    ALIAS = "alias"
    USAGE_ERROR = "usage_error"
    UNKNOWN_ELEMENT = "unknown_element"
    INPUT_TOO_LARGE = "input_too_large"
    DEPTH_EXCEEDED = "depth_exceeded"
    ALREADY_APPLIED = "already_applied"
    INVALID_CORPUS = "invalid_corpus"
    INVALID_CORPUS_ID = "invalid_corpus_id"
    DUPLICATE_NATIVE_REF = "duplicate_native_ref"
    NON_NFC_TEXT = "non_nfc_text"
    NON_CANONICAL_TEXT = "non_canonical_text"
    HASH_MISMATCH = "hash_mismatch"
    NORM_MISMATCH = "norm_mismatch"
    PASSAGE_ID_MISMATCH = "passage_id_mismatch"
    INVALID_CITATION = "invalid_citation"
    FLOAT_VALUE = "float_value"
    INVALID_RECORD = "invalid_record"
    INVALID_MANIFEST = "invalid_manifest"
    INVALID_SIDECAR = "invalid_sidecar"
    UNKNOWN_PASSAGE = "unknown_passage"
    SEGMENTATION_DIFFERS = "segmentation_differs"
    INVALID_APPROVAL = "invalid_approval"
    RETIRED_REF = "retired_ref"
    REVIEW_REQUIRED = "review_required"
    INVALID_LINE = "invalid_line"
    DUPLICATE_SOURCE_REF = "duplicate_source_ref"
    INVALID_STRUCTURE = "invalid_structure"
    EMPTY_PASSAGE = "empty_passage"
    INVALID_XML = "invalid_xml"
    UNSAFE_XML = "unsafe_xml"


@dataclass(frozen=True)
class DiagnosticCode:
    exit_code: ExitCode
    meaning: str


DIAGNOSTIC_CODES: dict[Code, DiagnosticCode] = {
    Code.EXISTS: DiagnosticCode(3, "Output exists with different bytes"),
    Code.BINDING_DRIFT: DiagnosticCode(4, "Bound state differs from the plan base"),
    Code.BUSY: DiagnosticCode(4, "Writer lock is held"),
    Code.DENIED: DiagnosticCode(5, "Operation refused by policy"),
    Code.ACTOR_REFUSED: DiagnosticCode(5, "Receipt actor is not permitted"),
    Code.PROMPT_REQUIRED: DiagnosticCode(2, "Required prompt cannot be shown"),
    Code.TIMESTAMP_REQUIRED: DiagnosticCode(2, "Explicit timestamp is required"),
    Code.UNSUPPORTED_VERSION: DiagnosticCode(3, "Document version is unsupported"),
    Code.STALE: DiagnosticCode(1, "Passage hash differs from the anchor"),
    Code.ORPHANED: DiagnosticCode(1, "Anchor segmentation was superseded"),
    Code.UNMATCHED: DiagnosticCode(1, "Anchored passage is missing"),
    Code.INTERNAL: DiagnosticCode(70, "Unexpected internal failure"),
    Code.UNREADABLE: DiagnosticCode(3, "Filesystem input or output is inaccessible"),
    Code.INVALID_JSON: DiagnosticCode(3, "Input is not valid JSON"),
    Code.INVALID_POLICY: DiagnosticCode(3, "Policy document is invalid"),
    Code.MISSING_HEADER: DiagnosticCode(3, "Stream header is missing"),
    Code.UNSUPPORTED_FORMAT: DiagnosticCode(3, "Document format is unsupported"),
    Code.ALIAS: DiagnosticCode(3, "Output aliases a protected path"),
    Code.USAGE_ERROR: DiagnosticCode(2, "Callback arguments are incompatible"),
    Code.UNKNOWN_ELEMENT: DiagnosticCode(3, "Source contains unknown elements"),
    Code.INPUT_TOO_LARGE: DiagnosticCode(3, "XML input exceeds 64 MiB"),
    Code.DEPTH_EXCEEDED: DiagnosticCode(3, "XML element depth exceeds 64"),
    Code.ALREADY_APPLIED: DiagnosticCode(0, "Operation was already applied"),
    Code.INVALID_CORPUS: DiagnosticCode(3, "Consumer rejected an invalid corpus"),
    Code.INVALID_CORPUS_ID: DiagnosticCode(1, "Corpus ID is malformed"),
    Code.DUPLICATE_NATIVE_REF: DiagnosticCode(1, "Native reference is repeated"),
    Code.NON_NFC_TEXT: DiagnosticCode(1, "Passage text is not NFC"),
    Code.NON_CANONICAL_TEXT: DiagnosticCode(1, "Passage whitespace is not canonical"),
    Code.HASH_MISMATCH: DiagnosticCode(1, "Passage hash differs from canonical text"),
    Code.NORM_MISMATCH: DiagnosticCode(1, "Passage norm differs from latin-fold/1"),
    Code.PASSAGE_ID_MISMATCH: DiagnosticCode(1, "Passage ID does not match corpus and native reference"),
    Code.INVALID_CITATION: DiagnosticCode(1, "Citation is invalid"),
    Code.FLOAT_VALUE: DiagnosticCode(1, "JSON floats are forbidden"),
    Code.INVALID_RECORD: DiagnosticCode(1, "Corpus record schema or order is invalid"),
    Code.INVALID_MANIFEST: DiagnosticCode(3, "Corpus manifest is invalid"),
    Code.INVALID_SIDECAR: DiagnosticCode(3, "Annotation sidecar is invalid"),
    Code.UNKNOWN_PASSAGE: DiagnosticCode(3, "Requested passage is absent from the corpus"),
    Code.SEGMENTATION_DIFFERS: DiagnosticCode(0, "Current segmentation differs from recomputed value"),
    Code.INVALID_APPROVAL: DiagnosticCode(3, "Approval is invalid or contradictory"),
    Code.RETIRED_REF: DiagnosticCode(3, "Approval names a retired reference"),
    Code.REVIEW_REQUIRED: DiagnosticCode(1, "Identity reconciliation requires review"),
    Code.INVALID_LINE: DiagnosticCode(3, "Text line does not match citation template"),
    Code.DUPLICATE_SOURCE_REF: DiagnosticCode(3, "Source label is repeated"),
    Code.INVALID_STRUCTURE: DiagnosticCode(3, "USFX structure is invalid"),
    Code.EMPTY_PASSAGE: DiagnosticCode(3, "Source passage has no canonical text"),
    Code.INVALID_XML: DiagnosticCode(3, "Source is not well-formed XML"),
    Code.UNSAFE_XML: DiagnosticCode(3, "Source XML contains a forbidden declaration"),
}


class Diagnostic(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    code: Code
    severity: Literal["error", "warning"] = "error"
    message: str
    item: int | None = None
    path: str | None = None
    ref: str | None = None


class CliResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    format: Literal["latinitas-cli-result"] = "latinitas-cli-result"
    version: Literal[1] = 1
    command: str
    ok: bool
    exit_code: ExitCode
    diagnostics: list[Diagnostic]
    result: JsonValue

    @field_validator("version", "exit_code", mode="before")
    @classmethod
    def strict_integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("must be an integer")
        return value

    @model_validator(mode="after")
    def consistent_ok(self) -> Self:
        if self.ok != (self.exit_code == 0):
            raise ValueError("ok must equal exit_code == 0")
        return self


def exit_code_for(diagnostics: Sequence[Diagnostic]) -> ExitCode:
    classes = {DIAGNOSTIC_CODES[d.code].exit_code for d in diagnostics if d.severity == "error"}
    precedence: tuple[ExitCode, ...] = (3, 4, 5, 1)
    for exit_code in precedence:
        if exit_code in classes:
            return exit_code
    return 0


class ContractError(Exception):
    def __init__(self, code: Code, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.diagnostic = Diagnostic(code=code, message=message, path=path)
        self.exit_code = DIAGNOSTIC_CODES[code].exit_code


def run_contract(
    command: str,
    json_output: bool,
    body: Callable[[], tuple[JsonValue, list[Diagnostic]]],
    *,
    render_text: Callable[[CliResult], None] | None = None,
) -> None:
    """Run a callback that returns result/diagnostics; only this boundary emits output."""
    try:
        with redirect_stdout(io.StringIO()):
            result, diagnostics = body()
        exit_code = exit_code_for(diagnostics)
        envelope = CliResult(
            command=command, ok=exit_code == 0, exit_code=exit_code, diagnostics=diagnostics, result=result
        )
        rendered = canonical_json(envelope.model_dump(mode="json"))
    except ContractError as error:
        envelope = CliResult(
            command=command,
            ok=error.exit_code == 0,
            exit_code=error.exit_code,
            diagnostics=[error.diagnostic],
            result=None,
        )
        rendered = canonical_json(envelope.model_dump(mode="json"))
    except Exception:
        envelope = CliResult(
            command=command,
            ok=False,
            exit_code=70,
            diagnostics=[Diagnostic(code=Code.INTERNAL, message="Unexpected internal failure")],
            result=None,
        )
        rendered = canonical_json(envelope.model_dump(mode="json"))
    if json_output:
        typer.echo(rendered)
    elif render_text is not None:
        render_text(envelope)
    else:
        if envelope.result is not None:
            typer.echo(encode_unsafe_controls(str(envelope.result), preserve_line_breaks=False))
        for diagnostic in envelope.diagnostics:
            typer.echo(
                encode_unsafe_controls(
                    f"{diagnostic.severity}: {diagnostic.code}: {diagnostic.message}", preserve_line_breaks=False
                ),
                err=True,
            )
    raise typer.Exit(envelope.exit_code)
