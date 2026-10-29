"""Normalized corpus inspection and normalization commands."""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue, ValidationError

from .. import atomic_io
from ..annotation_sidecar import Annotation, Retraction, read_sidecar
from ..annotation_state import annotation_states, inspect_annotations
from ..canonical import digest_bytes
from ..cli_contract import CliResult, Code, ContractError, Diagnostic, run_contract
from ..corpus_claims import join_corpus_decisions
from ..corpus_io import CorpusFinding, read_corpus, write_corpus
from ..corpus_manifest import Approvals, materialize, read_optional
from ..corpus_model import CorpusDescriptor, CorpusPassage, CorpusRecord, encode_ref_component, passage_hash
from ..corpus_text import build_corpus, parse_template, read_lines
from ..corpus_validation import validate_corpus, validate_records
from ..effective_config import build_report
from ..profile_setup import encode_unsafe_controls
from ..tei_adapter import WORK_ID, build_tei
from ..usfx_adapter import build_usfx

app = typer.Typer(help="Normalize and inspect versioned corpora.")


@app.command("config")
def corpus_config(
    policy: Annotated[Path | None, typer.Option(help="Report this write policy.")] = None,
    corpus: Annotated[Path | None, typer.Option(help="Report this normalized corpus.")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Report effective configuration and value origins without writing files."""
    run_contract("corpus config", json_output, lambda: build_report(policy, corpus))


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
    annotations: Annotated[Path | None, typer.Option(help="Check annotation anchors.")] = None,
    claims: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit the versioned result envelope.")] = False,
) -> None:
    """Validate record schemas, ordering, identity and canonical passage content."""

    def body() -> tuple[JsonValue, list[Diagnostic]]:
        if claims is not None and annotations is None:
            raise ContractError(Code.USAGE_ERROR, "--claims requires --annotations")
        if annotations is not None:
            result, diagnostics = inspect_annotations(Path(corpus), annotations, check=True)
            if claims is not None and not any(d.path == corpus and d.severity == "error" for d in diagnostics):
                _, findings = join_corpus_decisions(Path(corpus), annotations, claims)
                diagnostics.extend(findings)
            return {"counts": result["counts"]}, diagnostics
        counts, diagnostics = validate_corpus(corpus)
        return {"counts": dict(counts)}, diagnostics

    run_contract("corpus validate", json_output, body, render_text=_render_text)


@app.command()
def show(
    corpus: Annotated[Path, typer.Argument(help="Normalized corpus JSONL.")],
    passage: Annotated[list[str] | None, typer.Option("--passage", help="Select IDs in corpus order.")] = None,
    limit: Annotated[int, typer.Option(min=1, max=1000)] = 20,
    annotations: Annotated[Path | None, typer.Option()] = None,
    claims: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show a bounded read-only view with only current, live annotations."""

    def body() -> tuple[JsonValue, list[Diagnostic]]:
        if claims is not None and annotations is None:
            raise ContractError(Code.USAGE_ERROR, "--claims requires --annotations")
        hashes: dict[str, str] = {}
        selected: list[CorpusPassage] = []
        requested = set(passage or [])
        matched = 0

        def records() -> Iterable[CorpusRecord | CorpusFinding]:
            nonlocal matched
            for row in read_corpus(corpus):
                if isinstance(row, CorpusPassage):
                    hashes[row.passage_id] = row.passage_hash
                    if not requested or row.passage_id in requested:
                        matched += 1
                        if len(selected) < limit:
                            selected.append(row)
                yield row

        _, diagnostics = validate_records(records(), path=str(corpus))
        if diagnostics:
            return None, [
                Diagnostic(code=Code.INVALID_CORPUS, message="Cannot show an invalid corpus", path=str(corpus)),
                *diagnostics,
            ]
        rows = (
            []
            if annotations is None
            else [row for row in read_sidecar(annotations) if isinstance(row, Annotation | Retraction)]
        )
        missing = requested - hashes.keys()
        if missing:
            return None, [
                Diagnostic(code=Code.UNKNOWN_PASSAGE, message=f"Unknown passage: {ref}", path=str(corpus), ref=ref)
                for ref in sorted(missing)
            ]
        states = annotation_states(hashes, rows)
        reviews: dict[str, str] = {}
        decision_diagnostics: list[Diagnostic] = []
        if claims is not None and annotations is not None:
            reviews, findings = join_corpus_decisions(corpus, annotations, claims)
            decision_diagnostics = [
                d.model_copy(update={"severity": "warning"}) for d in findings if d.code == Code.INVALID_INPUT
            ]
        by_passage: dict[str, list[JsonValue]] = {}
        for row in rows:
            state = states[row.annotation_id]
            if (
                isinstance(row, Annotation)
                and state.state == "current"
                and not state.superseded
                and not state.retracted
            ):
                by_passage.setdefault(row.anchor.passage_id, []).append(
                    {
                        "annotation_id": row.annotation_id,
                        "layer": row.layer,
                        "value": row.value,
                        "producer": row.producer.model_dump(mode="json"),
                        **({"review_status": reviews.get(row.annotation_id, "open")} if claims is not None else {}),
                    }
                )
        return {
            "passages": [
                {
                    "passage_id": row.passage_id,
                    "citation": row.citation.model_dump(mode="json"),
                    "text": row.text,
                    "annotations": by_passage.get(row.passage_id, []),
                }
                for row in selected
            ],
            "truncated": matched > limit,
        }, decision_diagnostics

    run_contract("corpus show", json_output, body)


@dataclass(frozen=True)
class NormalizeRequest:
    """`corpus normalize` options after the adapter-independent checks."""

    source: Path
    corpus_id: str
    output: Path
    language: str
    descriptor: Path | None
    manifest: Path | None
    template: tuple[str, list[str]] | None
    approve_reuse: list[str]
    approve_allocation: list[int]
    approve_removal: list[str]
    allow_unknown: bool
    work: str | None


def _normalize_xml(
    request: NormalizeRequest,
    name: str,
    build: Callable[[CorpusDescriptor | None, list[Diagnostic]], list[CorpusRecord]],
) -> tuple[JsonValue, list[Diagnostic]]:
    """Source-native XML adapters write only the corpus; they never use templates, manifests or approvals."""
    if (
        request.template is not None
        or request.manifest is not None
        or request.approve_reuse
        or request.approve_allocation
        or request.approve_removal
    ):
        raise ContractError(Code.USAGE_ERROR, f"{name} cannot use templates, manifests or approvals")
    source, descriptor = request.source, request.descriptor
    try:
        parsed_descriptor = CorpusDescriptor.model_validate_json(descriptor.read_bytes()) if descriptor else None
        diagnostics: list[Diagnostic] = []
        records = build(parsed_descriptor, diagnostics)
        write_corpus(request.output, records, [source, *([descriptor] if descriptor else [])])
    except ValidationError as error:
        raise ContractError(Code.INVALID_CORPUS, f"Invalid descriptor: {error}") from error
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error)) from error
    return {"review_items": []}, diagnostics


def _normalize_usfx(request: NormalizeRequest) -> tuple[JsonValue, list[Diagnostic]]:
    if request.work is not None:
        raise ContractError(Code.USAGE_ERROR, "--work requires the TEI adapter")
    return _normalize_xml(
        request,
        "USFX",
        lambda descriptor, diagnostics: build_usfx(
            request.source,
            request.corpus_id,
            request.language,
            descriptor,
            allow_unknown=request.allow_unknown,
            diagnostics=diagnostics,
        ),
    )


def _normalize_tei(request: NormalizeRequest) -> tuple[JsonValue, list[Diagnostic]]:
    if request.allow_unknown:
        raise ContractError(Code.USAGE_ERROR, "--allow-unknown is not yet supported by the TEI adapter")
    if request.work is not None and not WORK_ID.fullmatch(request.work):
        raise ContractError(Code.USAGE_ERROR, "Malformed --work ID")
    return _normalize_xml(
        request,
        "TEI",
        lambda descriptor, diagnostics: build_tei(
            request.source,
            request.corpus_id,
            request.language,
            descriptor,
            work=request.work,
            diagnostics=diagnostics,
        ),
    )


def _normalize_text(request: NormalizeRequest) -> tuple[JsonValue, list[Diagnostic]]:
    source, output, manifest, descriptor, template = (
        request.source,
        request.output,
        request.manifest,
        request.descriptor,
        request.template,
    )
    if request.allow_unknown:
        raise ContractError(Code.USAGE_ERROR, "--allow-unknown requires the USFX adapter")
    if request.work is not None:
        raise ContractError(Code.USAGE_ERROR, "--work requires the TEI adapter")
    if template and (
        manifest is not None or request.approve_reuse or request.approve_allocation or request.approve_removal
    ):
        raise ContractError(Code.USAGE_ERROR, "Templates cannot use manifests or approvals")
    approvals = Approvals.parse(request.approve_reuse, request.approve_allocation, request.approve_removal)
    targets = [output] if template else [output, manifest or output.with_suffix(".manifest.json")]
    inputs = [source, *([descriptor] if descriptor else [])]
    locks = [Path(f"{spelling}.lock") for path in targets for spelling in (path.absolute(), path.resolve())]

    def aliases() -> None:
        for i, target in enumerate(targets):
            atomic_io.reject_aliases(target, [*targets[:i], *targets[i + 1 :], *inputs, *locks])
        for lock in locks:
            atomic_io.reject_aliases(lock, inputs)

    try:
        aliases()
        with atomic_io.writer_locks(targets):
            aliases()
            raw = source.read_bytes()
            parsed_descriptor = None
            if descriptor:
                try:
                    parsed_descriptor = CorpusDescriptor.model_validate_json(descriptor.read_bytes())
                except ValidationError as error:
                    raise ContractError(Code.INVALID_CORPUS, f"Invalid descriptor: {error}") from error
            texts, labels = read_lines(raw, template)

            def build(refs: list[str]) -> bytes:
                return build_corpus(
                    source, raw, request.corpus_id, request.language, parsed_descriptor, texts, refs, template, labels
                )

            if template:
                target_bytes = build([".".join(encode_ref_component(p) for p in parts) for parts in labels])
                if read_optional(output) != target_bytes:
                    atomic_io.atomic_replace(output.resolve(), target_bytes)
                return {"review_items": []}, []
            request_record: dict[str, JsonValue] = {
                "source": {"sha256": digest_bytes(raw), "file_name": source.name},
                "corpus_id": request.corpus_id,
                "adapter": "text/1",
                "text_canon": 1,
                "norm_rules": "latin-fold/1",
                "language": request.language,
                "descriptor": parsed_descriptor.model_dump(mode="json") if parsed_descriptor else None,
                "approvals": approvals.model_dump(mode="json"),
            }
            return materialize(
                output.resolve(),
                targets[1].resolve(),
                request.corpus_id,
                [passage_hash(t) for t in texts],
                request_record,
                approvals,
                build,
            )
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error)) from error


# The single dispatch point for `corpus normalize --adapter NAME`.
ADAPTERS: dict[str, Callable[[NormalizeRequest], tuple[JsonValue, list[Diagnostic]]]] = {
    "tei": _normalize_tei,
    "text": _normalize_text,
    "usfx": _normalize_usfx,
}


@app.command()
def normalize(
    source: Annotated[Path, typer.Argument(help="UTF-8 source file.")],
    adapter: Annotated[str, typer.Option()],
    corpus_id: Annotated[str, typer.Option("--corpus-id")],
    output: Annotated[Path, typer.Option()],
    language: Annotated[str, typer.Option()] = "la",
    descriptor: Annotated[Path | None, typer.Option()] = None,
    manifest: Annotated[Path | None, typer.Option()] = None,
    citation_template: Annotated[str | None, typer.Option("--citation-template")] = None,
    approve_reuse: Annotated[list[str] | None, typer.Option("--approve-reuse")] = None,
    approve_allocation: Annotated[list[int] | None, typer.Option("--approve-allocation")] = None,
    approve_removal: Annotated[list[str] | None, typer.Option("--approve-removal")] = None,
    allow_unknown: Annotated[bool, typer.Option("--allow-unknown")] = False,
    work: Annotated[str | None, typer.Option("--work", help="TEI work ID.")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Normalize text with stable reviewed ordinal refs, or explicit source labels."""

    def body() -> tuple[JsonValue, list[Diagnostic]]:
        build = ADAPTERS.get(adapter)
        if build is None:
            raise ContractError(Code.USAGE_ERROR, f"Adapter {adapter!r} is not yet supported")
        if not re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*", language):
            raise ContractError(Code.USAGE_ERROR, "Malformed language tag")
        return build(
            NormalizeRequest(
                source=source,
                corpus_id=corpus_id,
                output=output,
                language=language,
                descriptor=descriptor,
                manifest=manifest,
                template=parse_template(citation_template),
                approve_reuse=approve_reuse or [],
                approve_allocation=approve_allocation or [],
                approve_removal=approve_removal or [],
                allow_unknown=allow_unknown,
                work=work,
            )
        )

    run_contract("corpus normalize", json_output, body, render_text=_render_text)
