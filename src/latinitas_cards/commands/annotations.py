"""The built-in deterministic annotation producer; no analyzer dependencies."""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import JsonValue

from ..annotation_apply import apply_plan
from ..annotation_plans import approve_plan, create_plan
from ..annotation_sidecar import Annotation, Retraction, annotation_id, append_records, resolve_created
from ..annotation_state import inspect_annotations
from ..cli_contract import Code, Diagnostic, run_contract
from ..corpus_claims import join_corpus_decisions
from ..corpus_io import CorpusFinding, read_corpus
from ..corpus_model import CorpusPassage
from ..corpus_segmentation import segment_work, tokenize
from ..corpus_validation import validate_records

app = typer.Typer(help="Produce versioned annotation sidecars.")


@app.command()
def apply(
    plan: Annotated[Path, typer.Argument()],
    corpus: Annotated[Path, typer.Option()],
    sidecar: Annotated[Path, typer.Option()],
    policy: Annotated[Path | None, typer.Option()] = None,
    receipt: Annotated[Path | None, typer.Option()] = None,
    timestamp: Annotated[str | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Atomically append automatic and explicitly receipted planned operations."""
    run_contract(
        "annotations apply", json_output, lambda: apply_plan(plan, corpus, sidecar, policy, receipt, timestamp)
    )


@app.command()
def plan(
    patch: Annotated[Path, typer.Argument()],
    corpus: Annotated[Path, typer.Option()],
    sidecar: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    policy: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Classify an entire patch without appending any annotations."""
    run_contract("annotations plan", json_output, lambda: create_plan(patch, corpus, sidecar, policy, output))


@app.command()
def approve(
    plan: Annotated[Path, typer.Argument()],
    operation: Annotated[list[str], typer.Option()],
    actor: Annotated[str, typer.Option()],
    review: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Record explicit review of selected operation IDs by a declared actor."""
    run_contract("annotations approve", json_output, lambda: approve_plan(plan, operation, actor, review, output))


@app.command()
def segment(
    corpus: Annotated[Path, typer.Argument(help="Normalized corpus JSONL.")],
    output: Annotated[Path | None, typer.Option()] = None,
    timestamp: Annotated[str | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Append latin-seg/1 proposals without replacing current segmentations."""

    def body() -> tuple[JsonValue, list[Diagnostic]]:
        if timestamp is not None:
            resolve_created(timestamp, os.environ)
        passages: list[CorpusPassage] = []

        def records() -> Iterable[Any]:
            for row in read_corpus(corpus):
                if not isinstance(row, CorpusFinding) and isinstance(row, CorpusPassage):
                    passages.append(row)
                yield row

        _, diagnostics = validate_records(records(), path=str(corpus))
        if diagnostics:
            return None, [
                Diagnostic(
                    code=Code.INVALID_CORPUS, message=f"Invalid corpus: {diagnostics[0].code}", path=str(corpus)
                ),
                *diagnostics,
            ]
        sentences = segment_work(passages)
        target = output if output is not None else corpus.with_suffix(".annotations.jsonl")
        counts = {"appended": 0, "skipped": 0}

        def additions(existing: list[Annotation | Retraction]) -> Iterable[Annotation]:
            known = {record.annotation_id for record in existing}
            removed = {record.retracts if isinstance(record, Retraction) else record.supersedes for record in existing}
            current_by_passage: dict[str, list[Annotation]] = {}
            for record in existing:
                if (
                    isinstance(record, Annotation)
                    and record.layer == "segmentation"
                    and record.annotation_id not in removed
                ):
                    current_by_passage.setdefault(record.anchor.passage_id, []).append(record)
            for passage, parts in zip(passages, sentences, strict=True):
                value = {
                    "tokenizer": "latin-tok/1",
                    "tokens": [
                        {"kind": token.kind, "span": [token.start, token.end]} for token in tokenize(passage.text)
                    ],
                    "sentences": [
                        {
                            "continued": part.continued,
                            "continues": part.continues,
                            "span": [part.start, part.end],
                            "tokens": [part.token_start, part.token_end],
                        }
                        for part in parts
                    ],
                }
                current = [
                    record
                    for record in current_by_passage.get(passage.passage_id, [])
                    if record.input_hash == passage.passage_hash
                ]
                if current:
                    counts["skipped"] += 1
                    if any(record.value != value for record in current):
                        diagnostics.append(
                            Diagnostic(
                                code=Code.SEGMENTATION_DIFFERS,
                                severity="warning",
                                message="Current segmentation differs; retained without appending",
                                ref=passage.passage_id,
                            )
                        )
                    continue
                raw: dict[str, Any] = {
                    "record": "annotation",
                    "layer": "segmentation",
                    "anchor": {"passage_id": passage.passage_id, "passage_hash": passage.passage_hash},
                    "value": value,
                    "input_hash": passage.passage_hash,
                    "producer": {"kind": "rule", "name": "latin-seg", "version": "1"},
                    "plan": None,
                }
                raw["annotation_id"] = annotation_id(raw)
                if raw["annotation_id"] in known:
                    counts["skipped"] += 1
                    continue
                raw["created"] = resolve_created(timestamp, os.environ)
                yield Annotation.model_validate(raw)

        counts["appended"] = append_records(
            target,
            additions,
            [
                corpus,
                Path(f"{corpus}.claims.json"),
                corpus.with_suffix(".manifest.json"),
                Path(f"{corpus.with_suffix('.manifest.json')}.lock"),
                Path(f"{corpus.with_suffix('.manifest.json').resolve()}.lock"),
                Path(f"{corpus}.lock"),
                Path(f"{corpus.resolve()}.lock"),
                Path(f"{corpus}.claims.json.lock"),
            ],
        )
        return dict(counts), diagnostics

    run_contract("annotations segment", json_output, body)


@app.command()
def inspect(
    sidecar: Annotated[Path, typer.Argument(help="Annotation sidecar JSONL.")],
    corpus: Annotated[Path, typer.Option(help="Normalized corpus JSONL.")],
    claims: Annotated[Path | None, typer.Option()] = None,
    check: Annotated[bool, typer.Option()] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List provenance and anchor states without changing corpus or sidecar."""

    def body() -> tuple[JsonValue, list[Diagnostic]]:
        result, diagnostics = inspect_annotations(corpus, sidecar, check=check)
        if claims is not None and not any(d.path == str(corpus) and d.severity == "error" for d in diagnostics):
            reviews, findings = join_corpus_decisions(corpus, sidecar, claims)
            assert isinstance(result["annotations"], list)
            for row in result["annotations"]:
                assert isinstance(row, dict) and isinstance(row["annotation_id"], str)
                row["review_status"] = reviews.get(row["annotation_id"], "open")
            diagnostics.extend(d.model_copy(update={"severity": "error" if check else "warning"}) for d in findings)
        return {"annotations": result["annotations"]}, diagnostics

    run_contract("annotations inspect", json_output, body)
