"""Read-only anchor states; never re-anchor a proposal or discard its history."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from .annotation_sidecar import Annotation, Retraction, read_sidecar
from .cli_contract import Code, Diagnostic
from .corpus_io import CorpusFinding, read_corpus
from .corpus_model import CorpusHeader, CorpusPassage, CorpusRecord
from .corpus_validation import validate_records


@dataclass(frozen=True)
class AnnotationState:
    state: Literal["current", "stale", "orphaned", "unmatched"]
    superseded: bool
    retracted: bool


def annotation_states(
    corpus: Mapping[str, str], sidecar: Sequence[Annotation | Retraction]
) -> dict[str, AnnotationState]:
    """Compare passage hashes, then segmentation lifecycle, in precedence order."""
    seen: set[str] = set()
    superseded: set[str] = set()
    retracted: set[str] = set()
    for row in sidecar:
        if isinstance(row, Annotation) and row.supersedes in seen:
            assert row.supersedes is not None
            superseded.add(row.supersedes)
        elif isinstance(row, Retraction) and row.retracts in seen:
            retracted.add(row.retracts)
        seen.add(row.annotation_id)
    segmentations = {
        row.annotation_id for row in sidecar if isinstance(row, Annotation) and row.layer == "segmentation"
    }
    inactive_segmentations = superseded | retracted
    result: dict[str, AnnotationState] = {}
    for row in sidecar:
        state: Literal["current", "stale", "orphaned", "unmatched"] = "current"
        if row.anchor.passage_id not in corpus:
            state = "unmatched"
        elif corpus[row.anchor.passage_id] != row.anchor.passage_hash:
            state = "stale"
        elif row.anchor.segmentation_id is not None and (
            row.anchor.segmentation_id not in segmentations or row.anchor.segmentation_id in inactive_segmentations
        ):
            state = "orphaned"
        result[row.annotation_id] = AnnotationState(
            state, row.annotation_id in superseded, row.annotation_id in retracted
        )
    return result


def dependents(sidecar: Sequence[Annotation | Retraction], annotation_id: str) -> list[Annotation | Retraction]:
    """Return records naming the segmentation, preserving sidecar order."""
    return [row for row in sidecar if row.anchor.segmentation_id == annotation_id]


def inspect_annotations(corpus: Path, sidecar: Path, *, check: bool) -> tuple[dict[str, JsonValue], list[Diagnostic]]:
    """Validate a single corpus stream, retaining only its ID/hash map, not text."""
    hashes: dict[str, str] = {}
    source_sha256: str | None = None

    def records() -> Iterable[CorpusRecord | CorpusFinding]:
        nonlocal source_sha256
        for row in read_corpus(corpus):
            if isinstance(row, CorpusHeader):
                source_sha256 = row.source.sha256
            elif isinstance(row, CorpusPassage):
                hashes[row.passage_id] = row.passage_hash
            yield row

    counts, diagnostics = validate_records(records(), path=str(corpus))
    rows = [row for row in read_sidecar(sidecar) if isinstance(row, Annotation | Retraction)]
    states = annotation_states(hashes, rows)
    descriptions = {
        "stale": "passage hash differs from the anchor",
        "unmatched": "anchored passage is missing",
        "orphaned": "anchor segmentation is superseded, retracted, or unknown",
    }
    annotations: list[JsonValue] = []
    for row in rows:
        state = states[row.annotation_id]
        annotations.append(
            {
                "annotation_id": row.annotation_id,
                "layer": row.layer,
                "state": state.state,
                "superseded": state.superseded,
                "retracted": state.retracted,
                "passage_id": row.anchor.passage_id,
                "source_sha256": source_sha256,
                "producer": row.producer.model_dump(mode="json"),
                "plan": row.plan,
                "created": row.created,
                "input_hash": row.input_hash,
            }
        )
        if state.state != "current":
            diagnostics.append(
                Diagnostic(
                    code=Code(state.state),
                    severity="error" if check else "warning",
                    path=str(sidecar),
                    ref=row.annotation_id,
                    message=(
                        f"Annotation {row.annotation_id} on passage {row.anchor.passage_id}: "
                        f"{descriptions[state.state]}"
                    ),
                )
            )
    return {"annotations": annotations, "counts": dict(counts)}, diagnostics
