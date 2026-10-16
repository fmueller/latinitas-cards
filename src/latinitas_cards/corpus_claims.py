"""Explicit corpus annotation review, separate from deck claims and authored values."""

import fcntl
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

from .annotation_sidecar import Annotation, Retraction, read_sidecar
from .annotation_state import annotation_states
from .atomic_io import atomic_replace, reject_aliases
from .canonical import canonical_json
from .claim_review import ReviewDecision
from .claim_sidecar import (
    CorpusBinding,
    DecisionItem,
    Diagnostic,
    SidecarV2,
    StoredDecisionV2,
    UnsupportedVersion,
    sidecar_path,
)
from .cli_contract import Code, ContractError
from .cli_contract import Diagnostic as CliDiagnostic
from .corpus_io import CorpusFinding, read_corpus
from .corpus_model import CorpusPassage, CorpusRecord
from .corpus_validation import validate_records


def corpus_claim_fingerprint(record: Annotation) -> str:
    raw = record.model_dump(mode="json")
    payload = {key: raw[key] for key in ("annotation_id", "layer", "value", "anchor", "producer")}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def current_annotations(corpus: Path, sidecar: Path) -> tuple[Annotation, ...]:
    """Use the shared anchor/lifecycle rules; stale and inactive proposals are not claims."""
    hashes: dict[str, str] = {}

    def records() -> Iterable[CorpusRecord | CorpusFinding]:
        for row in read_corpus(corpus):
            if isinstance(row, CorpusPassage):
                hashes[row.passage_id] = row.passage_hash
            yield row

    _, diagnostics = validate_records(records(), path=str(corpus))
    if diagnostics:
        raise ValueError("; ".join(f"{d.code}: {d.message}" for d in diagnostics))
    rows = [row for row in read_sidecar(sidecar) if isinstance(row, Annotation | Retraction)]
    states = annotation_states(hashes, rows)
    return tuple(
        row
        for row in rows
        if isinstance(row, Annotation)
        and states[row.annotation_id].state == "current"
        and not states[row.annotation_id].superseded
        and not states[row.annotation_id].retracted
    )


def _binding(record: Annotation) -> CorpusBinding:
    return CorpusBinding(
        corpus_id=record.anchor.passage_id.split(":", 1)[0],
        passage_id=record.anchor.passage_id,
        passage_hash=record.anchor.passage_hash,
        annotation_id=record.annotation_id,
        layer=record.layer,
    )


def load_corpus_decisions(path: Path, *, missing_ok: bool = True) -> SidecarV2:
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if not missing_ok:
            raise
        return SidecarV2()
    try:
        raw = json.loads(payload)
    except RecursionError as error:
        raise ValueError("corpus decision JSON is nested too deeply") from error
    if isinstance(raw, dict) and (type(raw.get("version")) is not int or raw["version"] != 2):
        raise UnsupportedVersion("unsupported_version: corpus claims require version 2")
    return SidecarV2.model_validate_json(payload)


def _resolve(
    records: Sequence[Annotation], sidecar: SidecarV2
) -> tuple[dict[str, StoredDecisionV2], tuple[Diagnostic, ...]]:
    current = {corpus_claim_fingerprint(row): row for row in records}
    decisions: dict[str, StoredDecisionV2] = {}
    seen: set[str] = set()
    diagnostics: list[Diagnostic] = []
    for index, entry in enumerate(sidecar.decisions):
        row = current.get(entry.base_fingerprint)
        if row is None or entry.binding != _binding(row):
            diagnostics.append(Diagnostic("stale", "Stale or unmatched decision ignored", index))
        elif entry.base_fingerprint in seen:
            diagnostics.append(Diagnostic("conflict", "Duplicate decisions ignored", index))
            decisions.pop(entry.base_fingerprint, None)
        else:
            decisions[entry.base_fingerprint] = entry
        seen.add(entry.base_fingerprint)
    return decisions, tuple(diagnostics)


def join_corpus_decisions(corpus: Path, sidecar: Path, claims: Path) -> tuple[dict[str, str], list[CliDiagnostic]]:
    """Join explicit v2 decisions to live annotations without locks or writes."""
    try:
        stored = load_corpus_decisions(claims, missing_ok=False)
    except UnsupportedVersion as error:
        raise ContractError(Code.UNSUPPORTED_VERSION, str(error), path=str(claims)) from error
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(claims)) from error
    except ValueError as error:
        raise ContractError(Code.INVALID_INPUT, str(error), path=str(claims)) from error
    rows = current_annotations(corpus, sidecar)
    decisions, findings = _resolve(rows, stored)
    statuses: dict[str, str] = {}
    for row in rows:
        entry = decisions.get(corpus_claim_fingerprint(row))
        statuses[row.annotation_id] = entry.decision.status if entry else "open"
    diagnostics = []
    for finding in findings:
        assert finding.item is not None
        diagnostics.append(
            CliDiagnostic(
                code=Code(finding.code),
                message=finding.message,
                path=str(claims),
                ref=stored.decisions[finding.item].base_fingerprint,
            )
        )
    return statuses, diagnostics


def corpus_claim_report(corpus: Path, sidecar: Path) -> dict[str, object]:
    rows = current_annotations(corpus, sidecar)
    decisions, diagnostics = _resolve(rows, load_corpus_decisions(sidecar_path(corpus)))
    claims: list[dict[str, object]] = []
    for row in rows:
        fingerprint = corpus_claim_fingerprint(row)
        entry = decisions.get(fingerprint)
        claims.append(
            {
                "identity": row.anchor.passage_id,
                "annotation_id": row.annotation_id,
                "kind": row.layer,
                "value": row.value,
                "alternatives": row.alternatives or [],
                "producer": row.producer.model_dump(mode="json"),
                "fingerprint": fingerprint,
                "status": entry.decision.status if entry else "withheld",
                "review_status": entry.decision.status if entry else "open",
                "reason": entry.decision.reason if entry else "Individual review required",
            }
        )
    return {"claims": claims, "diagnostics": [asdict(d) for d in diagnostics]}


def record_corpus_batch(corpus: Path, sidecar: Path, items: Sequence[object]) -> tuple[Diagnostic, ...]:
    """Blocking persistent claims lock; re-read live inputs and validate all before replace."""
    path = sidecar_path(corpus)
    lock_path = Path(f"{path}.lock")
    protected = (corpus, sidecar, corpus.with_suffix(".manifest.json"), Path(f"{corpus}.lock"), Path(f"{sidecar}.lock"))
    reject_aliases(path, (*protected, lock_path))
    reject_aliases(lock_path, protected)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reject_aliases(path, (*protected, lock_path))
        rows = current_annotations(corpus, sidecar)
        stored = load_corpus_decisions(path)
        _, diagnostics = _resolve(rows, stored)
        if diagnostics:
            return diagnostics
        lookup = {corpus_claim_fingerprint(row): row for row in rows}
        entries = {entry.base_fingerprint: entry for entry in stored.decisions}
        errors: list[Diagnostic] = []
        for index, raw in enumerate(items):
            try:
                if isinstance(raw, dict) and {"value", "evidence", "selected_sense"}.intersection(raw):
                    raise ValueError("corpus decisions cannot author values, evidence or selected senses")
                item = DecisionItem.model_validate(raw)
                row = lookup.get(item.fingerprint)
                if row is None:
                    errors.append(Diagnostic("stale", "Fingerprint is not a current claim", index))
                    continue
                entry = StoredDecisionV2(
                    base_fingerprint=item.fingerprint,
                    decision=ReviewDecision(item.fingerprint, item.status, item.reviewer, item.reason),
                    binding=_binding(row),
                )
                previous = entries.get(item.fingerprint)
                if previous is not None and previous != entry:
                    errors.append(Diagnostic("conflict", "Conflicting decision for the same claim", index))
                else:
                    entries[item.fingerprint] = entry
            except ValueError as error:
                errors.append(Diagnostic("invalid", str(error), index))
        if errors:
            return tuple(errors)
        payload = (SidecarV2(decisions=tuple(entries.values())).model_dump_json(indent=2) + "\n").encode("utf-8")
        atomic_replace(path, payload)
    return ()
