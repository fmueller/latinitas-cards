"""Ordinal identity reconciliation and locked recoverable two-file materialization."""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, JsonValue, ValidationError, model_validator

from . import atomic_io
from .canonical import canonical_json, digest, digest_bytes
from .cli_contract import Code, ContractError, Diagnostic
from .corpus_model import CorpusId, CorpusModel, Sha256, VersionOne


class Materialization(CorpusModel):
    request_digest: Sha256
    corpus_digest: Sha256
    previous_corpus_digest: Sha256 | None


class ManifestEntry(CorpusModel):
    ref: str = Field(pattern=r"^p[0-9]{6,}$")
    passage_hash: Sha256
    ordinal: int = Field(ge=0)
    active: bool


class CorpusManifest(VersionOne):
    format: Literal["latinitas-corpus-manifest"]
    corpus_id: CorpusId
    next_ref: int = Field(ge=1)
    entries: list[ManifestEntry]
    snapshot: list[Sha256]
    materialization: Materialization

    @model_validator(mode="after")
    def consistent_assignments(self) -> Self:
        refs = [entry.ref for entry in self.entries]
        active = sorted((e for e in self.entries if e.active), key=lambda e: e.ordinal)
        if len(refs) != len(set(refs)) or any(int(ref[1:]) >= self.next_ref or int(ref[1:]) < 1 for ref in refs):
            raise ValueError("Manifest refs must be unique, positive and below next_ref")
        if any(ref != f"p{int(ref[1:]):06d}" for ref in refs):
            raise ValueError("Manifest refs must use canonical allocator spelling")
        if [e.ordinal for e in active] != list(range(len(self.snapshot))) or [
            e.passage_hash for e in active
        ] != self.snapshot:
            raise ValueError("Active assignments must exactly describe the ordered snapshot")
        return self


class Approvals(CorpusModel):
    reuse: list[tuple[int, str]] = Field(default_factory=list)
    allocation: list[int] = Field(default_factory=list)
    removal: list[str] = Field(default_factory=list)

    @classmethod
    def parse(cls, reuse: list[str], allocation: list[int], removal: list[str]) -> Self:
        try:
            pairs = []
            for flag in reuse:
                ordinal, separator, ref = flag.partition("=")
                if not separator or not ref:
                    raise ValueError("Reuse requires N=REF")
                pairs.append((int(ordinal), ref))
            if (
                len({n for n, _ in pairs}) != len(pairs)
                or len(set(allocation)) != len(allocation)
                or len(set(removal)) != len(removal)
                or {n for n, _ in pairs} & set(allocation)
                or len({ref for _, ref in pairs}) != len(pairs)
                or {ref for _, ref in pairs} & set(removal)
            ):
                raise ValueError("Duplicate or contradictory approvals")
            return cls(reuse=sorted(pairs), allocation=sorted(allocation), removal=sorted(removal))
        except ValueError as error:
            raise ContractError(Code.INVALID_APPROVAL, str(error)) from error

    def any(self) -> bool:
        return bool(self.reuse or self.allocation or self.removal)


def reconcile(
    hashes: list[str],
    manifest: CorpusManifest | None,
    approvals: Approvals,
) -> tuple[list[ManifestEntry], int, list[dict[str, JsonValue]]]:
    """Reuse only unique exact matches (or an unchanged snapshot); review all identity changes."""
    entries = manifest.entries if manifest else []
    by_ref = {e.ref: e for e in entries}
    active = [e for e in entries if e.active]
    reuse = dict(approvals.reuse)
    allocations = set(approvals.allocation)
    removals = set(approvals.removal)
    if manifest is None and approvals.any():
        raise ContractError(Code.INVALID_APPROVAL, "Approvals require an existing manifest")
    if any(n < 0 or n >= len(hashes) for n in [*reuse, *allocations]):
        raise ContractError(Code.INVALID_APPROVAL, "Approval ordinal is out of range")
    for ref in [*reuse.values(), *removals]:
        if ref not in by_ref:
            raise ContractError(Code.INVALID_APPROVAL, "Approval names an unknown ref")
        if not by_ref[ref].active:
            raise ContractError(Code.RETIRED_REF, "Approval names a retired ref")
    assignments: dict[int, str] = {}
    counts = Counter(hashes)
    if manifest and hashes == manifest.snapshot:
        assignments = {e.ordinal: e.ref for e in active}
    else:
        for n, fingerprint in enumerate(hashes):
            candidates = [e.ref for e in active if e.passage_hash == fingerprint]
            if counts[fingerprint] == 1 and len(candidates) == 1:
                assignments[n] = candidates[0]
    for n, ref in reuse.items():
        if n in assignments and assignments[n] != ref or any(i != n and r == ref for i, r in assignments.items()):
            raise ContractError(Code.INVALID_APPROVAL, "Reuse conflicts with an exact assignment")
        assignments[n] = ref
    next_ref = manifest.next_ref if manifest else 1
    for n in sorted(allocations if manifest else range(len(hashes))):
        if n in assignments:
            raise ContractError(Code.INVALID_APPROVAL, "Allocation conflicts with an assignment")
        assignments[n] = f"p{next_ref:06d}"
        next_ref += 1
    if removals & set(assignments.values()):
        raise ContractError(Code.INVALID_APPROVAL, "Removal conflicts with an assignment")
    reviews: list[dict[str, JsonValue]] = []
    reviewed: set[str] = set()
    for n, fingerprint in enumerate(hashes):
        if n in assignments:
            continue
        candidates = [e.ref for e in active if e.passage_hash == fingerprint]
        if candidates or counts[fingerprint] > 1:
            kind = "duplicate"
        else:
            candidates = [e.ref for e in active if e.ordinal == n and e.ref not in assignments.values()]
            kind = "edited" if candidates else "inserted"
        reviewed.update(candidates)
        reviews.append({"kind": kind, "ordinal": n, "candidates": list(candidates)})
    for entry in active:
        if entry.ref not in assignments.values() and entry.ref not in reviewed and entry.ref not in removals:
            reviews.append({"kind": "removed", "ordinal": None, "candidates": [entry.ref]})
    if reviews:
        return [], next_ref, reviews
    # All unused old identities must have explicit removal approval, never implicit retirement.
    result = [
        ManifestEntry(ref=ref, passage_hash=hashes[n], ordinal=n, active=True) for n, ref in sorted(assignments.items())
    ]
    result.extend(e.model_copy(update={"active": False}) for e in entries if e.ref not in assignments.values())
    return result, next_ref, []


def read_optional(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def materialize(
    output: Path,
    manifest_path: Path,
    corpus_id: str,
    hashes: list[str],
    request: dict[str, JsonValue],
    approvals: Approvals,
    build: Callable[[list[str]], bytes],
) -> tuple[JsonValue, list[Diagnostic]]:
    """Called under BOTH writer locks, including all source and old-state reads."""
    manifest_bytes = read_optional(manifest_path)
    old_corpus = read_optional(output)
    old_digest = digest_bytes(old_corpus) if old_corpus is not None else None
    request_digest = digest(request)
    manifest: CorpusManifest | None = None
    if manifest_bytes is not None:
        try:
            manifest = CorpusManifest.model_validate_json(manifest_bytes)
        except ValidationError as error:
            raise ContractError(Code.INVALID_MANIFEST, str(error)) from error
        receipt = manifest.materialization
        if request_digest == receipt.request_digest:
            if old_digest not in {None, receipt.previous_corpus_digest, receipt.corpus_digest}:
                raise ContractError(
                    Code.BINDING_DRIFT, "Pending materialization requires the identical request and old corpus"
                )
            if manifest.snapshot != hashes or manifest.corpus_id != corpus_id:
                raise ContractError(Code.BINDING_DRIFT, "Committed assignments do not match the request")
            refs = [e.ref for e in sorted(manifest.entries, key=lambda e: e.ordinal) if e.active]
            target = build(refs)
            if digest_bytes(target) != receipt.corpus_digest:
                raise ContractError(Code.BINDING_DRIFT, "Cannot reproduce committed corpus digest")
            if old_digest != receipt.corpus_digest:
                try:
                    with atomic_io.stage_bytes(output, target) as staged:
                        atomic_io.replace_staged(output, staged)
                except (OSError, ContractError) as error:
                    raise ContractError(Code.UNREADABLE, f"{error}; rerun the identical request to recover") from error
            return {"review_items": []}, []
        if old_digest != receipt.corpus_digest:
            raise ContractError(
                Code.BINDING_DRIFT, "Pending materialization requires the identical request and old corpus"
            )
        if manifest.corpus_id != corpus_id:
            raise ContractError(Code.BINDING_DRIFT, "Manifest corpus ID differs")
    entries, next_ref, reviews = reconcile(hashes, manifest, approvals)
    if reviews:
        diagnostics = [
            Diagnostic(
                code=Code.REVIEW_REQUIRED,
                message=str(item["kind"]),
                item=item["ordinal"] if isinstance(item["ordinal"], int) else None,
                ref=str(item["candidates"][0]) if isinstance(item["candidates"], list) and item["candidates"] else None,
            )
            for item in reviews
        ]
        return {"review_items": list[JsonValue](reviews)}, diagnostics
    target = build([e.ref for e in entries if e.active])
    committed = CorpusManifest(
        format="latinitas-corpus-manifest",
        version=1,
        corpus_id=corpus_id,
        next_ref=next_ref,
        entries=entries,
        snapshot=hashes,
        materialization=Materialization(
            request_digest=request_digest, corpus_digest=digest_bytes(target), previous_corpus_digest=old_digest
        ),
    )
    target_manifest = (canonical_json(committed.model_dump(mode="json")) + "\n").encode("utf-8")
    try:
        with (
            atomic_io.stage_bytes(manifest_path, target_manifest) as staged_manifest,
            atomic_io.stage_bytes(output, target) as staged_corpus,
        ):
            if manifest_bytes != target_manifest:
                atomic_io.replace_staged(manifest_path, staged_manifest)
            if old_corpus != target:
                atomic_io.replace_staged(output, staged_corpus)
    except (OSError, ContractError) as error:
        # Rename may have succeeded even when its directory fsync failed. Retrying
        # the exact request is safe both before and after the manifest commit.
        raise ContractError(Code.UNREADABLE, f"{error}; rerun the identical request to recover") from error
    return {"review_items": []}, []
