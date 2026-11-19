"""Versioned, offline curated candidates and content-bound human review.

These records do not grant rendered-card approval, export or provider permission.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from .acceptance_records import Hash, Record, Text, Timestamp, canonical, decode_json, digest

type Language = Literal["de", "en"]


class Evidence(Record):
    source: Text
    reference: Text
    version: Text
    text: Text


class Provenance(Record):
    document: Text
    section: Text
    reference: Text
    wording: Literal["quoted", "agent_proposed", "author_composed"]


class Rule(Record):
    text: Text
    language: Language
    applicability: Text
    caveats: tuple[Text, ...]
    evidence: tuple[Evidence, ...] = Field(min_length=1)


class Mnemonic(Record):
    key: Text
    language: Language
    text: Text
    rule_key: Text
    provenance: Provenance


class Example(Record):
    latin: Text
    explanation: Text
    evidence: tuple[Evidence, ...] = Field(min_length=1)


class Item(Record):
    key: Text
    rule: Rule
    mnemonics: tuple[Mnemonic, ...]
    examples: tuple[Example, ...]
    tags: tuple[Text, ...]
    provenance: Provenance

    @model_validator(mode="after")
    def associations(self) -> Self:
        identities = [(m.language, m.key) for m in self.mnemonics]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate mnemonic language/key")
        if any(m.rule_key != self.key for m in self.mnemonics):
            raise ValueError("mnemonic rule_key must match item key")
        return self


class CuratedInput(Record):
    format: Literal["latinitas-curated"]
    version: Literal[1]
    source_scope: Text
    items: tuple[Item, ...]

    @model_validator(mode="after")
    def item_keys(self) -> Self:
        if len({item.key for item in self.items}) != len(self.items):
            raise ValueError("duplicate item key")
        return self


class SourcePermission(Record):
    document: Text
    source_sha256: Hash
    rights: Text
    extraction_permission: Text
    processing: Literal["manual", "local", "hosted"]
    provider: Text | None
    provider_permission: Text | None

    @model_validator(mode="after")
    def processing_permission(self) -> Self:
        if self.processing == "manual":
            if self.provider is not None or self.provider_permission is not None:
                raise ValueError("manual extraction cannot declare a provider")
        elif self.provider is None or self.provider_permission is None:
            raise ValueError("local/hosted extraction requires explicit provider and provider_permission")
        return self


class Permission(Record):
    """Operator attestations of prior extraction; never authority to make a call."""

    format: Literal["latinitas-curated-permission"]
    version: Literal[1]
    source_scope: Text
    sources: tuple[SourcePermission, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def source_documents(self) -> Self:
        if len({s.document for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate source document")
        return self


def load_curated(path: Path, permissions: Path) -> tuple[CuratedInput, Permission]:
    data = CuratedInput.read_json(path.read_bytes())
    permission = Permission.read_json(permissions.read_bytes())
    if data.source_scope != permission.source_scope:
        raise ValueError("permission source_scope mismatch")
    documents = {item.provenance.document for item in data.items}
    documents.update(m.provenance.document for item in data.items for m in item.mnemonics)
    if not documents <= {s.document for s in permission.sources}:
        raise ValueError("missing source document permission")
    return data, permission


def rule_fingerprint(item: Item) -> str:
    # Example explanations are rule content too, never an unreviewed side channel.
    return digest({"key": item.key, "rule": item.rule, "examples": item.examples, "provenance": item.provenance})


def mnemonic_fingerprint(item: Item, mnemonic: Mnemonic) -> str:
    if mnemonic not in item.mnemonics:
        raise ValueError("mnemonic is not associated with this item")
    return digest({"mnemonic": mnemonic, "rule_fingerprint": rule_fingerprint(item)})


class Approval(Record):
    format: Literal["latinitas-curated-approval"]
    version: Literal[1]
    approval_id: Hash
    supersedes: tuple[Hash, ...]
    source_scope: Text
    item_key: Text
    subject: Literal["rule", "mnemonic", "rendered_card"]
    subject_key: Text
    language: Language
    fingerprint: Hash
    decision: Literal["approved", "rejected"]
    reviewer: Text
    reason: Text
    reviewed_at: Timestamp

    @field_validator("reviewed_at")
    @classmethod
    def valid_time(cls, value: str) -> str:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @model_validator(mode="after")
    def identity(self) -> Self:
        if self.approval_id != digest(self.model_dump(mode="json", exclude={"approval_id"})):
            raise ValueError("approval digest mismatch")
        if self.subject == "rule" and self.subject_key != self.item_key:
            raise ValueError("rule subject_key must equal item_key")
        return self


def make_approval(**fields: object) -> Approval:
    """Seal an explicit operator decision; hashing does not authenticate a human."""
    raw = {"format": "latinitas-curated-approval", "version": 1, **fields}
    return Approval.read_json(canonical({**raw, "approval_id": digest(raw)}))


def _subject(approval: Approval) -> tuple[str, ...]:
    return (approval.source_scope, approval.item_key, approval.subject, approval.subject_key, approval.language)


def validate_history(history: tuple[Approval, ...]) -> None:
    seen: dict[str, Approval] = {}
    for record in history:
        record = Approval.read_json(record.model_dump_json())
        if record.approval_id in seen:
            raise ValueError("duplicate approval_id")
        for identity in record.supersedes:
            previous = seen.get(identity)
            if previous is None:
                raise ValueError("supersedes must reference an earlier retained approval")
            if _subject(previous) != _subject(record):
                raise ValueError("superseded approval must have the same subject")
        seen[record.approval_id] = record


def load_approvals(path: Path | None, previous: tuple[Approval, ...] = ()) -> tuple[Approval, ...]:
    if path is None:
        return ()
    raw = decode_json(path.read_bytes())
    if not isinstance(raw, list):
        raise ValueError("approval history must be an array")
    result = tuple(Approval.read_json(canonical(value)) for value in raw)
    validate_history((*previous, *result))
    return result


def current_subject(
    item: Item,
    subject: str,
    subject_key: str,
    language: str,
) -> str | None:
    if subject == "rule" and subject_key == item.key and language == item.rule.language:
        return rule_fingerprint(item)
    if subject == "mnemonic":
        for mnemonic in item.mnemonics:
            if (mnemonic.key, mnemonic.language) == (subject_key, language):
                return mnemonic_fingerprint(item, mnemonic)
    return None


def approval_status(
    data: CuratedInput,
    item: Item,
    history: tuple[Approval, ...],
    *,
    subject: str,
    subject_key: str | None = None,
    language: str | None = None,
) -> str:
    validate_history(history)
    subject_key = item.key if subject_key is None else subject_key
    language = item.rule.language if language is None else language
    fingerprint = current_subject(item, subject, subject_key, language)
    if fingerprint is None:
        return "missing_language_variant" if subject == "mnemonic" else "unreviewed"
    matching = matching_decisions(data.source_scope, item.key, subject, subject_key, language, fingerprint, history)
    if len(matching) > 1:
        return "conflict"
    return matching[0].decision if matching else "unreviewed"


def matching_decisions(
    scope: str,
    item_key: str,
    subject: str,
    subject_key: str,
    language: str,
    fingerprint: str,
    history: tuple[Approval, ...],
) -> tuple[Approval, ...]:
    """Exact unsuperseded heads; callers validate the retained history first."""
    superseded = {identity for record in history for identity in record.supersedes}
    return tuple(
        record
        for record in history
        if record.approval_id not in superseded
        and _subject(record) == (scope, item_key, subject, subject_key, language)
        and record.fingerprint == fingerprint
    )


def record_reviews(
    data: CuratedInput,
    history: tuple[Approval, ...],
    decisions: tuple[Approval, ...],
    output: Path,
    *,
    rendered_subjects: dict[tuple[str, str, str], str] | None = None,
) -> None:
    """Write a new retained history only after every submitted decision matches."""
    if not decisions:
        raise ValueError("explicit decisions required")
    items = {item.key: item for item in data.items}
    for decision in decisions:
        item = items.get(decision.item_key)
        if decision.subject == "rendered_card":
            fingerprint = (rendered_subjects or {}).get((decision.item_key, decision.subject_key, decision.language))
        elif item is not None:
            fingerprint = current_subject(item, decision.subject, decision.subject_key, decision.language)
        else:
            fingerprint = None
        if decision.source_scope != data.source_scope or item is None or fingerprint != decision.fingerprint:
            raise ValueError("stale/unknown subject; rendered-card review belongs to the downstream card workflow")
    combined = (*history, *decisions)
    validate_history(combined)
    with output.open("xb") as stream:
        stream.write(canonical(combined) + b"\n")


def preview_candidates(data: CuratedInput, permission: Permission, history: tuple[Approval, ...]) -> dict[str, object]:
    validate_history(history)
    items: list[dict[str, object]] = []
    for item in data.items:
        items.append(
            {
                **item.model_dump(mode="json"),
                "rule": {
                    **item.rule.model_dump(mode="json"),
                    "fingerprint": rule_fingerprint(item),
                    "status": approval_status(data, item, history, subject="rule"),
                },
                "mnemonics": [
                    {
                        **m.model_dump(mode="json"),
                        "fingerprint": mnemonic_fingerprint(item, m),
                        "status": approval_status(
                            data, item, history, subject="mnemonic", subject_key=m.key, language=m.language
                        ),
                    }
                    for m in item.mnemonics
                ],
            }
        )
    return {
        "source_scope": data.source_scope,
        "permissions": permission.model_dump(mode="json"),
        "items": items,
        "candidate_only": True,
        "export_authorized": False,
        "limits": "No rendered-card approval or export; any displayed card mnemonic requires separate approval.",
    }
