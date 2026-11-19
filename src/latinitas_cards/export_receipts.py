"""Export commitments and manual installed-content handoffs, never Anki authority.

Decision IDs commit to assertions, evidence, binding and policy. Curated consumers
must supply their exact approval IDs and current authority assessment separately;
this ledger cannot approve either linguistic content or a destination operation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import model_validator

from .acceptance_records import Hash, Record, Text, Timestamp, canonical, decode_json, digest
from .pipeline_batches import CardRef

MANUAL_ACTION = (
    "Installation is presumed from export, not observed. In a backup/disposable collection, locate this note/card "
    "identity and inspect the actual destination. Review and manually remove/update the stale assertion and status "
    "or suspend the affected card as appropriate. Preserve user fields, all unowned tags (including case), sibling "
    "content, history and scheduling. Record the resolution and verify preservation after manual application. "
    "CSV omission does not remove assertions, update badges, suspend or retire cards. No action has been applied."
)
IMPORT_ACTION = (
    "Initial import and note-type installation require separate confirmation. Before any existing-note reimport, "
    "inspect the fresh destination, preserve its user fields and full actual tag set, and explicitly resolve/adopt "
    "namespace collisions before approving the exact update. These files are not approved for direct reimport. "
    "Unsupported clearing, retirement, suspension and template changes remain blocked; no Anki writes are performed."
)


class ReceiptCardRef(CardRef):
    note_id: Text
    card_key: Text


class ExportCard(ReceiptCardRef):
    decision_ids: tuple[Hash, ...]
    assertion_ids: tuple[Text, ...]
    rendered_sha256: Hash
    acceptance: Literal["human", "automatic", "mixed"]
    owned_fields: tuple[Text, ...]
    owned_tags: tuple[Text, ...]

    @model_validator(mode="after")
    def commitments(self) -> Self:
        if not self.decision_ids or not self.assertion_ids:
            raise ValueError("exported card requires assertion and authority commitments")
        for values in (self.decision_ids, self.assertion_ids, self.owned_fields, self.owned_tags):
            if len(set(values)) != len(values):
                raise ValueError("duplicate export commitment")
        if any(not t.startswith(("latinitas::acceptance::", "latinitas::basis::")) for t in self.owned_tags):
            raise ValueError("unowned export tag")
        return self


class Handoff(ReceiptCardRef):
    previous_decision_ids: tuple[Hash, ...]
    reason: Text
    resolution: Literal["manual", "managed_retained_fields", "blocked"]
    required_action: Text


class ExportFile(Record):
    path: Text
    sha256: Hash


class Cursor(Record):
    selection_sha256: Hash
    ordered_card_refs_sha256: Hash
    emitted_cards: tuple[ReceiptCardRef, ...]


class ExportReceipt(Record):
    format: Literal["latinitas-export-receipt"] = "latinitas-export-receipt"
    version: Literal[1] = 1
    receipt_id: Hash
    run_id: Text
    previous_receipt_id: Hash | None
    selection_sha256: Hash
    layout_sha256: Hash
    files: tuple[ExportFile, ...]
    cards: tuple[ExportCard, ...]
    ineligible: tuple[Handoff, ...]
    batch_cursor: Cursor | None
    created_at: Timestamp

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.receipt_id != digest(self.model_dump(exclude={"receipt_id"})):
            raise ValueError("export receipt digest mismatch")
        for rows in (self.cards, self.ineligible):
            refs = [(c.note_id, c.card_key) for c in rows]
            if len(set(refs)) != len(refs):
                raise ValueError("duplicate export CardRef")
        return self


@dataclass(frozen=True)
class Assessment:
    card: ExportCard | None
    reason: str


def invalidated_cards(previous: Sequence[ExportReceipt], current: Mapping[CardRef, Assessment]) -> tuple[Handoff, ...]:
    """Compare every historical export, not only the most recent batch or selection.

    Missing current coverage is unresolved, not proof of invalidity or removal.
    In particular a selection/processing limit cannot resolve an installed card.
    Retain old authority IDs even if a newer export has different accepted content.
    """
    affected: dict[CardRef, tuple[set[str], set[str]]] = {}
    for receipt in previous:
        for old in receipt.cards:
            ref = CardRef(note_id=old.note_id, card_key=old.card_key)
            assessment = current.get(ref)
            if assessment is not None and assessment.card == old:
                continue
            if assessment is None:
                reason = "current_eligibility_unavailable"
            elif assessment.card is not None:
                reason = "acceptance_or_rendering_changed"
            else:
                reason = assessment.reason
            ids, reasons = affected.setdefault(ref, (set(), set()))
            ids.update(old.decision_ids)
            reasons.add(reason)
        # Unresolved actions persist even through an empty subsequent export.
        for action in receipt.ineligible:
            ref = CardRef(note_id=action.note_id, card_key=action.card_key)
            ids, reasons = affected.setdefault(ref, (set(), set()))
            ids.update(action.previous_decision_ids)
            # Multiple causes use a JSON string array, so carrying them forward
            # stays idempotent without changing the versioned reason field.
            try:
                prior = decode_json(action.reason)
            except ValueError:
                prior = None
            if isinstance(prior, list) and prior and all(isinstance(r, str) and r.strip() for r in prior):
                reasons.update(prior)
            else:
                reasons.add(action.reason)
    for _, reasons in affected.values():
        if len(reasons) > 1:
            reasons.discard("current_eligibility_unavailable")
    return tuple(
        Handoff(
            **ref.model_dump(),
            previous_decision_ids=tuple(sorted(ids)),
            reason=next(iter(reasons)) if len(reasons) == 1 else canonical(sorted(reasons)).decode(),
            resolution="manual",
            required_action=MANUAL_ACTION,
        )
        for ref, (ids, reasons) in sorted(affected.items(), key=lambda item: (item[0].note_id, item[0].card_key))
    )
