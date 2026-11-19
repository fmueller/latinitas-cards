"""Receipt-backed offline curated CSV handoff; no destination writes or retirement."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .acceptance_records import canonical, digest
from .curated import Approval, CuratedInput, load_approvals
from .curated_cards import (
    CURATED_FIELDS,
    CURATED_LAYOUT,
    TEMPLATE_SHA256,
    CuratedCard,
    RecipePlan,
    Settings,
    render_cards,
)
from .export_receipts import IMPORT_ACTION, Assessment, ExportCard, ExportFile, ExportReceipt, invalidated_cards
from .pipeline_batches import Batches, CardRef, Receipt, _atomic, run_lock
from .preview_export import serialize_anki_csv

type Loader = Callable[[], tuple[CuratedInput, RecipePlan, tuple[Approval, ...]]]

IMPORT_ACTION_DE = (
    "Erstimport und Installation des Notiztyps benötigen eine gesonderte Bestätigung. Vor erneutem Import "
    "bestehender Notizen das aktuelle Ziel prüfen, Benutzerfelder und alle tatsächlichen Tags erhalten und "
    "Namensraumkollisionen ausdrücklich auflösen/übernehmen, bevor die genaue Änderung freigegeben wird. "
    "Diese Dateien sind nicht zum direkten Wiederimport freigegeben. Nicht unterstütztes Leeren, Stilllegen, "
    "Aussetzen und Ändern von Vorlagen bleibt gesperrt; es erfolgen keine Schreibzugriffe auf Anki."
)
MANUAL_ACTION_DE = (
    "Eine Installation wird nach dem Export nur vermutet, nicht beobachtet. In einer Sicherung/Testsammlung "
    "diese Notiz-/Kartenidentität suchen und das tatsächliche Ziel prüfen. Veraltete Aussagen und Status "
    "manuell prüfen, entfernen/aktualisieren oder die betroffene Karte gegebenenfalls aussetzen. Benutzerfelder, "
    "alle nicht verwalteten Tags einschließlich Groß-/Kleinschreibung, Geschwisterinhalte, Verlauf und Planung "
    "erhalten. Die Lösung protokollieren und den Erhalt nach manueller Anwendung prüfen. Eine fehlende CSV-Zeile "
    "entfernt keine Aussagen, ändert keine Statusanzeigen und setzt/stilllegt keine Karten. Nichts wurde angewendet."
)


def _commitment(card: CuratedCard) -> ExportCard | None:
    if card.reason != "accepted":
        return None
    slot = next(t.slot for t in CURATED_LAYOUT if t.slot.recipe == card.recipe)
    return ExportCard(
        note_id=card.note_id,
        card_key=card.card_key,
        decision_ids=tuple(dict.fromkeys(identity for r in card.requirements for identity in r.decision_ids)),
        assertion_ids=(
            "rule",
            "rendered_card",
            *(f"mnemonic:{r.subject_key}" for r in card.requirements if r.subject == "mnemonic"),
        ),
        rendered_sha256=card.fingerprint,
        acceptance="human",
        owned_fields=("LatinitasID", "CardKey", slot.enabled_field, slot.prompt_field, slot.answer_field),
        owned_tags=(),
    )


def export_curated(
    root: Path,
    output: Path,
    load: Loader,
    settings: Settings,
    *,
    after_receipt: str | None = None,
    reconcile_receipt: str | None = None,
) -> dict[str, object]:
    if root.is_symlink() or output.is_symlink() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("choose an output outside the nonsymlink receipt store")
    root.mkdir(exist_ok=True)
    with run_lock(root):
        return _export(root, output, load, settings, after_receipt, reconcile_receipt)


def _export(
    root: Path, output: Path, load: Loader, settings: Settings, after_receipt: str | None, reconcile_receipt: str | None
) -> dict[str, object]:
    data, plan, history = load()
    cards, skipped = render_cards(data, plan, history, settings)
    # Remember every review seen by this receipt store, including rejections and
    # supersession history. An older approved file cannot erase a seen rejection.
    history_dir = root / "curated-history"
    if history_dir.is_symlink():
        raise ValueError("symlink curated history")
    history_dir.mkdir(exist_ok=True)
    ids = {record.approval_id for record in history}
    for previous in history_dir.glob("*.json"):
        if previous.is_symlink() or not {r.approval_id for r in load_approvals(previous)} <= ids:
            raise ValueError("approval history truncated; retain every previously seen decision")
    final_data, final_plan, final_history = load()
    if (data, plan, history) != (final_data, final_plan, final_history):
        raise ValueError("curated input or approval changed before export")
    history_path = history_dir / (digest(history) + ".json")
    if history_path.is_symlink() or (history_path.exists() and history_path.read_bytes() != canonical(history)):
        raise ValueError("curated history conflict")
    # Retain decisions even on replay. Atomic installation keeps interrupted
    # writes out of the authoritative history glob without discarding rejections.
    if not history_path.exists():
        _atomic(history_path, canonical(history))
    rendered_items = {card.item_key for card in cards}
    snapshot = digest(
        {
            "cards": [card.report() for card in cards],
            "source_tags": {item.key: item.tags for item in data.items if item.key in rendered_items},
            "skipped": skipped,
            "configuration": settings.snapshot,
            "scope": data.source_scope,
        }
    )
    batches = Batches(root, digest((str(root.resolve()), "curated", data.source_scope)))
    batches.recover(snapshot)
    previous_receipts = []
    for identity in sorted(set(batches.index.requests.values())):
        batch = batches.read(identity)
        previous_record = ExportReceipt.read_json(canonical(batch.result["export_receipt"]))
        if previous_record.run_id != batches.run_id or previous_record.files != (
            ExportFile(path=batch.output, sha256=batch.output_sha256),
        ):
            raise ValueError("curated receipt publication mismatch")
        previous_receipts.append(previous_record)
    current = {CardRef(note_id=c.note_id, card_key=c.card_key): Assessment(_commitment(c), c.reason) for c in cards}
    handoffs = invalidated_cards(previous_receipts, current)
    if settings.language == "de":
        handoffs = tuple(h.model_copy(update={"required_action": MANUAL_ACTION_DE}) for h in handoffs)
    blocked_notes = {h.note_id for h in handoffs}
    limit = settings.profile.selection.output_limit if settings.profile else None
    request_id, replay, emitted = batches.request(snapshot, limit, TEMPLATE_SHA256, after_receipt, reconcile_receipt)
    if replay is not None:
        prior = ExportReceipt.read_json(canonical(replay.result["export_receipt"]))
        if prior.ineligible != handoffs or any(c.note_id in blocked_notes for c in prior.cards):
            raise ValueError("installed content changed; explicit reconciliation required")
        return batches.result(replay)
    if output.exists():
        raise ValueError("choose a new output; existing files are never overwritten")
    eligible = [c for c in cards if c.reason == "accepted" and c.note_id not in blocked_notes]

    def ref(card: CuratedCard) -> CardRef:
        return CardRef(note_id=card.note_id, card_key=card.card_key)

    remaining = [c for c in eligible if ref(c) not in emitted]
    chosen = remaining[:limit] if limit is not None else remaining
    touched = {c.note_id for c in chosen}
    carried = [c for c in eligible if ref(c) in emitted and c.note_id in touched]
    notes: dict[str, dict[str, str]] = {}
    item_tags = {item.key: " ".join(item.tags) for item in data.items}
    for card in (*chosen, *carried):
        fields = notes.setdefault(card.note_id, dict.fromkeys(CURATED_FIELDS, ""))
        fields["Tags"] = item_tags[card.item_key]
        fields["LatinitasID"] = card.note_id
        fields["CardKey"] = " ".join(filter(None, (fields["CardKey"], card.card_key)))
        slot = next(t.slot for t in CURATED_LAYOUT if t.slot.recipe == card.recipe)
        fields[slot.enabled_field] = "1"
        fields[slot.prompt_field] = card.front_html
        fields[slot.answer_field] = card.back_html
    payload = serialize_anki_csv(
        "Latinitas Curated v1",
        "Latinitas Curated",
        (*CURATED_FIELDS, "Tags"),
        (tuple(row[field] for field in (*CURATED_FIELDS, "Tags")) for row in notes.values()),
    )
    output_sha256 = hashlib.sha256(payload).hexdigest()
    raw = dict(
        format="latinitas-export-receipt",
        version=1,
        run_id=batches.run_id,
        previous_receipt_id=None,
        selection_sha256=snapshot,
        layout_sha256=TEMPLATE_SHA256,
        files=(ExportFile(path=str(output.resolve()), sha256=output_sha256),),
        cards=tuple(_commitment(c) for c in (*chosen, *carried)),
        ineligible=handoffs,
        batch_cursor=None,
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    previous_id = after_receipt or reconcile_receipt
    if previous_id is not None:
        raw["previous_receipt_id"] = ExportReceipt.read_json(
            canonical(batches.read(previous_id).result["export_receipt"])
        ).receipt_id
    export_receipt = ExportReceipt.model_validate({**raw, "receipt_id": digest(raw)})
    result: dict[str, object] = {
        "exported": len(chosen),
        "notes": len(notes),
        "effective_configuration": settings.snapshot,
        "withheld": [c.report() for c in cards if c.reason != "accepted"],
        "skipped": skipped,
        "export_receipt": export_receipt.model_dump(mode="json"),
        "unresolved_note_rows": sorted(blocked_notes),
        "accepted_unselected": [ref(c).model_dump() for c in remaining[len(chosen) :]],
        "application": {
            "state": "not_applied",
            "resolution": "manual",
            "required_action": IMPORT_ACTION_DE if settings.language == "de" else IMPORT_ACTION,
        },
        "summary": "CSV-Übergabe; keine Installation oder Aktualisierung in Anki."
        if settings.language == "de"
        else "CSV handoff; no Anki installation or update.",
    }
    receipt = Receipt(
        format="latinitas-selection-receipt",
        version=1,
        run_id=batches.run_id,
        request_id=request_id,
        selection_sha256=snapshot,
        previous_receipt_id=after_receipt,
        reconciled_receipt_id=reconcile_receipt,
        ordered_card_refs_sha256=digest([ref(c) for c in eligible]),
        emitted_cards=tuple(dict.fromkeys((*emitted, *(ref(c) for c in chosen)))),
        output=str(output.resolve()),
        output_sha256=output_sha256,
        result=result,
    )
    final_data, final_plan, final_history = load()
    if (data, plan, history) != (final_data, final_plan, final_history):
        raise ValueError("curated input or approval changed before export")
    return batches.publish(receipt, payload)
