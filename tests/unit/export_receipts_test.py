"""Ledger controls only; curated authority checking belongs to its consumer."""

import pytest

from latinitas_cards.acceptance_records import canonical, digest
from latinitas_cards.export_receipts import Assessment, ExportCard, ExportReceipt, invalidated_cards
from latinitas_cards.pipeline_batches import CardRef


def receipt(cards: tuple[ExportCard, ...]) -> ExportReceipt:
    raw = dict(
        format="latinitas-export-receipt",
        version=1,
        run_id="synthetic",
        previous_receipt_id=None,
        selection_sha256=digest("selection"),
        layout_sha256=digest("layout"),
        files=[],
        cards=cards,
        ineligible=[],
        batch_cursor=None,
        created_at="2026-11-19T00:00:00Z",
    )
    return ExportReceipt.read_json(canonical({**raw, "receipt_id": digest(raw)}))


@pytest.mark.parametrize(
    "reason", ["curated_approval_withdrawn", "curated_rule_changed", "curated_presentation_changed"]
)
def test_curated_adapter_handoff_keeps_prior_authorities_and_distinct_card_refs(reason: str) -> None:
    old = ExportCard(
        note_id="curated:rule-a",
        card_key="rule-only",
        decision_ids=(digest("approved rule"), digest("approved rendering")),
        assertion_ids=("rule-a",),
        rendered_sha256=digest("approved displayed wording"),
        acceptance="human",
        owned_fields=("Rule",),
        owned_tags=(),
    )
    other_note = old.model_copy(update={"note_id": "curated:rule-b"})
    sibling = old.model_copy(update={"card_key": "mnemonic"})
    before = receipt((old, other_note, sibling))
    ref = CardRef(note_id=old.note_id, card_key=old.card_key)
    current = {
        ref: Assessment(None, reason),
        CardRef(note_id=other_note.note_id, card_key=other_note.card_key): Assessment(other_note, "accepted"),
        CardRef(note_id=sibling.note_id, card_key=sibling.card_key): Assessment(sibling, "accepted"),
    }
    actions = invalidated_cards((before,), current)
    assert len(actions) == 1
    assert (actions[0].note_id, actions[0].card_key) == (old.note_id, old.card_key)
    assert actions[0].reason == reason
    assert set(actions[0].previous_decision_ids) == set(old.decision_ids)
    assert actions[0].resolution == "manual"
    assert "not observed" in actions[0].required_action
    assert "No action has been applied" in actions[0].required_action
    assert "history and scheduling" in actions[0].required_action
    # A later successful export is not evidence that the old installed item was fixed.
    later = before.model_dump(exclude={"receipt_id"})
    later.update(cards=[], ineligible=actions)
    unresolved = ExportReceipt.read_json(canonical({**later, "receipt_id": digest(later)}))
    assert invalidated_cards((unresolved,), current) == actions
    altered = before.model_dump(mode="json")
    altered["cards"][0]["decision_ids"] = [digest("unapproved replacement")]
    with pytest.raises(ValueError, match="digest mismatch"):
        ExportReceipt.read_json(canonical(altered))


def test_handoff_reasons_and_authorities_survive_history_order_and_repeated_carry() -> None:
    old = ExportCard(
        note_id="curated:rule-a",
        card_key="rule-only",
        decision_ids=(digest("first"),),
        assertion_ids=("rule-a",),
        rendered_sha256=digest("wording"),
        acceptance="human",
        owned_fields=("Rule",),
        owned_tags=(),
    )
    ref = CardRef(note_id=old.note_id, card_key=old.card_key)
    initial = receipt((old,))
    changed = receipt((old.model_copy(update={"decision_ids": (digest("second"),)}),))
    actions = invalidated_cards((initial, changed), {ref: Assessment(None, "curated_approval_withdrawn")})
    raw = initial.model_dump(exclude={"receipt_id"})
    raw.update(cards=[], ineligible=actions)
    withdrawn = ExportReceipt.read_json(canonical({**raw, "receipt_id": digest(raw)}))
    for history in ((initial, withdrawn), (withdrawn, initial)):
        pending = invalidated_cards(history, {})
        assert pending == actions
        assert set(pending[0].previous_decision_ids) == {digest("first"), digest("second")}
        combined = invalidated_cards(history, {ref: Assessment(None, "checker_changed")})
        assert combined[0].reason == canonical(["checker_changed", "curated_approval_withdrawn"]).decode()
        raw.update(ineligible=combined)
        carried = ExportReceipt.read_json(canonical({**raw, "receipt_id": digest(raw)}))
        assert invalidated_cards((carried, *history), {ref: Assessment(None, "checker_changed")}) == combined


@pytest.mark.parametrize("location", ["cards", "ineligible", "cursor"])
@pytest.mark.parametrize("field", ["note_id", "card_key"])
@pytest.mark.parametrize("value", ["", "   "])
def test_export_receipt_rejects_empty_semantic_identity(location: str, field: str, value: str) -> None:
    old = ExportCard(
        note_id="note",
        card_key="slot",
        decision_ids=(digest("authority"),),
        assertion_ids=("claim",),
        rendered_sha256=digest("wording"),
        acceptance="human",
        owned_fields=("Rule",),
        owned_tags=(),
    )
    raw = receipt((old,)).model_dump(mode="json", exclude={"receipt_id"})
    if location == "cards":
        raw["cards"][0][field] = value
    elif location == "ineligible":
        action = invalidated_cards((receipt((old,)),), {})[0].model_dump(mode="json")
        action[field] = value
        raw["ineligible"] = [action]
    else:
        raw["batch_cursor"] = dict(
            selection_sha256=digest("selection"),
            ordered_card_refs_sha256=digest("order"),
            emitted_cards=[{"note_id": "note", "card_key": "slot", field: value}],
        )
    with pytest.raises(ValueError):
        ExportReceipt.read_json(canonical({**raw, "receipt_id": digest(raw)}))
