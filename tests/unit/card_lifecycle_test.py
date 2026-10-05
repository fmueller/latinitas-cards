"""Synthetic lifecycle planning: eligibility is not destination card existence."""

from copy import deepcopy
from typing import Any

import pytest

from latinitas_cards.card_lifecycle import plan_card_lifecycle
from latinitas_cards.cards import render_cards
from latinitas_cards.destination_state import (
    BoundDestination,
    ReconciliationRequired,
    adopt,
    begin_observation,
    read_snapshot,
    schema_contract,
)
from latinitas_cards.notes import GeneratedNote, GeneratedNoteProvenance, GenerationMetadata, ManagedNoteContent
from latinitas_cards.principal_parts import ParsedPrincipalParts, PrincipalPartValue

PRESENT = "principal_part_recognition:present_1s"
PERFECT = "principal_part_recognition:perfect_1s"


def generated(
    *,
    perfect: str | None = "amavi",
    recipes: tuple[str, ...] = ("principal_part_recognition",),
    sense: str = "love",
    meaning: str = "old",
    noun: bool = False,
) -> GeneratedNote:
    parts = (
        (PrincipalPartValue("noun_nominative", "amor", "amor"),)
        if noun
        else (
            PrincipalPartValue("present_1s", "amo", "amo"),
            PrincipalPartValue("perfect_1s", perfect, perfect),
        )
    )
    parsed = ParsedPrincipalParts("amo", "amo", parts)
    return GeneratedNote.create(
        source_identity="entry",
        source_scope="scope",
        object_key=sense,
        provenance=GeneratedNoteProvenance("csv", "row 1", source_scope="scope"),
        metadata=GenerationMetadata("profile"),
        content=ManagedNoteContent("amo", "amo, amavi", meaning),
        cards=render_cards(parsed, selected_recipes=recipes),
    )


def fixture(note: GeneratedNote, sibling: GeneratedNote | None = None) -> tuple[Any, dict[str, Any]]:
    notes = [note] if sibling is None else [note, sibling]
    data: dict[str, Any] = {
        "version": 1,
        "snapshot_id": "before",
        "captured_at": "2026-10-05T00:00:00Z",
        "artifact_sha256": "a" * 64,
        "client": "synthetic",
        "export_method": "offline",
        "export_options": {},
        "destination": "collection",
        "profile": "profile",
        "schema": schema_contract("model"),
        "managed_set": {"scope": "scope", "members": [[n.latinitas_id, "scope", "entry", n.object_key] for n in notes]},
        "selection": "all",
        "exclusions": [],
        "expected_count": len(notes),
        "complete": True,
        "fresh": True,
        "level": "collection",
        "deck_options": {"bury": True},
        "notes": [
            {
                "identity": n.latinitas_id,
                "source": ["scope", "entry", n.object_key],
                "note_type_id": "model",
                "guid": f"guid-{n.object_key}",
                "local_id": f"note-{n.object_key}",
                "fields": {k: v for k, v in n.to_anki_fields() if k not in ("Personal Notes", "Tags")},
                "tags": [],
                "personal_digest": "personal",
                "cards": {
                    "complete": True,
                    "rows": [
                        {
                            "id": f"card-{n.object_key}-{c.slot.ordinal}",
                            "semantic_key": c.slot.semantic_key,
                            "ordinal": c.slot.ordinal,
                            "template_name": c.slot.template_name,
                            "suspended": False,
                            "due": 7 + c.slot.ordinal,
                        }
                        for c in n.cards
                        if c.eligible
                    ],
                },
                "history": {"complete": True, "rows": [{"id": f"log-{n.object_key}", "ease": 2}]},
            }
            for n in notes
        ],
    }
    snapshot = capture(data)
    return snapshot, adopt(
        snapshot,
        {
            n.latinitas_id: {
                "source_tags": [],
                "configured_tags": [],
                "keep_tags": [],
                "keep_fields": [],
            }
            for n in notes
        },
        "adopt",
    )


def capture(data: dict[str, Any]) -> Any:
    return read_snapshot(data, BoundDestination("collection", "profile", data["schema"], data["managed_set"]))


def test_sibling_retirement_retains_safe_content_and_binding() -> None:
    old = generated()
    snapshot, state = fixture(old)
    proposal = generated(perfect=None, meaning="new")
    result = plan_card_lifecycle(snapshot, state, old.latinitas_id, proposal, approvals={PERFECT: "review"})
    assert [(e["semantic_key"], e["effect"]) for e in result["effects"]] == [(PRESENT, "retain"), (PERFECT, "retire")]
    effect = result["effects"][1]
    assert effect["card_id"] == "card-love-7"
    assert effect["ordinal"] == 7
    assert effect["pre_suspended"] is False
    assert effect["retention"]["fields"]["RecognitionPerfectAnswer"] == "Perfekt, 1. Person Singular"
    assert effect["retention"]["fields"]["Meaning"] == "old"
    assert effect["retention"]["newly_approved"] is False
    assert effect["retention"]["stale"] is True
    assert effect["suspension_owner"] == "pending_tool"
    assert result["lifecycle_tags"] == []
    assert "target" not in result
    assert result["unsupported"] == [PERFECT]
    assert state["anchors"][old.latinitas_id]["cards"] == snapshot.payload["notes"][0]["cards"]


@pytest.mark.parametrize("status", ["missing", "ambiguous", "withheld", "inapplicable", "parser_failure"])
def test_uncertainty_never_retires_or_adds(status: str) -> None:
    old = generated()
    snapshot, state = fixture(old)
    result = plan_card_lifecycle(snapshot, state, old.latinitas_id, None, proposal_status=status)
    assert [e["effect"] for e in result["effects"]] == ["retain", "retain"]
    assert result["blocked"]
    assert "target" not in result


def test_absent_slot_addition_is_not_a_sibling_content_update() -> None:
    old = generated(perfect=None)
    snapshot, state = fixture(old)
    result = plan_card_lifecycle(snapshot, state, old.latinitas_id, generated(meaning="new"))
    assert [e["effect"] for e in result["effects"]] == ["retain", "add"]
    assert result["effects"][1]["card_id"] is None
    assert result["unsupported"] == [PERFECT]
    assert "target" not in result


def test_enabled_slot_missing_actual_card_blocks_even_unchanged_guards() -> None:
    old = generated()
    snapshot, state = fixture(old)
    data = snapshot.payload
    data["notes"][0]["cards"]["rows"].pop()
    snapshot = capture(data)
    # Reviewed adoption of the actual observed absence must not turn eligibility into existence.
    state = adopt(snapshot, {old.latinitas_id: state["anchors"][old.latinitas_id]}, "review")
    result = plan_card_lifecycle(snapshot, state, old.latinitas_id, generated(meaning="new"))
    assert result["effects"][1]["effect"] == "add"
    assert "target" not in result
    target: dict[str, Any] = {
        "fields": dict(data["notes"][0]["fields"]),
        "tags": [],
        "source_tags": [],
        "configured_tags": [],
        "keep_tags": [],
        "keep_fields": [],
    }
    target["fields"]["Meaning"] = "new"
    with pytest.raises(ReconciliationRequired, match="card set"):
        begin_observation(state, snapshot, "plan", {old.latinitas_id: target}, "approve")


@pytest.mark.parametrize(
    "pre,owned,approved,unsuspend",
    [
        (False, True, True, True),
        (True, False, True, False),
        (False, None, True, False),
        (False, True, False, False),
    ],
)
def test_reactivation_preserves_identity_and_only_reverses_known_tool_suspension(
    pre: bool,
    owned: bool | None,
    approved: bool,
    unsuspend: bool,
) -> None:
    old = generated()
    snapshot, state = fixture(old)
    data = snapshot.payload
    row = data["notes"][0]["cards"]["rows"][1]
    row["suspended"] = True
    row["retirement"] = {
        "approval": "retire-review",
        "pre_suspended": pre,
        "tool_suspended": owned,
        "observed": snapshot.reference,
    }
    snapshot = capture(data)
    state = adopt(snapshot, {old.latinitas_id: state["anchors"][old.latinitas_id]}, "review")
    result = plan_card_lifecycle(
        snapshot, state, old.latinitas_id, old, approvals={PERFECT: "reactivate-review"} if approved else {}
    )
    effect = result["effects"][1]
    assert effect["card_id"] == "card-love-7"
    assert effect["effect"] == ("reactivate" if approved else "retain")
    assert effect["unsuspend"] is unsuspend
    assert result["effects"][0]["effect"] == "retain"
    assert "target" not in result
    if owned is None or not approved:
        assert result["blocked"]


def test_two_senses_equal_lemma_never_merge_and_mutable_content_keeps_ids() -> None:
    first, second = generated(), generated(sense="like")
    snapshot, state = fixture(first, second)
    for note in (first, second):
        result = plan_card_lifecycle(
            snapshot, state, note.latinitas_id, generated(sense=note.object_key, meaning="new")
        )
        assert result["identity"] == note.latinitas_id
        assert [e["card_id"] for e in result["effects"]] == [f"card-{note.object_key}-5", f"card-{note.object_key}-7"]
        assert result["target"]["fields"]["Meaning"] == "new"
    assert first.latinitas_id != second.latinitas_id
    with pytest.raises(ReconciliationRequired, match="identity"):
        plan_card_lifecycle(snapshot, state, first.latinitas_id, second)


def test_noun_no_verb_additions_and_reenable_without_retirement_approval() -> None:
    noun = generated(noun=True)
    snapshot, state = fixture(noun)
    assert plan_card_lifecycle(snapshot, state, noun.latinitas_id, noun)["effects"] == []
    old = generated()
    snapshot, state = fixture(old)
    result = plan_card_lifecycle(snapshot, state, old.latinitas_id, generated(recipes=()))
    assert all(e["effect"] == "retain" for e in result["effects"])
    assert result["blocked"]
    assert "target" not in result


def test_whole_object_retirement_distinct_and_unrepresentable_retention_blocks() -> None:
    old = generated()
    snapshot, state = fixture(old)
    result = plan_card_lifecycle(snapshot, state, old.latinitas_id, None, retire_object="object-review")
    assert result["lifecycle_tags"] == ["latinitas::retired"]
    assert all(e["effect"] == "retire" for e in result["effects"])
    assert "target" not in result
    bad = deepcopy(state)
    bad["anchors"][old.latinitas_id]["fields"]["RecognitionPerfectAnswer"] = ""
    result = plan_card_lifecycle(
        snapshot, bad, old.latinitas_id, generated(perfect=None), approvals={PERFECT: "review"}
    )
    assert any("retention" in reason for reason in result["blocked"])
    assert "target" not in result


def test_whole_retirement_does_not_add_newly_eligible_slots() -> None:
    old = generated(perfect=None)
    snapshot, state = fixture(old)
    result = plan_card_lifecycle(snapshot, state, old.latinitas_id, generated(), retire_object="object-review")
    assert [e["effect"] for e in result["effects"]] == ["retire"]


@pytest.mark.parametrize("change", ["enable", "clear", "tag"])
def test_journal_refuses_indirect_structural_csv_effects(change: str) -> None:
    old = generated(perfect=None)
    snapshot, state = fixture(old)
    target: dict[str, Any] = {
        "fields": dict(snapshot.payload["notes"][0]["fields"]),
        "tags": [],
        "source_tags": [],
        "configured_tags": [],
        "keep_tags": [],
        "keep_fields": [],
    }
    if change == "enable":
        target["fields"].update(
            RecognitionPerfectEnabled="1", RecognitionPerfectPrompt="amavi", RecognitionPerfectAnswer="Perfekt"
        )
    elif change == "clear":
        target["fields"]["RecognitionPresentEnabled"] = ""
    else:
        target["tags"] = target["lifecycle_tags"] = ["latinitas::retired"]
    with pytest.raises(ReconciliationRequired, match="unsupported"):
        begin_observation(state, snapshot, "plan", {old.latinitas_id: target}, "approve")


@pytest.mark.parametrize("legacy_first", [False, True])
def test_mixed_legacy_and_keyed_senses_require_confirmation(legacy_first: bool) -> None:
    first, second = generated(), generated(sense="like")
    snapshot, _ = fixture(first, second)
    data = snapshot.payload
    members = data["managed_set"]["members"]
    members.insert(0 if legacy_first else len(members), ["legacy-id", "scope", "entry"])
    with pytest.raises(ReconciliationRequired, match="ambiguous"):
        capture(data)
