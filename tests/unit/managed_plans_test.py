"""Sanitized offline plans; no fixture constitutes native import proof."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from card_lifecycle_test import capture, fixture, generated
from typer.testing import CliRunner

from latinitas_cards.cli import app
from latinitas_cards.destination_state import DestinationSnapshot, ReconciliationRequired, _digest, _encoded, adopt
from latinitas_cards.managed_plans import approve_plan, compose_plan, verify_approval


def request(snapshot: DestinationSnapshot, *, meaning: str = "new", lemma: str = "new lemma") -> dict[str, Any]:
    fields = snapshot.payload["notes"][0]["fields"].copy()
    fields.update(Meaning=meaning, Lemma=lemma)
    return {"identity": fields["LatinitasID"], "fields": fields, "contributions": {}}


def test_deterministic_subset_uses_destination_values_and_full_tags() -> None:
    old = generated()
    sibling = generated(sense="other")
    snapshot, state = fixture(old, sibling)
    data = snapshot.payload
    data["notes"][0]["fields"]["Lemma"] = "destination B"
    data["notes"][0]["tags"] = ["user-only"]
    snapshot = capture(data)
    proposed = request(snapshot)
    other_fields = snapshot.payload["notes"][1]["fields"].copy()
    other_fields["Meaning"] = "unapproved other note"
    other = {"identity": other_fields["LatinitasID"], "fields": other_fields}
    plan = compose_plan(snapshot, state, [proposed, other], {"theme": "muted"})
    assert plan == compose_plan(snapshot, state, [other, proposed], {"theme": "muted"})
    note = plan["notes"][0]
    assert note["classification"] == "conflict"
    operation = next(op["id"] for op in note["operations"] if op.get("field") == "Meaning")
    approval = approve_plan(plan, [operation], "explicit apply review")
    target = approval["targets"][proposed["identity"]]
    assert target["fields"]["Meaning"] == "new"
    assert target["fields"]["Lemma"] == "destination B"
    assert target["tags"] == ["user-only"]
    assert "Personal Notes" not in target["fields"]
    assert len(approval["targets"]) == 1
    columns = approval["import_columns"]
    row = approval["import_rows"][0]
    assert row[columns.index("Meaning")] == "new"
    assert row[columns.index("Lemma")] == "destination B"
    assert row[columns.index("Tags")] == "user-only"
    assert "Personal Notes" not in columns
    assert len(approval["import_rows"]) == 1
    verify_approval(plan, approval, snapshot, state)
    changed = deepcopy(plan)
    changed["effective_profile"]["theme"] = "monochrome"
    with pytest.raises(ReconciliationRequired, match="changed|identity"):
        verify_approval(changed, approval, snapshot, state)
    data["notes"][0]["tags"].append("later")
    with pytest.raises(ReconciliationRequired, match="stale"):
        verify_approval(plan, approval, capture(data), state)


def test_noop_and_convergence_have_no_writes() -> None:
    snapshot, state = fixture(generated())
    proposal = request(snapshot, meaning="old", lemma="amo")
    plan = compose_plan(snapshot, state, [proposal], {})
    assert plan["notes"][0]["classification"] == "unchanged"
    assert not plan["notes"][0]["operations"]
    assert approve_plan(plan, [], "review")["targets"] == {}
    data = snapshot.payload
    data["notes"][0]["fields"]["Meaning"] = "convergent"
    snapshot = capture(data)
    proposal = request(snapshot, meaning="convergent", lemma="amo")
    assert compose_plan(snapshot, state, [proposal], {})["notes"][0]["classification"] == "unchanged"


def test_anchored_absence_is_conflict_but_never_anchored_absence_is_create() -> None:
    snapshot, state = fixture(generated())
    proposal = request(snapshot)
    data = snapshot.payload
    data["notes"] = []
    data["expected_count"] = 0
    empty = capture(data)
    before = deepcopy(state)
    entry = compose_plan(empty, state, [proposal], {})["notes"][0]
    assert entry["classification"] == "conflict"
    assert entry["reasons"] == ["previously anchored note absent; destination reconciliation required"]
    assert entry["blocked"] == entry["reasons"]
    assert entry["operations"] == entry["card_effects"] == []
    assert state == before
    never_anchored = deepcopy(state)
    never_anchored["anchors"] = {}
    assert compose_plan(empty, never_anchored, [proposal], {})["notes"][0]["classification"] == "create"
    data["complete"] = False
    with pytest.raises(ReconciliationRequired, match="incomplete"):
        capture(data)


def test_retire_keeps_underlying_field_and_tag_conflicts_visible() -> None:
    snapshot, state = fixture(generated())
    identity = generated().latinitas_id
    state["anchors"][identity]["configured_tags"] = ["managed"]
    state["anchors"][identity]["tags"] = ["managed"]
    data = snapshot.payload
    data["notes"][0]["fields"]["Meaning"] = "user meaning"
    snapshot = capture(data)
    proposal = request(snapshot)
    proposal.update(retire=True, contributions={"configured_tags": ["managed"]})
    entry = compose_plan(snapshot, state, [proposal], {})["notes"][0]
    assert entry["classification"] == "retire"
    assert "field:Meaning" in entry["reasons"]
    assert "tag:managed" in entry["reasons"]
    assert all(not operation["supported"] for operation in entry["operations"])


def test_card_reactivation_is_inspectable_but_refused_at_approval() -> None:
    old = generated()
    snapshot, state = fixture(old)
    data = snapshot.payload
    row = data["notes"][0]["cards"]["rows"][1]
    row.update(
        suspended=True,
        retirement={
            "approval": "retire-review",
            "pre_suspended": False,
            "tool_suspended": True,
            "observed": snapshot.reference,
        },
    )
    snapshot = capture(data)
    state = adopt(snapshot, {old.latinitas_id: state["anchors"][old.latinitas_id]}, "review")
    before = deepcopy(state)
    entry = compose_plan(snapshot, state, [request(snapshot, meaning="old", lemma="amo")], {})
    note = entry["notes"][0]
    effect = note["card_effects"][1]
    assert (effect["effect"], effect["card_id"], effect["unsuspend"]) == ("reactivate", "card-love-7", True)
    operation = next(op for op in note["operations"] if op["kind"] == "reactivate")
    assert operation["id"] == f"{old.latinitas_id}/card/principal_part_recognition:perfect_1s"
    assert not operation["supported"]
    with pytest.raises(ReconciliationRequired, match="unsupported"):
        approve_plan(entry, [operation["id"]], "review")
    assert state == before
    assert snapshot.payload == data


def test_missing_baseline_and_create_retire_are_inspectable_not_applicable() -> None:
    snapshot, state = fixture(generated())
    proposal = request(snapshot)
    plan = compose_plan(snapshot, None, [proposal], {})
    assert plan["notes"][0]["classification"] == "conflict"
    assert "baseline" in " ".join(plan["notes"][0]["reasons"])
    data = snapshot.payload
    data["notes"] = []
    data["expected_count"] = 0
    empty = capture(data)
    plan = compose_plan(empty, None, [proposal], {})
    assert plan["notes"][0]["classification"] == "create"
    with pytest.raises(ReconciliationRequired, match="unsupported"):
        approve_plan(plan, [plan["notes"][0]["operations"][0]["id"]], "review")
    proposal["retire"] = True
    plan = compose_plan(snapshot, state, [proposal], {})
    assert plan["notes"][0]["classification"] == "retire"
    assert any(effect["effect"] == "retire" for effect in plan["notes"][0]["card_effects"])
    assert plan["notes"][0]["proposed_tags"]["lifecycle_tags"] == ["latinitas::retired"]
    operation = next(op for op in plan["notes"][0]["operations"] if op["kind"] == "tags")
    assert "latinitas::retired" in operation["value"]
    assert not operation["supported"]
    with pytest.raises(ReconciliationRequired, match="unsupported"):
        approve_plan(plan, [operation["id"]], "review")


@pytest.mark.parametrize("mutation", ["wrong", "duplicate", "incomplete", "stale"])
def test_cli_rejects_unsafe_snapshot(tmp_path: Path, mutation: str) -> None:
    snapshot, state = fixture(generated())
    data = snapshot.payload
    binding = state["binding"]
    if mutation == "wrong":
        data["destination"] = "elsewhere"
    elif mutation == "duplicate":
        data["notes"].append(deepcopy(data["notes"][0]))
        data["expected_count"] = 2
    else:
        data["complete" if mutation == "incomplete" else "fresh"] = False
    payload = {"binding": binding, "snapshot": data, "baseline": state, "proposals": [], "effective_profile": {}}
    source = tmp_path / "request.json"
    source.write_text(json.dumps(payload))
    result = CliRunner().invoke(app, ["managed", "plan", str(source)])
    assert result.exit_code == 1, result.output
    assert "error" in result.output.lower()


def test_cli_review_and_explicit_approval(tmp_path: Path) -> None:
    snapshot, state = fixture(generated())
    payload = {
        "binding": state["binding"],
        "snapshot": snapshot.payload,
        "baseline": state,
        "proposals": [request(snapshot, lemma="amo")],
        "effective_profile": {"presentation_version": 1},
    }
    source = tmp_path / "request.json"
    source.write_text(json.dumps(payload))
    runner = CliRunner()
    result = runner.invoke(app, ["managed", "plan", str(source)])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["notes"][0]["classification"] == "update"
    operation = plan["notes"][0]["operations"][0]["id"]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(result.output)
    result = runner.invoke(app, ["managed", "approve", str(plan_path), "--operation", operation])
    assert result.exit_code != 0
    result = runner.invoke(
        app, ["managed", "approve", str(plan_path), "--operation", operation, "--review", "explicit apply review"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["selected_operations"] == [operation]


def test_cli_unicode_review_is_escaped_without_changing_canonical_data(tmp_path: Path) -> None:
    field_controls = "\u2028\u2029\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
    controls = "".join(chr(code) for code in range(0x80, 0xA0)) + field_controls
    proposed = "proposed ā Ω " + field_controls + " end A"
    destination = "destination é " + field_controls[::-1] + " end B"
    profile = {"theme": "profile 漢 " + controls + " end C"}
    review = "review ü " + controls[::-1] + " end D"
    snapshot, state = fixture(generated())
    data = snapshot.payload
    data["notes"][0]["fields"]["Lemma"] = destination
    snapshot = capture(data)
    proposal = request(snapshot, meaning=proposed, lemma=destination)
    expected_plan = compose_plan(snapshot, state, [proposal], profile)
    canonical = json.dumps(expected_plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert _encoded(expected_plan) == canonical
    assert _digest(expected_plan) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    source = tmp_path / "request.json"
    source.write_text(
        json.dumps(
            {
                "binding": state["binding"],
                "snapshot": snapshot.payload,
                "baseline": state,
                "proposals": [proposal],
                "effective_profile": profile,
            }
        )
    )
    runner = CliRunner()
    result = runner.invoke(app, ["managed", "plan", str(source)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == expected_plan
    assert all(char.encode("utf-8") not in result.stdout_bytes for char in controls)
    assert result.stdout_bytes == (json.dumps(expected_plan) + "\n").encode("ascii")
    source.write_text(result.stdout)
    operation = next(op["id"] for op in expected_plan["notes"][0]["operations"] if op.get("field") == "Meaning")
    result = runner.invoke(app, ["managed", "approve", str(source), "--operation", operation, "--review", review])
    assert result.exit_code == 0, result.output
    approval = json.loads(result.stdout)
    expected_approval = approve_plan(expected_plan, [operation], review)
    assert approval == expected_approval
    assert approval["review"] == review
    assert approval["targets"][proposal["identity"]]["fields"]["Meaning"] == proposed
    assert approval["targets"][proposal["identity"]]["fields"]["Lemma"] == destination
    assert all(char.encode("utf-8") not in result.stdout_bytes for char in controls)
    assert result.stdout_bytes == (json.dumps(expected_approval) + "\n").encode("ascii")
    verify_approval(expected_plan, approval, snapshot, state)


def test_verification_recomputes_capabilities_not_only_an_editable_hash() -> None:
    snapshot, state = fixture(generated())
    proposal = request(snapshot, lemma="amo")
    plan = compose_plan(snapshot, state, [proposal], {})
    plan["notes"][0]["operations"][0]["value"] = "unreviewed replacement"
    del plan["plan_id"]
    plan["plan_id"] = _digest(plan)
    operation = plan["notes"][0]["operations"][0]["id"]
    approval = approve_plan(plan, [operation], "review")
    with pytest.raises(ReconciliationRequired, match="recomputed|derived"):
        verify_approval(plan, approval, snapshot, state)


def test_resolution_and_presentation_require_new_approval() -> None:
    snapshot, state = fixture(generated())
    data = snapshot.payload
    data["notes"][0]["fields"]["Meaning"] = "destination edit"
    snapshot = capture(data)
    proposal = request(snapshot, lemma="amo")
    proposal["decisions"] = {"field:Meaning": {"action": "accept_proposal", "approval": "claim review"}}
    plan = compose_plan(snapshot, state, [proposal], {"theme": "muted"})
    assert plan["notes"][0]["classification"] == "update"
    operation = plan["notes"][0]["operations"][0]["id"]
    approval = approve_plan(plan, [operation], "apply review")
    proposal["decisions"]["field:Meaning"].update(action="replacement", value="replacement")
    changed = compose_plan(snapshot, state, [proposal], {"theme": "muted"})
    with pytest.raises(ReconciliationRequired, match="changed"):
        verify_approval(changed, approval, snapshot, state)
    changed = compose_plan(snapshot, state, [proposal], {"theme": "monochrome"})
    assert changed["plan_id"] != plan["plan_id"]


def test_personal_and_kept_fields_never_become_approved_writes() -> None:
    snapshot, state = fixture(generated())
    proposal = request(snapshot)
    proposal["fields"]["Personal Notes"] = "unowned"
    with pytest.raises(ReconciliationRequired, match="user-owned"):
        compose_plan(snapshot, state, [proposal], {})
    del proposal["fields"]["Personal Notes"]
    state["anchors"][proposal["identity"]]["keep_fields"] = ["Meaning"]
    plan = compose_plan(snapshot, state, [proposal], {})
    assert not any(op.get("field") == "Meaning" for op in plan["notes"][0]["operations"])


def test_card_loss_subset_retains_actual_card_payload_and_blocks_slot_writes() -> None:
    snapshot, state = fixture(generated())
    proposal = request(snapshot, lemma="amo")
    from latinitas_cards.cards import TEMPLATE_REGISTRY

    slot = next(slot for slot in TEMPLATE_REGISTRY if slot.role == "perfect_1s" and slot.recipe.endswith("recognition"))
    for field in (slot.enabled_field, slot.prompt_field, slot.answer_field):
        proposal["fields"][field] = ""
    plan = compose_plan(snapshot, state, [proposal], {})
    note = plan["notes"][0]
    assert any(effect["effect"] == "retire" for effect in note["card_effects"])
    content = next(op["id"] for op in note["operations"] if op.get("field") == "Meaning")
    approval = approve_plan(plan, [content], "content-only review")
    assert approval["targets"][proposal["identity"]]["fields"][slot.enabled_field] == "1"
    verify_approval(plan, approval, snapshot, state)
    structural = next(op["id"] for op in note["operations"] if op["kind"] == "retire")
    with pytest.raises(ReconciliationRequired, match="unsupported"):
        approve_plan(plan, [structural], "review")
    proposal["fields"][slot.enabled_field] = "1"
    proposal["fields"][slot.prompt_field] = snapshot.payload["notes"][0]["fields"][slot.prompt_field]
    proposal["fields"][slot.answer_field] = "changed answer presentation"
    plan = compose_plan(snapshot, state, [proposal], {})
    answer = next(op for op in plan["notes"][0]["operations"] if op.get("field") == slot.answer_field)
    assert answer["supported"] is False


def test_tag_diff_exposes_baseline_removals_and_preserved_destination_tags() -> None:
    snapshot, state = fixture(generated())
    identity = snapshot.payload["notes"][0]["identity"]
    state["anchors"][identity].update(tags=["source-old"], source_tags=["source-old"])
    data = snapshot.payload
    data["notes"][0]["tags"] = ["source-old", "user-only"]
    snapshot = capture(data)
    proposal = request(snapshot, meaning="old", lemma="amo")
    proposal["contributions"] = {"source_tags": ["source-new"]}
    plan = compose_plan(snapshot, state, [proposal], {})
    note = plan["notes"][0]
    assert note["baseline_tags"] == ["source-old"]
    assert note["tag_diff"] == {
        "additions": ["source-new"],
        "removals": ["source-old"],
        "removal_candidates": ["source-old"],
    }
    assert note["resolved_tags"] == ["source-new", "user-only"]
    approval = approve_plan(plan, [note["operations"][0]["id"]], "review")
    assert approval["targets"][identity]["tags"] == ["source-new", "user-only"]


@pytest.mark.parametrize("tag", ["user tag", "user\ttag", "user\x1btag", "user\x7ftag"])
def test_unrepresentable_csv_tag_fails_closed(tag: str) -> None:
    snapshot, state = fixture(generated())
    data = snapshot.payload
    data["notes"][0]["tags"] = [tag]
    snapshot = capture(data)
    with pytest.raises(ReconciliationRequired, match="tag.*whitespace|tag.*control"):
        compose_plan(snapshot, state, [request(snapshot, lemma="amo")], {})
