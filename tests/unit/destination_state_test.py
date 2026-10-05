"""Synthetic offline evidence; no Anki collection or native-import claims."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from latinitas_cards.destination_state import (
    BoundDestination,
    ReconciliationRequired,
    adopt,
    begin_observation,
    load_state,
    observe,
    read_snapshot,
    reconcile,
    review_reconciliation,
    save_state,
    schema_contract,
)
from latinitas_cards.notes import AUTHORITATIVE_NOTE_FIELDS


def test_reconciled_note_targets_observe_origins_and_decisions(evidence: dict[str, Any], tmp_path: Path) -> None:
    from latinitas_cards.destination_reconciliation import reconcile_note

    evidence["notes"][0]["cards"]["rows"].append({"id": "sibling-id-A", "due": 19})
    before = read_snapshot(evidence, bound(evidence))
    state = adopt(before, ownership(), "adopt")
    proposal = dict(before.payload["notes"][0]["fields"])
    proposal["Meaning"] = "new-A"
    result = reconcile_note(
        before,
        state,
        "id-A",
        proposal,
        {"source_tags": ["source"], "lifecycle_tags": ["lifecycle"]},
        {"field:Meaning": {"action": "replacement", "value": "reviewed-A", "approval": "review"}},
    )
    assert result["conflicts"] == []
    assert result["target"]["tags"] == ["lifecycle", "manual", "source"]
    journal = begin_observation(state, before, "plan", {"id-A": result["target"]}, "approve")
    after_data = deepcopy(evidence)
    after_data["snapshot_id"] = "after"
    after_data["notes"][0]["fields"] = result["fields"]
    after_data["notes"][0]["tags"] = result["tags"]
    after = read_snapshot(after_data, bound(evidence))
    observed = observe(journal, "plan", after, "offline report", interval_confirmed=True)
    save_state(tmp_path / "state.json", observed)
    loaded = load_state(tmp_path / "state.json")
    assert loaded["anchors"]["id-A"]["lifecycle_tags"] == ["lifecycle"]
    assert loaded["anchors"]["id-A"]["decisions"]["field:Meaning"]["result"] == "reviewed-A"
    shared_note = loaded["anchors"]["id-A"]
    assert shared_note["cards"]["rows"] == [{"id": "card-id-A", "due": 7}, {"id": "sibling-id-A", "due": 19}]
    assert shared_note["tags"] == ["lifecycle", "manual", "source"]
    assert loaded["anchors"]["id-B"]["tags"] == ["manual", "source"]
    noop = reconcile_note(
        after, loaded, "id-A", result["fields"], {"source_tags": ["source"], "lifecycle_tags": ["lifecycle"]}
    )
    assert noop["writes"] == {}
    assert noop["tag_write"] is False
    assert noop["tags"] == ["lifecycle", "manual", "source"]


@pytest.fixture
def evidence() -> dict[str, Any]:
    notes = []
    for identity, meaning in (("id-A", "old-A"), ("id-B", "old-B")):
        fields = {field.name: "" for field in AUTHORITATIVE_NOTE_FIELDS if field.ownership == "managed"}
        fields.update({"LatinitasID": identity, "Meaning": meaning, "Note Schema": "3"})
        notes.append(
            {
                "identity": identity,
                "source": ["scope", identity],
                "note_type_id": "model-1",
                "guid": f"guid-{identity}",
                "local_id": identity,
                "fields": fields,
                "tags": ["source", "manual"],
                "personal_digest": "synthetic-personal-digest",
                "cards": {"rows": [{"id": f"card-{identity}", "due": 7}], "complete": True},
                "history": {"rows": [{"id": f"log-{identity}", "ease": 2}], "complete": True},
            }
        )
    return {
        "version": 1,
        "snapshot_id": "capture-before",
        "captured_at": "2026-10-05T00:00:00Z",
        "artifact_sha256": "a" * 64,
        "client": "synthetic, not native proof",
        "export_method": "closed synthetic fixture",
        "export_options": {"selection": "whole managed set including retired"},
        "destination": "collection-1",
        "profile": "profile-1",
        "schema": schema_contract("model-1"),
        "managed_set": {"scope": "scope", "members": [["id-A", "scope", "id-A"], ["id-B", "scope", "id-B"]]},
        "selection": "all, including retired",
        "exclusions": [],
        "expected_count": 2,
        "complete": True,
        "fresh": True,
        "level": "collection",
        "deck_options": {"bury": True},
        "notes": notes,
    }


def bound(evidence: dict[str, Any]) -> BoundDestination:
    return BoundDestination("collection-1", "profile-1", evidence["schema"], evidence["managed_set"])


def ownership() -> dict[str, Any]:
    return {
        identity: {"source_tags": ["source"], "configured_tags": [], "keep_tags": ["manual"], "keep_fields": []}
        for identity in ("id-A", "id-B")
    }


def test_fingerprint_is_canonical_but_bound_to_actual_evidence(evidence: dict[str, Any]) -> None:
    first = read_snapshot(evidence, bound(evidence))
    reordered = deepcopy(evidence)
    reordered["notes"].reverse()
    reordered["notes"][0]["tags"].reverse()
    reordered["managed_set"]["members"].reverse()
    reordered["captured_at"] = "2026-10-05T01:00:00Z"
    reordered["artifact_sha256"] = "b" * 64
    assert read_snapshot(reordered, bound(evidence)).fingerprint == first.fingerprint
    reordered["notes"][0]["cards"]["rows"][0]["due"] = 19
    assert read_snapshot(reordered, bound(evidence)).fingerprint != first.fingerprint


@pytest.mark.parametrize("case", ["missing", "duplicate", "destination", "schema", "incomplete", "count", "source"])
def test_untrusted_evidence_never_becomes_empty_or_adopted(evidence: dict[str, Any], case: str) -> None:
    expected = bound(evidence)
    if case == "missing":
        evidence["notes"][0]["identity"] = ""
    elif case == "duplicate":
        evidence["notes"][1] = deepcopy(evidence["notes"][0])
    elif case == "destination":
        evidence["destination"] = "other"
    elif case == "schema":
        evidence["schema"]["version"] = "unknown"
    elif case == "incomplete":
        evidence["complete"] = False
    elif case == "count":
        evidence["expected_count"] = 3
    else:
        evidence["notes"][0]["source"] = ["scope", "other-object"]
    with pytest.raises(ReconciliationRequired):
        read_snapshot(evidence, expected)


def test_absence_requires_complete_evidence_and_baseline(evidence: dict[str, Any]) -> None:
    expected = bound(evidence)
    evidence["notes"] = []
    evidence["expected_count"] = 0
    snapshot = read_snapshot(evidence, expected)
    assert snapshot.presence("id-A") == "absent"
    with pytest.raises(ReconciliationRequired, match="baseline"):
        reconcile(snapshot, None)
    evidence["complete"] = False
    with pytest.raises(ReconciliationRequired):
        read_snapshot(evidence, expected)


def test_explicit_adoption_and_atomic_roundtrip(evidence: dict[str, Any], tmp_path: Path) -> None:
    snapshot = read_snapshot(evidence, bound(evidence))
    with pytest.raises(ReconciliationRequired, match="approval"):
        adopt(snapshot, ownership(), "")
    incomplete = ownership()
    incomplete["id-A"]["keep_tags"] = []
    with pytest.raises(ReconciliationRequired, match="ownership"):
        adopt(snapshot, incomplete, "review-1")
    state = adopt(snapshot, ownership(), "review-1")
    assert state["anchors"]["id-A"]["source_tags"] == ["source"]
    assert state["anchors"]["id-A"]["keep_tags"] == ["manual"]
    assert "Personal Notes" not in state["anchors"]["id-A"]["fields"]
    path = tmp_path / "state.json"
    save_state(path, state)
    assert load_state(path) == state
    assert reconcile(snapshot, load_state(path)) == ()


def operations(evidence: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for note in evidence["notes"]:
        fields = dict(note["fields"], Meaning=f"new-{note['identity'][-1]}")
        result[note["identity"]] = {
            "fields": fields,
            "tags": ["source", "manual", "configured"],
            "source_tags": ["source"],
            "configured_tags": ["configured"],
            "keep_tags": ["manual"],
            "keep_fields": [],
        }
    return result


def test_partial_note_receipts_keep_unresolved_anchor_and_detect_backup(evidence: dict[str, Any]) -> None:
    expected = bound(evidence)
    before = read_snapshot(evidence, expected)
    state = adopt(before, ownership(), "adopt")
    pending = begin_observation(state, before, "plan-1", operations(evidence), "approved-1")
    assert pending["plans"]["plan-1"]["status"] == "pending"
    after = deepcopy(evidence)
    after["snapshot_id"] = "capture-after"
    after["notes"][0]["fields"]["Meaning"] = "new-A"
    after["notes"][0]["tags"].append("configured")
    # Mixed fields/tags within B must not advance any of B's anchor.
    after["notes"][1]["fields"]["Meaning"] = "new-B"
    result = observe(pending, "plan-1", read_snapshot(after, expected), "report-1", interval_confirmed=True)
    assert result["version"] == 2
    assert result["anchors"]["id-A"]["fields"]["Meaning"] == "new-A"
    assert result["anchors"]["id-B"]["fields"]["Meaning"] == "old-B"
    assert result["plans"]["plan-1"]["status"] == "partial"
    assert result["plans"]["plan-1"]["operations"]["id-A"]["status"] == "confirmed"
    assert result["plans"]["plan-1"]["operations"]["id-B"]["status"] == "unresolved"
    assert reconcile(before, result) == ("id-A",)
    with pytest.raises(ReconciliationRequired):
        begin_observation(result, before, "retry", operations(evidence), "new-approval")


def test_import_before_persistence_interruption_reacquires_and_never_replays(
    evidence: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = bound(evidence)
    before = read_snapshot(evidence, expected)
    state = begin_observation(adopt(before, ownership(), "adopt"), before, "plan", operations(evidence), "approved")
    path = tmp_path / "state.json"
    save_state(path, state)
    after = deepcopy(evidence)
    for note in after["notes"]:
        note["fields"]["Meaning"] = f"new-{note['identity'][-1]}"
        note["tags"].append("configured")
    observed = read_snapshot(after, expected)
    confirmed = observe(state, "plan", observed, "report", interval_confirmed=True)
    with monkeypatch.context() as patch:
        patch.setattr("latinitas_cards.destination_state.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk")))
        with pytest.raises(OSError):
            save_state(path, confirmed)
    recovered = load_state(path)
    assert recovered["plans"]["plan"]["status"] == "pending"
    assert reconcile(observed, recovered) == ("id-A", "id-B")
    with pytest.raises(ReconciliationRequired):
        begin_observation(recovered, observed, "blind-retry", operations(evidence), "approval")
    recovered = observe(recovered, "plan", observed, "reacquired-report", interval_confirmed=True)
    save_state(path, recovered)
    assert load_state(path)["plans"]["plan"]["status"] == "complete"
    assert observe(recovered, "plan", observed, "again", interval_confirmed=True)["version"] == 2


@pytest.mark.parametrize("case", ["unknown-cards", "changed-history", "personal", "no-interval", "missing-observation"])
def test_preservation_unknown_or_changed_never_confirms(evidence: dict[str, Any], case: str) -> None:
    expected = bound(evidence)
    before = read_snapshot(evidence, expected)
    state = begin_observation(adopt(before, ownership(), "adopt"), before, "plan", operations(evidence), "approved")
    after = deepcopy(evidence)
    for note in after["notes"]:
        note["fields"]["Meaning"] = f"new-{note['identity'][-1]}"
        note["tags"].append("configured")
    if case == "unknown-cards":
        after["notes"][0]["cards"] = None
    elif case == "changed-history":
        after["notes"][0]["history"]["rows"][0]["ease"] = 4
    elif case == "personal":
        after["notes"][0]["personal_digest"] = "changed"
    result = observe(
        state,
        "plan",
        None if case == "missing-observation" else read_snapshot(after, expected),
        "report",
        interval_confirmed=case != "no-interval",
    )
    assert result["anchors"]["id-A"]["fields"]["Meaning"] == "old-A"
    assert result["plans"]["plan"]["status"] != "complete"


def test_note_only_is_not_collection_proof(evidence: dict[str, Any]) -> None:
    before = read_snapshot(evidence, bound(evidence))
    evidence["level"] = "note-only"
    evidence["deck_options"] = None
    for note in evidence["notes"]:
        note["cards"] = None
        note["history"] = None
    snapshot = read_snapshot(evidence, bound(evidence))
    assert snapshot.fingerprint != before.fingerprint
    assert not snapshot.preservation_available
    state = adopt(snapshot, ownership(), "review")
    with pytest.raises(ReconciliationRequired, match="preservation"):
        begin_observation(state, snapshot, "plan", operations(evidence), "approved")


def test_reviewed_recovery_keeps_receipts_and_allows_new_plan(evidence: dict[str, Any]) -> None:
    expected = bound(evidence)
    before = read_snapshot(evidence, expected)
    pending = begin_observation(adopt(before, ownership(), "adopt"), before, "plan", operations(evidence), "approve")
    after = deepcopy(evidence)
    after["notes"][0]["fields"]["Meaning"] = "new-A"
    after["notes"][0]["tags"].append("configured")
    partial = observe(pending, "plan", read_snapshot(after, expected), "report", interval_confirmed=True)
    # Explicitly accept restored backup values; retain historical successful receipt.
    recovered = review_reconciliation(partial, before, ownership(), "review-restored-backup")
    assert recovered["version"] == 3
    assert recovered["plans"]["plan"]["status"] == "abandoned"
    assert recovered["plans"]["plan"]["operations"]["id-A"]["receipt"] is not None
    assert recovered["anchors"]["id-A"]["fields"]["Meaning"] == "old-A"
    assert reconcile(before, recovered) == ()
    assert begin_observation(recovered, before, "new-plan", operations(evidence), "new-approval")


@pytest.mark.parametrize("case", ["personal-field", "slot-guard", "keep-field", "prior-export"])
def test_ownership_and_structural_boundaries(evidence: dict[str, Any], case: str) -> None:
    before = read_snapshot(evidence, bound(evidence))
    origins = ownership()
    if case == "keep-field":
        origins["id-A"]["keep_fields"] = ["Meaning"]
    state = adopt(before, origins, "review")
    proposed = operations(evidence)
    if case == "personal-field":
        proposed["id-A"]["fields"]["Personal Notes"] = "must never be owned"
    elif case == "slot-guard":
        guard = next(name for name in proposed["id-A"]["fields"] if name.endswith("Enabled"))
        proposed["id-A"]["fields"][guard] = "1"
    elif case == "prior-export":
        state = {"version": 1, "entries": []}
    with pytest.raises(ReconciliationRequired):
        begin_observation(state, before, "plan", proposed, "approved")


def test_collection_options_restore_requires_review(evidence: dict[str, Any]) -> None:
    expected = bound(evidence)
    before = read_snapshot(evidence, expected)
    pending = begin_observation(adopt(before, ownership(), "adopt"), before, "plan", operations(evidence), "approved")
    after = deepcopy(evidence)
    for note in after["notes"]:
        note["fields"]["Meaning"] = f"new-{note['identity'][-1]}"
        note["tags"].append("configured")
    confirmed = observe(pending, "plan", read_snapshot(after, expected), "report", interval_confirmed=True)
    after["deck_options"]["bury"] = False
    restored = read_snapshot(after, expected)
    with pytest.raises(ReconciliationRequired, match="deck options"):
        reconcile(restored, confirmed)
    with pytest.raises(ReconciliationRequired):
        begin_observation(confirmed, restored, "retry", operations(evidence), "new-approval")


def test_inconsistent_whole_plan_status_rejected_at_every_boundary(evidence: dict[str, Any], tmp_path: Path) -> None:
    import json

    before = read_snapshot(evidence, bound(evidence))
    state = begin_observation(adopt(before, ownership(), "adopt"), before, "plan", operations(evidence), "approved")
    state["plans"]["plan"]["status"] = "complete"
    path = tmp_path / "inconsistent.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ReconciliationRequired, match="status"):
        load_state(path)
    with pytest.raises(ReconciliationRequired, match="status"):
        save_state(path, state)
    with pytest.raises(ReconciliationRequired, match="status"):
        begin_observation(state, before, "blind-retry", operations(evidence), "new-approval")


@pytest.mark.parametrize("exclusion", ["id-A", "tag:active", "unknown-query"])
def test_exclusion_is_not_proven_absence(evidence: dict[str, Any], exclusion: str) -> None:
    expected = bound(evidence)
    evidence["notes"] = []
    evidence["expected_count"] = 0
    evidence["exclusions"] = [exclusion]
    with pytest.raises(ReconciliationRequired, match="exclusion"):
        read_snapshot(evidence, expected)


@pytest.mark.parametrize("case", ["incomplete", "stale", "excluded", "malformed"])
def test_direct_snapshot_construction_cannot_bypass_validation(evidence: dict[str, Any], case: str) -> None:
    import json

    from latinitas_cards.destination_state import DestinationSnapshot

    if case == "incomplete":
        evidence["complete"] = False
    elif case == "stale":
        evidence["fresh"] = False
    elif case == "excluded":
        evidence["exclusions"] = ["id-A"]
    with pytest.raises(ReconciliationRequired):
        DestinationSnapshot("not-json" if case == "malformed" else json.dumps(evidence))


def test_empty_attempt_does_not_create_an_inconsistent_pending_journal(evidence: dict[str, Any]) -> None:
    before = read_snapshot(evidence, bound(evidence))
    with pytest.raises(ReconciliationRequired, match="empty"):
        begin_observation(adopt(before, ownership(), "adopt"), before, "empty-plan", {}, "approved")
