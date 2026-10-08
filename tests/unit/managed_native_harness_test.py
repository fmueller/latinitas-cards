"""Exercise the owning native harness checks without requiring its optional backend."""

import runpy
from copy import deepcopy
from pathlib import Path

from card_lifecycle_test import capture, fixture, generated


def test_harness_distinguishes_anchored_absence_from_unsupported_create() -> None:
    harness = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/check-managed-anki.py"))
    initial, state = fixture(generated(), generated(sense="retained"))
    proposal = harness["propose"](initial, Meaning="cannot create")
    data = initial.payload
    data["notes"] = data["notes"][1:]
    data["expected_count"] = 1
    absent = capture(data)
    before = deepcopy(state)

    result = harness["check_absent_note"](absent, state, proposal)

    anchored = result["anchored"]
    entry = anchored["plan"]["notes"][0]
    assert entry["classification"] == "conflict"
    assert entry["blocked"] == ["previously anchored note absent; destination reconciliation required"]
    assert entry["operations"] == entry["card_effects"] == []
    assert anchored["approval"]["targets"] == {}
    assert anchored["approval"]["selected_operations"] == anchored["approval"]["import_rows"] == []
    never_anchored = result["never_anchored"]
    entry = never_anchored["plan"]["notes"][0]
    assert entry["classification"] == "create"
    assert [op["kind"] for op in entry["operations"]] == ["create"]
    assert never_anchored["selected_operations"] == [f"{proposal['identity']}/create"]
    assert never_anchored["rejection"] == "unsupported or unresolved selected operation"
    assert state == before
