"""Asymmetric offline three-way fixtures, not native transport proof."""

import pytest

from latinitas_cards.destination_reconciliation import reconcile_values
from latinitas_cards.destination_state import ReconciliationRequired


def test_fields_three_way_and_explicit_resolutions() -> None:
    baseline = {"Meaning": "old", "Lemma": "amo", "Generator": "old-tool"}
    destination = {"Meaning": "local", "Lemma": "amo", "Generator": "new-tool", "Personal Notes": "secret"}
    proposal = {"Meaning": "old", "Lemma": "amare", "Generator": "new-tool", "Personal Notes": "overwrite"}
    result = reconcile_values(baseline, destination, proposal, [], [], {}, {})
    assert result["conflicts"] == ["field:Meaning"]
    assert result["writes"] == {"Lemma": "amare"}
    assert result["fields"] == {"Meaning": "local", "Lemma": "amare", "Generator": "new-tool"}
    for action, value in (("keep_destination", "local"), ("accept_proposal", "old"), ("replacement", "reviewed")):
        decision = {"action": action, "approval": "review-1"}
        if action == "replacement":
            decision["value"] = "reviewed"
        resolved = reconcile_values(baseline, destination, proposal, [], [], {}, {}, {"field:Meaning": decision})
        assert resolved["conflicts"] == []
        assert resolved["fields"]["Meaning"] == value
        assert resolved["decisions"]["field:Meaning"]["result"] == value
    with pytest.raises(ReconciliationRequired, match="decision"):
        reconcile_values(baseline, destination, proposal, [], [], {}, {}, {"field:Personal Notes": decision})


def test_tag_origins_removal_and_destination_additions() -> None:
    result = reconcile_values(
        {},
        {},
        {},
        ["both", "gone", "life"],
        ["both", "gone", "life", "managed:user"],
        {"source_tags": ["both", "gone"], "configured_tags": ["both"], "lifecycle_tags": ["life"]},
        {"source_tags": ["added"], "configured_tags": ["both"], "lifecycle_tags": ["life"]},
    )
    assert result["tags"] == ["added", "both", "life", "managed:user"]
    assert result["removals"] == ["gone"]
    assert result["conflicts"] == []
    assert result["ownership"] == {
        "source_tags": ["added"],
        "configured_tags": ["both"],
        "lifecycle_tags": ["life"],
        "keep_tags": ["managed:user"],
        "keep_fields": [],
    }


def test_deleted_required_tag_keep_override_and_noop() -> None:
    origins = {"source_tags": ["required", "remove"]}
    proposal = {"source_tags": ["required"]}
    result = reconcile_values({}, {}, {}, ["required", "remove"], ["remove"], origins, proposal)
    assert result["conflicts"] == ["tag:required"]
    assert result["tags"] == []
    resolved = reconcile_values(
        {},
        {},
        {},
        ["required", "remove"],
        ["remove"],
        origins,
        proposal,
        {
            "tag:required": {"action": "keep_destination", "approval": "review"},
            "tag:remove": {"action": "keep_as_user_owned", "approval": "review"},
        },
    )
    assert resolved["tags"] == ["remove"]
    assert resolved["ownership"]["keep_tags"] == ["remove"]
    assert resolved["ownership"]["source_tags"] == ["required"]
    assert resolved["ownership"]["suppressed_tags"] == ["required"]
    again = reconcile_values({}, {}, {}, resolved["tags"], resolved["tags"], resolved["ownership"], proposal)
    assert again["tags"] == ["remove"]
    assert again["conflicts"] == []
    assert again["tag_write"] is False


def test_unknown_ownership_and_reserved_collision() -> None:
    with pytest.raises(ReconciliationRequired, match="ownership"):
        reconcile_values({}, {}, {}, ["unknown"], ["unknown"], {}, {})
    with pytest.raises(ReconciliationRequired, match="baseline"):
        reconcile_values(None, {}, {}, [], [], {}, {})
    result = reconcile_values({}, {}, {}, [], ["retired"], {}, {"lifecycle_tags": ["retired"]})
    assert result["conflicts"] == ["tag:retired"]
    resolved = reconcile_values(
        {},
        {},
        {},
        [],
        ["retired"],
        {},
        {"lifecycle_tags": ["retired"]},
        {"tag:retired": {"action": "keep_as_user_owned", "approval": "review"}},
    )
    assert resolved["ownership"]["lifecycle_tags"] == []
    assert resolved["ownership"]["keep_tags"] == ["retired"]
    assert resolved["tags"] == ["retired"]
    assert resolved["tag_write"] is False


def test_readded_suppressed_tag_becomes_user_owned() -> None:
    result = reconcile_values(
        {},
        {},
        {},
        [],
        ["required"],
        {"source_tags": ["required"], "suppressed_tags": ["required"]},
        {"source_tags": ["required"]},
    )
    assert result["tags"] == ["required"]
    assert result["ownership"]["source_tags"] == []
    assert result["ownership"]["keep_tags"] == ["required"]
    assert "suppressed_tags" not in result["ownership"]


@pytest.mark.parametrize("remaining", ["source_tags", "configured_tags", "lifecycle_tags"])
def test_overlap_retains_exact_tag_set(remaining: str) -> None:
    origins = {"source_tags": ["shared"], "configured_tags": ["shared"], "lifecycle_tags": ["shared"]}
    result = reconcile_values({}, {}, {}, ["shared"], ["shared", "user"], origins, {remaining: ["shared"]})
    assert result["tags"] == ["shared", "user"]
    assert result["removals"] == []
    assert result["ownership"][remaining] == ["shared"]
    assert result["tag_write"] is False


@pytest.mark.parametrize("action,value", [("accept_proposal", None), ("replacement", True)])
def test_deleted_required_tag_can_be_explicitly_restored(action: str, value: bool | None) -> None:
    choice: dict[str, object] = {"action": action, "approval": "review"}
    if value is not None:
        choice["value"] = value
    result = reconcile_values(
        {},
        {},
        {},
        ["missing"],
        [],
        {"configured_tags": ["missing"]},
        {"configured_tags": ["missing"]},
        {"tag:missing": choice},
    )
    assert result["tags"] == ["missing"]
    assert result["conflicts"] == []
    assert result["decisions"]["tag:missing"]["result"] is True
    assert result["tag_write"] is True


def test_managed_looking_addition_is_not_claimed_and_user_fields_are_excluded() -> None:
    result = reconcile_values(
        {"Personal Notes": "B", "Custom": "B"},
        {"Personal Notes": "D", "Custom": "D"},
        {"Personal Notes": "P", "Custom": "P"},
        [],
        ["managed:new"],
        {},
        {"configured_tags": ["managed:new"]},
    )
    assert result["fields"] == result["writes"] == {}
    assert result["conflicts"] == []
    assert result["tags"] == ["managed:new"]
    assert result["ownership"]["configured_tags"] == []
    assert result["ownership"]["keep_tags"] == ["managed:new"]


def test_kept_field_is_not_a_write_or_overridable_conflict() -> None:
    result = reconcile_values(
        {"Meaning": "B"},
        {"Meaning": "B"},
        {"Meaning": "P"},
        [],
        [],
        {"keep_fields": ["Meaning"]},
        {},
    )
    assert result["fields"] == {"Meaning": "B"}
    assert result["writes"] == {}
    assert result["conflicts"] == []
    with pytest.raises(ReconciliationRequired, match="cannot be overwritten"):
        reconcile_values(
            {"Meaning": "B"},
            {"Meaning": "D"},
            {"Meaning": "P"},
            [],
            [],
            {"keep_fields": ["Meaning"]},
            {},
            {"field:Meaning": {"action": "accept_proposal", "approval": "review"}},
        )


def test_explicit_accept_proposal_can_review_tag_ownership_override() -> None:
    result = reconcile_values(
        {},
        {},
        {},
        ["reserved"],
        ["reserved"],
        {"keep_tags": ["reserved"]},
        {"lifecycle_tags": ["reserved"]},
        {"tag:reserved": {"action": "accept_proposal", "approval": "review"}},
    )
    assert result["tags"] == ["reserved"]
    assert result["ownership"]["lifecycle_tags"] == ["reserved"]
    assert result["ownership"]["keep_tags"] == []
    assert result["decisions"]["tag:reserved"]["result"] is True
    assert result["tag_write"] is False
