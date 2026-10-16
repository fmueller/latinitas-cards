"""Policy document and routing contracts (write authorization, not acceptance)."""

import ast
import inspect
from typing import Any

import pytest
from pydantic import ValidationError

from latinitas_cards import policy
from latinitas_cards.policy import Policy
from latinitas_cards.profile import DeckProfile


def document(**changes: Any) -> dict[str, Any]:
    return {"format": "latinitas-policy", "version": 1, "rules": [], **changes}


@pytest.mark.parametrize("version", [1.0, True, "1", 2, None])
def test_version_is_strict_one(version: Any) -> None:
    with pytest.raises(ValidationError):
        Policy.model_validate(document(version=version))


@pytest.mark.parametrize("field", ["threshold", "min_confidence", "calibration", "unknown"])
@pytest.mark.parametrize("nested", [False, True])
def test_unknown_fields_are_forbidden(field: str, nested: bool) -> None:
    data = document()
    rule = {"operation": "add", "layer": "*", "producer_kind": "*", "producer": "*", "action": "review"}
    if nested:
        rule[field] = "forbidden"
        data["rules"] = [rule]
    else:
        data[field] = "forbidden"
    with pytest.raises(ValidationError):
        Policy.model_validate(data)


@pytest.mark.parametrize("operation", ["supersede", "retract", "*"])
def test_auto_requires_explicit_add(operation: str) -> None:
    with pytest.raises(ValidationError):
        Policy.model_validate(
            document(
                rules=[{"operation": operation, "layer": "*", "producer_kind": "*", "producer": "*", "action": "auto"}]
            )
        )


def test_required_fields_and_receipt_actor() -> None:
    assert Policy.model_validate(document()).receipt_actor == "human"
    assert Policy.model_validate(document(receipt_actor="any")).receipt_actor == "any"
    invalid_changes: list[dict[str, Any]] = [{"receipt_actor": "agent"}, {"format": "other"}, {"rules": [{}]}]
    for changes in invalid_changes:
        with pytest.raises(ValidationError):
            Policy.model_validate(document(**changes))
    rule = {"operation": "add", "layer": "morphology", "producer_kind": "agent", "producer": "*", "action": "auto"}
    for key in rule:
        with pytest.raises(ValidationError):
            Policy.model_validate(document(rules=[{k: v for k, v in rule.items() if k != key}]))


def test_policy_stays_outside_claim_and_profile_contracts() -> None:
    tree = ast.parse(inspect.getsource(policy))
    imported = {
        node.module.split(".")[-1] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {name.name.split(".")[-1] for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names}
    assert imported.isdisjoint({"claim_review", "claim_sidecar", "profile"})
    assert set(DeckProfile.model_fields) == {
        "schema_version",
        "source_identity",
        "note_type",
        "fields",
        "principal_parts",
        "language_tag",
        "generated_note_type",
        "target_deck",
        "tags",
        "selected_recipes",
        "morphology",
    }
