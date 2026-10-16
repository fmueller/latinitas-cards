import hashlib
from pathlib import Path

import pytest

from latinitas_cards.policy import Action, Layer, Operation, ProducerKind, load_policy, policy_digest, route

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/policy"


@pytest.mark.parametrize(
    ("policy_name", "operation", "layer", "kind", "name", "destructive", "conflicting", "action", "reasons"),
    [
        (None, "add", "translation", "human", "x", False, False, "auto", ["no_policy"]),
        (None, "supersede", "translation", "human", "x", False, False, "review", ["no_policy"]),
        (None, "retract", "correction", "llm", "x", True, True, "review", ["no_policy", "destructive", "conflicting"]),
        (None, "add", "correction", "llm", "x", False, True, "review", ["no_policy", "conflicting"]),
        ("first-match", "add", "morphology", "agent", "fixture-one", False, False, "auto", ["rule:0"]),
        ("first-match", "add", "morphology", "agent", "Fixture-one", False, False, "review", ["default_review"]),
        ("first-match", "add", "translation", "agent", "fixture-one", False, False, "review", ["rule:1"]),
        ("first-match", "add", "morphology", "human", "fixture-one", False, False, "review", ["rule:1"]),
        ("first-match", "add", "morphology", "agent", "other", False, False, "review", ["default_review"]),
        ("first-match", "add", "morphology", "agent", "fixture-one", False, True, "review", ["rule:0", "conflicting"]),
        (
            "deny-precedence",
            "supersede",
            "translation",
            "human",
            "fixture-one",
            True,
            False,
            "deny",
            ["rule:0", "destructive"],
        ),
        (
            "deny-precedence",
            "add",
            "translation",
            "agent",
            "fixture-one",
            False,
            True,
            "deny",
            ["rule:0", "conflicting"],
        ),
        (
            "deny-precedence",
            "retract",
            "translation",
            "rule",
            "fixture-one",
            True,
            True,
            "deny",
            ["rule:0", "destructive", "conflicting"],
        ),
        (
            "deny-precedence",
            "supersede",
            "correction",
            "rule",
            "fixture-one",
            True,
            True,
            "review",
            ["default_review", "destructive", "conflicting"],
        ),
    ],
)
def test_literal_routes(
    policy_name: str | None,
    operation: Operation,
    layer: Layer,
    kind: ProducerKind,
    name: str,
    destructive: bool,
    conflicting: bool,
    action: Action,
    reasons: list[str],
) -> None:
    policy = load_policy(FIXTURES / f"{policy_name}.json") if policy_name else None
    result = route(operation, layer, kind, name, destructive, conflicting, policy)
    assert result.action == action
    assert list(result.reasons) == reasons


def test_raw_byte_digest(tmp_path: Path) -> None:
    assert policy_digest(None) is None
    path = tmp_path / "policy.json"
    raw = b'{"version": 1}\r\n'
    path.write_bytes(raw)
    assert policy_digest(path) == "sha256:" + hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw + b" ")
    assert policy_digest(path) != "sha256:" + hashlib.sha256(raw).hexdigest()
