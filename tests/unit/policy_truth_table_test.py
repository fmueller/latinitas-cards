import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from latinitas_cards.policy import load_policy, route

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/policy"
ROWS = json.loads((FIXTURES / "truth-table.json").read_text())


def test_independent_table_is_complete_and_reproducible() -> None:
    assert len(ROWS) == 3 * 200
    assert Counter((r["policy"], r["operation"]) for r in ROWS) == {
        (policy, operation): count
        for policy in (None, "first-match", "deny-precedence")
        for operation, count in (("add", 40), ("supersede", 80), ("retract", 80))
    }
    spec = importlib.util.spec_from_file_location("policy_truth_generator", FIXTURES / "generate_truth_table.py")
    assert spec and spec.loader
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert generator.render() == (FIXTURES / "truth-table.json").read_text()


@pytest.mark.parametrize("row", ROWS)
def test_truth_table(row: dict[str, Any]) -> None:
    policy = load_policy(FIXTURES / f"{row['policy']}.json") if row["policy"] else None
    result = route(
        row["operation"],
        row["layer"],
        row["producer_kind"],
        row["producer_name"],
        row["destructive"],
        row["conflicting"],
        policy,
    )
    assert result.action == row["action"]
    assert list(result.reasons) == row["reasons"]
