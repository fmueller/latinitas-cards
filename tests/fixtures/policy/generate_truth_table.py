"""Independent expected rows for the two fixed policies; no application imports.

Run with Python from any directory. Expectations are predicates specific to the
fixtures, not a second implementation of a general rule interpreter.
"""

import json
from itertools import product
from pathlib import Path


def render() -> str:
    rows = []
    for policy, operation, layer, kind, destructive, conflicting in product(
        (None, "first-match", "deny-precedence"),
        ("add", "supersede", "retract"),
        ("segmentation", "morphology", "translation", "correction"),
        ("rule", "analyzer", "agent", "llm", "human"),
        (False, True),
        (False, True),
    ):
        if operation == "add" and destructive:
            continue
        safe_add = operation == "add" and not conflicting and not destructive
        if policy is None:
            action = "auto" if safe_add else "review"
            reason = "no_policy"
        elif policy == "first-match":
            narrow = operation == "add" and layer == "morphology" and kind == "agent"
            action = "auto" if narrow and safe_add else "review"
            reason = "rule:0" if narrow else "rule:1"
        elif layer == "translation":
            action, reason = "deny", "rule:0"
        else:
            action = "auto" if safe_add else "review"
            reason = "rule:1" if operation == "add" else "default_review"
        reasons = [reason] + (["destructive"] if destructive else []) + (["conflicting"] if conflicting else [])
        rows.append(
            {
                "policy": policy,
                "operation": operation,
                "layer": layer,
                "producer_kind": kind,
                "producer_name": "fixture-one",
                "destructive": destructive,
                "conflicting": conflicting,
                "action": action,
                "reasons": reasons,
            }
        )
    assert len(rows) == 3 * 200
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


if __name__ == "__main__":
    Path(__file__).with_name("truth-table.json").write_text(render(), encoding="utf-8")
