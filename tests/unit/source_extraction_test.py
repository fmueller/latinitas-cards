"""Independent reviewed expectations, never regenerated from the extractor."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from latinitas_cards.principal_parts import PrincipalPartParseSuccess, parse_principal_part_value
from latinitas_cards.profile import PrincipalPartLayout

FIXTURES = [
    json.loads(line) for line in Path("tests/fixtures/source-extraction/reviewed.jsonl").read_text().splitlines()
]
ROLES = ("present_infinitive", "present_1s", "perfect_1s", "perfect_passive_participle")


def reviewed_layout(name: str) -> PrincipalPartLayout:
    roles: tuple[str, ...] = ROLES
    if name.startswith("supine"):
        roles = (*ROLES[:3], "supine")
    if name == "supine-dash":
        roles = ("present_1s", "present_infinitive", "perfect_1s", "supine")
    if name == "short-dash":
        roles = ("present_1s", "present_infinitive", "perfect_1s")
    separator = (
        ";"
        if name.endswith("semicolon")
        else " / "
        if name.endswith("slash")
        else " — "
        if name.endswith("dash")
        else ","
    )
    return PrincipalPartLayout(
        roles=roles,
        separators=(separator,),
        pipe_alternatives=name.endswith("pipe"),
        trailing_poet_hint=name.endswith("poet"),
    )


@pytest.mark.parametrize("fixture", FIXTURES, ids=[case["id"] for case in FIXTURES])
def test_reviewed_exact_extraction(fixture: dict[str, Any]) -> None:
    result = parse_principal_part_value(
        fixture["raw"],
        reviewed_layout(fixture["profile"]),
        lexical_entry="fixture",
        source_identity=fixture["source"],
        source_location="reviewed fixture",
    )
    evidence = result.evidence
    assert evidence is not None
    assert evidence.raw == fixture["raw"]
    assert evidence.status == fixture["status"]
    assert evidence.candidates == (
        None if fixture["candidates"] is None else tuple(tuple(slot) for slot in fixture["candidates"])
    )
    assert [asdict(hint) for hint in evidence.hints] == fixture["hints"]
    assert list(evidence.rules) == fixture["rules"]
    assert evidence.roles == reviewed_layout(fixture["profile"]).roles
    if isinstance(result, PrincipalPartParseSuccess):
        assert result.value.source_identity == fixture["source"]
        assert result.value.source_location == "reviewed fixture"
        assert tuple(part.raw for part in result.value.parts) == tuple(
            fixture["raw"].split(reviewed_layout(fixture["profile"]).separators[0])
        )
    else:
        assert result.source_identity == fixture["source"]
        assert result.source_location == "reviewed fixture"
        assert result.message


@pytest.mark.parametrize(
    "raw", ["dīcere — dīcō — dīxī — dictum", "dīcere; dīcō; dīxī; dictum", "dīcere / dīcō / dīxī / dictum"]
)
def test_confirmed_comma_never_autodetects(raw: str) -> None:
    result = parse_principal_part_value(raw, reviewed_layout("ppp-comma"), lexical_entry="dīcere")
    assert not isinstance(result, PrincipalPartParseSuccess)
    assert result.code == "separator_mismatch"


@pytest.mark.parametrize(
    "raw",
    [
        "amāre||amare, amō, amāvī, amātum",
        "amāre, amō, amāvī<br>poet. extra, amātum",
        "amāre, amō, amāvī<br>poet.<br>poet., amātum",
    ],
)
def test_unreviewed_alternative_and_hint_grammar_withheld(raw: str) -> None:
    layout = reviewed_layout("ppp-comma-pipe" if "|" in raw else "ppp-comma-poet")
    result = parse_principal_part_value(raw, layout, lexical_entry="amāre")
    assert not isinstance(result, PrincipalPartParseSuccess)
    assert result.evidence is not None
    assert result.evidence.candidates is None
    assert "review" in result.message.lower()


def test_combined_confirmed_rules_are_all_recorded() -> None:
    layout = reviewed_layout("ppp-comma-poet").model_copy(update={"pipe_alternatives": True})
    raw = "amāre|amare, amō, mōnī<br>poet., monitum"
    result = parse_principal_part_value(raw, layout, lexical_entry="amāre")
    assert isinstance(result, PrincipalPartParseSuccess)
    assert result.evidence is not None
    assert result.evidence.candidates == (("amāre", "amare"), ("amō",), ("mōnī",), ("monitum",))
    assert result.evidence.rules == (
        "literal comma",
        "safe HTML line boundary",
        "confirmed trailing poet. hint",
        "confirmed pipe alternatives within each slot",
        "trim",
        "preserve alternative order",
    )


def test_rejected_leading_omission_retains_positional_evidence() -> None:
    raw = ", amō, amāvī, amātum"
    result = parse_principal_part_value(
        raw, reviewed_layout("ppp-comma"), lexical_entry="amāre", source_identity="omitted", source_location="row 2"
    )
    assert not isinstance(result, PrincipalPartParseSuccess)
    assert result.code == "required_role_omitted"
    assert result.evidence is not None
    assert result.evidence.raw == raw
    assert result.evidence.candidates == ((), ("amō",), ("amāvī",), ("amātum",))
    assert result.evidence.rules == ("literal comma", "explicit omission")
    assert (result.source_identity, result.source_location) == ("omitted", "row 2")


@pytest.mark.parametrize("segment", ['amāvī<br class="hint">poet.', "amāvī<p>poet.</p>"])
def test_unreviewed_hint_boundaries_require_review_not_mislabelled_raw(segment: str) -> None:
    raw = f"amāre, amō, {segment}, amātum"
    result = parse_principal_part_value(raw, reviewed_layout("ppp-comma-poet"), lexical_entry="amāre")
    assert not isinstance(result, PrincipalPartParseSuccess)
    assert result.status == "unsupported"
    assert result.evidence is not None
    assert result.evidence.raw == raw
    assert result.evidence.hints == ()
    assert "review" in result.message
