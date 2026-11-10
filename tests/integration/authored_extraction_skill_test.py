"""Executable contracts for the agent-only extraction skill and synthetic examples."""

import json
from pathlib import Path

from typer.testing import CliRunner

from latinitas_cards.authored_identity import derive_authored_id
from latinitas_cards.authored_import import load_authored_import
from latinitas_cards.cli import app

FIXTURES = Path("tests/fixtures/authored-extraction")
SKILL = Path(".agents/skills/extracting-authored-notes/SKILL.md")


def test_extraction_skill_is_mirrored_and_documents_identity_and_safe_workflow() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert text == Path(".claude/skills/extracting-authored-notes/SKILL.md").read_text(encoding="utf-8")
    for contract in (
        "name: extracting-authored-notes",
        "vocab/<lemma>/<sense>",
        "form/<text-label>/<occurrence-label>",
        "qa/<text-label>/<reviewed-label>",
        "explicit, reviewed keys",
        "never silently rename",
        "new, changed, and missing",
        "preserve existing `skip`",
        "separate solutions",
        "authored validate",
        "authored preview",
        "never write to Anki",
    ):
        assert contract in text


def test_synthetic_reextraction_contract_and_read_only_cli() -> None:
    first = load_authored_import(FIXTURES / "first.expected.jsonl").require_valid()
    revised = load_authored_import(FIXTURES / "reextracted.expected.jsonl").require_valid()
    old = {item.key: item for item in first}
    new = {item.key: item for item in revised}
    assert len(old) == 6 and len(new) == 7
    assert {item.provenance.document for item in first} == {"Harbour Tale", "Garden Dialogue"}
    assert new["vocab/navis/ship"].status == "skip"
    assert new["qa/harbour/cargo"].model_dump()["answer"] == "A basket of roses."
    assert old["qa/harbour/cargo"].model_dump()["answer"] == "A basket of apples."
    assert new["form/garden/speaker"].provenance.reference == "scene B"
    assert old["form/garden/speaker"].provenance.reference == "scene A"
    assert new["form/garden/speaker"].key != new["form/harbour/sailor"].key
    for key in old:
        assert derive_authored_id("synthetic-course", old[key]) == derive_authored_id("synthetic-course", new[key])
        assert old[key].status == new[key].status
    # Missing material is retained, not silently deleted or reclassified.
    assert old["qa/garden/path"] == new["qa/garden/path"]
    report = json.loads((FIXTURES / "changes.expected.json").read_text(encoding="utf-8"))
    assert report == {
        "new": ["vocab/porta/gate"],
        "changed": ["form/garden/speaker", "qa/harbour/cargo"],
        "missing": ["qa/garden/path"],
        "unchanged": ["form/harbour/sailor", "vocab/navis/ship", "vocab/rosa/rose"],
        "missing_policy": "retained unchanged pending review",
    }
    changed = {
        key
        for key in old
        if old[key].model_dump(exclude={"line_number"}) != new[key].model_dump(exclude={"line_number"})
    }
    assert changed == set(report["changed"])
    assert new.keys() - old.keys() == set(report["new"])
    runner = CliRunner()
    for filename, selected in (("first.expected.jsonl", 5), ("reextracted.expected.jsonl", 6)):
        source = FIXTURES / filename
        before = source.read_bytes()
        for command in ("validate", "preview"):
            result = runner.invoke(app, ["authored", command, str(source), "--namespace", "synthetic-course"])
            assert result.exit_code == 0, result.output
            if command == "preview":
                assert f"Effective selection: {selected}" in result.output
                assert "Merged duplicates: 0" in result.output
        assert source.read_bytes() == before
