"""Public preview, initial export and inspection of reviewed corpus exercises."""

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
from corpus_exercises_test import profile
from corpus_facts_test import Fixture, morph, text
from typer.testing import CliRunner

from latinitas_cards import preview_export
from latinitas_cards.cli import app
from latinitas_cards.review_inspection import read_export_csv

RUNNER = CliRunner()


def invoke(root: Path, facts: Fixture, command: str, *flags: str, code: int = 0) -> dict[str, Any]:
    result = RUNNER.invoke(
        app,
        [
            "corpus-cards",
            command,
            str(root / "profile.json"),
            "--corpus",
            str(facts.corpus),
            "--selection",
            str(root / "selection.json"),
            *flags,
            "--json",
        ],
    )
    assert result.exit_code == code, result.output
    return cast(dict[str, Any], json.loads(result.output))


def selected(root: Path, language: str = "en") -> Fixture:
    facts = Fixture(root, ("Marcus amicum videt.",))
    annotations = [facts.add(0, "morphology", morph("amicus", case="accusativus"), (7, 13))]
    for span, value in (
        ((7, 13), 'friend & "ally"'),
        ((0, 13), "Marcus and his friend"),
        (None, "Marcus sees a friend."),
    ):
        annotations.append(facts.add(0, "translation", text(language, value), span))
    facts.decide("accepted", *annotations)
    (root / "profile.json").write_text(profile(language).model_dump_json())
    invoke(root, facts, "select", "--report", str(root / "report.json"))
    return facts


@pytest.mark.parametrize("language", ["de", "en"])
def test_preview_export_inspect_and_claim_slots(tmp_path: Path, language: str) -> None:
    facts = selected(tmp_path, language)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    preview = invoke(tmp_path, facts, "preview")["result"]
    assert len(preview["exercises"]) == 4
    assert {e["recipe"] for e in preview["exercises"]} == {
        "corpus_lexical_recognition",
        "corpus_guided_analysis",
        "corpus_phrase_translation",
        "corpus_sentence_translation",
    }
    for exercise in preview["exercises"]:
        assert exercise["status"] == "eligible" and exercise["reasons"] == []
        assert exercise["prompt"] and exercise["answer"] and exercise["citation"]
        assert all(f["claim_status"] == "accepted" and f["producer"]["name"] == "fixture" for f in exercise["facts"])
    assert len(invoke(tmp_path, facts, "preview", "--limit", "1")["result"]["exercises"]) == 1
    invoke(tmp_path, facts, "inspect")
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    output = tmp_path / "cards.csv"
    assert invoke(tmp_path, facts, "export", "--output", str(output))["result"]["exported"] == 4
    rows = read_export_csv(output)
    assert len(rows) == 4
    assert all(sum(v == "1" for k, v in row.items() if k.endswith("Enabled")) == 1 for row in rows)
    assert "friend &amp; &quot;ally&quot;" in output.read_text()
    invoke(tmp_path, facts, "inspect", "--check", str(output))
    claims = RUNNER.invoke(app, ["claims", "inspect", str(output), "--json"])
    assert claims.exit_code == 0, claims.output
    cards = json.loads(claims.output)["cards"]
    assert {c["recipe"] for c in cards} == {e["recipe"] for e in preview["exercises"]}
    assert sum(c["eligible"] for c in cards) == 4
    original = output.read_bytes()
    invoke(tmp_path, facts, "export", "--output", str(output), code=3)
    assert output.read_bytes() == original
    output.write_text(output.read_text().replace("reader-a", "another-profile"))
    invoke(tmp_path, facts, "inspect", "--check", str(output))
    output.write_text(output.read_text().replace("Marcus sees a friend.", "Tampered"))
    assert invoke(tmp_path, facts, "inspect", "--check", str(output), code=1)["diagnostics"][-1]["code"] == "card_drift"


def test_deselect_restore_output_journey(tmp_path: Path) -> None:
    from corpus_selection_test import deselect

    facts = selected(tmp_path)
    state = tmp_path / "selection.json"
    original = json.loads(state.read_bytes())
    removed = original["selected"][0]
    identity = removed["latinitas_id"]
    baseline = tmp_path / "baseline.csv"
    invoke(tmp_path, facts, "export", "--output", str(baseline))
    deselect(tmp_path, identity)
    invoke(tmp_path, facts, "select", "--report", str(tmp_path / "report.json"))
    before, mtime = state.read_bytes(), state.stat().st_mtime_ns
    preview = invoke(tmp_path, facts, "preview")["result"]["exercises"]
    assert {e["latinitas_id"] for e in preview} == {e["latinitas_id"] for e in original["selected"][1:]}
    drift = invoke(tmp_path, facts, "inspect", "--check", str(baseline), code=1)
    assert drift["diagnostics"][-1]["code"] == "card_drift"
    assert (state.read_bytes(), state.stat().st_mtime_ns) == (before, mtime)
    output = tmp_path / "remaining.csv"
    assert invoke(tmp_path, facts, "export", "--output", str(output))["result"]["exported"] == 3
    assert {row["LatinitasID"] for row in read_export_csv(output)} == {e["latinitas_id"] for e in preview}
    deselect(tmp_path, identity, restore=True)
    restored = json.loads(state.read_bytes())
    assert next(e for e in restored["selected"] if e["latinitas_id"] == identity) == removed | {"run": 3}
    invoke(tmp_path, facts, "select", "--report", str(tmp_path / "report.json"))
    assert len(invoke(tmp_path, facts, "preview")["result"]["exercises"]) == 4
    invoke(tmp_path, facts, "inspect", "--check", str(baseline))


def test_review_withholding_and_binding_guards(tmp_path: Path) -> None:
    facts = selected(tmp_path)
    output = tmp_path / "cards.csv"
    invoke(tmp_path, facts, "export", "--output", str(output))
    Path(f"{facts.corpus}.claims.json").unlink()
    preview = invoke(tmp_path, facts, "preview")["result"]["exercises"]
    assert all(e["status"] == "withheld" and "open_fact" in e["reasons"] and not e["answer"] for e in preview)
    assert invoke(tmp_path, facts, "inspect", "--check", str(output), code=1)["diagnostics"][-1]["code"] == "card_drift"
    empty = tmp_path / "empty.csv"
    assert (
        invoke(tmp_path, facts, "export", "--output", str(empty), code=1)["diagnostics"][-1]["code"]
        == "nothing_to_export"
    )
    assert not empty.exists()
    settings = tmp_path / "profile.json"
    settings.write_text(profile(profile_id="other-reader").model_dump_json())
    for command in ("preview", "inspect", "export"):
        flags = ("--output", str(empty)) if command == "export" else ()
        assert invoke(tmp_path, facts, command, *flags, code=4)["diagnostics"][-1]["code"] == "binding_drift"


def test_missing_target_context_and_reading(tmp_path: Path) -> None:
    facts = selected(tmp_path)
    path = tmp_path / "selection.json"
    state = json.loads(path.read_text())
    state["selected"][0]["context_digest"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(state))
    exercises = invoke(tmp_path, facts, "preview")["result"]["exercises"]
    assert sum("context_changed" in e["reasons"] for e in exercises) == 1
    assert all(not e["answer"] for e in exercises if e["reasons"])
    configured = profile(recipes=[{"recipe": "corpus_sentence_translation", "budget": 10}])
    (tmp_path / "profile.json").write_text(configured.model_dump_json())
    exercises = invoke(tmp_path, facts, "preview")["result"]["exercises"]
    assert sum("target_missing" in e["reasons"] for e in exercises) == 3
    configured = profile(reading=[{"corpus_id": "facts", "work": "absent", "from": "no", "to": "no"}])
    (tmp_path / "profile.json").write_text(configured.model_dump_json())
    assert invoke(tmp_path, facts, "preview", code=3)["diagnostics"][-1]["code"] == "invalid_reading"
    (tmp_path / "profile.json").write_text(profile(corpora=["other"]).model_dump_json())
    for command in ("preview", "inspect", "export"):
        flags = ("--output", str(tmp_path / "new.csv")) if command == "export" else ()
        assert invoke(tmp_path, facts, command, *flags, code=3)["diagnostics"][-1]["code"] == "profile_mismatch"


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_check_inaccessible_csv_is_input_error(tmp_path: Path, kind: str) -> None:
    facts = selected(tmp_path)
    check = tmp_path / "check.csv"
    if kind == "directory":
        check.mkdir()
    before = set(tmp_path.iterdir())
    result = invoke(tmp_path, facts, "inspect", "--check", str(check), code=3)
    assert result["diagnostics"][0]["code"] == "unreadable"
    assert result["diagnostics"][0]["path"] == str(check)
    assert set(tmp_path.iterdir()) == before


def test_export_preserves_file_created_during_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    facts = selected(tmp_path)
    output = tmp_path / "cards.csv"
    original_stage = preview_export._stage_bytes

    def competing_creation(destination: Path, payload: bytes) -> Path:
        staged = original_stage(destination, payload)
        destination.write_bytes(b"External writer's existing file\n")
        return staged

    monkeypatch.setattr(preview_export, "_stage_bytes", competing_creation)
    result = invoke(tmp_path, facts, "export", "--output", str(output), code=3)
    assert result["diagnostics"][0]["code"] == "exists"
    assert output.read_bytes() == b"External writer's existing file\n"
    assert not list(tmp_path.glob(".cards.csv.*"))


def test_export_staging_failure_is_input_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    facts = selected(tmp_path)
    output = tmp_path / "cards.csv"

    def no_space(_: int) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(os, "fsync", no_space)
    result = invoke(tmp_path, facts, "export", "--output", str(output), code=3)
    assert result["diagnostics"][0]["code"] == "unreadable"
    assert result["diagnostics"][0]["path"] == str(output)
    assert not output.exists() and not list(tmp_path.glob(".cards.csv.*"))


@pytest.mark.parametrize("directive", ["#tags column:1", "#notetype column:24", ""])
def test_inspect_rejects_transport_routing_drift(tmp_path: Path, directive: str) -> None:
    facts = selected(tmp_path)
    output = tmp_path / "cards.csv"
    invoke(tmp_path, facts, "export", "--output", str(output))
    original = output.read_text()
    output.write_text(
        original.replace("#tags column:25", directive) if directive else original.replace("#tags column:25\n", "")
    )
    result = invoke(tmp_path, facts, "inspect", "--check", str(output), code=1)
    assert result["diagnostics"][-1]["code"] == "card_drift"


@pytest.mark.parametrize("surface", ["preview", "claims"])
def test_preview_and_claim_inspection_include_full_context(tmp_path: Path, surface: str) -> None:
    facts = Fixture(tmp_path, ("Marcus amicum videt. Rhinoceros purpureus dormit.",))
    facts.decide("accepted", facts.add(0, "translation", text("en", "Marcus sees a friend."), (0, 20)))
    (tmp_path / "profile.json").write_text(
        profile(recipes=[{"recipe": "corpus_sentence_translation", "budget": 10}]).model_dump_json()
    )
    invoke(tmp_path, facts, "select", "--report", str(tmp_path / "report.json"))
    if surface == "preview":
        exercise = invoke(tmp_path, facts, "preview")["result"]["exercises"][0]
        assert exercise["latin"] == "Marcus amicum videt."
        assert exercise["passage"] == "Marcus amicum videt. Rhinoceros purpureus dormit."
        return
    output = tmp_path / "cards.csv"
    invoke(tmp_path, facts, "export", "--output", str(output))
    result = RUNNER.invoke(app, ["claims", "inspect", str(output), "--json"])
    assert result.exit_code == 0, result.output
    card = next(c for c in json.loads(result.output)["cards"] if c["eligible"])
    assert "Marcus amicum videt." in card["prompt"]
    assert "Rhinoceros purpureus dormit." in card["answer"]
    assert "p000001 (facts)" in card["answer"]
    assert card["reasons"] == ["Artifact backing evidence is unverified without current source"]
