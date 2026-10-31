"""Public preview, initial export and inspection of reviewed corpus exercises."""

import ast
import html
import json
import os
import re
from pathlib import Path
from typing import Any, cast

import pytest
from corpus_exercises_test import profile
from corpus_facts_test import Fixture, morph, text
from typer.testing import CliRunner

from latinitas_cards import preview_export
from latinitas_cards.annotation_sidecar import Annotation, read_sidecar
from latinitas_cards.canonical import canonical_json
from latinitas_cards.cli import app
from latinitas_cards.review_inspection import read_export_csv

RUNNER = CliRunner()


@pytest.mark.parametrize("language,unit", [("de", "Abschnitt"), ("en", "Passage")])
def test_source_provenance_survives_csv_without_changing_selection(tmp_path: Path, language: str, unit: str) -> None:
    facts = selected(tmp_path, language)
    selection = (tmp_path / "selection.json").read_bytes()
    claims = Path(f"{facts.corpus}.claims.json").read_bytes()
    sidecar = facts.sidecar.read_bytes()
    before = invoke(tmp_path, facts, "preview")["result"]["exercises"]
    lines = [json.loads(line) for line in facts.corpus.read_text().splitlines()]
    revision = 'rev<&"\x7f\x85\u2028\u202e\\x7f\x00'
    lines[0].update(revision=revision, url='javascript:alert("not a link")', edition="Test edition")
    lines[2].update(author="Author </pre><script>alert(1)</script>", title='Work & "title"')
    facts.corpus.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    preview_result = invoke(tmp_path, facts, "preview")
    preview = preview_result["result"]["exercises"]
    assert [row["latinitas_id"] for row in preview] == [row["latinitas_id"] for row in before]
    assert all(row["status"] == "eligible" and unit in row["citation"] for row in preview)
    assert all("revision" not in row["citation"] for row in preview)
    output = tmp_path / "provenance.csv"
    exported = invoke(tmp_path, facts, "export", "--output", str(output))
    reasons = ["missing_edition_urn", "missing_source_license", "no_verified_passage_link"]
    for result in (preview_result, exported):
        warnings = [row for row in result["diagnostics"] if row["code"] == "incomplete_provenance"]
        assert len(warnings) == len(preview)
        assert all(row["severity"] == "warning" for row in warnings)
        assert all(row["ref"] == lines[3]["passage_id"] for row in warnings)
        assert all(row["message"] == f"Source provenance limitations: {reasons}" for row in warnings)
    for note in read_export_csv(output):
        citation = note["Citation"]
        assert '<details class="corpus-provenance">' in citation
        assert "<script>" not in citation and "href=" not in citation
        match = re.search(r"<pre>(.*?)</pre>", citation)
        assert match is not None
        serialized = html.unescape(match[1])
        provenance = json.loads(serialized)
        assert serialized == canonical_json(provenance).replace("\x7f", "\\u007f").replace("\x85", "\\u0085")
        assert provenance["header"]["revision"] == revision
        assert provenance["header"]["source"] == lines[0]["source"]
        assert provenance["header"]["url"] == 'javascript:alert("not a link")'
        assert provenance["work"]["author"] == lines[2]["author"]
        assert provenance["passage"]["passage_id"] == note["Source ID"]
        assert provenance["derived_cts_reference"] is None
        assert provenance["resolver"] is None and provenance["limitations"] == reasons
    invoke(tmp_path, facts, "inspect", "--check", str(output))
    invoke(tmp_path, facts, "select", "--report", str(tmp_path / "report.json"))
    assert (tmp_path / "selection.json").read_bytes() == selection
    assert Path(f"{facts.corpus}.claims.json").read_bytes() == claims
    assert facts.sidecar.read_bytes() == sidecar


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


@pytest.mark.parametrize("known_word", ["AMĪCUS", "ignotus"])
@pytest.mark.parametrize("featureless", [False, True])
def test_known_deck_links_in_report_and_preview_only(tmp_path: Path, known_word: str, featureless: bool) -> None:
    from known_lemmas_test import _csv, _profile

    from latinitas_cards.manifest import CsvIdentityManifest

    facts = selected(tmp_path)
    if featureless:
        old = next(
            row for row in read_sidecar(facts.sidecar) if isinstance(row, Annotation) and row.layer == "morphology"
        )
        facts.retract(old)
        facts.add(0, "morphology", morph("amicus"), (7, 13))
        Path(f"{facts.corpus}.claims.json").unlink()
        facts.decide(
            "accepted",
            *(row for row in read_sidecar(facts.sidecar) if isinstance(row, Annotation) and row != old),
        )
        invoke(tmp_path, facts, "select", "--report", str(tmp_path / "report.json"))
    original_report = json.loads((tmp_path / "report.json").read_bytes())
    original_preview = invoke(tmp_path, facts, "preview")["result"]["exercises"]
    assert all(row["deck_links"] == [] for row in original_report["candidates"])
    assert all(row["deck_links"] == [] for row in original_preview)
    invoke(tmp_path, facts, "export", "--output", str(tmp_path / "before.csv"))
    known, source = tmp_path / "known.json", tmp_path / "known.csv"
    _profile(known)
    _csv(source, [("friend", known_word), ("other-sense", known_word), ("unrelated", "legere")])
    Path(f"{source}.latinitas.json").write_text(
        CsvIdentityManifest.scoped(("ID", "Entry", "Forms", "Meaning"), "deck-scope").to_json()
    )
    protected = [
        facts.corpus,
        facts.sidecar,
        Path(f"{facts.corpus}.claims.json"),
        tmp_path / "profile.json",
        known,
        source,
        Path(f"{source}.latinitas.json"),
        tmp_path / "selection.json",
    ]
    before = {path: path.read_bytes() for path in protected}
    flags = ("--known-lemmas", str(known), "--known-lemmas-source", str(source))
    invoke(tmp_path, facts, "select", "--report", str(tmp_path / "report.json"), *flags)
    report = json.loads((tmp_path / "report.json").read_bytes())
    preview = invoke(tmp_path, facts, "preview", *flags)["result"]["exercises"]
    expected = [{"identity": "friend", "scope": "deck-scope"}, {"identity": "other-sense", "scope": "deck-scope"}]
    for row in report["candidates"]:
        single = row["recipe"] in {"corpus_lexical_recognition", "corpus_guided_analysis"}
        assert row["deck_links"] == (expected if single and known_word == "AMĪCUS" else [])
        if featureless and row["recipe"] == "corpus_guided_analysis":
            assert row["eligible"] is False and row["reasons"] == ["no_features"] and row["outcome"] == "kept"
            assert row["factors"] is row["contributions"] is row["score"] is None
    if featureless:
        guided = next(row for row in preview if row["recipe"] == "corpus_guided_analysis")
        assert guided["status"] == "withheld" and guided["reasons"] == ["no_features"] and not guided["answer"]
    assert {row["latinitas_id"]: row["deck_links"] for row in preview} == {
        row["latinitas_id"]: row["deck_links"] for row in report["candidates"]
    }
    assert {row["latinitas_id"]: {k: v for k, v in row.items() if k != "deck_links"} for row in preview} == {
        row["latinitas_id"]: {k: v for k, v in row.items() if k != "deck_links"} for row in original_preview
    }
    invoke(tmp_path, facts, "export", "--output", str(tmp_path / "after.csv"))
    assert (tmp_path / "after.csv").read_bytes() == (tmp_path / "before.csv").read_bytes()
    assert before == {path: path.read_bytes() for path in protected}


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
    assert "Author not supplied, facts, Passage p000001" in card["answer"]
    assert '"source"' in card["answer"] and '"passage_hash"' in card["answer"]
    assert card["reasons"] == ["Artifact backing evidence is unverified without current source"]


def translation_equalities(source: str) -> list[tuple[str, str]]:
    """Inventory payload equality and simple local aliases, not whole-program dataflow.

    Translation.text feeds Fact/Resolution.value and rendered gloss/translation
    answers. Generic text/value names deliberately overapproximate these owners.
    No inference across helpers, closures, arbitrary containers or dynamic keys;
    aliases are flow-insensitive and only common string transforms are followed.
    """
    payloads = {"text", "value", "translation", "reference_translation", "gloss", "answer", "candidate_text"}
    scopes: dict[str, list[ast.AST]] = {}

    def collect(node: ast.AST, owner: str = "") -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            owner = f"{owner}.{node.name}".strip(".")
        scopes.setdefault(owner, []).append(node)
        for child in ast.iter_child_nodes(node):
            collect(child, owner)

    collect(ast.parse(source))
    found: list[tuple[str, str]] = []
    for owner, nodes in scopes.items():
        aliases = payloads.copy()

        def is_payload(node: ast.AST, names: set[str]) -> bool:
            if isinstance(node, ast.Name):
                return node.id in names
            if isinstance(node, ast.Attribute):
                return node.attr in payloads
            if isinstance(node, ast.Subscript):
                return isinstance(node.slice, ast.Constant) and node.slice.value in payloads
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "str" and node.args:
                    return is_payload(node.args[0], names)
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"strip", "lower", "casefold", "replace"}:
                        return is_payload(node.func.value, names)
                    return (
                        node.func.attr == "get"
                        and bool(node.args)
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value in payloads
                    )
            return False

        while True:
            previous = aliases.copy()
            for node in nodes:
                if (
                    isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr)
                    and node.value is not None
                    and is_payload(node.value, aliases)
                ):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    aliases.update(target.id for target in targets if isinstance(target, ast.Name))
            if aliases == previous:
                break
        found.extend(
            (owner, ast.unparse(node))
            for node in nodes
            if isinstance(node, ast.Compare)
            and any(isinstance(op, ast.Eq | ast.NotEq) for op in node.ops)
            and any(is_payload(operand, aliases) for operand in [node.left, *node.comparators])
        )
    return found


def test_no_translation_string_grading_in_source() -> None:
    root = Path(__file__).resolve().parents[2] / "src"
    found = [
        (str(path.relative_to(root)), owner, expression)
        for path in root.rglob("*.py")
        for owner, expression in translation_equalities(path.read_text())
    ]
    # Exact expression/owner inventory, including counts: no module is exempt.
    reviewed = [
        # Persisted export drift, never learner-answer grading.
        ("corpus_output.py", "inspect_corpus", "row.get(field) != value"),
        ("review_inspection.py", "inspect_workspace", "current.get(field, '') != value"),
        # Evidence freshness, corpus identity/NFC and required morphology validation.
        ("principal_relationships.py", "compare_principal_parts", "item.claim.candidate_text != part.display"),
        ("corpus_validation.py", "validate_records", "raw.get('corpus_id') != corpus_id"),
        ("corpus_validation.py", "validate_records", "text != nfc"),
        ("phrase_inputs.py", "bind_selections", "getattr(form.features, feature) != value"),
        # Annotation conflicts and retained segmentation differences.
        ("annotation_plans.py", "evaluate_operations", "row.value != record.value"),
        ("commands/annotations.py", "segment.body.additions", "record.value != value"),
        # Authored-import emptiness/conflicts and offline destination reconciliation.
        ("authored_identity.py", "reconcile_authored_import", "value == ''"),
        ("authored_identity.py", "reconcile_authored_import", "previous[0] != value"),
        ("destination_reconciliation.py", "reconcile_values", "value != d"),
        ("managed_plans.py", "compose_plan", "d == value"),
        ("managed_plans.py", "compose_plan", "d == value"),
        # Import controls, canonical role labels, XML dispatch and metadata conflicts.
        (
            "preview_export.py",
            "_validate_import_metadata",
            "encode_unsafe_controls(value, preserve_line_breaks=False) != value",
        ),
        ("claim_sidecar.py", "record_batch", "claim.value != base.feature"),
        ("usfx_adapter.py", "legacy_projection", "initial_tag == 've'"),
        ("tei_adapter.py", "build_tei", "value != override"),
    ]
    assert sorted(found) == sorted((f"latinitas_cards/{module}", owner, expr) for module, owner, expr in reviewed)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("return renamed.text == submitted", ["renamed.text == submitted"]),
        (
            "wording = candidate.resolutions[-1].value\ncopy = str(wording).casefold()\nreturn submitted != copy",
            ["submitted != copy"],
        ),
        ("alias = record['value']['text']\nreturn alias == submitted", ["alias == submitted"]),
        ("alias: str = record.get('text')\nreturn '' == alias == submitted", ["'' == alias == submitted"]),
        ("return translation.language == 'en'", []),
        ("return annotation.layer == 'translation'", []),
        ("return len(translation) == 1", []),
        ("return f'<div>{renamed.text}</div>'", []),
        ("# translation == submitted\nreturn 'translation == submitted'", []),
    ],
)
def test_translation_equality_controls(body: str, expected: list[str]) -> None:
    source = "def grade(candidate, submitted):\n" + "\n".join(f"    {line}" for line in body.splitlines())
    assert translation_equalities(source) == [("grade", expression) for expression in expected]
