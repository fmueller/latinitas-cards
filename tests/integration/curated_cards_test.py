"""Public curated recipes with synthetic decisions, not linguistic approval."""

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
from curated_test import candidate, permission
from typer.testing import CliRunner

from latinitas_cards.acceptance_records import canonical
from latinitas_cards.cli import app
from latinitas_cards.curated import make_approval
from latinitas_cards.curated_cards import CURATED_LAYOUT, RecipePlan

runner = CliRunner()


def write(path: Path, value: object) -> None:
    path.write_bytes(canonical(value))


def csv_note(path: Path) -> dict[str, str]:
    payload = path.read_text()
    columns = next(line.removeprefix("#columns:") for line in payload.splitlines() if line.startswith("#columns:"))
    rows = list(csv.reader(io.StringIO("\n".join(line for line in payload.splitlines() if not line.startswith("#")))))
    return dict(zip(next(csv.reader([columns])), rows[0], strict=True))


def fixture(root: Path, language: str = "en") -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    data = candidate()
    item = data["items"][0]
    item["rule"].update(
        language=language,
        text="Präsensstamm + -bā- + aktive Personalendungen."
        if language == "de"
        else "Present stem + -bā- + active personal endings.",
        applicability="Regelmäßige a-Konjugation, Imperfekt Indikativ Aktiv."
        if language == "de"
        else "Regular first conjugation, imperfect indicative active.",
        caveats=["Vor -m/-t wird ā gekürzt." if language == "de" else "Shorten ā before -m/-t."],
    )
    item["examples"][0].update(
        latin="amābās",
        explanation="amā-: Präsensstamm; -bā-: Imperfektzeichen; -s: 2. Person Singular."
        if language == "de"
        else "amā-: present stem; -bā-: imperfect marker; -s: second person singular.",
    )
    # This is original engineering material, with cited synthetic references only.
    item["rule"]["evidence"][0]["text"] = item["rule"]["text"]
    item["examples"][0]["evidence"][0]["text"] = item["examples"][0]["explanation"]
    write(root / "input.json", data)
    write(root / "permissions.json", permission())
    write(
        root / "plan.json",
        {
            "format": "latinitas-curated-recipes",
            "version": 1,
            "targets": [
                {
                    "key": "imperfect",
                    "item_key": "imperfect-a",
                    "formation": item["rule"]["applicability"],
                    "example": "amābās",
                    "analysis": "supported",
                    "mnemonic_key": None,
                    "mnemonic_side": "back",
                    "related_production": [],
                }
            ],
        },
    )
    return [
        str(root / "input.json"),
        "--permissions",
        str(root / "permissions.json"),
        "--plan",
        str(root / "plan.json"),
        "--language",
        language,
    ]


def call(command: str, args: list[str], *extra: str) -> dict[str, Any]:
    result = runner.invoke(app, ["curated", command, *args, *extra])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return dict(json.loads(result.stdout))


def decisions(root: Path, report: dict[str, Any], subjects: tuple[str, ...] = ("rule", "rendered_card")) -> Path:
    raw = []
    seen = set()
    for card in report["cards"]:
        for requirement in card["requirements"]:
            key = (card["item_key"], requirement["subject"], requirement["subject_key"], requirement["fingerprint"])
            if requirement["subject"] not in subjects or key in seen:
                continue
            seen.add(key)
            raw.append(
                make_approval(
                    source_scope=report["source_scope"],
                    item_key=card["item_key"],
                    **{k: requirement[k] for k in ("subject", "subject_key", "language", "fingerprint")},
                    decision="approved",
                    reviewer="isolated-synthetic-engineering-reviewer",
                    reason="Synthetic engineering test only; not real linguistic approval.",
                    reviewed_at="2026-11-19T00:00:00Z",
                    supersedes=(),
                )
            )
    path = root / "decisions.json"
    write(path, raw)
    return path


def approve(
    root: Path, args: list[str], extra: list[str], subjects: tuple[str, ...] = ("rule", "rendered_card")
) -> Path:
    report = call("preview", args, *extra)
    path = decisions(root, report, subjects)
    history = root / "approved.json"
    call("review", args, *extra, "--decisions", str(path), "--output", str(history))
    return history


@pytest.mark.parametrize("language", ["de", "en"])
def test_distinct_recipes_rule_only_approval_and_exact_rerun(tmp_path: Path, language: str) -> None:
    args = fixture(tmp_path, language)
    preview = call("preview", args)
    assert [c["recipe"] for c in preview["cards"]] == ["rule_application"]
    assert preview["cards"][0]["state"] == "withheld"
    for recipe in ("formation_rule_recall", "component_explanation"):
        extra = ["--exclude-recipe", "rule_application", "--include-recipe", recipe]
        report = call("preview", args, *extra)
        assert [c["recipe"] for c in report["cards"]] == [recipe]
        card = report["cards"][0]
        assert "Mit ba" not in card["front_html"]
        assert "-bā-" not in card["front_html"]
        if recipe == "component_explanation":
            assert "amābās" in card["front_html"]
            assert "-s:" in card["back_html"]
        else:
            assert "amābās" not in card["front_html"]
        history = approve(tmp_path, args, extra)
        approved = [*extra, "--approvals", str(history)]
        result = call("preview", args, *approved)
        assert result["cards"][0]["state"] == "accepted"
        output = tmp_path / f"{recipe}.csv"
        run = tmp_path / f"{recipe}-run"
        exported = call("export", args, *approved, "--run", str(run), "--output", str(output))
        assert exported["exported"] == 1
        assert "Personal Notes" not in output.read_text()
        assert exported["export_receipt"]["cards"][0]["acceptance"] == "human"
        repeated = call("export", args, *approved, "--run", str(run), "--output", str(output))
        assert repeated["receipt_id"] == exported["receipt_id"]
        history.unlink()


@pytest.mark.parametrize(
    "subjects,expected", [((), 0), (("rule",), 0), (("rendered_card",), 0), (("rule", "rendered_card"), 1)]
)
def test_export_requires_both_exact_approvals(tmp_path: Path, subjects: tuple[str, ...], expected: int) -> None:
    args = fixture(tmp_path)
    history = tmp_path / "approved.json"
    if subjects:
        history = approve(tmp_path, args, [], subjects)
    else:
        write(history, [])
    result = call(
        "export",
        args,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "out.csv"),
    )
    assert result["exported"] == expected


@pytest.mark.parametrize(
    "recipe", ["rule_application", "formation_rule_recall", "component_explanation", "mnemonic_recall"]
)
@pytest.mark.parametrize("side", ["front", "back"])
def test_every_displayed_mnemonic_needs_language_specific_approval(tmp_path: Path, recipe: str, side: str) -> None:
    args = fixture(tmp_path, "de")
    plan = json.loads((tmp_path / "plan.json").read_text())
    plan["targets"][0].update(mnemonic_key="ba", mnemonic_side=side)
    write(tmp_path / "plan.json", plan)
    extra = (
        ["--include-recipe", recipe]
        if recipe == "rule_application"
        else ["--exclude-recipe", "rule_application", "--include-recipe", recipe]
    )
    if recipe == "formation_rule_recall" and side == "front":
        report = call("preview", args, *extra)
        assert report["cards"] == []
        assert report["skipped"][0]["reason"] == "mnemonic_cue_not_allowed"
        return
    history = approve(tmp_path, args, extra)
    report = call("preview", args, *extra, "--approvals", str(history))
    assert report["cards"][0]["state"] == "withheld"
    assert "mnemonic" in report["cards"][0]["reason"]
    unapproved = call(
        "export",
        args,
        *extra,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "missing-run"),
        "--output",
        str(tmp_path / "missing.csv"),
    )
    assert unapproved["exported"] == 0
    history.unlink()
    history = approve(tmp_path, args, extra, ("rule", "mnemonic", "rendered_card"))
    assert call("preview", args, *extra, "--approvals", str(history))["cards"][0]["state"] == "accepted"
    exported = call(
        "export",
        args,
        *extra,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "approved-run"),
        "--output",
        str(tmp_path / "approved.csv"),
    )
    assert exported["exported"] == 1
    approved_card = call("preview", args, *extra, "--approvals", str(history))["cards"][0]
    note = csv_note(tmp_path / "approved.csv")
    for template in CURATED_LAYOUT:
        slot = template.slot
        if slot.recipe == recipe:
            assert note[slot.enabled_field] == "1"
            assert note[slot.prompt_field] == approved_card["front_html"]
            assert note[slot.answer_field] == approved_card["back_html"]
        else:
            assert not any(note[field] for field in (slot.enabled_field, slot.prompt_field, slot.answer_field))
    mnemonic = json.loads((tmp_path / "input.json").read_text())["items"][0]["mnemonics"][0]["text"]
    assert mnemonic in approved_card["front_html" if side == "front" or recipe == "mnemonic_recall" else "back_html"]
    for card in exported["export_receipt"]["cards"]:
        assert "Personal Notes" not in card["owned_fields"] and "Tags" not in card["owned_fields"]
        assert card["owned_tags"] == []


def test_mnemonic_edit_preserves_rule_only_cards_and_receipts_withhold_stale_rows(tmp_path: Path) -> None:
    args = fixture(tmp_path, "de")
    extra = ["--include-recipe", "mnemonic_recall"]
    history = approve(tmp_path, args, extra, ("rule", "mnemonic", "rendered_card"))
    before = call("preview", args, *extra, "--approvals", str(history))
    run = tmp_path / "run"
    first = call(
        "export", args, *extra, "--approvals", str(history), "--run", str(run), "--output", str(tmp_path / "before.csv")
    )
    data = json.loads((tmp_path / "input.json").read_text())
    data["items"][0]["mnemonics"][0]["text"] += " geändert"
    write(tmp_path / "input.json", data)
    after = call("preview", args, *extra, "--approvals", str(history))
    assert before["cards"][0] == after["cards"][0]
    assert after["cards"][1]["state"] == "withheld"
    assert [c["note_id"] for c in before["cards"]] == [c["note_id"] for c in after["cards"]]
    result = call(
        "export",
        args,
        *extra,
        "--approvals",
        str(history),
        "--run",
        str(run),
        "--reconcile-receipt",
        first["receipt_id"],
        "--output",
        str(tmp_path / "after.csv"),
    )
    assert result["exported"] == 0  # whole-note sibling safety, not loss of rule approval
    assert len(result["export_receipt"]["ineligible"]) == 1
    action = result["export_receipt"]["ineligible"][0]
    assert action["resolution"] == "manual" and "vermutet" in action["required_action"]
    assert result["export_receipt"]["cards"] == []


@pytest.mark.parametrize("change", ["rule", "caveat", "explanation", "theme"])
def test_exact_content_and_presentation_invalidation(tmp_path: Path, change: str) -> None:
    args = fixture(tmp_path)
    history = approve(tmp_path, args, [])
    before = call("preview", args, "--approvals", str(history))
    data = json.loads((tmp_path / "input.json").read_text())
    if change == "rule":
        data["items"][0]["rule"]["text"] += " Correction."
    elif change == "caveat":
        data["items"][0]["rule"]["caveats"].append("New limit.")
    elif change == "explanation":
        data["items"][0]["examples"][0]["explanation"] += " Correction."
    write(tmp_path / "input.json", data)
    extra = ["--theme", "dark"] if change == "theme" else []
    after = call("preview", args, "--approvals", str(history), *extra)
    assert after["cards"][0]["state"] == "withheld"
    assert before["cards"][0]["note_id"] == after["cards"][0]["note_id"]
    assert before["cards"][0]["card_key"] == after["cards"][0]["card_key"]
    rule = next(r for r in after["cards"][0]["requirements"] if r["subject"] == "rule")
    assert (rule["status"] == "approved") == (change == "theme")


@pytest.mark.parametrize("analysis", ["unsupported", "ambiguous"])
def test_uncertain_analysis_is_withheld_without_invented_content(tmp_path: Path, analysis: str) -> None:
    args = fixture(tmp_path)
    plan = json.loads((tmp_path / "plan.json").read_text())
    plan["targets"][0]["analysis"] = analysis
    write(tmp_path / "plan.json", plan)
    report = call("preview", args, "--include-recipe", "component_explanation")
    assert report["cards"] == []
    assert all(s["reason"] == analysis for s in report["skipped"])


def test_nominal_contrast_keeps_legitimate_alternative_explanations(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    data = json.loads((tmp_path / "input.json").read_text())
    item = data["items"][0]
    item["key"] = "first-declension"
    item["mnemonics"] = []
    item["rule"]["text"] = "First-declension genitive plural: base + -ārum."
    item["rule"]["applicability"] = "First-declension nouns, genitive plural."
    item["rule"]["caveats"] = ["Alternative segmentation depends on the stated base convention."]
    item["examples"][0].update(latin="puellārum", explanation="puell- + -ārum; alternatively puellā- + -rum.")
    item["rule"]["evidence"][0]["text"] = item["rule"]["text"]
    item["examples"][0]["evidence"][0]["text"] = item["examples"][0]["explanation"]
    write(tmp_path / "input.json", data)
    plan = json.loads((tmp_path / "plan.json").read_text())
    plan["targets"][0].update(item_key=item["key"], formation="first-declension genitive plural", example="puellārum")
    write(tmp_path / "plan.json", plan)
    extra = ["--include-recipe", "component_explanation"]
    history = approve(tmp_path, args, extra)
    report = call("preview", args, *extra, "--approvals", str(history))
    assert len(report["cards"]) == 2
    assert all(c["state"] == "accepted" for c in report["cards"])
    assert all("alternatively" in c["back_html"] for c in report["cards"])
    assert all("genitive plural" in c["back_html"] and "base convention" in c["back_html"] for c in report["cards"])
    assert all("amābās" not in c["back_html"] and "Shorten" not in c["back_html"] for c in report["cards"])


def test_html_escaping_and_personal_fields_never_exported(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    data = json.loads((tmp_path / "input.json").read_text())
    data["items"][0]["examples"][0]["explanation"] += '<img src=x onerror="alert(1)">'
    write(tmp_path / "input.json", data)
    history = approve(tmp_path, args, [])
    report = call(
        "export",
        args,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "out.csv"),
    )
    assert report["exported"] == 1
    payload = (tmp_path / "out.csv").read_text()
    assert "&lt;img" in payload and "<img" not in payload
    rows = list(csv.reader(io.StringIO("\n".join(line for line in payload.splitlines() if not line.startswith("#")))))
    assert rows and "Personal Notes" not in payload


def test_conflicting_slots_withhold_all_targets_but_keep_unrelated_items(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    data = json.loads((tmp_path / "input.json").read_text())
    unrelated = copy.deepcopy(data["items"][0])
    unrelated["key"] = "unrelated"
    unrelated["mnemonics"] = []
    data["items"].append(unrelated)
    write(tmp_path / "input.json", data)
    plan = json.loads((tmp_path / "plan.json").read_text())
    target = plan["targets"][0]
    plan["targets"].extend([{**target, "key": "second-target"}, {**target, "item_key": "unrelated"}])
    write(tmp_path / "plan.json", plan)

    report = call("preview", args, "--include-recipe", "component_explanation")
    assert [(c["item_key"], c["recipe"]) for c in report["cards"]] == [
        ("unrelated", "rule_application"),
        ("unrelated", "component_explanation"),
    ]
    assert report["skipped"] == [
        {"item_key": "imperfect-a", "recipe": recipe, "reason": "conflicting_slot"}
        for recipe in ("rule_application", "component_explanation", "rule_application", "component_explanation")
    ]


def test_recipe_plan_rejects_duplicate_targets() -> None:
    target: dict[str, Any] = dict(
        key="same",
        item_key="x",
        formation="first",
        example="x",
        analysis="supported",
        mnemonic_key=None,
        mnemonic_side="back",
        related_production=[],
    )
    with pytest.raises(ValueError, match="duplicate"):
        RecipePlan.read_json(
            canonical(dict(format="latinitas-curated-recipes", version=1, targets=[target, copy.deepcopy(target)]))
        )


@pytest.mark.parametrize("language", ["de", "en"])
@pytest.mark.parametrize("publication_fault", [False, True])
def test_saved_grammar_study_integrates_production_without_approval_substitution(
    tmp_path: Path, language: str, publication_fault: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from grammar_pipeline_test import call as pipeline_call
    from grammar_pipeline_test import fixture as production_fixture

    args = fixture(tmp_path / "curated", language)
    # An English alternative is original fixture text, not a translation authorized by German review.
    data = json.loads((tmp_path / "curated/input.json").read_text())
    data["items"][0]["mnemonics"].append(
        {
            **data["items"][0]["mnemonics"][0],
            "language": "en",
            "text": "The BA marker points to the past.",
        }
    )
    write(tmp_path / "curated/input.json", data)
    profile, objectives, production_run = production_fixture(tmp_path / "production", language, preset=None)
    shared = json.loads(profile.read_text())
    source = tmp_path / "curated/input.json"
    shared["inputs"].append(
        dict(kind="curated", path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    )
    write(profile, shared)
    saved = tmp_path / "saved.json"
    report = call("preview", args, "--profile", str(profile), "--preset", "grammar-study", "--save-profile", str(saved))
    config = report["effective_configuration"]
    assert config["preset"] == "grammar-study" and config["language"] == language
    assert not any("cloze" in r for r in config["recipes"])
    args = args[:-2]  # Reload must retain the language without repeating the CLI flag.
    pipeline_call(
        "adapt",
        "--profile",
        str(saved),
        "--objectives",
        str(objectives),
        "--kind",
        "phrase-inputs",
        "--output",
        str(production_run),
    )
    production_preview = pipeline_call("preview", "--run", str(production_run))
    plan_path = tmp_path / "curated/plan.json"
    plan = json.loads(plan_path.read_text())
    related_id = production_preview["cards"][0]["note_id"]
    plan["targets"][0]["related_production"] = [related_id]
    write(plan_path, plan)
    extra = ["--profile", str(saved)]
    history = approve(tmp_path / "curated", args, extra, ("rule", "mnemonic", "rendered_card"))
    combined = call("preview", args, *extra, "--approvals", str(history), "--production-run", str(production_run))
    assert {c["recipe"] for c in combined["cards"]} == {
        "formation_rule_recall",
        "component_explanation",
        "mnemonic_recall",
    }
    assert all(c["state"] == "accepted" for c in combined["cards"])
    assert all(related_id in c["back_html"] for c in combined["cards"])
    assert combined["production"]["cards"] and all(c["state"] == "withheld" for c in combined["production"]["cards"])
    # Supply an independent, exact synthetic human claim decision for ONE production drill.
    from grammar_pipeline_test import EVIDENCE

    from latinitas_cards.acceptance import check_envelope, save_acceptance, save_stage
    from latinitas_cards.acceptance_records import Check, Human, digest
    from latinitas_cards.pipeline_consumers import Objectives, current_cards
    from latinitas_cards.pipeline_profile import PipelineProfile

    production, _ = current_cards(
        saved,
        PipelineProfile.read_json(saved.read_bytes()),
        Objectives.read_json(objectives.read_bytes()),
        production_run,
    )
    card = production[0]
    envelope = check_envelope(card.binding, card.versions, "human", card.claims.assertions)
    stage = save_stage(
        production_run,
        run_id="synthetic-only",
        stage_id="review",
        stage="analyze",
        binding=card.binding,
        versions=card.versions,
        envelope=envelope,
    )
    check = Check(
        id="human",
        assertion_ids=tuple(a.id for a in card.claims.assertions),
        method="human",
        version=card.versions.checker,
        result="pass",
        evidence=(EVIDENCE,),
        alternatives=(),
        objections=(),
        input_sha256=digest(envelope),
    )
    decision = card.claims.decisions[0].reader.decide(
        checks=(check,),
        origin="human",
        stage_ids=(stage.stage_id,),
        human=Human(
            reviewer="isolated-synthetic-engineering-reviewer",
            reason="Not linguistic approval",
            reviewed_sha256=digest({"binding": card.binding, "assertions": card.claims.assertions}),
        ),
    )
    save_acceptance(card.claims.decisions[0].path, decision)
    if publication_fault:
        from latinitas_cards import pipeline_batches

        original_atomic = pipeline_batches._atomic

        def fail_after_commit(path: Path, payload: bytes) -> None:
            original_atomic(path, payload)
            if path == production_run / "selection/index.json":
                raise OSError("synthetic post-index interruption")

        monkeypatch.setattr(pipeline_batches, "_atomic", fail_after_commit)
    output = call(
        "export",
        args,
        *extra,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "out.csv"),
        "--production-run",
        str(production_run),
        "--production-output",
        str(tmp_path / "production.csv"),
    )
    assert output["exported"] == 3
    if publication_fault:
        assert (tmp_path / "production.csv").exists()
        assert output["production"]["state"] == "recovery_required"
        assert "exported" not in output["production"]
        monkeypatch.setattr(pipeline_batches, "_atomic", original_atomic)
        recovered = call(
            "export",
            args,
            *extra,
            "--approvals",
            str(history),
            "--run",
            str(tmp_path / "run"),
            "--output",
            str(tmp_path / "out.csv"),
            "--production-run",
            str(production_run),
            "--production-output",
            str(tmp_path / "production.csv"),
        )
        assert recovered["receipt_id"] == output["receipt_id"]
        assert recovered["production"]["exported"] == 1
    else:
        assert output["production"]["exported"] == 1
        assert output["production"]["export_receipt"]["cards"][0]["acceptance"] == "human"
    overridden = call(
        "preview",
        args,
        *extra,
        "--approvals",
        str(history),
        "--exclude-recipe",
        "mnemonic_recall",
        "--production-run",
        str(production_run),
    )
    assert {c["recipe"] for c in overridden["cards"]} == {"formation_rule_recall", "component_explanation"}
    assert json.loads(saved.read_text())["language"] == language
    assert json.loads(profile.read_text())["preset"] is None
    if not publication_fault:
        data["items"] = []
        write(source, data)
        confirmed = json.loads(saved.read_text())
        next(r for r in confirmed["inputs"] if r["kind"] == "curated")["sha256"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        write(saved, confirmed)
        empty = call(
            "preview",
            args,
            *extra,
            "--approvals",
            str(history),
            "--production-run",
            str(production_run),
        )
        assert empty["cards"] == [] and any(c["state"] == "accepted" for c in empty["production"]["cards"])
        partial = call(
            "export",
            args,
            *extra,
            "--approvals",
            str(history),
            "--run",
            str(tmp_path / "run"),
            "--output",
            str(tmp_path / "empty.csv"),
            "--reconcile-receipt",
            output["receipt_id"],
            "--production-run",
            str(production_run),
            "--production-output",
            str(tmp_path / "production-empty.csv"),
            "--production-reconcile-receipt",
            output["production"]["receipt_id"],
        )
        assert partial["exported"] == 0 and len(partial["export_receipt"]["ineligible"]) == 3
        # Its eligible drill was already emitted: reaching production reconciliation
        # must preserve that cursor instead of duplicating it or withholding it.
        assert partial["production"]["exported"] == 0
        assert partial["production"]["export_receipt"]["ineligible"] == []


def test_mnemonic_withdrawal_history_cannot_be_rolled_back(tmp_path: Path) -> None:
    args = fixture(tmp_path, "de")
    extra = ["--include-recipe", "mnemonic_recall"]
    history = approve(tmp_path, args, extra, ("rule", "mnemonic", "rendered_card"))
    first = call(
        "export",
        args,
        *extra,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "first.csv"),
    )
    records = json.loads(history.read_text())
    old = next(r for r in records if r["subject"] == "mnemonic")
    rejection = make_approval(
        **{k: v for k, v in old.items() if k not in ("approval_id", "decision", "supersedes")},
        decision="rejected",
        supersedes=(old["approval_id"],),
    )
    write(tmp_path / "rejection.json", [rejection])
    revised = tmp_path / "revised.json"
    call(
        "review",
        args,
        *extra,
        "--approvals",
        str(history),
        "--decisions",
        str(tmp_path / "rejection.json"),
        "--output",
        str(revised),
    )
    withdrawn = call(
        "export",
        args,
        *extra,
        "--approvals",
        str(revised),
        "--run",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "withdrawn.csv"),
        "--reconcile-receipt",
        first["receipt_id"],
    )
    assert withdrawn["exported"] == 0 and len(withdrawn["export_receipt"]["ineligible"]) == 1
    rollback = runner.invoke(
        app,
        [
            "curated",
            "export",
            *args,
            *extra,
            "--approvals",
            str(history),
            "--run",
            str(tmp_path / "run"),
            "--output",
            str(tmp_path / "rollback.csv"),
            "--reconcile-receipt",
            withdrawn["receipt_id"],
        ],
    )
    assert rollback.exit_code == 1 and "history truncated" in rollback.output
    assert not (tmp_path / "rollback.csv").exists()


@pytest.mark.parametrize("mnemonics", ["absent", "german_only"])
def test_english_rule_only_export_does_not_require_or_borrow_a_mnemonic(tmp_path: Path, mnemonics: str) -> None:
    args = fixture(tmp_path)
    data = json.loads((tmp_path / "input.json").read_text())
    if mnemonics == "absent":
        data["items"][0]["mnemonics"] = []
    write(tmp_path / "input.json", data)
    extra = [
        "--exclude-recipe",
        "rule_application",
        "--include-recipe",
        "formation_rule_recall",
        "--include-recipe",
        "component_explanation",
        "--include-recipe",
        "mnemonic_recall",
    ]
    history = approve(tmp_path, args, extra)
    if mnemonics == "german_only":
        from latinitas_cards.curated import CuratedInput, mnemonic_fingerprint

        parsed = CuratedInput.read_json(canonical(data))
        item = parsed.items[0]
        german = make_approval(
            source_scope=parsed.source_scope,
            item_key=item.key,
            subject="mnemonic",
            subject_key="ba",
            language="de",
            fingerprint=mnemonic_fingerprint(item, item.mnemonics[0]),
            decision="approved",
            reviewer="isolated-synthetic-engineering-reviewer",
            reason="Not linguistic approval",
            reviewed_at="2026-11-19T00:00:00Z",
            supersedes=(),
        )
        write(history, [*json.loads(history.read_text()), german])
    preview = call("preview", args, *extra, "--approvals", str(history))
    assert {c["recipe"] for c in preview["cards"]} == {"formation_rule_recall", "component_explanation"}
    assert all("Eselsbrücke" not in c["front_html"] + c["back_html"] for c in preview["cards"])
    assert preview["skipped"][0]["reason"] == "missing_language_variant"
    result = call(
        "export",
        args,
        *extra,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "out.csv"),
    )
    assert result["exported"] == 2


def test_profile_source_hash_is_checked_before_preview_and_save(tmp_path: Path) -> None:
    from grammar_pipeline_test import fixture as production_fixture

    args = fixture(tmp_path / "curated")
    profile, _, _ = production_fixture(tmp_path / "production", "en")
    raw = json.loads(profile.read_text())
    raw["inputs"].append(dict(kind="curated", path=str(tmp_path / "curated/input.json"), sha256="0" * 64))
    write(profile, raw)
    result = runner.invoke(
        app, ["curated", "preview", *args, "--profile", str(profile), "--save-profile", str(tmp_path / "saved.json")]
    )
    assert result.exit_code == 1 and "curated profile input" in result.output
    assert not (tmp_path / "saved.json").exists()


def test_empty_intake_reconciles_last_exported_item(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    history = approve(tmp_path, args, [])
    extra = ["--approvals", str(history), "--run", str(tmp_path / "run")]
    first = call("export", args, *extra, "--output", str(tmp_path / "first.csv"))
    data = json.loads((tmp_path / "input.json").read_text())
    data["items"] = []
    write(tmp_path / "input.json", data)
    assert call("preview", args)["cards"] == []
    result = call(
        "export",
        args,
        *extra,
        "--output",
        str(tmp_path / "empty.csv"),
        "--reconcile-receipt",
        first["receipt_id"],
    )
    assert result["exported"] == 0
    old = first["export_receipt"]["cards"][0]
    action = result["export_receipt"]["ineligible"][0]
    assert (action["note_id"], action["card_key"]) == (old["note_id"], old["card_key"])
    assert action["previous_decision_ids"] == sorted(old["decision_ids"])
    assert action["resolution"] == "manual"


def test_replay_retains_rejection_for_unselected_never_exported_item(tmp_path: Path) -> None:
    from grammar_pipeline_test import fixture as production_fixture

    args = fixture(tmp_path / "curated")
    source = tmp_path / "curated/input.json"
    data = json.loads(source.read_text())
    second = copy.deepcopy(data["items"][0])
    second["key"] = "second-item"
    second["mnemonics"] = []
    data["items"].append(second)
    write(source, data)
    plan_path = tmp_path / "curated/plan.json"
    plan = json.loads(plan_path.read_text())
    plan["targets"].append({**plan["targets"][0], "item_key": second["key"]})
    write(plan_path, plan)
    history = approve(tmp_path / "curated", args, [])
    records = json.loads(history.read_text())
    old = next(r for r in records if r["item_key"] == second["key"] and r["subject"] == "rendered_card")
    rejection = make_approval(
        **{k: v for k, v in old.items() if k not in ("approval_id", "decision", "supersedes")},
        decision="rejected",
        supersedes=(old["approval_id"],),
    )
    revised = tmp_path / "revised.json"
    write(revised, [*records, rejection])
    profile, _, _ = production_fixture(tmp_path / "production", "en", preset=None)
    raw = json.loads(profile.read_text())
    raw["inputs"].append(dict(kind="curated", path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    raw["selection"]["entries"] = [data["items"][0]["key"]]
    write(profile, raw)
    common = ["--profile", str(profile), "--run", str(tmp_path / "run")]
    first = call("export", args, *common, "--approvals", str(history), "--output", str(tmp_path / "first.csv"))
    replay = call("export", args, *common, "--approvals", str(revised), "--output", str(tmp_path / "first.csv"))
    assert replay["receipt_id"] == first["receipt_id"]
    raw["selection"]["entries"].append(second["key"])
    write(profile, raw)
    result = runner.invoke(
        app,
        [
            "curated",
            "export",
            *args,
            *common,
            "--approvals",
            str(history),
            "--output",
            str(tmp_path / "rollback.csv"),
            "--reconcile-receipt",
            first["receipt_id"],
        ],
    )
    assert result.exit_code == 1 and "history truncated" in result.output
    assert not (tmp_path / "rollback.csv").exists()


def test_interrupted_history_write_can_retry_without_losing_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    args = fixture(tmp_path)
    history = approve(tmp_path, args, [])
    extra = ["--approvals", str(history), "--run", str(tmp_path / "run"), "--output", str(tmp_path / "out.csv")]
    original = os.fsync

    def fail_sync(fd: int) -> None:
        raise OSError("synthetic history interruption")

    monkeypatch.setattr(os, "fsync", fail_sync)
    result = runner.invoke(app, ["curated", "export", *args, *extra])
    assert result.exit_code == 1
    assert not list((tmp_path / "run/curated-history").glob("*.json"))
    monkeypatch.setattr(os, "fsync", original)
    assert call("export", args, *extra)["exported"] == 1


def test_german_safety_guidance_and_review_labels(tmp_path: Path) -> None:
    args = fixture(tmp_path, "de")
    page = tmp_path / "preview.html"
    call("preview", args, "--html", str(page))
    assert "Regelanwendung" in page.read_text() and "Zurückgehalten" in page.read_text()
    history = approve(tmp_path, args, [])
    result = call(
        "export",
        args,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "out.csv"),
    )
    assert "Erstimport" in result["application"]["required_action"]
    assert "Tags" in result["application"]["required_action"]
    records = json.loads(history.read_text())
    old = next(r for r in records if r["subject"] == "rule")
    rejection = make_approval(
        **{k: v for k, v in old.items() if k not in ("approval_id", "decision", "supersedes")},
        decision="rejected",
        supersedes=(old["approval_id"],),
    )
    revised = tmp_path / "rejected.json"
    write(revised, [*records, rejection])
    rejected_page = tmp_path / "rejected.html"
    call("preview", args, "--approvals", str(revised), "--html", str(rejected_page))
    assert "rule:rejected" in rejected_page.read_text()
    assert "Freigaben fehlen" not in rejected_page.read_text()


def test_curated_continuation_retains_approved_sibling_slots(tmp_path: Path) -> None:
    from grammar_pipeline_test import fixture as production_fixture

    args = fixture(tmp_path / "curated")
    source = tmp_path / "curated/input.json"
    profile, _, _ = production_fixture(tmp_path / "production", "en", preset=None)
    raw = json.loads(profile.read_text())
    raw["inputs"].append(dict(kind="curated", path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    raw["selection"]["output_limit"] = 1
    write(profile, raw)
    extra = ["--profile", str(profile), "--include-recipe", "component_explanation"]
    history = approve(tmp_path / "curated", args, extra)
    cards = call("preview", args, *extra, "--approvals", str(history))["cards"]
    common = [*extra, "--approvals", str(history), "--run", str(tmp_path / "run")]
    first = call("export", args, *common, "--output", str(tmp_path / "first.csv"))
    second_args = [*common, "--output", str(tmp_path / "second.csv"), "--after-receipt", first["receipt_id"]]
    second = call("export", args, *second_args)
    assert first["exported"] == second["exported"] == 1
    assert len(second["export_receipt"]["cards"]) == 2 and second["export_receipt"]["ineligible"] == []
    note = csv_note(tmp_path / "second.csv")
    for card in cards:
        slot = next(t.slot for t in CURATED_LAYOUT if t.slot.recipe == card["recipe"])
        assert note[slot.enabled_field] == "1"
        assert note[slot.prompt_field] == card["front_html"]
        assert note[slot.answer_field] == card["back_html"]
    before = (tmp_path / "second.csv").read_bytes()
    assert call("export", args, *second_args)["receipt_id"] == second["receipt_id"]
    assert (tmp_path / "second.csv").read_bytes() == before


def test_formation_target_edit_requires_new_rendered_review(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    extra = ["--exclude-recipe", "rule_application", "--include-recipe", "formation_rule_recall"]
    history = approve(tmp_path, args, extra)
    before = call("preview", args, *extra, "--approvals", str(history))["cards"][0]
    plan = json.loads((tmp_path / "plan.json").read_text())
    plan["targets"][0]["formation"] += " (edited target)"
    write(tmp_path / "plan.json", plan)
    after = call("preview", args, *extra, "--approvals", str(history))["cards"][0]
    assert before["front_html"] != after["front_html"] and after["state"] == "withheld"
    assert next(r for r in after["requirements"] if r["subject"] == "rule")["status"] == "approved"
    result = call(
        "export",
        args,
        *extra,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "out.csv"),
    )
    assert result["exported"] == 0
