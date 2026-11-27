"""Source-supported component and context diagnostics, never whole-card approval."""

import json
from pathlib import Path
from typing import Any

import pytest
from grammar_pipeline_test import call, fixture, rebind, write

from latinitas_cards.acceptance_records import canonical
from latinitas_cards.pipeline_consumers import Analysis, FormObjective, current_cards, load_run
from latinitas_cards.pipeline_profile import PipelineProfile


def unsupported_siblings(task: dict[str, Any], prompt: dict[str, str], answer: str) -> dict[str, str]:
    """Well-formed but unsupported source/rendered values that follow the task's fixed presentation."""
    objective = json.loads(task["objective"])
    supplied = next(e for e in task["source_evidence"] if e["source"] == "unverified-form-entry")
    entry = json.loads(supplied["text"])["entry"]
    template = objective["presentation"][objective["direction"] + "_prompt"]
    return {
        "source": canonical(dict(entry=entry, sense=entry["senses"][0])).decode(),
        "rendered": canonical(dict(prompt=template.format(**prompt), answer=answer)).decode(),
    }


@pytest.mark.parametrize("entry,ending", [("v", "mus"), ("n", "rum")])
def test_named_source_stem_reconstruction(tmp_path: Path, entry: str, ending: str) -> None:
    profile, objectives, _ = fixture(tmp_path)
    data = json.loads((tmp_path / "inputs.json").read_text())
    noun = data["entries"][1]
    noun["forms"] = [{**f, "text": f["text"].replace("puell", "port")} for f in noun["forms"]]
    (tmp_path / "deck.csv").write_text((tmp_path / "deck.csv").read_text().replace("puell", "port"))
    rebind(profile, data)
    raw = json.loads(objectives.read_text())
    objective = next(f for f in raw["forms"] if f["entry"] == entry)
    objective["target"]["number"] = "plural"
    if entry == "n":
        objective["target"]["case"] = "genitive"
    else:
        objective["target"]["person"] = "1"
    raw["forms"] = [objective]
    raw["phrases"] = []
    write(objectives, raw)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    cards, _ = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root)
    card = next(c for c in cards if isinstance(c.objective, FormObjective) and c.objective.entry == entry)
    analysis = Analysis.read_json(card.analysis_path.read_bytes())
    stem = next(a for a in analysis.assertions if a.id == "stem")
    value = json.loads(stem.value)
    assert value == {
        "convention": "bennett-present-stem/1" if entry == "v" else "bennett-declensional-stem/1",
        "components": [
            {"role": "stem", "text": "amā" if entry == "v" else "portā"},
            {"role": "personal-ending" if entry == "v" else "case-ending", "text": ending},
        ],
        "operation": "concatenation",
        "surface": "amāmus" if entry == "v" else "portārum",
    }
    checks = {c.id: c for c in analysis.checks}
    assert checks["target"].result == checks["stem"].result == "pass"
    assert all(c.result != "pass" for c in analysis.checks if c.id not in ("target", "stem"))
    assert any("sha256:" in e.version for e in checks["stem"].evidence)
    assert report["file_export"]["exported"] == 0


@pytest.mark.parametrize(
    "control",
    [
        "correct",
        "wrong-context",
        "wrong-case",
        "alternative",
        "missing-sibling",
        "saved-context",
        "saved-reference",
        "saved-checker",
    ],
)
def test_context_model_proof_reconstructs_original_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    import grammar_pipeline_test as fixtures
    from grammar_pipeline_test import configure_pipeline_http, synthetic_pipeline_http

    profile, objectives = context_fixture(tmp_path, "form_cloze", "____ es")
    target: dict[str, Any] = dict(
        surface="amātus",
        features=dict(
            case="accusative" if control == "wrong-case" else "nominative",
            number="singular",
            gender="masculine",
            tense="perfect",
            voice="passive",
        ),
        source_role="ppp",
        anchors=[],
        rule=None,
        context="____ sunt" if control == "wrong-context" else "____ es",
    )

    def answer(body: dict[str, Any]) -> bytes:
        task = json.loads(body["messages"][1]["content"])
        values = unsupported_siblings(task, dict(lemma="MODEL-UNSUPPORTED", meaning="unsupported"), "amātus")
        assertions = [
            dict(
                id=identity,
                kind="label",
                value=canonical(target).decode() if identity == "target" else values[identity],
                selected_sense=None,
                alternatives=[],
                evidence=[fixtures.EVIDENCE],
            )
            for identity in ("source", "target", "rendered")
            if control != "missing-sibling" or identity != "rendered"
        ]
        result = dict(
            format="latinitas-model-answer",
            version=1,
            task_sha256=task["task_sha256"],
            assertions=assertions,
            alternatives=["perhaps vocative"] if control == "alternative" else [],
            objections=[],
        )
        return canonical({"choices": [{"finish_reason": "stop", "message": {"content": canonical(result).decode()}}]})

    monkeypatch.setattr(fixtures, "provider_answer", answer)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["p"]
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        cards, _ = current_cards(
            profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root
        )
        checks, _ = cards[0].claims.decisions[0].reader.model_checks()
        by_id = {c.id: c for c in checks}
        assert by_id["semantic/target"].result == (
            "fail"
            if control in ("wrong-context", "wrong-case")
            else "unresolved"
            if control == "alternative"
            else "pass"
        )
        assert by_id["semantic/rendered"].result != "pass"
        assert by_id["semantic/source"].result != "pass"
        assert by_id["blind/transport"].assertion_ids == ("source", "target", "rendered")
        if control == "missing-sibling":
            assert by_id["blind/transport"].result != "pass"
        assert len(sent) == 2 and sent[0]["messages"][1]["content"] == sent[1]["messages"][1]["content"]
        assert "MODEL-UNSUPPORTED" not in sent[1]["messages"][1]["content"]
        assert report["file_export"]["exported"] == 0
        if control.startswith("saved-"):
            import shutil

            from latinitas_cards import acceptance_policy, latin_library
            from latinitas_cards.pipeline_orchestration import read_source_comparison

            old = cards[0]
            assert read_source_comparison(root, old) is not None
            before = len(sent)
            replay = call("resume", "--run", str(root))
            assert len(sent) == before and replay["file_export"]["exported"] == 0
            unchanged, _ = current_cards(
                profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root
            )
            assert read_source_comparison(root, unchanged[0]) is not None
            baseline_checks, _ = unchanged[0].claims.decisions[0].reader.model_checks()
            assert next(c for c in baseline_checks if c.id == "semantic/target").result == "pass"
            if control == "saved-context":
                state = load_run(root)
                changed = state.objectives.model_copy(
                    update={"forms": (state.objectives.forms[0].model_copy(update={"context": "____ est"}),)}
                )
                current, _ = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), changed, root)
            else:
                package = Path(acceptance_policy.__file__).parent
                copied = tmp_path / "installed-package"
                shutil.copytree(package, copied, ignore=shutil.ignore_patterns("__pycache__"))
                monkeypatch.setattr(acceptance_policy, "__file__", str(copied / "acceptance_policy.py"))
                reference = copied / "references" / "bennett-1918.html"
                monkeypatch.setattr(latin_library, "BENNETT_PATH", reference)
                path = reference if control == "saved-reference" else copied / "form_evidence.py"
                path.write_bytes(path.read_bytes() + b"\n# changed installed commitment\n")
                current, _ = current_cards(
                    profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root
                )
            assert current[0].binding != old.binding or current[0].versions != old.versions
            assert read_source_comparison(root, current[0]) is None
            if control == "saved-context":
                with pytest.raises(ValueError, match="complete current card"):
                    current[0].claims.decisions[0].reader.model_checks()
            else:
                checks, _ = current[0].claims.decisions[0].reader.model_checks()
                assert not any(c.result == "pass" for c in checks)
            assert len(sent) == before


@pytest.mark.parametrize(
    "recipe,context",
    [
        ("ppp_recognition", "amātus es"),
        ("ppp_recognition", "amātus est"),
        ("form_cloze", "____ es"),
        ("adjective_declension", ""),
    ],
)
def test_contextual_models_receive_selected_task(tmp_path: Path, recipe: str, context: str) -> None:
    from grammar_pipeline_test import configure_pipeline_http, synthetic_pipeline_http

    profile, objectives = context_fixture(tmp_path, recipe, context)
    production_target = dict(case="accusative", number="plural", gender="feminine")
    if recipe == "adjective_declension":
        raw = json.loads(objectives.read_text())
        raw["forms"][0]["target"] = production_target
        write(objectives, raw)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["p"]
        write(profile, settings)
        report = call(
            "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(tmp_path / "workflow")
        )
        assert len(sent) == 2
        task = json.loads(sent[0]["messages"][1]["content"])
        if recipe == "ppp_recognition":
            objective = json.loads(task["objective"])
            assert objective["recipe"] == recipe
            assert objective["context"] == context
            assert objective["selected_form"] == "ppp"
        elif recipe == "form_cloze":
            assert task["target"]["context"] == context
        if recipe == "adjective_declension":
            assert task["target"] == production_target
        else:
            assert task["target"]["features"] == dict(
                case="nominative", number="singular", gender="masculine", tense="perfect", voice="passive"
            )
        assert sent[0]["messages"][1]["content"] == sent[1]["messages"][1]["content"]
        assert "attested-context-binding" not in sent[1]["messages"][1]["content"]
        assert report["file_export"]["exported"] == 0


@pytest.mark.parametrize(
    "control",
    ["correct", "convention", "components", "same-spelling", "sound-change", "malformed", "alternative", "objection"],
)
def test_segmentation_model_proof_is_source_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    import grammar_pipeline_test as fixtures
    from grammar_pipeline_test import configure_pipeline_http, synthetic_pipeline_http

    profile, objectives, _ = fixture(tmp_path)
    raw = json.loads(objectives.read_text())
    raw["forms"] = [
        {
            **raw["forms"][0],
            "target": {"person": "1", "number": "plural", "tense": "present", "mood": "indicative", "voice": "active"},
        }
    ]
    write(objectives, raw)
    anchors = json.loads((tmp_path / "inputs.json").read_text())["entries"][0]["forms"]
    target = dict(
        surface="amāmus",
        features=raw["forms"][0]["target"],
        source_role="derived",
        anchors=anchors,
        rule="regular-first-present-active/1",
    )
    segmentation: dict[str, Any] = dict(
        convention="bennett-present-stem/1",
        components=[dict(role="stem", text="amā"), dict(role="personal-ending", text="mus")],
        operation="concatenation",
        surface="amāmus",
    )
    if control == "convention":
        segmentation["convention"] = "invented-equivalent/1"
    elif control == "components":
        segmentation["components"][0]["text"] = "ama"
    elif control == "same-spelling":
        segmentation["components"] = [dict(role="stem", text="am"), dict(role="personal-ending", text="āmus")]
    elif control == "sound-change":
        segmentation["operation"] = "lengthening"

    def answer(body: dict[str, Any]) -> bytes:
        task = json.loads(body["messages"][1]["content"])
        stem_line = json.loads(task["objective"])["presentation"]["stem_line"]
        stem = segmentation["components"][0]["text"]
        siblings = unsupported_siblings(task, dict(cue="MODEL-UNSUPPORTED"), "amāmus\n" + stem_line.format(stem=stem))
        values = {
            "source": siblings["source"],
            "target": canonical(target).decode(),
            "stem": "not JSON" if control == "malformed" else canonical(segmentation).decode(),
            "rendered": siblings["rendered"],
        }
        assertions = [
            dict(
                id=identity,
                kind="segmentation" if identity == "stem" else "label",
                value=value,
                selected_sense=None,
                alternatives=[],
                evidence=[fixtures.EVIDENCE],
            )
            for identity, value in values.items()
        ]
        result = dict(
            format="latinitas-model-answer",
            version=1,
            task_sha256=task["task_sha256"],
            assertions=assertions,
            alternatives=["another convention"] if control == "alternative" else [],
            objections=["unresolved quantity"] if control == "objection" else [],
        )
        return canonical({"choices": [{"finish_reason": "stop", "message": {"content": canonical(result).decode()}}]})

    monkeypatch.setattr(fixtures, "provider_answer", answer)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"]
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        # Broken literal concatenation and malformed values are processing failures, not verdicts.
        failure = {
            "components": "contract_violation",
            "sound-change": "malformed_output",
            "malformed": "malformed_output",
        }
        if control in failure:
            assert [(c["state"], c["reason"]) for c in report["provider_calls"]] == [("failed", failure[control])]
            assert len(sent) == 1 and report["file_export"]["exported"] == 0
            return
        card = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root)[
            0
        ][0]
        checks, _ = card.claims.decisions[0].reader.model_checks()
        check = next(c for c in checks if c.id == "semantic/stem")
        assert check.result == (
            "pass" if control == "correct" else "unresolved" if control in ("alternative", "objection") else "fail"
        )
        assert set(next(c for c in checks if c.id == "blind/transport").assertion_ids) == {
            "source",
            "target",
            "stem",
            "rendered",
        }
        assert len(sent) == 2 and sent[0]["messages"][1]["content"] == sent[1]["messages"][1]["content"]
        assert "MODEL-UNSUPPORTED" not in sent[1]["messages"][1]["content"]
        assert "amāmus" not in sent[1]["messages"][1]["content"]
        assert report["file_export"]["exported"] == 0
        before = {p: p.read_bytes() for p in (root / "stages").glob("*.json")}
        call("resume", "--run", str(root), "--offline")
        assert len(sent) == 2 and all(path.read_bytes() == value for path, value in before.items())


def context_fixture(tmp_path: Path, recipe: str, context: str) -> tuple[Path, Path]:
    profile, objectives, _ = fixture(tmp_path, language="en", preset=None)
    settings = json.loads(profile.read_text())
    settings["recipes"] = [recipe]
    settings["recipe_overrides"] = {
        "include": [recipe],
        "exclude": ["form_translation", "ppp_translation", "phrase_translation"],
    }
    write(profile, settings)
    raw = json.loads(objectives.read_text())
    template = raw["forms"][0]
    raw["forms"] = [{**template, "entry": "p", "form": "ppp", "anchors": [], "recipe": recipe, "context": context}]
    raw["phrases"] = []
    write(objectives, raw)
    return profile, objectives


@pytest.mark.parametrize("recipe", ["ppp_recognition", "form_cloze"])
@pytest.mark.parametrize(
    "surface,auxiliary,person,number",
    [
        ("amātus", "sum", "1", "singular"),
        ("amātus", "es", "2", "singular"),
        ("amātus", "est", "3", "singular"),
        ("amātī", "sumus", "1", "plural"),
        ("amātī", "estis", "2", "plural"),
        ("amātī", "sunt", "3", "plural"),
    ],
)
def test_attested_contextual_ppp_diagnostic(
    tmp_path: Path, recipe: str, surface: str, auxiliary: str, person: str, number: str
) -> None:
    context = ("____" if recipe == "form_cloze" else surface) + " " + auxiliary
    profile, objectives = context_fixture(tmp_path, recipe, context)
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][3]["forms"][0].update(text=surface)
    data["entries"][3]["forms"][0]["features"]["number"] = number
    if number == "plural":
        (tmp_path / "deck.csv").write_text((tmp_path / "deck.csv").read_text().replace("amātus", "amātī"))
    rebind(profile, data)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    cards, skipped = current_cards(
        profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root
    )
    assert not [s for s in skipped if "work_id" in s]
    card = next(c for c in cards if isinstance(c.objective, FormObjective) and c.objective.entry == "p")
    analysis = Analysis.read_json(card.analysis_path.read_bytes())
    target = next(a for a in analysis.assertions if a.id == "target")
    value = json.loads(target.value)
    assert value["surface"] == surface and value["context"] == context
    assert value["features"] == {
        "case": "nominative",
        "number": number,
        "gender": "masculine",
        "tense": "perfect",
        "voice": "passive",
        "person": None,
        "mood": None,
    }
    check = next(c for c in analysis.checks if c.id == "target")
    assert check.result == "pass"
    assert {c.id for c in analysis.checks} == {a.id for a in analysis.assertions}
    assert all(c.result != "pass" for c in analysis.checks if c.id != "target")
    assert context in card.prompt
    context_proof = next(e for e in check.evidence if e.source == "attested-context-binding")
    proof = json.loads(context_proof.text)
    assert proof["context"] == surface + " " + auxiliary and proof["target_span"] == [0, len(surface)]
    assert proof["auxiliary"]["surface"] == auxiliary and proof["auxiliary"]["person"] == person
    assert {int(e.reference.rsplit("sect", 1)[1].split("/")[0]) for e in check.evidence if "#sect" in e.reference} >= {
        63,
        95,
        100,
        101,
        102,
        166,
        233,
        234,
        254,
    }
    assert proof["nominal_basis"] == {
        "classification": "adjective-form",
        "construction": "subject-agreement",
        "case": "nominative",
        "gender": "masculine",
        "number": number,
        "paradigm_exemplar": "bonus" if number == "singular" else "bonī",
        "rules": ["95", "166", "233.3", "234", "254.1", "254.2"],
    }
    assert report["file_export"]["exported"] == 0
    assert call("export", "--run", str(root), "--output", str(tmp_path / "denied.csv"))["exported"] == 0


@pytest.mark.parametrize(
    "control",
    [
        "isolated",
        "number",
        "case",
        "voice",
        "unknown-context",
        "contradictory-context",
        "wrong-lemma",
        "supine",
        "supine-as-ppp",
        "missing-resource",
        "tampered-resource",
    ],
)
def test_context_cannot_be_self_certified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str) -> None:
    from latinitas_cards import latin_library

    context = {"isolated": "", "unknown-context": "amātus vidētur", "contradictory-context": "amātus sunt"}.get(
        control, "amātus es"
    )
    profile, objectives = context_fixture(tmp_path, "ppp_recognition", context)
    data = json.loads((tmp_path / "inputs.json").read_text())
    form = data["entries"][3]["forms"][0]
    if control == "number":
        form["features"]["number"] = "plural"
    elif control == "case":
        form["features"]["case"] = "vocative"
    elif control == "voice":
        form["features"]["voice"] = "active"
    elif control == "supine":
        form["source_role"] = "supine"
    elif control == "supine-as-ppp":
        form["text"] = "amātum"
        (tmp_path / "deck.csv").write_text((tmp_path / "deck.csv").read_text().replace("amātus", "amātum"))
        raw = json.loads(objectives.read_text())
        raw["forms"][0]["context"] = "amātum es"
        write(objectives, raw)
    elif control == "wrong-lemma":
        (tmp_path / "deck.csv").write_text((tmp_path / "deck.csv").read_text().replace(",amō,", ",moneō,"))
    rebind(profile, data)
    if control in ("missing-resource", "tampered-resource"):
        reference = tmp_path / "missing.html"
        if control == "tampered-resource":
            reference.write_bytes(latin_library.BENNETT_PATH.read_bytes().replace(b"sumus", b"sunt "))
        monkeypatch.setattr(latin_library, "BENNETT_PATH", reference)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    cards, skipped = current_cards(
        profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root
    )
    if control in ("voice", "supine"):
        assert not cards and any("PPP" in str(s["reason"]) for s in skipped)
    else:
        assert len(cards) == 1
    assert all(
        next(c for c in Analysis.read_json(card.analysis_path.read_bytes()).checks if c.id == "target").result != "pass"
        for card in cards
    )
    assert report["file_export"]["exported"] == 0
