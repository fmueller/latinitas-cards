"""Synthetic transport and confirmed source roles, not linguistic qualification."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from grammar_pipeline_test import (
    EVIDENCE,
    call,
    configure_pipeline_http,
    fixture,
    rebind,
    synthetic_pipeline_http,
    write,
)

from latinitas_cards.acceptance import load_stage, save_stage, stage_artifact
from latinitas_cards.acceptance_records import canonical, digest

PRESENT = dict(person="1", number="singular", tense="present", mood="indicative", voice="active")


@pytest.mark.parametrize("control", ["correct", "wrong-cell", "missing-sibling", "alternative", "objection"])
def test_attested_cell_reconstructs_complete_model_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    import grammar_pipeline_test as fixtures

    from latinitas_cards.pipeline_consumers import current_cards, load_run
    from latinitas_cards.pipeline_profile import PipelineProfile

    profile, objectives, _ = fixture(tmp_path)
    raw = json.loads(objectives.read_text())
    raw["forms"] = [{**raw["forms"][0], "target": {**PRESENT, "number": "plural"}}]
    write(objectives, raw)
    anchors = json.loads((tmp_path / "inputs.json").read_text())["entries"][0]["forms"]

    def answer(body: dict[str, Any]) -> bytes:
        task = json.loads(body["messages"][1]["content"])
        value = dict(
            surface="amāmus" if control != "wrong-cell" else "amātis",
            features={**PRESENT, "number": "plural"},
            source_role="derived",
            anchors=anchors,
            rule="regular-first-present-active/1",
        )
        assertions = [
            dict(
                id=identity,
                kind=kind,
                value=canonical(value).decode() if identity == "target" else "unsupported sibling",
                selected_sense=None,
                alternatives=["perhaps another voice"] if identity == "target" and control == "alternative" else [],
                evidence=[EVIDENCE],
            )
            for identity, kind in (
                ("source", "label"),
                ("target", "label"),
                ("stem", "segmentation"),
                ("rendered", "label"),
            )
            if control != "missing-sibling" or identity != "stem"
        ]
        content = dict(
            format="latinitas-model-answer",
            version=1,
            task_sha256=task["task_sha256"],
            assertions=assertions,
            alternatives=[],
            objections=["unresolved exception"] if control == "objection" else [],
        )
        return canonical({"choices": [{"finish_reason": "stop", "message": {"content": canonical(content).decode()}}]})

    monkeypatch.setattr(fixtures, "provider_answer", answer)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"]
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        card = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root)[
            0
        ][0]
        checks, _ = card.claims.decisions[0].reader.model_checks()
        by_id = {c.id: c for c in checks}
        if control == "wrong-cell":
            expected_result = "fail"
        elif control in ("alternative", "objection"):
            expected_result = "unresolved"
        else:
            expected_result = "pass"
        assert by_id["semantic/target"].result == expected_result
        assert all(
            c.result != "pass" for c in checks if c.method == "semantic_comparison" and c.id != "semantic/target"
        )
        assert by_id["blind/transport"].assertion_ids == tuple(a.id for a in card.claims.assertions)
        if control == "missing-sibling":
            assert by_id["blind/transport"].result != "pass"
        elif control == "correct":
            assert by_id["blind/transport"].result == "pass"
        assert report["file_export"]["exported"] == 0
        assert len(sent) == 2 and sent[0]["messages"][1]["content"] == sent[1]["messages"][1]["content"]
        assert "amāmus" not in sent[1]["messages"][1]["content"]
        before = {p: p.read_bytes() for p in (root / "stages").glob("*.json")}
        call("resume", "--run", str(root), "--offline")
        assert len(sent) == 2 and all(path.read_bytes() == data for path, data in before.items())


def source_answer(body: dict[str, Any]) -> bytes:
    task = json.loads(body["messages"][1]["content"])
    # Written from the disposable mapped source, never from the renderer.
    target: dict[str, object] = dict(surface="amō", features=PRESENT, source_role="present_1s", anchors=[], rule=None)
    answer = dict(
        format="latinitas-model-answer",
        version=1,
        task_sha256=task["task_sha256"],
        assertions=[
            dict(
                id="target",
                kind="label",
                value=canonical(target).decode(),
                selected_sense=None,
                alternatives=[],
                evidence=[EVIDENCE],
            ),
            dict(
                id="translation",
                kind="translation",
                value="synthetic unsupported prose",
                selected_sense="sense",
                alternatives=[],
                evidence=[EVIDENCE],
            ),
        ],
        alternatives=[],
        objections=[],
    )
    return canonical({"choices": [{"finish_reason": "stop", "message": {"content": canonical(answer).decode()}}]})


def setup_source(root: Path) -> tuple[Path, Path]:
    profile, objectives, _ = fixture(root, language="en", preset=None)
    settings = json.loads(profile.read_text())
    settings["recipes"] = ["form_translation"]
    settings["recipe_overrides"] = {"include": [], "exclude": ["ppp_translation", "phrase_translation"]}
    write(profile, settings)
    data = json.loads(objectives.read_text())
    data["forms"] = [
        {
            **data["forms"][0],
            "recipe": "form_translation",
            "form": "present",
            "anchors": [],
            "target": {},
            "translations": [],
            "explanation": "",
        }
    ]
    write(objectives, data)
    inputs = json.loads((root / "inputs.json").read_text())
    inputs["entries"][0]["forms"] = inputs["entries"][0]["forms"][:1]
    write(root / "inputs.json", inputs)
    rebind(profile, inputs)
    return profile, objectives


def test_source_comparison_proof_through_offline_consumers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import grammar_pipeline_test as fixtures

    monkeypatch.setattr(fixtures, "provider_answer", source_answer)
    profile, objectives = setup_source(tmp_path)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"]
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        assert len(sent) == 2
        row = report["stages"]["counter_check"][0]
        proof = cast(dict[str, Any], stage_artifact(root, load_stage(root, row["stage_id"])))["semantic_comparison"]
        by_id = {item["assertion_id"]: item for item in proof["comparisons"]}
        assert by_id["target"]["relation"] == "equivalent"
        assert by_id["target"]["check"]["result"] == "pass"
        assert proof["versions"]["proposal_prompt"] == "proposal/1"
        assert proof["versions"]["counter_prompt"] == "blind/1"
        assert proof["versions"]["proposal_model"] and proof["versions"]["counter_model"]
        assert by_id["translation"]["relation"] == "unresolved"
        assert by_id["source"]["check"]["result"] == "unresolved"
        assert by_id["rendered"]["check"]["result"] == "unresolved"
        assert proof["relation"] == "unresolved" and proof["authorizes_acceptance"] is False
        assert report["file_export"]["exported"] == 0
        assert report["file_export"]["export_receipt"]["cards"] == []
        assert json.loads(sent[0]["messages"][1]["content"]) == json.loads(sent[1]["messages"][1]["content"])
        assert "synthetic unsupported prose" not in sent[1]["messages"][1]["content"]
        decision = call("stage", "decide", "--run", str(root))["stages"]["decide"][0]
        assert decision["decisions"][0]["semantic_comparison"] == "unresolved"
        assert decision["decisions"][0]["state"] == "withheld"
        assert call("preview", "--run", str(root))["cards"][0]["state"] == "withheld"
        call("resume", "--run", str(root), "--offline")
        call("resume", "--run", str(root))
        assert len(sent) == 2


@pytest.mark.parametrize("mutation", ["feature", "surface", "missing", "objection", "alternative", "sense", "context"])
def test_agreement_does_not_supply_source_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    import grammar_pipeline_test as fixtures

    def answer(body: dict[str, Any]) -> bytes:
        wrapper = json.loads(source_answer(body))
        content = json.loads(wrapper["choices"][0]["message"]["content"])
        target = content["assertions"][0]
        value = json.loads(target["value"])
        if mutation == "feature":
            value["features"]["person"] = "3"
        elif mutation == "surface":
            value["surface"] = "amant"
        elif mutation == "missing":
            content["assertions"] = content["assertions"][1:]
        elif mutation == "objection":
            content["objections"] = ["unresolved applicability"]
        elif mutation == "alternative":
            target["alternatives"] = ["perhaps passive"]
        elif mutation == "sense":
            content["assertions"][1]["selected_sense"] = "different-sense"
        target["value"] = canonical(value).decode()
        wrapper["choices"][0]["message"]["content"] = canonical(content).decode()
        return canonical(wrapper)

    monkeypatch.setattr(fixtures, "provider_answer", answer)
    profile, objectives = setup_source(tmp_path)
    if mutation == "context":
        raw = json.loads(objectives.read_text())
        raw["forms"][0]["context"] = "a context requiring independent applicability"
        write(objectives, raw)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"].update(entries=["v"], variants_per_binding=32)
        settings["processing"]["max_source_units"] = None
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        proofs = [
            cast(dict[str, Any], stage_artifact(root, load_stage(root, row["stage_id"])))["semantic_comparison"]
            for row in report["stages"]["counter_check"]
        ]
        assert proofs and all(p["relation"] != "equivalent" for p in proofs)
        expected = "contradiction" if mutation in ("feature", "surface") else "unresolved"
        targets = [c for p in proofs for c in p["comparisons"] if c["assertion_id"] == "target"]
        if mutation == "sense":
            assert all(c["relation"] == "equivalent" for c in targets)
            translations = [c for p in proofs for c in p["comparisons"] if c["assertion_id"] == "translation"]
            assert all(c["relation"] == "unresolved" and c["check"]["result"] != "pass" for c in translations)
            assert all(
                "translation_requires_sense_government_and_semantic_evidence" in c["check"]["objections"]
                for c in translations
            )
        elif mutation == "context":
            # Automatic enumeration's context-free sibling is distinct.
            assert any(c["relation"] == "unresolved" for c in targets)
        else:
            assert all(c["relation"] == expected for c in targets)
        assert report["file_export"]["exported"] == 0
        before = len(sent)
        call("resume", "--run", str(root), "--offline")
        assert len(sent) == before


def recommit_stage_tree(root: Path, identity: str) -> None:
    """An attacker can recompute hashes, but cannot make a false proof true."""
    stage = load_stage(root, identity)
    assert stage.artifact is not None
    artifact_path = root / stage.artifact
    artifact = json.loads(artifact_path.read_text())
    if stage.prerequisite_ids:
        artifact["prerequisite_output_sha256s"] = [load_stage(root, p).output_sha256 for p in stage.prerequisite_ids]
        write(artifact_path, artifact)
    record = stage.model_dump()
    record.update(input_sha256=digest(artifact), output_sha256=digest(artifact))
    write(root / "stages" / (identity + ".json"), record)
    for path in (root / "stages").glob("*.json"):
        parent = json.loads(path.read_text())
        if identity in parent["prerequisite_ids"]:
            recommit_stage_tree(root, parent["stage_id"])


@pytest.mark.parametrize(
    "damage",
    [
        "request",
        "response",
        "proof",
        "request_recommit",
        "response_recommit",
        "proof_recommit",
        "omit_sibling_recommit",
        "stage_result_recommit",
        "missing_result_recommit",
        "redirect_reference",
        "proposal_response",
        "proposal_response_recommit",
        "scope",
        "wrapper_stage_type",
    ],
)
def test_source_comparison_tampering_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    import hashlib

    import grammar_pipeline_test as fixtures

    from latinitas_cards.pipeline_consumers import current_cards, load_run

    monkeypatch.setattr(fixtures, "provider_answer", source_answer)
    profile, objectives = setup_source(tmp_path)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"]
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        assert len(sent) == 2 and report["stages"]["counter_check"][0]["semantic_comparison"] == "unresolved"
        saved = load_run(root)
        card = current_cards(profile, saved.profile, saved.objectives, root)[0][0]
        destination = card.claims.decisions[0]
        assert destination.reader.model_checks()[0]
        ledger_path = root / "providers" / "provider-ledger.json"
        ledger = json.loads(ledger_path.read_text())
        counter = ledger["calls"][0 if damage.startswith("proposal") else -1]
        stage_damage = damage in (
            "proof",
            "proof_recommit",
            "omit_sibling_recommit",
            "stage_result_recommit",
            "missing_result_recommit",
            "redirect_reference",
        )
        if damage == "wrapper_stage_type":
            index = json.loads((root / "stage-index.json").read_text())
            identity = next(value for key, value in index.items() if key.startswith("counter_check/"))
            artifact = root / "stages" / (identity + ".json")
        elif stage_damage:
            stage = load_stage(root, report["stages"]["counter_check"][0]["stage_id"])
            assert stage.artifact is not None
            artifact = root / stage.artifact
        elif damage == "scope":
            artifact = root / "providers" / (counter["call_id"] + ".scope.json")
        else:
            artifact = (
                root
                / "providers"
                / counter["request_artifact" if damage.startswith("request") else "response_artifact"]
            )
        raw = json.loads(artifact.read_text())
        if damage == "wrapper_stage_type":
            assert raw["stage"] == "counter_check"
            raw["stage"] = "analyze"
        elif damage == "scope":
            raw[0]["fields"] = ["unselected-field"]
        elif damage.startswith("request"):
            raw["messages"].append({"role": "user", "content": "leaked proposal answer"})
        elif "response" in damage:
            content = json.loads(raw["choices"][0]["message"]["content"])
            content["objections"] = ["late contradiction"]
            raw["choices"][0]["message"]["content"] = canonical(content).decode()
        elif damage in ("omit_sibling_recommit", "missing_result_recommit", "redirect_reference"):
            proof = raw["semantic_comparison"]
            proof["comparisons"] = [c for c in proof["comparisons"] if c["assertion_id"] == "target"]
            proof["relation"] = "equivalent"
            proof["complete_assertion_coverage"] = True
            if damage == "missing_result_recommit":
                raw.pop("result")
        elif damage == "stage_result_recommit":
            raw["result"]["objections"] = ["edited cached result"]
        else:
            raw["semantic_comparison"]["comparisons"][0]["relation"] = "equivalent"
            raw["semantic_comparison"]["authorizes_acceptance"] = True
        if damage == "redirect_reference":
            forged = save_stage(
                root,
                run_id=stage.run_id,
                stage_id="forged-counter",
                stage="counter_check",
                binding=stage.binding,
                versions=stage.versions,
                envelope=raw,
            )
            wrapper = load_stage(root, report["stages"]["counter_check"][0]["execution_stage_id"])
            assert wrapper.artifact is not None
            wrapper_path = root / wrapper.artifact
            content = json.loads(wrapper_path.read_text())
            content["outcomes"][0]["stage_id"] = forged.stage_id
            write(wrapper_path, content)
            recommit_stage_tree(root, wrapper.stage_id)
        else:
            write(artifact, raw)
        if stage_damage and damage.endswith("recommit"):
            recommit_stage_tree(root, stage.stage_id)
        elif damage.endswith("recommit"):
            key = "request_sha256" if damage.startswith("request") else "response_sha256"
            counter[key] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            write(ledger_path, ledger)
        with pytest.raises(ValueError):
            destination.reader.model_checks()
        _, eligibility = destination.reader.read(destination.path)
        assert not eligibility.eligible and eligibility.reason == "stale_evidence"
        decision = call("stage", "decide", "--run", str(root))["stages"]["decide"][0]
        assert decision["analysis_errors"] and not decision["decisions"]
        replay = call("resume", "--run", str(root), "--offline")
        assert replay["file_export"]["exported"] == 0 and len(sent) == 2
        assert replay["stages"]["decide"][0]["analysis_errors"]
        assert not replay["stages"]["decide"][0]["decisions"]


@pytest.mark.parametrize("change", ["prompt", "model"])
def test_source_comparison_invalidates_changed_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    import grammar_pipeline_test as fixtures

    from latinitas_cards.pipeline_consumers import current_cards, load_run
    from latinitas_cards.pipeline_orchestration import read_source_comparison
    from latinitas_cards.pipeline_profile import load_pipeline_profile
    from latinitas_cards.providers import PROMPT_VERSIONS

    monkeypatch.setattr(fixtures, "provider_answer", source_answer)
    profile, objectives = setup_source(tmp_path)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"]
        write(profile, settings)
        output = tmp_path / "workflow"
        call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        saved = load_run(root)
        old = current_cards(profile, saved.profile, saved.objectives, root)[0][0]
        assert read_source_comparison(root, old) is not None
        if change == "prompt":
            monkeypatch.setitem(PROMPT_VERSIONS, "counter_check", "blind/changed-synthetic")
        else:
            settings["providers"]["counter_check"]["model"] = "changed-synthetic-deployment"
            write(profile, settings)
        current = current_cards(profile, load_pipeline_profile(profile), saved.objectives, root)[0][0]
        assert current.versions != old.versions
        assert read_source_comparison(root, current) is None
        replay = call("resume", "--run", str(root), "--profile", str(profile), "--offline")
        assert replay["file_export"]["exported"] == 0 and len(sent) == 2
        assert all(d["semantic_comparison"] == "not_available" for d in replay["stages"]["decide"][0]["decisions"])


def complete_source_answer(body: dict[str, Any]) -> bytes:
    """Cover every requested sibling, without claiming unsupported prose is true."""
    wrapper = json.loads(source_answer(body))
    answer = json.loads(wrapper["choices"][0]["message"]["content"])
    for identity in ("source", "rendered"):
        answer["assertions"].append(
            dict(
                id=identity,
                kind="label",
                value="synthetic unsupported sibling",
                selected_sense=None,
                alternatives=[],
                evidence=[EVIDENCE],
            )
        )
    wrapper["choices"][0]["message"]["content"] = canonical(answer).decode()
    return canonical(wrapper)


def test_acceptance_reader_regenerates_complete_model_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    import grammar_pipeline_test as fixtures

    from latinitas_cards.acceptance import Acceptance, save_acceptance
    from latinitas_cards.pipeline_consumers import current_cards, load_run

    monkeypatch.setattr(fixtures, "provider_answer", complete_source_answer)
    profile, objectives = setup_source(tmp_path)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"]
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        saved = load_run(root)
        card = current_cards(profile, saved.profile, saved.objectives, root)[0][0]
        destination = card.claims.decisions[0]
        reader = destination.reader
        checks, stages = reader.model_checks()
        assert len(sent) == 2
        assert {c.id for c in checks} == {
            "blind/transport",
            "semantic/source",
            "semantic/target",
            "semantic/translation",
            "semantic/rendered",
        }
        blind = next(c for c in checks if c.method == "blind_model")
        assert blind.assertion_ids == tuple(a.id for a in reader.assertions)
        assert blind.result == "pass"
        target = next(c for c in checks if c.id == "semantic/target")
        assert target.result == "pass" and not target.objections
        assert all(c.result == "unresolved" for c in checks if c.id not in (blind.id, target.id))
        stored, eligibility = reader.read(destination.path)
        assert set(checks) <= set(stored.checks)
        assert set(stages) <= set(stored.stage_ids)
        assert not eligibility.eligible and eligibility.reason == "policy_not_enabled"
        assert report["file_export"]["exported"] == 0
        assert report["file_export"]["export_receipt"]["cards"] == []
        partial = replace(reader, assertions=(reader.assertions[1],), cells=(reader.cells[1],))
        with pytest.raises(ValueError, match="complete current card"):
            partial.model_checks()
        assert partial.decide_automatic(checks=checks, stage_ids=stages).state == "withheld"
        for damage in (
            "blind_only",
            "omit_semantic_sibling",
            "forged_check",
            "forged_state",
            "missing_graph",
            "human_mix",
        ):
            damaged_checks = checks
            fields: dict[str, Any] = stored.model_dump(exclude={"decision_id"})
            if damage == "blind_only":
                damaged_checks = (blind,)
            elif damage == "omit_semantic_sibling":
                damaged_checks = tuple(c for c in checks if c.id != "semantic/translation")
            elif damage == "forged_check":
                damaged_checks = tuple(c.model_copy(update={"version": "forged"}) if c == target else c for c in checks)
            elif damage == "forged_state":
                damaged_checks = tuple(
                    c.model_copy(update={"result": "pass", "objections": (), "alternatives": ()}) for c in checks
                )
                fields.update(state="accepted", reason="accepted")
            elif damage == "missing_graph":
                fields["stage_ids"] = tuple(s for s in stored.stage_ids if s not in stages)
            elif damage == "human_mix":
                damaged_checks = (*checks, target.model_copy(update={"id": "human", "method": "human"}))
            forged = Acceptance.create(**{**fields, "checks": damaged_checks})
            forged_path = root / (damage + "-decision.json")
            save_acceptance(forged_path, forged)
            _, rejected = reader.read(forged_path)
            assert not rejected.eligible
            assert rejected.reason == ("origin_mismatch" if damage == "human_mix" else "stale_evidence")
        call("resume", "--run", str(root), "--offline")
        assert len(sent) == 2
        for analysis in (root / "analyses").rglob("*.json"):
            analysis.unlink()
        for malformed_index in ("[]", "null", '"not an index"'):
            (root / "stage-index.json").write_text(malformed_index)
            with pytest.raises(ValueError, match="stage index must be an object"):
                reader.model_checks()
            _, rejected = reader.read(destination.path)
            assert not rejected.eligible and rejected.reason == "stale_evidence"
