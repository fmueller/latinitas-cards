"""Finite source-derived compatibility, not a semantic-acceptance campaign."""

import json
from pathlib import Path
from typing import Any

import pytest
from grammar_pipeline_test import bennett_phrase_fixture, call, phrase_fixture, rebind, write

from latinitas_cards.acceptance_records import canonical
from latinitas_cards.pipeline_consumers import Analysis, current_cards, load_run, load_source, source_claims


def manifest_phrase_inputs(profile: Path) -> None:
    from latinitas_cards.manifest import reconcile_csv_manifest
    from latinitas_cards.sources import read_source_records

    base = profile.parent
    deck = json.loads((base / "deck-profile.json").read_text())
    deck["source_identity"] = {"strategy": "manifest", "field": None}
    write(base / "deck-profile.json", deck)
    records = read_source_records(base / "deck.csv")
    reconciled = reconcile_csv_manifest(
        records,
        None,
        source_scope="synthetic-scope",
        approved_allocations={index: record.fields["ID"] for index, record in enumerate(records)},
    )
    assert reconciled.complete and not reconciled.reviews
    reconciled.manifest.save(base / "deck.csv.latinitas.json")
    assert not reconcile_csv_manifest(records, reconciled.manifest).reviews
    rebind(profile, json.loads((base / "inputs.json").read_text()))


@pytest.mark.parametrize("control", ["plain", "html", "whitespace", "manifest"])
def test_phrase_public_source_encodings_preserve_independent_support(tmp_path: Path, control: str) -> None:
    from latinitas_cards.pipeline_consumers import Objectives
    from latinitas_cards.pipeline_orchestration import _task
    from latinitas_cards.pipeline_profile import PipelineProfile

    profile, objectives, _ = bennett_phrase_fixture(tmp_path, "noun-genitive")
    if control == "manifest":
        manifest_phrase_inputs(profile)
    elif control != "plain":
        deck = tmp_path / "deck.csv"
        value = "<b>hortus</b>" if control == "html" else " hortus "
        deck.write_text(deck.read_text().replace("h,hortus,", "h," + value + ","))
        rebind(profile, json.loads((tmp_path / "inputs.json").read_text()))
    parsed = PipelineProfile.read_json(profile.read_bytes())
    source = load_source(profile, parsed)
    card = current_cards(profile, parsed, Objectives.read_json(objectives.read_bytes()), tmp_path)[0][0]
    task, scopes = _task(card, source, parsed)
    assert task.reference_constraints == () and scopes
    expected, _, reason = source_claims(source, card.objective, card.versions)
    assert "construction" in expected, reason
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    current = current_cards(profile, parsed, load_run(output / "run").objectives, output / "run")[0][0]
    assert (
        next(c for c in Analysis.read_json(current.analysis_path.read_bytes()).checks if c.id == "construction").result
        == "pass"
    )
    assert report["file_export"]["exported"] == 0


def test_phrase_counter_task_exposes_input_facts_not_expected_proof(tmp_path: Path) -> None:
    from latinitas_cards.pipeline_consumers import Objectives
    from latinitas_cards.pipeline_orchestration import _task
    from latinitas_cards.pipeline_profile import PipelineProfile

    profile, objectives, _ = bennett_phrase_fixture(tmp_path, "noun-genitive")
    parsed = PipelineProfile.read_json(profile.read_bytes())
    source = load_source(profile, parsed)
    card = current_cards(profile, parsed, Objectives.read_json(objectives.read_bytes()), tmp_path)[0][0]
    task, scope = _task(card, source, parsed)
    supplied = {
        e.reference.rsplit("/", 1)[-1]: json.loads(e.text)
        for e in task.source_evidence
        if e.source == "unverified-phrase-entry"
    }
    assert set(supplied) == {"n", "h"}
    assert supplied["h"]["entry"]["senses"][0]["meaning"] == "garden"
    assert any(f["text"] == "hortī" for f in supplied["h"]["entry"]["forms"])
    assert supplied["h"]["identity"]["value"] == "h"
    pattern = next(json.loads(e.text) for e in task.source_evidence if e.source == "unverified-phrase-pattern")
    assert pattern["family"] == "noun-genitive" and pattern["roles"]["dependent"]["requires"]["case"] == "genitive"
    objective = json.loads(task.objective)
    assert {b["role"]: b["entry"] for b in objective["bindings"]} == {"head": "n", "dependent": "h"}
    assert {a["id"] for a in objective["assertions"]} == {
        "construction",
        "translation-0",
        "explanation-0",
        "roles-0",
        "rendered",
    }
    assert all(set(a) == {"id", "kind"} for a in objective["assertions"])
    assert any(s.fields == ("phrase-input/entry",) and s.source_identity == "h" for s in scope)
    assert any(s.fields == ("phrase-input/pattern",) for s in scope)
    assert "source-bound-constituent" not in task.model_dump_json() and "Bennett" not in task.model_dump_json()
    assert task.reference_constraints == ()


@pytest.mark.parametrize("control", ["sense-key", "meaning", "noun-valency", "unsupported-lemma", "morphology"])
def test_phrase_source_claims_are_not_self_corroborating(tmp_path: Path, control: str) -> None:
    profile, objectives, _ = bennett_phrase_fixture(tmp_path, "preposition-noun")
    data = json.loads((tmp_path / "inputs.json").read_text())
    raw = json.loads(objectives.read_text())
    noun = data["entries"][1]
    if control == "sense-key":
        noun["senses"][0]["key"] = "unresolved"
        data["patterns"][0]["roles"]["object"]["senses"] = ["n/unresolved"]
        raw["phrases"][0]["selection"]["bindings"][0]["sense"] = "unresolved"
    elif control == "meaning":
        noun["senses"][0]["meaning"] = "door"
        deck = tmp_path / "deck.csv"
        deck.write_text(deck.read_text().replace("gate", "door"))
    elif control == "noun-valency":
        noun["senses"][0]["valency"] = {"object": "dative"}
    elif control == "unsupported-lemma":
        deck = tmp_path / "deck.csv"
        deck.write_text(deck.read_text().replace("porta,", "puella,"))
    else:
        next(f for f in noun["forms"] if f["key"] == "abl")["features"]["gender"] = "masculine"
        noun["gender"] = "masculine"
        data["patterns"][0]["roles"]["object"]["requires"]["gender"] = "masculine"
        for reading in raw["phrases"][0]["reference"]["readings"]:
            reading["features"]["object"]["gender"] = "masculine"
    rebind(profile, data)
    write(objectives, raw)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    run = load_run(root)
    cards, _ = current_cards(profile, run.profile, run.objectives, root)
    assert len(cards) == 1
    analysis = Analysis.read_json(cards[0].analysis_path.read_bytes())
    check = next(c for c in analysis.checks if c.id == "construction")
    assert check.result != "pass" and check.objections
    assert report["file_export"]["exported"] == 0


@pytest.mark.parametrize("control", ["government", "dual-government", "agreement", "passive", "missing"])
def test_incompatible_phrase_inputs_withhold_at_public_boundary(tmp_path: Path, control: str) -> None:
    family = {"agreement": "adjective-noun", "passive": "verb-noun", "missing": "noun-genitive"}.get(
        control, "preposition-noun"
    )
    profile, objectives, _ = bennett_phrase_fixture(tmp_path, family)
    data = json.loads((tmp_path / "inputs.json").read_text())
    raw = json.loads(objectives.read_text())
    pattern = data["patterns"][0]
    phrase = raw["phrases"][0]
    if control == "government":
        pattern["function_words"] = ["ad"]
    elif control == "dual-government":
        pattern["function_words"] = ["in"]
    elif control == "agreement":
        selected = next(b for b in phrase["selection"]["bindings"] if b["role"] == "modifier")
        selected["form"] = "fem"
        pattern["roles"]["modifier"]["requires"]["gender"] = "feminine"
        for reading in phrase["reference"]["readings"]:
            reading["features"]["modifier"]["gender"] = "feminine"
    elif control == "passive":
        next(f for f in data["entries"][0]["forms"] if f["key"] == "third")["features"]["voice"] = "passive"
        pattern["roles"]["verb"]["requires"]["voice"] = "passive"
        for reading in phrase["reference"]["readings"]:
            reading["features"]["verb"]["voice"] = "passive"
    else:
        phrase["selection"]["bindings"] = [b for b in phrase["selection"]["bindings"] if b["role"] != "dependent"]
    rebind(profile, data)
    write(objectives, raw)
    from latinitas_cards.pipeline_consumers import PhraseObjective, _versions
    from latinitas_cards.pipeline_profile import PipelineProfile

    parsed = PipelineProfile.read_json(profile.read_bytes())
    source = load_source(profile, parsed)
    objective = PhraseObjective.model_validate(phrase)
    if control == "missing":
        with pytest.raises(ValueError, match="roles"):
            source_claims(source, objective, _versions(parsed))
    else:
        expected, evidence, reason = source_claims(source, objective, _versions(parsed))
        assert not expected and evidence
        assert {
            "government": "government",
            "dual-government": "government",
            "agreement": "gender",
            "passive": "constituent_evidence_unresolved",
        }[control] in reason
    report = call(
        "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(tmp_path / "workflow")
    )
    assert report["file_export"]["exported"] == 0
    assert report["skipped"]


@pytest.mark.parametrize("family", ["preposition-noun", "noun-genitive", "verb-noun", "adjective-noun"])
def test_phrase_proof_reconstructs_without_stage_timestamps(tmp_path: Path, family: str) -> None:
    profile, objectives, _ = bennett_phrase_fixture(tmp_path, family)
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    run = load_run(root)
    card = current_cards(profile, run.profile, run.objectives, root)[0][0]
    source = load_source(profile, run.profile)
    actual = source_claims(source, card.objective, card.versions, root)
    assert actual[0]["construction"] == json.loads(card.claims.assertions[0].value)
    assert canonical(source_claims(source, card.objective, card.versions)) == canonical(actual)
    assert canonical(source_claims(source, card.objective, card.versions)) == canonical(actual)


@pytest.mark.parametrize(
    "family,control,dirty_side",
    [
        ("preposition-noun", "correct", "both"),
        ("noun-genitive", "correct", "both"),
        ("verb-noun", "correct", "both"),
        ("adjective-noun", "correct", "both"),
        ("noun-genitive", "wrong-identity", "proposal"),
        ("noun-genitive", "wrong-identity", "counter"),
        ("noun-genitive", "missing-sibling", "proposal"),
        ("noun-genitive", "missing-sibling", "counter"),
        ("noun-genitive", "alternative", "counter"),
        ("noun-genitive", "manifest", "both"),
        ("noun-genitive", "html", "both"),
        ("noun-genitive", "whitespace", "both"),
        ("verb-noun", "same-entry", "both"),
        ("noun-genitive", "same-entry", "both"),
    ],
)
def test_phrase_saved_comparison_uses_independent_source_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, control: str, dirty_side: str
) -> None:
    import grammar_pipeline_test as fixtures
    from grammar_pipeline_test import configure_pipeline_http, synthetic_pipeline_http

    from latinitas_cards.pipeline_orchestration import read_source_comparison

    if control == "same-entry":  # Several roles bind forms of one German source entry.
        profile, objectives, _ = phrase_fixture(tmp_path, "de", family)
    else:
        profile, objectives, _ = bennett_phrase_fixture(tmp_path, family)
    if control == "manifest":
        manifest_phrase_inputs(profile)
    elif control in ("html", "whitespace"):
        deck = tmp_path / "deck.csv"
        value = "<b>hortus</b>" if control == "html" else " hortus "
        deck.write_text(deck.read_text().replace("h,hortus,", "h," + value + ","))
        rebind(profile, json.loads((tmp_path / "inputs.json").read_text()))
    # Controlled answers reconstruct supplied input facts, not an out-of-band renderer result.
    invocations: list[str] = []

    def answer(body: dict[str, Any]) -> bytes:
        task = json.loads(body["messages"][1]["content"])
        objective = json.loads(task["objective"])
        inputs = {
            e["reference"].rsplit("/", 1)[-1]: json.loads(e["text"])
            for e in task["source_evidence"]
            if e["source"] == "unverified-phrase-entry"
        }
        pattern = next(
            json.loads(e["text"]) for e in task["source_evidence"] if e["source"] == "unverified-phrase-pattern"
        )
        components = {}
        for selected in objective["bindings"]:
            supplied = inputs[selected["entry"]]
            entry = supplied["entry"]
            components[selected["role"]] = dict(
                identity=supplied["identity"],
                entry=entry,
                sense=next(s for s in entry["senses"] if s["key"] == selected["sense"]),
                form=next(f for f in entry["forms"] if f["key"] == selected["form"]),
            )
        value = dict(pattern=pattern, components=components)
        side = "proposal" if not invocations else "counter"
        invocations.append(side)
        dirty = dirty_side in (side, "both")
        assertions = [
            dict(
                id=a["id"],
                kind=a["kind"],
                value="proposal-only-sentinel" if side == "proposal" else "unsupported response",
                selected_sense=canonical({b["role"]: b["sense"] for b in objective["bindings"]}).decode()
                if a["kind"] == "translation"
                else None,
                alternatives=[],
                evidence=[fixtures.EVIDENCE],
            )
            for a in objective["assertions"]
        ]
        for assertion in assertions:
            if assertion["id"] == "construction":
                if dirty and control == "wrong-identity":
                    value["components"]["dependent"]["entry"]["identity"] = "impostor"
                assertion["value"] = canonical(value).decode()
                if dirty and control == "alternative":
                    assertion["alternatives"] = ["another attachment"]
        if dirty and control == "missing-sibling":
            assertions = [a for a in assertions if a["id"] != "rendered"]
        content = dict(
            format="latinitas-model-answer",
            version=1,
            task_sha256=task["task_sha256"],
            assertions=assertions,
            alternatives=[],
            objections=[],
        )
        return canonical({"choices": [{"finish_reason": "stop", "message": {"content": canonical(content).decode()}}]})

    monkeypatch.setattr(fixtures, "provider_answer", answer)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = []
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        run = load_run(root)
        card = current_cards(profile, run.profile, run.objectives, root)[0][0]
        proof = read_source_comparison(root, card)
        assert proof is not None
        comparisons = {c["assertion_id"]: c for c in json.loads(canonical(proof))["comparisons"]}
        # The plain German fixture's senses are not Bennett-qualified, so its comparison stays unresolved.
        relations = {"wrong-identity": "contradiction", "alternative": "unresolved", "same-entry": "unresolved"}
        expected_relation = relations.get(control, "equivalent")
        assert comparisons["construction"]["relation"] == expected_relation
        assert all(c["check"]["result"] != "pass" for key, c in comparisons.items() if key != "construction")
        assert report["file_export"]["exported"] == 0
        assert len(sent) == 2 and sent[0]["messages"][1]["content"] == sent[1]["messages"][1]["content"]
        assert (
            "impostor" not in sent[1]["messages"][1]["content"]
            and "proposal-only-sentinel" not in sent[1]["messages"][1]["content"]
        )
        assert json.loads(sent[1]["messages"][1]["content"])["reference_constraints"] == []
        assert proof["proposal_output_sha256"] != proof["counter_output_sha256"]
        if control == "missing-sibling":
            assert proof["complete_assertion_coverage"] is False
        if control == "same-entry":
            evidence = json.loads(sent[0]["messages"][1]["content"])["source_evidence"]
            references = [(e["source"], e["reference"].rsplit("/", 1)[-1]) for e in evidence]
            assert report["status"] == "completed"
            assert references.count(("confirmed-deck-mapping", "n")) == 2  # One per binding.
            assert references.count(("unverified-phrase-entry", "n")) == 1
    # The server is stopped: normal saved-call reuse must not dispatch again.
    replay = call("resume", "--run", str(root))
    assert replay["file_export"]["exported"] == 0 and len(sent) == 2


def test_added_constituent_tables_preserve_all_qualified_alternatives() -> None:
    from latinitas_cards.lexical_evidence import BennettLexicon

    library = BennettLexicon()
    expected: dict[str, set[tuple[str, str, str]]] = {}
    cases = ("nominative", "genitive", "dative", "accusative", "vocative", "ablative")
    for lemma, singular, plural in (
        ("hortus", "hortus hortī hortō hortum horte hortō", "hortī hortōrum hortīs hortōs hortī hortīs"),
        ("puer", "puer puerī puerō puerum puer puerō", "puerī puerōrum puerīs puerōs puerī puerīs"),
    ):
        for number, forms in (("singular", singular), ("plural", plural)):
            for case, surface in zip(cases, forms.split(), strict=True):
                expected.setdefault(surface, set()).add((lemma, case, number))
    for surface, alternatives in expected.items():
        readings = library.analyze(surface)
        assert {(r.lemma, r.features.case, r.features.number) for r in readings} == alternatives
        assert all(r.pos == "noun" and r.features.gender == "masculine" for r in readings)
    assert library.analyze("hortūs") == library.analyze("puella") == ()
    assert library.senses("porta", "en")[0].meaning == "gate"
    assert library.senses("hortus", "en")[0].meaning == "garden"
    assert library.senses("puer", "en")[0].meaning == "boy"
    assert library.senses("bonus", "en")[0].meaning == "good"
    assert library.senses("amō", "en")[0].valency == {"subject": "nominative", "object": "accusative"}
    for lemma, meaning in (
        ("porta", "Tor"),
        ("hortus", "Garten"),
        ("puer", "Junge"),
        ("bonus", "gut"),
        ("amō", "lieben"),
    ):
        sense = library.senses(lemma, "de")[0]
        assert sense.meaning == meaning and sense.evidence
    assert library.senses("amō", "de")[0].valency == {"subject": "nominative", "object": "accusative"}
    assert library.senses("porta", "fr") == library.senses("puella", "de") == ()
    assert library.ambiguity_complete is False


@pytest.mark.parametrize("damage", ["reference", "checker", "constituent", "source", "context"])
def test_phrase_positive_proof_rejects_real_dependency_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    import shutil

    from latinitas_cards import acceptance_policy as policy
    from latinitas_cards import latin_library
    from latinitas_cards.pipeline_consumers import _analysis_decision

    installed = tmp_path / "installed-package"
    shutil.copytree(Path(policy.__file__).parent, installed, ignore=shutil.ignore_patterns("__pycache__"))
    reference = installed / "references" / "bennett-1918.html"
    original = policy.implementation_digest()
    monkeypatch.setattr(policy, "__file__", str(installed / "acceptance_policy.py"))
    monkeypatch.setattr(latin_library, "BENNETT_PATH", reference)
    assert policy.implementation_digest() == original
    profile, objectives, _ = bennett_phrase_fixture(tmp_path / "inputs", "noun-genitive")
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    run = load_run(root)
    card = current_cards(profile, run.profile, run.objectives, root)[0][0]
    assert (
        next(c for c in Analysis.read_json(card.analysis_path.read_bytes()).checks if c.id == "construction").result
        == "pass"
    )
    assert _analysis_decision(card).state == "withheld"
    replay = call("stage", "analyze", "--run", str(root))
    assert all(row["reuse"] == "reused" for row in replay["stages"]["analyze"])
    if damage == "constituent":
        from grammar_pipeline_test import runner

        from latinitas_cards.cli import app

        path = next((root / "artifacts").glob("deck-*-report.json"))
        path.write_text("{}")
        rejected = runner.invoke(app, ["pipeline", "stage", "analyze", "--run", str(root), "--json"])
        assert rejected.exit_code == 1 and "artifact digest mismatch" in rejected.output
        return
    if damage in ("reference", "checker"):
        changed = reference if damage == "reference" else installed / "deck_evidence.py"
        changed.write_bytes(changed.read_bytes() + b"\n")
        assert policy.implementation_digest() != original
        current = current_cards(profile, run.profile, run.objectives, root)[0][0]
    else:
        from latinitas_cards.pipeline_consumers import Objectives
        from latinitas_cards.pipeline_profile import PipelineProfile

        if damage == "source":
            deck = profile.parent / "deck.csv"
            deck.write_text(deck.read_text().replace("h,hortus,", "h,hortus ,"))
            rebind(profile, json.loads((profile.parent / "inputs.json").read_text()))
        else:
            raw = json.loads(objectives.read_text())
            raw["phrases"][0]["reference"]["readings"][0]["explanation"] = "changed contextual claim"
            write(objectives, raw)
        current = current_cards(
            profile,
            PipelineProfile.read_json(profile.read_bytes()),
            Objectives.read_json(objectives.read_bytes()),
            root,
        )[0][0]
    assert current.binding != card.binding or current.versions != card.versions
    with pytest.raises(ValueError, match="stale analysis context"):
        _analysis_decision(current)
