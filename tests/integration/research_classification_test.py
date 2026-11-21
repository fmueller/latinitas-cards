"""Research classification uses stock disabled policy and real saved proof readers."""

import json
from pathlib import Path
from typing import Any

import pytest
from grammar_pipeline_test import call, configure_pipeline_http, rebind, synthetic_pipeline_http, write
from meaning_evidence_test import captured_answer, meaning_fixture


@pytest.mark.parametrize("language", ["de", "en"])
@pytest.mark.parametrize("family", ["form", "preposition-noun", "noun-genitive", "adjective-noun", "verb-noun"])
def test_research_complete_card_and_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str, family: str
) -> None:
    import grammar_pipeline_test as fixtures

    from latinitas_cards.acceptance import load_acceptance
    from latinitas_cards.pipeline_consumers import current_cards, load_run
    from latinitas_cards.research_classification import classify, read, save

    invocations: list[dict[str, Any]] = []

    def answer(body: dict[str, Any]) -> bytes:
        result = captured_answer(body, paraphrase=bool(invocations))
        invocations.append(body)
        return result

    monkeypatch.setattr(fixtures, "provider_answer", answer)
    profile, objectives = meaning_fixture(tmp_path, language, family)
    if family == "form":
        data = json.loads((tmp_path / "inputs.json").read_text())
        data["entries"][0]["forms"] = data["entries"][0]["forms"][:1]
        rebind(profile, data)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"] if family == "form" else []
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        records = classify(root)
        assert len(records) == 1
        record = records[0]
        assert record.classification == "would_accept", record
        assert not record.failures
        assert len(record.obligations) == len(record.assertions)
        assert all(o.result == "pass" and o.checks for o in record.obligations)
        assert all(c.result == "pass" for c in record.checks)
        assert len(sent) == 2
        assert sent[0]["messages"][1]["content"] == sent[1]["messages"][1]["content"]
        destination = tmp_path / "research.json"
        save(destination, record)
        assert read(root, destination) == record
        with pytest.raises(ValueError):
            load_acceptance(destination)
        run = load_run(root)
        card = current_cards(profile, run.profile, run.objectives, root)[0][0]
        production = card.claims.decisions[0]
        assert production.reader.preview(production.path)["reason"] == "policy_not_enabled"
        with pytest.raises(ValueError, match="policy_not_enabled"):
            production.reader.export(production.path)
        with pytest.raises(ValueError):
            production.reader.export(destination)
        assert report["file_export"]["exported"] == 0
    assert call("resume", "--run", str(root))["file_export"]["exported"] == 0
    assert read(root, destination) == record
    response = next(root.rglob("*.response.json"))
    before = response.read_bytes()
    response.write_bytes(before + b" ")
    damaged = classify(root)[0]
    assert damaged.classification == "unscorable"
    assert len(damaged.obligations) == len(record.assertions)
    assert damaged.failures
    with pytest.raises(ValueError):
        read(root, destination)
    if family == "form" and language == "de":
        unscorable = tmp_path / "unscorable.json"
        save(unscorable, damaged)
        with pytest.raises(ValueError, match="unscorable"):
            read(root, unscorable)
    response.write_bytes(before)
    assert read(root, destination) == record
    if family == "form" and language == "de":
        from latinitas_cards.acceptance_records import canonical

        saved = destination.read_bytes()
        for field, value in (
            ("implementation_sha256", "0" * 64),
            ("support_registry_sha256", "1" * 64),
            ("obligations", []),
            ("classification", "would_withhold"),
        ):
            raw = json.loads(saved)
            raw[field] = value
            destination.write_bytes(canonical(raw))
            with pytest.raises(ValueError):
                read(root, destination)
        destination.write_bytes(saved)
        request = next(root.rglob("*.request.json"))
        index = root / "stage-index.json"
        for path in (request, index):
            original = path.read_bytes()
            path.write_bytes(original + b" " if path == request else b"{}")
            with pytest.raises(ValueError):
                read(root, destination)
            path.write_bytes(original)
            assert read(root, destination) == record
        original = index.read_bytes()
        raw_index = json.loads(original)
        counter_key = next(k for k in raw_index if k.startswith("counter_check/"))
        index_value: object
        for index_value in ([], {**raw_index, counter_key: None}, {**raw_index, counter_key: ""}):
            index.write_bytes(canonical(index_value))
            damaged_records = classify(root)
            assert len(damaged_records) == 1
            assert damaged_records[0].classification == "unscorable"
            assert damaged_records[0].assertions == record.assertions
            assert [o.assertion_id for o in damaged_records[0].obligations] == [a.id for a in record.assertions]
            index.write_bytes(original)
            assert read(root, destination) == record


@pytest.mark.parametrize("case", ["contradiction", "compatible_alternatives", "ambiguous", "incomplete", "unsupported"])
def test_research_retains_every_failed_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    import grammar_pipeline_test as fixtures

    from latinitas_cards.acceptance_records import canonical
    from latinitas_cards.research_classification import classify

    def answer(body: dict[str, Any]) -> bytes:
        wrapper = json.loads(captured_answer(body))
        response = json.loads(wrapper["choices"][0]["message"]["content"])
        translation = next(a for a in response["assertions"] if a["id"] == "translation")
        if case == "incomplete":
            response["assertions"] = [a for a in response["assertions"] if a["id"] != "translation"]
        else:
            translation["value"] = {
                "contradiction": "we love",
                "compatible_alternatives": "I feel love",
                "ambiguous": "I love / we love",
                "unsupported": "An uninterpreted utterance.",
            }[case]
            if case == "ambiguous":
                translation["alternatives"] = ["we love"]
        wrapper["choices"][0]["message"]["content"] = canonical(response).decode()
        return canonical(wrapper)

    monkeypatch.setattr(fixtures, "provider_answer", answer)
    profile, objectives = meaning_fixture(tmp_path, "en")
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][0]["forms"] = data["entries"][0]["forms"][:1]
    rebind(profile, data)
    with synthetic_pipeline_http() as (url, _, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"]
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        record = classify(output / "run")[0]
        assert record.classification == "would_withhold"
        assert [o.assertion_id for o in record.obligations] == [a.id for a in record.assertions]
        target = next(o for o in record.obligations if o.assertion_id == "translation")
        assert target.result == "withhold"
        semantic = next(c for c in target.checks if c.method == "semantic_comparison")
        assert semantic.result == ("fail" if case == "contradiction" else "unresolved")
        expected_reason = (
            "incomplete_model_assertion_coverage"
            if case == "incomplete"
            else "bounded_meaning_unresolved"
            if case == "unsupported"
            else "bounded_meaning_" + case
        )
        assert expected_reason in target.reasons
        assert report["file_export"]["exported"] == 0


def test_research_readable_application_negative_is_not_processing_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import grammar_pipeline_test as fixtures

    from latinitas_cards.research_classification import classify

    monkeypatch.setattr(fixtures, "provider_answer", captured_answer)
    profile, objectives = meaning_fixture(tmp_path, "en")
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][0]["forms"] = data["entries"][0]["forms"][:1]
    rebind(profile, data)
    raw = json.loads(objectives.read_text())
    raw["forms"][0]["context"] = "an additional context without qualified applicability"
    write(objectives, raw)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"]
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        record = classify(output / "run")[0]
        assert len(sent) == 2 and report["file_export"]["exported"] == 0
        assert record.classification == "would_withhold"
        assert not record.failures
        assert any(c.result == "not_applicable" for c in record.checks if c.method == "rule")
        assert any(o.result == "withhold" for o in record.obligations)
        assert all(o.result != "unscorable" for o in record.obligations)
