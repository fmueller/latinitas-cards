"""Corpus selections handed off to the offline managed update workflow."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from card_lifecycle_test import capture, fixture, generated
from corpus_exercises_test import profile
from corpus_facts_test import Fixture, text
from corpus_output_test import RUNNER, invoke, selected

from latinitas_cards.annotation_sidecar import Annotation, read_sidecar
from latinitas_cards.cli import app
from latinitas_cards.corpus_claims import current_annotations
from latinitas_cards.destination_state import adopt, save_state, schema_contract
from latinitas_cards.managed_layouts import layout_version
from latinitas_cards.managed_plans import compose_plan
from latinitas_cards.review_inspection import read_export_csv


def destination(root: Path, facts: Fixture) -> dict[str, Any]:
    export = root / "initial.csv"
    invoke(root, facts, "export", "--output", str(export))
    fields = read_export_csv(export)
    layout = layout_version("corpus-v1")
    data = cast(dict[str, Any], fixture(generated())[0].payload)
    prototype = data["notes"][0]
    data["schema"] = schema_contract("model", "corpus-v1")
    data["managed_set"] = {
        "scope": "corpus-passage-v1",
        "members": [[f["LatinitasID"], f["Source Scope"], f["Source ID"], f["Source Location"]] for f in fields],
    }
    data["notes"] = []
    for index, row in enumerate(fields):
        note = deepcopy(prototype)
        slot = next(s for s in layout.slots if row[s.enabled_field] == "1")
        note.update(
            identity=row["LatinitasID"],
            source=[row["Source Scope"], row["Source ID"], row["Source Location"]],
            guid=f"guid-{index}",
            local_id=f"note-{index}",
            fields={name: row[name] for name in layout.managed_fields},
            tags=["user-tag"],
        )
        note["cards"]["rows"] = [
            {
                "id": f"card-{index}",
                "semantic_key": slot.semantic_key,
                "ordinal": slot.ordinal,
                "template_name": slot.template_name,
                "suspended": False,
                "due": index + 13,
            }
        ]
        data["notes"].append(note)
    data["expected_count"] = len(fields)
    save_destination(root, data)
    return data


def save_destination(root: Path, data: dict[str, Any]) -> None:
    snapshot = capture(data)
    ownership = {
        note["identity"]: {"source_tags": [], "configured_tags": [], "keep_tags": note["tags"], "keep_fields": []}
        for note in data["notes"]
    }
    (root / "snapshot.json").write_text(json.dumps(snapshot.payload))
    save_state(root / "state.json", adopt(snapshot, ownership, "reviewed synthetic adoption"))


def plan(root: Path, facts: Fixture, *, output: Path | None = None, code: int = 0) -> dict[str, Any]:
    return invoke(
        root,
        facts,
        "plan",
        str(root / "snapshot.json"),
        str(root / "state.json"),
        "--output",
        str(output or root / "request.json"),
        code=code,
    )


def managed_plan(root: Path) -> dict[str, Any]:
    result = RUNNER.invoke(app, ["managed", "plan", str(root / "request.json")])
    assert result.exit_code == 0, result.output
    return cast(dict[str, Any], json.loads(result.output))


def test_reviewed_correction_is_approvable_retained_update(tmp_path: Path) -> None:
    facts = selected(tmp_path)
    data = destination(tmp_path, facts)
    original = next(
        r
        for r in read_sidecar(facts.sidecar)
        if isinstance(r, Annotation) and r.layer == "translation" and r.anchor.span is None
    )
    facts.add(0, "translation", text("en", "Marcus watches a friend."), supersedes=original.annotation_id)
    Path(f"{facts.corpus}.claims.json").unlink()
    facts.decide("accepted", *current_annotations(facts.corpus, facts.sidecar))
    before = {p: p.read_bytes() for p in (tmp_path / "state.json", tmp_path / "snapshot.json", facts.corpus)}
    result = plan(tmp_path, facts)["result"]
    assert result["not_imported"] == []
    request = json.loads((tmp_path / "request.json").read_bytes())
    assert request["snapshot"] == capture(data).payload
    planned = managed_plan(tmp_path)
    changes = [op for n in planned["notes"] for op in n["operations"]]
    assert {op["field"] for op in changes} == {"CorpusSentenceAnswer", "CorpusAnalysisAnswer"}
    assert all(op["supported"] for op in changes)
    operation = next(op["id"] for op in changes if op["field"] == "CorpusSentenceAnswer")
    assert all(e["effect"] == "retain" for n in planned["notes"] for e in n["card_effects"])
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(planned))
    approval = RUNNER.invoke(
        app, ["managed", "approve", str(plan_path), "--operation", operation, "--review", "review"]
    )
    assert approval.exit_code == 0, approval.output
    backup = tmp_path / "backup.bin"
    backup.write_bytes(b"synthetic backup, not native proof")
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "binding": request["binding"],
                "snapshot": request["snapshot"],
                "plan": planned,
                "approval": json.loads(approval.output),
                "backup": str(backup),
                "recovery": "restore fixture",
                "note_type": "Latinitas Corpus Context v1",
                "deck": "Latin::A",
            }
        )
    )
    assert all(p.read_bytes() == raw for p, raw in before.items())
    emitted = RUNNER.invoke(
        app,
        [
            "managed",
            "emit",
            str(handoff),
            "--state",
            str(tmp_path / "state.json"),
            "--output",
            str(tmp_path / "updates.csv"),
        ],
    )
    assert emitted.exit_code == 0, emitted.output
    rows = read_export_csv(tmp_path / "updates.csv")
    assert len(rows) == 1 and "Marcus watches a friend." in rows[0]["CorpusSentenceAnswer"]
    assert rows[0]["Tags"] == "user-tag"
    assert "Personal Notes" not in rows[0]
    assert (tmp_path / "snapshot.json").read_bytes() == before[tmp_path / "snapshot.json"]


def test_withheld_explicit_retirement_and_managed_membership(tmp_path: Path) -> None:
    facts = selected(tmp_path)
    data = destination(tmp_path, facts)
    rows = {n["fields"]["CorpusSentenceEnabled"]: n for n in data["notes"]}
    sentence = rows["1"]["identity"]
    # A bound member not yet imported may be created; an unbound exercise must be omitted.
    missing = next(n for n in data["notes"] if n["fields"]["CorpusPhraseEnabled"] == "1")
    outside = next(n for n in data["notes"] if n["fields"]["CorpusAnalysisEnabled"] == "1")
    data["notes"] = [n for n in data["notes"] if n not in (missing, outside)]
    data["managed_set"]["members"] = [m for m in data["managed_set"]["members"] if m[0] != outside["identity"]]
    data["expected_count"] = len(data["notes"])
    save_destination(tmp_path, data)
    report = plan(tmp_path, facts)["result"]
    assert report["not_imported"] == [outside["identity"]]
    notes = {n["identity"]: n for n in managed_plan(tmp_path)["notes"]}
    assert notes[missing["identity"]]["classification"] == "create"
    assert notes[missing["identity"]]["operations"][0]["supported"] is False
    Path(f"{facts.corpus}.claims.json").unlink()
    report = plan(tmp_path, facts, output=tmp_path / "withheld.json")["result"]
    assert set(report["not_imported"]) == {missing["identity"], outside["identity"]}
    request = json.loads((tmp_path / "withheld.json").read_bytes())
    assert all(p["status"] == "withheld" for p in request["proposals"])
    assert {p["identity"]: p["fields"] for p in request["proposals"]} == {
        n["identity"]: n["fields"] for n in data["notes"]
    }
    assert all(
        not n["operations"]
        for n in compose_plan(capture(data), request["baseline"], request["proposals"], request["effective_profile"])[
            "notes"
        ]
    )
    result = RUNNER.invoke(
        app,
        [
            "corpus-cards",
            "deselect",
            str(tmp_path / "selection.json"),
            "--exercise",
            sentence,
            "--reason",
            "manual withdrawal",
        ],
    )
    assert result.exit_code == 0, result.output
    plan(tmp_path, facts, output=tmp_path / "retire.json")
    request = json.loads((tmp_path / "retire.json").read_bytes())
    retire = [p for p in request["proposals"] if p.get("retire")]
    assert [p["identity"] for p in retire] == [sentence]
    composed = compose_plan(capture(data), request["baseline"], request["proposals"], request["effective_profile"])
    retired = next(n for n in composed["notes"] if n["identity"] == sentence)
    assert retired["classification"] == "retire"
    assert any(op["kind"] == "retire" for op in retired["operations"])
    assert not any(op["supported"] for op in retired["operations"])


def test_passage_correction_requires_review_and_context_confirmation(tmp_path: Path) -> None:
    facts = selected(tmp_path)
    data = destination(tmp_path, facts)
    facts.edit(0, "Marcus amicum videt!")
    result = RUNNER.invoke(
        app, ["annotations", "segment", str(facts.corpus), "--timestamp", "2026-01-02T00:00:00Z", "--json"]
    )
    assert result.exit_code == 0, result.output
    segmentation = next(r for r in current_annotations(facts.corpus, facts.sidecar) if r.layer == "segmentation")
    facts.add(0, "translation", text("en", "Marcus sees a friend."), (0, 20), segmentation=segmentation.annotation_id)
    Path(f"{facts.corpus}.claims.json").unlink()
    facts.decide("accepted", *current_annotations(facts.corpus, facts.sidecar))
    plan(tmp_path, facts, output=tmp_path / "before-confirm.json")
    request = json.loads((tmp_path / "before-confirm.json").read_bytes())
    assert all(p["status"] == "withheld" for p in request["proposals"])
    sentence = next(n["identity"] for n in data["notes"] if n["fields"]["CorpusSentenceEnabled"] == "1")
    invoke(tmp_path, facts, "select", "--report", str(tmp_path / "report.json"), "--confirm-context", sentence)
    plan(tmp_path, facts)
    changes = [op for n in managed_plan(tmp_path)["notes"] for op in n["operations"]]
    assert {op["field"] for op in changes} == {"Latin", "Passage"}
    assert all(op["supported"] for op in changes)
    assert next(op["value"] for op in changes if op["field"] == "Passage") == "Marcus amicum videt!"


def test_shared_profile_and_new_language_do_not_rewrite_or_retire(tmp_path: Path) -> None:
    facts = selected(tmp_path)
    data = destination(tmp_path, facts)
    for note in data["notes"]:
        note["fields"]["Profile"] = "first-exporter"
    save_destination(tmp_path, data)
    plan(tmp_path, facts)
    assert all(not n["operations"] for n in managed_plan(tmp_path)["notes"])
    (tmp_path / "profile.json").write_text(profile("de").model_dump_json())
    (tmp_path / "selection.json").unlink()
    facts.decide("accepted", facts.add(0, "translation", text("de", "Marcus sieht einen Freund.")))
    invoke(tmp_path, facts, "select", "--report", str(tmp_path / "german-report.json"))
    result = plan(tmp_path, facts, output=tmp_path / "german.json")["result"]
    assert result["not_imported"]
    assert json.loads((tmp_path / "german.json").read_bytes())["proposals"] == []


@pytest.mark.parametrize(
    "change,code,diagnostic",
    [("selection", 4, "binding_drift"), ("corpus", 3, "profile_mismatch"), ("destination", 4, "binding_drift")],
)
def test_plan_binding_guards(tmp_path: Path, change: str, code: int, diagnostic: str) -> None:
    facts = selected(tmp_path)
    data = destination(tmp_path, facts)
    if change == "selection":
        (tmp_path / "profile.json").write_text(profile(profile_id="another").model_dump_json())
    elif change == "corpus":
        (tmp_path / "profile.json").write_text(profile(corpora=["missing"]).model_dump_json())
    else:
        data["destination"] = "different-collection"
        (tmp_path / "snapshot.json").write_text(json.dumps(data))
    assert plan(tmp_path, facts, code=code)["diagnostics"][-1]["code"] == diagnostic
    assert not (tmp_path / "request.json").exists()


def test_plan_artifact_provenance_idempotence_and_aliases(tmp_path: Path) -> None:
    from latinitas_cards.atomic_io import writer_lock
    from latinitas_cards.canonical import digest_bytes

    facts = selected(tmp_path)
    destination(tmp_path, facts)
    plan(tmp_path, facts)
    raw = (tmp_path / "request.json").read_bytes()
    inputs = json.loads(raw)["effective_profile"]["inputs"]
    assert inputs[str(facts.corpus)] == digest_bytes(facts.corpus.read_bytes())
    assert inputs[str(tmp_path / "selection.json")] == digest_bytes((tmp_path / "selection.json").read_bytes())
    assert plan(tmp_path, facts)["result"]["written"] is False
    assert (tmp_path / "request.json").read_bytes() == raw
    for target in (facts.corpus, Path(f"{facts.corpus}.lock"), tmp_path / "state.json"):
        assert plan(tmp_path, facts, output=target, code=3)["diagnostics"][-1]["code"] == "alias"
    other = tmp_path / "busy.json"
    with writer_lock(other):
        assert plan(tmp_path, facts, output=other, code=4)["diagnostics"][-1]["code"] == "busy"


@pytest.mark.parametrize("changed", range(7))
def test_plan_uses_captured_input_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: int) -> None:
    from latinitas_cards import corpus_plan
    from latinitas_cards.canonical import digest_bytes
    from latinitas_cards.corpus_output import CorpusOutput, load_output

    facts = selected(tmp_path)
    data = destination(tmp_path, facts)
    paths = [tmp_path / name for name in ("profile.json", "selection.json", "snapshot.json", "state.json")]
    paths += [facts.corpus, facts.sidecar, Path(f"{facts.corpus}.claims.json")]
    expected = {str(p): digest_bytes(p.read_bytes()) for p in paths}
    baseline = json.loads((tmp_path / "state.json").read_bytes())

    def mutate_originals(*args: Any, **kwargs: Any) -> CorpusOutput:
        paths[changed].write_bytes(b"not valid JSON")
        return load_output(*args, **kwargs)

    monkeypatch.setattr(corpus_plan, "load_output", mutate_originals)
    plan(tmp_path, facts)
    request = json.loads((tmp_path / "request.json").read_bytes())
    assert request["effective_profile"]["inputs"] == expected
    assert request["snapshot"] == capture(data).payload
    assert request["baseline"] == baseline
    assert {p["identity"]: p["fields"] for p in request["proposals"]} == {
        n["identity"]: n["fields"] for n in data["notes"]
    }
    assert all(p["status"] == "approved" for p in request["proposals"])


def test_plan_reports_previously_anchored_disappearance(tmp_path: Path) -> None:
    facts = selected(tmp_path)
    data = destination(tmp_path, facts)
    missing = data["notes"].pop()["identity"]
    data["expected_count"] -= 1
    (tmp_path / "snapshot.json").write_text(json.dumps(data))
    before = (tmp_path / "state.json").read_bytes()
    plan(tmp_path, facts)
    note = next(n for n in managed_plan(tmp_path)["notes"] if n["identity"] == missing)
    assert note["classification"] == "conflict"
    assert note["reasons"] == ["previously anchored note absent; destination reconciliation required"]
    assert note["operations"] == note["card_effects"] == []
    assert (tmp_path / "state.json").read_bytes() == before


def test_plan_preserves_nonempty_managed_tag_origins(tmp_path: Path) -> None:
    facts = selected(tmp_path)
    data = destination(tmp_path, facts)
    origins = {
        "source_tags": ["shared", "source"],
        "configured_tags": ["configured", "shared"],
        "lifecycle_tags": ["lifecycle"],
    }
    for note in data["notes"]:
        note["tags"] = ["source", "shared", "configured", "lifecycle", "user-tag"]
    observed = capture(data)
    ownership = {n["identity"]: {**origins, "keep_tags": ["user-tag"], "keep_fields": []} for n in data["notes"]}
    save_state(tmp_path / "state.json", adopt(observed, ownership, "explicit origin review"))
    (tmp_path / "snapshot.json").write_text(json.dumps(observed.payload))
    plan(tmp_path, facts)
    assert all(not n["operations"] for n in managed_plan(tmp_path)["notes"])
    Path(f"{facts.corpus}.claims.json").unlink()
    plan(tmp_path, facts, output=tmp_path / "withheld.json")
    identity = data["notes"][0]["identity"]
    result = RUNNER.invoke(
        app,
        ["corpus-cards", "deselect", str(tmp_path / "selection.json"), "--exercise", identity, "--reason", "withdraw"],
    )
    assert result.exit_code == 0, result.output
    plan(tmp_path, facts, output=tmp_path / "retired.json")
    for filename in ("request.json", "withheld.json", "retired.json"):
        request = json.loads((tmp_path / filename).read_bytes())
        assert all(p["contributions"] == origins for p in request["proposals"])
    retired = json.loads((tmp_path / "retired.json").read_bytes())
    assert [p["identity"] for p in retired["proposals"] if p.get("retire")] == [identity]
