"""Corpus decisions are explicit, version-bound attestations, never authored edits."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from latinitas_cards.annotation_sidecar import Annotation, read_sidecar
from latinitas_cards.claim_sidecar import load_sidecar, sidecar_path
from latinitas_cards.cli import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RUNNER = CliRunner()


def test_deck_loader_rejects_v2_before_validation(tmp_path: Path) -> None:
    path = tmp_path / "claims.json"
    path.write_text('{"version": 2, "decisions": []}')
    with pytest.raises(ValueError, match="unsupported_version"):
        load_sidecar(path)


def test_v2_binding_is_strict_and_matches_decision() -> None:
    from latinitas_cards.claim_sidecar import CorpusBinding, SidecarV2, StoredDecisionV2

    binding = {
        "kind": "corpus",
        "corpus_id": "wsl",
        "passage_id": "wsl:p000001",
        "passage_hash": "sha256:" + "1" * 64,
        "annotation_id": "sha256:" + "2" * 64,
        "layer": "morphology",
    }
    entry = {
        "base_fingerprint": "a" * 64,
        "decision": {"claim_fingerprint": "a" * 64, "status": "accepted", "reviewer": "R", "reason": "X"},
        "binding": binding,
    }
    assert StoredDecisionV2.model_validate(entry).binding == CorpusBinding.model_validate(binding)
    assert SidecarV2().model_dump() == {"version": 2, "decisions": ()}
    for update in ({"kind": "deck"}, {"corpus_id": "OTHER"}, {"passage_id": "other:p000001"}, {"layer": "syntax"}):
        with pytest.raises(ValueError):
            CorpusBinding.model_validate(binding | update)
    invalid_entries: tuple[dict[str, object], ...] = (
        {"base_fingerprint": "b" * 64},
        {"value": "invented"},
        {"evidence": []},
    )
    for invalid in invalid_entries:
        with pytest.raises(ValueError):
            StoredDecisionV2.model_validate(entry | invalid)
    with pytest.raises(ValueError):
        SidecarV2.model_validate({"version": True, "decisions": []})


@pytest.mark.parametrize("family", ["principal-parts", "form-parsing", "phrases"])
def test_deck_commands_reject_v2_without_export(tmp_path: Path, family: str, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(FIXTURES / "goldens/claims-v1" / family, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)
    for path in tmp_path.glob("*.claims.json"):
        path.write_text('{"version": 2, "decisions": []}')
    cases = json.loads((tmp_path / "cases.json").read_text())
    replay = cases["replay"][0]["args"]
    result = RUNNER.invoke(app, ["claims", "list", *replay, "--json"])
    assert result.exit_code == 1, result.output
    assert json.loads(result.output)["diagnostics"][0]["code"] == "unsupported_version"
    for export in cases["exports"]:
        output = tmp_path / export["output"]
        output.unlink()
        result = RUNNER.invoke(app, export["argv"])
        assert result.exit_code != 0 and "unsupported_version" in result.output, result.output
        assert not output.exists()
    previews = {
        "principal-parts": ["preview", "--input", "source.csv", "--profile", "profile.json"],
        "form-parsing": ["form-parsing", "preview", "parsing.json"],
        "phrases": ["phrases", "inspect", "inputs.json"],
    }
    result = RUNNER.invoke(app, previews[family])
    assert result.exit_code != 0 and "unsupported_version" in result.output, result.output


def test_deeply_nested_v1_retains_legacy_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(FIXTURES / "goldens/claims-v1/principal-parts", tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)
    payload = '{"version":1,"decisions":' + "[" * 100_000 + "0" + "]" * 100_000 + "}"
    sidecar = tmp_path / "source.csv.claims.json"
    sidecar.write_text(payload)
    result = RUNNER.invoke(app, ["claims", "list", "source.csv", "--profile", "profile.json", "--json"])
    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert set(report) == {"diagnostics"}
    assert report["diagnostics"][0]["code"] == "invalid"
    assert sidecar.read_text() == payload


def test_corpus_fingerprint_literal() -> None:
    from latinitas_cards.corpus_claims import corpus_claim_fingerprint

    record = list(read_sidecar(FIXTURES / "annotations/fixture-morph.annotations.jsonl"))[2]
    assert isinstance(record, Annotation)
    # Independently calculated with canonical fixture bytes piped to sha256sum.
    expected = "0abc30c213eaaafe24de770bc93cd0c89d5434df4d002dbfe5d12445470b01c2"
    assert corpus_claim_fingerprint(record) == expected
    raw = record.model_dump(mode="json")
    data = {key: raw[key] for key in ("annotation_id", "layer", "value", "anchor", "producer")}
    assert hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != expected


@pytest.fixture
def corpus_review(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes((FIXTURES / "corpus/wsl-three.corpus.jsonl").read_bytes())
    sidecar = tmp_path / "annotations.jsonl"
    sidecar.write_bytes((FIXTURES / "annotations/wsl.annotations.jsonl").read_bytes())
    return corpus, sidecar


def invoke(command: str, corpus: Path, sidecar: Path, *args: str) -> Result:
    return RUNNER.invoke(app, ["claims", command, str(corpus), "--annotations", str(sidecar), "--json", *args])


def test_corpus_list_record_inspect_and_stale(corpus_review: tuple[Path, Path], tmp_path: Path) -> None:
    corpus, sidecar = corpus_review
    result = invoke("list", corpus, sidecar)
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert set(report) == {"claims", "diagnostics"} and not report["diagnostics"]
    claims = report["claims"]
    assert len(claims) == 6
    assert {c["review_status"] for c in claims} == {"open"}
    assert {c["status"] for c in claims} == {"withheld"}
    assert set(claims[0]) == {
        "identity",
        "annotation_id",
        "kind",
        "value",
        "alternatives",
        "producer",
        "fingerprint",
        "status",
        "review_status",
        "reason",
    }
    items = [dict(fingerprint=c["fingerprint"], status="accepted", reviewer="R", reason="Checked") for c in claims]
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps(items))
    for _ in range(2):
        result = invoke("record", corpus, sidecar, "--batch", str(batch))
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"diagnostics": [], "recorded": 6}
        payload = sidecar_path(corpus).read_bytes()
    assert len(json.loads(payload)["decisions"]) == 6
    assert json.loads(payload)["version"] == 2
    rows = {r.annotation_id: r for r in read_sidecar(sidecar) if isinstance(r, Annotation)}
    for entry in json.loads(payload)["decisions"]:
        row = rows[entry["binding"]["annotation_id"]]
        assert entry["base_fingerprint"] == entry["decision"]["claim_fingerprint"]
        assert entry["binding"] == dict(
            kind="corpus",
            corpus_id="wsl",
            passage_id=row.anchor.passage_id,
            passage_hash=row.anchor.passage_hash,
            annotation_id=row.annotation_id,
            layer=row.layer,
        )
    result = invoke("inspect", corpus, sidecar, "--check")
    assert result.exit_code == 0, result.output
    assert {c["review_status"] for c in json.loads(result.output)["claims"]} == {"accepted"}
    corpus.write_bytes((FIXTURES / "corpus/wsl-three-edited.corpus.jsonl").read_bytes())
    result = invoke("inspect", corpus, sidecar, "--check")
    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert [c["identity"] for c in report["claims"]] == ["wsl:poem.1.1", "wsl:poem.1.3"] * 2
    assert {c["review_status"] for c in report["claims"]} == {"accepted"}
    assert [d["code"] for d in report["diagnostics"]] == ["stale", "stale"]
    # Unknown and now-stale fingerprints both reject every item, leaving bytes unchanged.
    for fingerprint in ("0" * 64, claims[4]["fingerprint"]):
        batch.write_text(json.dumps([items[0], items[0] | {"fingerprint": fingerprint}]))
        result = invoke("record", corpus, sidecar, "--batch", str(batch))
        assert result.exit_code == 1, result.output
        assert sidecar_path(corpus).read_bytes() == payload
    assert Path(f"{sidecar_path(corpus)}.lock").exists()


@pytest.mark.parametrize("option", ["value", "evidence", "selected-sense"])
def test_corpus_authored_options_rejected(corpus_review: tuple[Path, Path], option: str, tmp_path: Path) -> None:
    corpus, sidecar = corpus_review
    result = invoke("record", corpus, sidecar, f"--{option}", "unused")
    assert result.exit_code == 3, result.output
    assert set(json.loads(result.output)) == {"diagnostics"}
    assert not sidecar_path(corpus).exists()
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            [
                {
                    "fingerprint": "a" * 64,
                    "status": "accepted",
                    "reviewer": "R",
                    "reason": "X",
                    option.replace("-", "_"): None,
                }
            ]
        )
    )
    result = invoke("record", corpus, sidecar, "--batch", str(batch))
    assert result.exit_code == 1, result.output
    assert json.loads(result.output)["diagnostics"][0]["code"] == "invalid"
    assert not sidecar_path(corpus).exists()


@pytest.mark.parametrize("payload", ['{"version":1,"decisions":[]}', "not json", "[]"])
def test_corpus_decision_errors_keep_legacy_shape(corpus_review: tuple[Path, Path], payload: str) -> None:
    corpus, sidecar = corpus_review
    sidecar_path(corpus).write_text(payload)
    result = invoke("list", corpus, sidecar)
    assert result.exit_code == 1, result.output
    assert set(json.loads(result.output)) == {"diagnostics"}
    assert sidecar_path(corpus).read_text() == payload
    result = invoke("list", corpus, sidecar, "--policy", "unused")
    assert result.exit_code == 2


def test_individual_review_and_atomic_invalid_batch(corpus_review: tuple[Path, Path], tmp_path: Path) -> None:
    corpus, sidecar = corpus_review
    claims = json.loads(invoke("list", corpus, sidecar).output)["claims"]
    result = invoke(
        "record",
        corpus,
        sidecar,
        "--fingerprint",
        claims[4]["fingerprint"],
        "--status",
        "withheld",
        "--reviewer",
        "R",
        "--reason",
        "Ambiguous",
    )
    assert result.exit_code == 0, result.output
    initial = sidecar_path(corpus).read_bytes()
    report = json.loads(invoke("inspect", corpus, sidecar).output)
    assert report["claims"][4]["review_status"] == "withheld"
    assert report["claims"][4]["reason"] == "Ambiguous"
    assert report["claims"][3]["review_status"] == "open"
    good = dict(fingerprint=claims[0]["fingerprint"], status="accepted", reviewer="R", reason="Checked")
    invalid_items: tuple[object, ...] = (
        good | {"fingerprint": "0" * 64},
        good | {"reviewer": " "},
        [],
        good | {"status": "auto"},
        good | {"status": "withheld"},
    )
    for bad in invalid_items:
        batch = tmp_path / "batch.json"
        batch.write_text(json.dumps([good, bad]))
        result = invoke("record", corpus, sidecar, "--batch", str(batch))
        assert result.exit_code == 1, result.output
        assert sidecar_path(corpus).read_bytes() == initial
    for field, value in (
        ("passage_hash", "sha256:" + "0" * 64),
        ("annotation_id", "sha256:" + "0" * 64),
        ("layer", "translation"),
    ):
        changed = json.loads(initial)
        changed["decisions"][0]["binding"][field] = value
        sidecar_path(corpus).write_text(json.dumps(changed))
        result = invoke("inspect", corpus, sidecar, "--check")
        assert result.exit_code == 1, result.output
        report = json.loads(result.output)
        assert report["claims"][4]["review_status"] == "open"
        assert report["diagnostics"][0]["code"] == "stale"


@pytest.mark.parametrize("option", ["--profile", "--phrase-input", "--selections", "--recipes", "--artifact"])
def test_corpus_conflicting_modes(corpus_review: tuple[Path, Path], option: str) -> None:
    corpus, sidecar = corpus_review
    command = "inspect" if option == "--artifact" else "list"
    args = [option] if option == "--phrase-input" else [option, "unused"]
    result = invoke(command, corpus, sidecar, *args)
    assert result.exit_code == 1, result.output
    assert not sidecar_path(corpus).exists()


def test_current_annotations_omit_inactive_and_orphaned(corpus_review: tuple[Path, Path]) -> None:
    from latinitas_cards.annotation_sidecar import annotation_id
    from latinitas_cards.canonical import canonical_json

    corpus, sidecar = corpus_review
    sidecar.write_bytes((FIXTURES / "annotations/wsl-superseded.annotations.jsonl").read_bytes())
    result = invoke("list", corpus, sidecar)
    assert result.exit_code == 0, result.output
    claims = json.loads(result.output)["claims"]
    assert len(claims) == 5
    assert [c["kind"] for c in claims if c["identity"] == "wsl:poem.1.1"] == ["segmentation"]
    # Retracting the line-2 segmentation hides both it and its orphaned morphology.
    target = next(
        r
        for r in read_sidecar(sidecar)
        if isinstance(r, Annotation) and r.layer == "segmentation" and r.anchor.passage_id == "wsl:poem.1.2"
    )
    raw = target.model_dump(mode="json", exclude={"value"})
    raw.update(record="retraction", retracts=target.annotation_id)
    raw["annotation_id"] = annotation_id(raw)
    with sidecar.open("a") as stream:
        stream.write(canonical_json(raw) + "\n")
    result = invoke("list", corpus, sidecar)
    assert result.exit_code == 0, result.output
    claims = json.loads(result.output)["claims"]
    assert len(claims) == 3
    assert all(c["identity"] != "wsl:poem.1.2" for c in claims)


def test_unreadable_decisions_are_legacy_exit_one(corpus_review: tuple[Path, Path]) -> None:
    corpus, sidecar = corpus_review
    sidecar_path(corpus).mkdir()
    result = invoke("list", corpus, sidecar)
    assert result.exit_code == 1
    assert json.loads(result.output)["diagnostics"][0]["code"] == "invalid"


@pytest.mark.parametrize("command", ["list", "record", "inspect"])
def test_deeply_nested_decisions_keep_legacy_diagnostics(corpus_review: tuple[Path, Path], command: str) -> None:
    corpus, sidecar = corpus_review
    payload = '{"version":2,"decisions":' + "[" * 100_000 + "0" + "]" * 100_000 + "}"
    sidecar_path(corpus).write_text(payload)
    result = invoke(command, corpus, sidecar)
    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert set(report) == {"diagnostics"}
    assert report["diagnostics"][0]["code"] == "invalid"
    assert sidecar_path(corpus).read_text() == payload
