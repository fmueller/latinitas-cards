import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from latinitas_cards.cli import app
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig

runner = CliRunner()


def fixture(tmp_path: Path) -> Path:
    data = json.loads(Path("docs/examples/phrase-inputs.json").read_text())
    source = tmp_path / "deck.csv"
    source.write_text(
        "ID,Word,Parts\nfriend,amicus,amici\ngirl,puella,puellae\nbook,liber,libri\nsee,video,videre\ngood,bonus,bona\n"
    )
    profile = DeckProfile.default(
        note_type="Vocabulary",
        lexical_entry_field="Word",
        principal_parts_field="Parts",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
    )
    (tmp_path / "profile.json").write_text(profile.model_dump_json())
    (tmp_path / "deck.csv.latinitas.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_columns": ["ID", "Word", "Parts"],
                "source_scope": "reviewed-fixture-scope",
                "entries": [],
            }
        )
    )
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(data))
    return path


def call(*args: str, success: bool = True) -> dict[str, Any]:
    result = runner.invoke(app, [*args, "--json"])
    assert (result.exit_code == 0) == success, result.output[:600]
    return dict(json.loads(result.stdout))


def accept(path: Path) -> None:
    claims = call("claims", "list", str(path), "--phrase-input")["claims"]
    batch = path.parent / "decisions.json"
    batch.write_text(
        json.dumps(
            [
                {
                    "fingerprint": c["fingerprint"],
                    "status": "accepted",
                    "reviewer": "Fixture reviewer",
                    "reason": "Individually checked synthetic evidence",
                }
                for c in claims
            ]
        )
    )
    call("claims", "record", str(path), "--phrase-input", "--batch", str(batch))


def test_reviewed_selection_identity_and_determinism(tmp_path: Path) -> None:
    path = fixture(tmp_path)
    source_before = (tmp_path / "deck.csv").read_bytes()
    report = call("phrases", "inspect", str(path))
    assert {d["code"] for d in report["diagnostics"]} >= {"uncertain_sense", "unreviewed"}
    request = tmp_path / "request.json"
    request.write_text(json.dumps(report["example_selection"]))
    output = tmp_path / "selected.json"
    call("phrases", "select", str(path), str(request), str(output), success=False)
    assert not output.exists()
    accept(path)
    call("phrases", "select", str(path), str(request), str(output))
    first = output.read_bytes()
    saved = json.loads(first)
    assert len(saved["selections"]) == 4
    assert {s["pattern"] for s in saved["selections"]} == {"with", "possessor", "sees", "quality"}
    clause = next(s for s in saved["selections"] if s["pattern"] == "sees")
    assert clause["source_identity"].startswith("phrase-source-v1-")
    assert clause["source_identity"] not in {"girl", "friend", "see"}
    data = json.loads(request.read_text())
    data["selections"].reverse()
    for selection in data["selections"]:
        selection["bindings"].reverse()
    request.write_text(json.dumps(data))
    call("phrases", "select", str(path), str(request), str(output))
    assert output.read_bytes() == first
    swap = next(s for s in data["selections"] if s["pattern"] == "sees")
    for binding in swap["bindings"]:
        if binding["role"] == "subject":
            binding.update(entry="friend", sense="person", form="nom-sg")
        elif binding["role"] == "object":
            binding.update(entry="girl", sense="person", form="acc-sg")
    request.write_text(json.dumps(data))
    call("phrases", "select", str(path), str(request), str(output))
    swapped = next(s for s in json.loads(output.read_text())["selections"] if s["pattern"] == "sees")
    assert swapped["source_identity"] != clause["source_identity"]
    assert (tmp_path / "deck.csv").read_bytes() == source_before


@pytest.mark.parametrize(
    ("pattern", "role", "form", "code"),
    [
        ("with", "object", "acc-sg", "government"),
        ("possessor", "dependent", "nom-sg", "government"),
        ("quality", "modifier", "fem-sg", "agreement"),
        ("sees", "verb", "present-pl", "agreement"),
        ("sees", "object", "abl-sg", "government"),
        ("with", "object", "unknown", "unsupported_form"),
    ],
)
def test_asymmetric_rejections(tmp_path: Path, pattern: str, role: str, form: str, code: str) -> None:
    path = fixture(tmp_path)
    accept(path)
    data = call("phrases", "inspect", str(path))["example_selection"]
    selection = next(s for s in data["selections"] if s["pattern"] == pattern)
    next(b for b in selection["bindings"] if b["role"] == role)["form"] = form
    request = tmp_path / "request.json"
    request.write_text(json.dumps(data))
    output = tmp_path / "selected.json"
    output.write_text("retain me")
    report = call("phrases", "select", str(path), str(request), str(output), success=False)
    assert code in {d["code"] for d in report["diagnostics"]}
    assert output.read_text() == "retain me"


def test_valency_compatibility_missing_data_and_limits(tmp_path: Path) -> None:
    path = fixture(tmp_path)
    accept(path)
    data = json.loads(path.read_text())
    data["entries"][3]["senses"][0]["valency"]["object"] = "dative"
    data["entries"][0]["forms"] = []
    data["entries"][1]["gender"] = None
    path.write_text(json.dumps(data))
    report = call("phrases", "inspect", str(path))
    assert {d["code"] for d in report["diagnostics"]} >= {"missing_paradigm", "missing_gender", "stale"}
    Path(f"{path}.claims.json").unlink()
    accept(path)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(report["example_selection"]))
    rejected = call("phrases", "select", str(path), str(request), str(tmp_path / "out.json"), success=False)
    assert "valency" in {d["code"] for d in rejected["diagnostics"]}
    request.write_text(json.dumps({"version": 1, "limits": {"max_exercises": 0}, "selections": []}))
    call("phrases", "select", str(path), str(request), str(tmp_path / "out.json"), success=False)


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_reject_noninteger_or_unknown_versions(tmp_path: Path, version: object) -> None:
    path = fixture(tmp_path)
    data = json.loads(path.read_text())
    data["version"] = version
    path.write_text(json.dumps(data))
    call("phrases", "inspect", str(path), success=False)


@pytest.mark.parametrize("failure", ["sense", "pattern", "entry", "limit", "alias", "missing-evidence"])
def test_fail_closed_selection_contract(tmp_path: Path, failure: str) -> None:
    path = fixture(tmp_path)
    accept(path)
    data = call("phrases", "inspect", str(path))["example_selection"]
    output = tmp_path / "out.json"
    if failure == "sense":
        # Known, reviewed but semantically excluded by this pattern.
        data["selections"][0]["bindings"][0].update(entry="book", sense="volume", form="nom-sg")
        code = "incompatible"
    elif failure == "pattern":
        data["selections"][0]["pattern"] = "unknown"
        code = "unknown_selection"
    elif failure == "entry":
        data["selections"][0]["bindings"][0]["entry"] = "unknown"
        code = "unknown_selection"
    elif failure == "limit":
        data["limits"]["max_exercises"] = 3
        code = "invalid_limits"
    elif failure == "alias":
        output.symlink_to(tmp_path / "deck.csv")
        code = "invalid"
    else:
        imported = json.loads(path.read_text())
        imported["patterns"][0]["evidence"] = []
        path.write_text(json.dumps(imported))
        Path(f"{path}.claims.json").unlink()
        code = "unreviewed"
        claims = call("claims", "list", str(path), "--phrase-input")["claims"]
        unsupported = next(c for c in claims if c["identity"] == "pattern:with")
        call(
            "claims",
            "record",
            str(path),
            "--phrase-input",
            "--fingerprint",
            unsupported["fingerprint"],
            "--status",
            "accepted",
            "--reviewer",
            "Reviewer",
            "--reason",
            "Cannot supply missing evidence",
            success=False,
        )
    request = tmp_path / "request.json"
    request.write_text(json.dumps(data))
    before = (tmp_path / "deck.csv").read_bytes()
    report = call("phrases", "select", str(path), str(request), str(output), success=False)
    assert code in {d["code"] for d in report["diagnostics"]}
    assert (tmp_path / "deck.csv").read_bytes() == before


def test_text_inspection_and_review_are_not_truncated(tmp_path: Path) -> None:
    path = fixture(tmp_path)
    text = runner.invoke(app, ["phrases", "inspect", str(path)])
    assert text.exit_code == 0
    for family in ("preposition-noun", "noun-genitive", "verb-noun", "adjective-noun"):
        assert family in text.stdout
    claims = call("claims", "list", str(path), "--phrase-input")["claims"]
    listed = runner.invoke(app, ["claims", "list", str(path), "--phrase-input"])
    assert listed.exit_code == 0 and claims[-1]["fingerprint"] in listed.stdout


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_selection_cannot_replace_shared_review_lock(tmp_path: Path, alias: str) -> None:
    path = fixture(tmp_path)
    accept(path)
    lock = Path(f"{path}.claims.json.lock")
    lock.write_bytes(b"preserve shared lock inode")
    before_inode = lock.stat().st_ino
    request = tmp_path / "request.json"
    request.write_text(json.dumps(call("phrases", "inspect", str(path))["example_selection"]))
    output = lock
    if alias != "direct":
        output = tmp_path / "lock-alias.json"
        if alias == "symlink":
            output.symlink_to(lock)
        else:
            output.hardlink_to(lock)
    rejected = call("phrases", "select", str(path), str(request), str(output), success=False)
    assert rejected["diagnostics"][0]["code"] == "invalid"
    assert lock.read_bytes() == b"preserve shared lock inode"
    assert lock.stat().st_ino == before_inode
    assert output.samefile(lock)
