import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from latinitas_cards.claim_review import Evidence
from latinitas_cards.cli import app
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig

runner = CliRunner()


def test_generate_explicit_reviewed_proposals(tmp_path: Path) -> None:
    path = fixture(tmp_path)
    accept(path)
    data = call("phrases", "inspect", str(path))["example_selection"]
    data["selections"].extend(
        [
            {
                "pattern": "towards",
                "bindings": [{"role": "object", "entry": "friend", "sense": "person", "form": "acc-sg"}],
            },
            {
                "pattern": "sees",
                "bindings": [
                    {"role": "subject", "entry": "girl", "sense": "person", "form": "nom-pl"},
                    {"role": "object", "entry": "friend", "sense": "person", "form": "acc-sg"},
                    {"role": "verb", "entry": "see", "sense": "perceive", "form": "present-pl"},
                ],
            },
        ]
    )
    request, saved = tmp_path / "request.json", tmp_path / "selected.json"
    request.write_text(json.dumps(data))
    call("phrases", "select", str(path), str(request), str(saved))
    report = call("phrases", "generate", str(path), str(saved))
    assert not report["skipped"]
    assert {c["text"] for c in report["candidates"]} == {
        "cum amico",
        "ad amicum",
        "liber amici",
        "amicus bonus",
        "puella amicum videt",
        "puellae amicum vident",
    }
    assert {c["family"] for c in report["candidates"]} == {
        "preposition-noun",
        "noun-genitive",
        "verb-noun",
        "adjective-noun",
    }
    for candidate in report["candidates"]:
        assert candidate["review_state"] == "proposal"
        assert candidate["pattern_version"] == 1
        assert candidate["pattern_evidence"] and candidate["input_claims"]
        assert candidate["candidate_key"] == candidate["object_key"]
        for binding in candidate["constituents"]:
            assert binding["source"]["scope"] == "reviewed-fixture-scope"
            assert binding["sense"]["meaning"] and binding["form"]["features"]
    clauses = {c["text"]: c for c in report["candidates"] if c["family"] == "verb-noun"}
    for text, number in [("puella amicum videt", "singular"), ("puellae amicum vident", "plural")]:
        verb = next(b for b in clauses[text]["constituents"] if b["role"] == "verb")
        assert verb["form"]["features"] == {
            "case": None,
            "number": number,
            "gender": None,
            "person": "3",
            "tense": "present",
            "mood": "indicative",
            "voice": "active",
        }
    original = saved.read_bytes()
    saved_data = json.loads(original)
    saved_data["selections"].reverse()
    saved.write_text(json.dumps(saved_data))
    assert call("phrases", "generate", str(path), str(saved)) == report
    assert (tmp_path / "deck.csv").read_text().startswith("ID,Word,Parts\n")


@pytest.mark.parametrize(
    ("pattern", "role", "changes", "code"),
    [
        ("with", "object", {"form": "acc-sg"}, "government"),
        ("possessor", "dependent", {"form": "nom-sg"}, "government"),
        ("quality", "modifier", {"form": "fem-sg"}, "agreement"),
        ("sees", "verb", {"form": "present-pl"}, "agreement"),
        ("with", "object", {"form": "missing"}, "unsupported_form"),
        ("with", "object", {"entry": "book", "sense": "volume", "form": "nom-sg"}, "incompatible"),
    ],
)
def test_generation_skips_invalid_bindings(
    tmp_path: Path, pattern: str, role: str, changes: dict[str, str], code: str
) -> None:
    path, saved = saved_example(tmp_path)
    data = json.loads(saved.read_text())
    choice = next(s for s in data["selections"] if s["pattern"] == pattern)
    next(b for b in choice["bindings"] if b["role"] == role).update(changes)
    saved.write_text(json.dumps(data))
    report = call("phrases", "generate", str(path), str(saved))
    assert len(report["candidates"]) == 3
    assert code in {d["code"] for s in report["skipped"] for d in s["diagnostics"]}
    assert all(s["action"] for s in report["skipped"])


def test_generation_bounds_attempts_and_deduplicates(tmp_path: Path) -> None:
    path, saved = saved_example(tmp_path)
    data = json.loads(saved.read_text())
    choice = next(s for s in data["selections"] if s["pattern"] == "with")
    data["selections"] = [choice] * 1000
    data["limits"] = {"max_exercises": 1, "variants_per_binding": 1}
    saved.write_text(json.dumps(data))
    report = call("phrases", "generate", str(path), str(saved))
    assert len(report["candidates"]) == 1 and not report["skipped"]
    assert report["visited"] == 1 and report["unvisited"] == 999 and report["limit_reason"]
    choice["bindings"][0]["form"] = "missing"
    saved.write_text(json.dumps(data))
    report = call("phrases", "generate", str(path), str(saved))
    assert not report["candidates"] and len(report["skipped"]) == 1
    assert report["visited"] == 1 and report["unvisited"] == 999
    choice["bindings"][0]["form"] = "abl-sg"
    data["selections"] = [choice, choice]
    data["limits"]["max_exercises"] = 2
    saved.write_text(json.dumps(data))
    report = call("phrases", "generate", str(path), str(saved))
    assert len(report["candidates"]) == 1 and not report["skipped"]


def test_generation_saved_file_byte_limit(tmp_path: Path) -> None:
    path, saved = saved_example(tmp_path)
    original = saved.read_bytes()
    saved.write_bytes(original + b" " * (4 * 1024 * 1024 - len(original)))
    assert len(call("phrases", "generate", str(path), str(saved))["candidates"]) == 4
    with saved.open("ab") as stream:
        stream.write(b" ")
    report = call("phrases", "generate", str(path), str(saved), success=False)
    assert "4 MiB" in report["diagnostics"][0]["message"]


@pytest.mark.parametrize(("field", "limit"), [("scope", 2048), ("pattern", 256)])
def test_generation_saved_string_limits(tmp_path: Path, field: str, limit: int) -> None:
    path, saved = saved_example(tmp_path)
    data = json.loads(saved.read_text())
    target = data["selections"][0] if field == "pattern" else data["selections"][0]["bindings"][0]["source"]
    target[field] = "x" * limit
    saved.write_text(json.dumps(data))
    assert len(call("phrases", "generate", str(path), str(saved))["skipped"]) == 1
    target[field] += "x"
    saved.write_text(json.dumps(data))
    report = call("phrases", "generate", str(path), str(saved), success=False)
    assert "at most" in report["diagnostics"][0]["message"]


def test_generation_current_review_and_source_binding(tmp_path: Path) -> None:
    path, saved = saved_example(tmp_path)
    before = (tmp_path / "deck.csv").read_bytes()
    original = saved.read_text()
    data = json.loads(original)
    for choice in data["selections"]:
        for binding in choice["bindings"]:
            binding["source"]["scope"] = "other-deck"
    saved.write_text(json.dumps(data))
    report = call("phrases", "generate", str(path), str(saved))
    assert not report["candidates"]
    assert {d["code"] for s in report["skipped"] for d in s["diagnostics"]} == {"stale_selection"}
    saved.write_text(original)
    Path(f"{path}.claims.json").unlink()
    report = call("phrases", "generate", str(path), str(saved))
    assert not report["candidates"] and len(report["skipped"]) == 4
    assert "unreviewed" in {d["code"] for s in report["skipped"] for d in s["diagnostics"]}
    assert (tmp_path / "deck.csv").read_bytes() == before and saved.read_text() == original


def test_generation_equal_gloss_senses_and_explicit_variant_limit(tmp_path: Path) -> None:
    path = fixture(tmp_path)
    inputs = json.loads(path.read_text())
    friend = inputs["entries"][0]
    friend["senses"].append(friend["senses"][0] | {"key": "companion"})
    inputs["patterns"][0]["roles"]["object"]["senses"].append("friend/companion")
    path.write_text(json.dumps(inputs))
    accept(path)
    data = inputs["example_selection"]
    clause = next(s for s in data["selections"] if s["pattern"] == "sees")
    plural = json.loads(json.dumps(clause))
    next(b for b in plural["bindings"] if b["role"] == "subject")["form"] = "nom-pl"
    next(b for b in plural["bindings"] if b["role"] == "verb")["form"] = "present-pl"
    data["selections"].extend(
        [
            plural,
            {
                "pattern": "with",
                "bindings": [{"role": "object", "entry": "friend", "sense": "companion", "form": "abl-sg"}],
            },
        ]
    )
    data["limits"]["variants_per_binding"] = 1
    request, saved = tmp_path / "request.json", tmp_path / "selected.json"
    request.write_text(json.dumps(data))
    call("phrases", "select", str(path), str(request), str(saved))
    report = call("phrases", "generate", str(path), str(saved))
    assert len(report["candidates"]) == 5
    assert len(report["skipped"]) == 1 and report["skipped"][0]["diagnostics"][0]["code"] == "variant_limit"
    friends = [c for c in report["candidates"] if c["text"] == "cum amico"]
    assert len({c["candidate_key"] for c in friends}) == 2
    from phrase_recipes_test import exercise

    plan = tmp_path / "recipes.json"
    plan.write_text(
        json.dumps(
            {
                "version": 1,
                "max_exercises": 2,
                "exercises": [
                    exercise().model_copy(update={"candidate_key": c["candidate_key"]}).model_dump(mode="json")
                    for c in friends
                ],
            }
        )
    )
    notes = call("phrases", "preview", str(path), str(saved), str(plan))["exercises"]
    assert len({n["latinitas_id"] for n in notes}) == 2
    keys = {c["candidate_key"] for c in report["candidates"]}
    friend["senses"][0]["meaning"] = "Freund (korrigiert)"
    path.write_text(json.dumps(inputs))
    assert not call("phrases", "generate", str(path), str(saved))["candidates"]
    Path(f"{path}.claims.json").unlink()
    accept(path)
    assert {c["candidate_key"] for c in call("phrases", "generate", str(path), str(saved))["candidates"]} == keys
    corrected = call("phrases", "preview", str(path), str(saved), str(plan))["exercises"]
    assert {(n["latinitas_id"], n["slot"]) for n in corrected} == {(n["latinitas_id"], n["slot"]) for n in notes}


def saved_example(tmp_path: Path) -> tuple[Path, Path]:
    path = fixture(tmp_path)
    accept(path)
    request, saved = tmp_path / "request.json", tmp_path / "selected.json"
    request.write_text(json.dumps(call("phrases", "inspect", str(path))["example_selection"]))
    call("phrases", "select", str(path), str(request), str(saved))
    return path, saved


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


def test_explicit_recipe_plan_cli_preview_and_reproducibility(tmp_path: Path) -> None:
    from phrase_recipes_test import exercise

    source, selections = saved_example(tmp_path)
    proposed = call("phrases", "generate", str(source), str(selections))["candidates"][0]
    chosen = exercise().model_copy(update={"candidate_key": proposed["candidate_key"]})
    request, plan = tmp_path / "recipes-request.json", tmp_path / "recipes.json"
    request.write_text(json.dumps({"version": 1, "max_exercises": 1, "exercises": [chosen.model_dump(mode="json")]}))
    call("phrases", "recipes", str(source), str(selections), str(request), str(plan))
    first_bytes = plan.read_bytes()
    call("phrases", "recipes", str(source), str(selections), str(request), str(plan))
    assert plan.read_bytes() == first_bytes
    report = call("phrases", "preview", str(source), str(selections), str(plan))
    assert len(report["exercises"]) == 1
    rendered = report["exercises"][0]
    assert rendered["recipe"] == "phrase_recognition"
    assert rendered["review_state"] == "proposed" and rendered["enabled"] == ""
    assert "cum amico" in rendered["prompt"] and "mit dem Freund" in rendered["answer"]
    assert call("phrases", "preview", str(source), str(selections), str(plan)) == report
    result = runner.invoke(app, ["phrases", "preview", str(source), str(selections), str(plan)])
    assert result.exit_code == 0 and "not asserted" in result.output
    bad = chosen.model_copy(update={"target": "missing"})
    request.write_text(json.dumps({"version": 1, "max_exercises": 1, "exercises": [bad.model_dump(mode="json")]}))
    call("phrases", "recipes", str(source), str(selections), str(request), str(plan), success=False)
    assert plan.read_bytes() == first_bytes


def test_phrase_review_independent_gates_corrections_and_staleness(tmp_path: Path) -> None:
    from phrase_recipes_test import exercise

    source, selections = saved_example(tmp_path)
    source_bytes = (tmp_path / "deck.csv").read_bytes()
    candidate = call("phrases", "generate", str(source), str(selections))["candidates"][0]
    chosen = exercise().model_copy(update={"candidate_key": candidate["candidate_key"]})
    plan = tmp_path / "recipes.json"
    plan.write_text(json.dumps({"version": 1, "max_exercises": 1, "exercises": [chosen.model_dump(mode="json")]}))
    args = [str(source), "--selections", str(selections), "--recipes", str(plan)]
    claims = call("claims", "list", *args)["claims"]
    assert {c["feature"] for c in claims} == {"combination", "answer"}
    by_feature = {c["feature"]: c for c in claims}
    initial = call("phrases", "preview", str(source), str(selections), str(plan))["exercises"][0]
    assert not initial["eligible"]
    # Answer attestation alone is not combination approval.
    call(
        "claims",
        "record",
        *args,
        "--fingerprint",
        by_feature["answer"]["fingerprint"],
        "--status",
        "accepted",
        "--reviewer",
        "teacher",
        "--reason",
        "Reference checked",
    )
    answer_only = call("claims", "inspect", *args)["cards"][0]
    assert not answer_only["eligible"] and any("combination" in r for r in answer_only["reasons"])
    answer_preview = call("phrases", "preview", str(source), str(selections), str(plan))["exercises"][0]
    assert answer_preview["review_state"] == "withheld"
    assert "combination" in answer_preview["reason"]
    sidecar = Path(f"{plan}.claims.json")
    sidecar.unlink()
    # Combination approval alone is not answer attestation.
    call(
        "claims",
        "record",
        *args,
        "--fingerprint",
        by_feature["combination"]["fingerprint"],
        "--status",
        "accepted",
        "--reviewer",
        "teacher",
        "--reason",
        "Correct and natural",
    )
    combination_only = call("phrases", "preview", str(source), str(selections), str(plan))["exercises"][0]
    assert not combination_only["eligible"] and any("answer" in r for r in combination_only["reasons"])
    corrected = chosen.reference.model_dump(mode="json")
    corrected["readings"][0]["translations"] = ["in Begleitung eines Freundes"]
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            [
                {
                    "fingerprint": by_feature["answer"]["fingerprint"],
                    "status": "accepted",
                    "reviewer": "teacher",
                    "reason": "Authored correction checked",
                    "value": json.dumps(corrected),
                    "evidence": corrected["evidence"],
                }
            ]
        )
    )
    call("claims", "record", *args, "--batch", str(batch))
    approved = call("phrases", "preview", str(source), str(selections), str(plan))["exercises"][0]
    assert approved["eligible"] and approved["enabled"] == "1"
    assert "in Begleitung eines Freundes" in approved["answer"]
    assert approved["latinitas_id"] == initial["latinitas_id"]
    assert approved["slot"] == initial["slot"] == 11
    assert approved["object_key"] == candidate["object_key"]
    assert call("claims", "inspect", *args, "--check")["cards"][0]["eligible"]
    assert call("phrases", "preview", str(source), str(selections), str(plan))["exercises"][0] == approved
    changed = chosen.model_dump(mode="json")
    changed["reference"]["readings"][0]["translations"] = ["mit einem anderen Freund"]
    plan.write_text(json.dumps({"version": 1, "max_exercises": 1, "exercises": [changed]}))
    stale = call("phrases", "preview", str(source), str(selections), str(plan))
    assert not stale["exercises"][0]["eligible"]
    assert stale["exercises"][0]["latinitas_id"] == initial["latinitas_id"]
    assert any(d["code"] == "stale" for d in stale["diagnostics"])
    call("claims", "inspect", *args, "--check", success=False)
    assert (tmp_path / "deck.csv").read_bytes() == source_bytes


def test_phrase_questions_keep_identity_and_review_across_presentation_changes(tmp_path: Path) -> None:
    from phrase_recipes_test import exercise

    source, selections = saved_example(tmp_path)
    candidate = call("phrases", "generate", str(source), str(selections))["candidates"][0]
    recognition = exercise().model_copy(update={"candidate_key": candidate["candidate_key"]})
    translation = exercise("phrase_translation").model_copy(update={"candidate_key": candidate["candidate_key"]})
    number = recognition.model_copy(update={"distinction": "number"})
    number = number.model_copy(
        update={
            "reference": number.reference.model_copy(
                update={
                    "evidence": (Evidence("fixture", "number-question", "1", "Separately checked number reference"),)
                }
            )
        }
    )
    plan = tmp_path / "recipes.json"
    plan.write_text(
        json.dumps(
            {
                "version": 1,
                "max_exercises": 3,
                "exercises": [e.model_dump(mode="json") for e in (recognition, translation, number)],
            }
        )
    )
    args = [str(source), "--selections", str(selections), "--recipes", str(plan)]
    claims = call("claims", "list", *args)["claims"]
    assert len([c for c in claims if c["feature"] == "combination"]) == 1
    assert len([c for c in claims if c["feature"] == "answer"]) == 3
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            [
                {
                    "fingerprint": c["fingerprint"],
                    "status": "accepted",
                    "reviewer": "teacher",
                    "reason": "Individually checked",
                }
                for c in claims
            ]
        )
    )
    call("claims", "record", *args, "--batch", str(batch))
    first = call("phrases", "preview", str(source), str(selections), str(plan))
    assert all(e["eligible"] for e in first["exercises"])
    assert len({e["latinitas_id"] for e in first["exercises"]}) == 3
    assert len({e["object_key"] for e in first["exercises"]}) == 1
    assert {e["slot"] for e in first["exercises"]} == {10, 11}
    profile_path = tmp_path / "profile.json"
    profile = json.loads(profile_path.read_text())
    profile["morphology"]["terminology"]["language"] = "user"
    profile["morphology"]["appearance"] = "dark"
    profile_path.write_text(json.dumps(profile))
    saved = json.loads(selections.read_text())
    saved["selections"].reverse()
    for selection in saved["selections"]:
        selection["bindings"].reverse()
    selections.write_text(json.dumps(saved))
    localized = call("phrases", "preview", str(source), str(selections), str(plan))
    assert not localized["diagnostics"]
    for before, after in zip(first["exercises"], localized["exercises"], strict=True):
        assert after["eligible"]
        assert (after["latinitas_id"], after["slot"], after["claim_fingerprint"]) == (
            before["latinitas_id"],
            before["slot"],
            before["claim_fingerprint"],
        )
    assert any("Kasus" in e["answer"] for e in localized["exercises"])
    answer = next(c for c in claims if c["feature"] == "answer")
    # T064 deliberately refuses conflicting decisions rather than overwriting.
    sidecar = Path(f"{plan}.claims.json")
    before_conflict = sidecar.read_bytes()
    call(
        "claims",
        "record",
        *args,
        "--fingerprint",
        answer["fingerprint"],
        "--status",
        "withheld",
        "--reviewer",
        "teacher",
        "--reason",
        "Needs another reading",
        success=False,
    )
    assert sidecar.read_bytes() == before_conflict
    sidecar.unlink()
    decisions = json.loads(batch.read_text())
    for decision in decisions:
        if decision["fingerprint"] == answer["fingerprint"]:
            decision.update(status="withheld", reason="Needs another reading")
    batch.write_text(json.dumps(decisions))
    call("claims", "record", *args, "--batch", str(batch))
    withheld = call("claims", "inspect", *args)
    assert sum(c["eligible"] for c in withheld["cards"]) == 2
    assert any("Needs another reading" in r for c in withheld["cards"] for r in c["reasons"])
    withheld_answer = next(c["answer"] for c in withheld["cards"] if not c["eligible"])
    assert withheld_answer.count("Withheld:") == 1
    assert withheld_answer.count("Needs another reading") == 1
