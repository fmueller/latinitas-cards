"""latinitas-corpus-profile v1 through the public `corpus-cards profile validate` command."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from latinitas_cards.cli import app
from latinitas_cards.cli_contract import CliResult, ContractError
from latinitas_cards.corpus_profile import load_corpus_profile, require_profile_corpora

EXAMPLE = Path(__file__).resolve().parents[2] / "docs/examples/corpus-profile.json"

MINIMAL: dict[str, Any] = {
    "format": "latinitas-corpus-profile",
    "version": 1,
    "profile_id": "aeneid-1",
    "corpora": ["vergil-aeneid"],
    "deck": "Latin::Aeneid",
    "recipes": [{"recipe": "corpus_sentence_translation", "budget": 7}],
}

# Defaults written out by hand from the spec, not taken from the model.
MINIMAL_EFFECTIVE: dict[str, Any] = {
    **MINIMAL,
    "target_language": "de",
    "note_type": "Latinitas Corpus Context v1",
    "budgets": {"per_run": 50, "per_work": 200},
    "reading": [],
    "reading_only": False,
    "targets": [],
    "limits": {"min_tokens": 3, "ideal_tokens": 12, "max_tokens": 30, "max_unknown": None},
    "weights": {"reading": 40, "target": 30, "novelty": 20, "length": 10, "unknown": 20, "coverage": 10},
    "morphology": {
        "version": 1,
        "theme": "muted",
        "appearance": "light",
        "comparison": "static",
        "terminology": {"version": 1, "language": "latin"},
    },
}


def independent_digest(value: object) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate(path: Path) -> tuple[int, CliResult, str]:
    result = CliRunner().invoke(app, ["corpus-cards", "profile", "validate", str(path), "--json"])
    assert result.stdout.count("\n") == 1 and result.stdout.endswith("\n"), result.stdout
    envelope = CliResult.model_validate_json(result.stdout)
    assert envelope.command == "corpus-cards profile validate"
    assert envelope.exit_code == result.exit_code
    return result.exit_code, envelope, result.stdout


def write(tmp_path: Path, document: object, name: str = "profile.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def with_change(path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    document = copy.deepcopy(MINIMAL)
    target: Any = document
    for part in path[:-1]:
        if isinstance(part, str) and part not in target:
            target[part] = {}
        target = target[part]
    target[path[-1]] = value
    return document


def test_documented_example_validates_with_canonical_digest() -> None:
    exit_code, envelope, _ = validate(EXAMPLE)
    assert exit_code == 0, envelope.diagnostics
    assert envelope.diagnostics == []
    effective = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert envelope.result == {
        "profile_id": "caesar-bg-reading",
        "profile_digest": independent_digest(effective),
        "profile": effective,
    }


def test_defaults_fill_the_effective_profile(tmp_path: Path) -> None:
    exit_code, envelope, _ = validate(write(tmp_path, MINIMAL))
    assert exit_code == 0, envelope.diagnostics
    assert isinstance(envelope.result, dict)
    assert envelope.result["profile"] == MINIMAL_EFFECTIVE
    assert envelope.result["profile_digest"] == independent_digest(MINIMAL_EFFECTIVE)


def test_digest_ignores_layout_and_explicit_defaults_but_not_values(tmp_path: Path) -> None:
    _, minimal, first = validate(write(tmp_path, MINIMAL, "minimal.json"))
    _, _, again = validate(write(tmp_path, MINIMAL, "minimal.json"))
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(dict(reversed(list(MINIMAL_EFFECTIVE.items()))), indent=4), encoding="utf-8")
    _, explicit, _ = validate(reordered)
    _, changed, _ = validate(write(tmp_path, with_change(("deck",), "Latin::Other"), "changed.json"))
    assert first == again
    assert isinstance(minimal.result, dict) and isinstance(explicit.result, dict) and isinstance(changed.result, dict)
    assert explicit.result["profile_digest"] == minimal.result["profile_digest"]
    assert changed.result["profile_digest"] != minimal.result["profile_digest"]


@pytest.mark.parametrize("tag", ["de", "en", "en-GB", "de-AT", "EN-us", "de-CH-1901"])
def test_german_and_english_tags_with_regions_validate(tmp_path: Path, tag: str) -> None:
    exit_code, envelope, _ = validate(write(tmp_path, with_change(("target_language",), tag)))
    assert exit_code == 0, envelope.diagnostics
    assert isinstance(envelope.result, dict) and isinstance(envelope.result["profile"], dict)
    assert envelope.result["profile"]["target_language"] == tag


@pytest.mark.parametrize(
    "change",
    [
        (("recipes", 0, "budget"), 1),
        (("recipes", 0, "budget"), 1000),
        (("budgets", "per_run"), 1),
        (("budgets", "per_work"), 1000),
        (("limits",), {"min_tokens": 1, "ideal_tokens": 1, "max_tokens": 1}),
        (("limits",), {"min_tokens": 200, "ideal_tokens": 200, "max_tokens": 200, "max_unknown": 200}),
        (("limits", "max_unknown"), 0),
        (("weights",), {"reading": -100, "length": 100}),
        (("corpora",), ["caesar-bg", "vergil-aeneid"]),
        (("targets",), ["amo", "Bellum", "līber"]),
        (("reading",), [{"corpus_id": "vergil-aeneid", "work": "aen", "from": "1.5", "to": "1.1"}]),
        (("reading_only",), True),
    ],
)
def test_boundary_values_validate(tmp_path: Path, change: tuple[tuple[str | int, ...], object]) -> None:
    exit_code, envelope, _ = validate(write(tmp_path, with_change(*change)))
    assert exit_code == 0, envelope.diagnostics


@pytest.mark.parametrize(
    "change",
    [
        pytest.param((("extra",), True), id="unknown-field"),
        pytest.param((("budgets", "per_day"), 1), id="unknown-nested-field"),
        pytest.param(
            (("reading",), [{"corpus_id": "vergil-aeneid", "work": "a", "from_": "1", "to": "2"}]), id="alias"
        ),
        pytest.param((("recipes", 0, "budget"), 7.0), id="float-budget"),
        pytest.param((("budgets", "per_run"), 50.0), id="float-default-value"),
        pytest.param((("weights", "length"), 1.5), id="float-weight"),
        pytest.param((("version",), 1.0), id="float-version"),
        pytest.param((("recipes", 0, "budget"), True), id="bool-budget"),
        pytest.param((("recipes", 0, "budget"), "7"), id="string-budget"),
        pytest.param((("reading_only",), 1), id="int-boolean"),
        pytest.param((("target_language",), "fr"), id="language-fr"),
        pytest.param((("target_language",), "la"), id="language-la"),
        pytest.param((("target_language",), "deu"), id="language-deu"),
        pytest.param((("target_language",), "x-de"), id="language-private"),
        pytest.param((("target_language",), "en_GB"), id="language-malformed"),
        pytest.param((("recipes",), [{"recipe": "corpus_cloze", "budget": 1}]), id="unknown-recipe"),
        pytest.param(
            (("recipes",), [{"recipe": "corpus_guided_analysis", "budget": 1}] * 2),
            id="duplicate-recipe",
        ),
        pytest.param((("recipes",), []), id="no-recipes"),
        pytest.param((("recipes", 0, "budget"), 0), id="budget-0"),
        pytest.param((("recipes", 0, "budget"), 1001), id="budget-1001"),
        pytest.param((("budgets", "per_run"), 0), id="per-run-0"),
        pytest.param((("budgets", "per_work"), 1001), id="per-work-1001"),
        pytest.param((("limits", "min_tokens"), 0), id="min-0"),
        pytest.param((("limits",), {"min_tokens": 13}), id="min-above-ideal"),
        pytest.param((("limits",), {"max_tokens": 11}), id="max-below-ideal"),
        pytest.param((("limits",), {"min_tokens": 5, "ideal_tokens": 4, "max_tokens": 30}), id="ideal-below-min"),
        pytest.param((("limits",), {"ideal_tokens": 30, "max_tokens": 201}), id="max-201"),
        pytest.param((("limits", "max_unknown"), -1), id="max-unknown-negative"),
        pytest.param((("limits", "max_unknown"), 201), id="max-unknown-201"),
        pytest.param((("weights", "reading"), 101), id="weight-101"),
        pytest.param((("weights", "unknown"), -101), id="weight-minus-101"),
        pytest.param((("corpora",), []), id="no-corpora"),
        pytest.param((("corpora",), ["vergil-aeneid", "vergil-aeneid"]), id="duplicate-corpus"),
        pytest.param((("corpora",), ["vergil-aeneid", "caesar-bg"]), id="unsorted-corpora"),
        pytest.param((("corpora",), ["Vergil"]), id="corpus-id-pattern"),
        pytest.param(
            (("reading",), [{"corpus_id": "caesar-bg", "work": "bg", "from": "1", "to": "2"}]),
            id="reading-corpus-not-listed",
        ),
        pytest.param((("reading",), [{"corpus_id": "vergil-aeneid", "work": "aen", "from": "1"}]), id="reading-to"),
        pytest.param((("targets",), ["pars", "bellum"]), id="unsorted-targets"),
        pytest.param((("targets",), ["bellum", "Bellum"]), id="targets-duplicate-after-fold"),
        pytest.param((("targets",), ["iam", "jam"]), id="targets-j-folds-to-i"),
        pytest.param((("targets",), ["liber", "līber"]), id="targets-macron-folds"),
        pytest.param((("targets",), ["amo "]), id="padded-target"),
        pytest.param((("deck",), " "), id="blank-deck"),
        pytest.param((("deck",), "Latin\n#notetype:x"), id="deck-inner-newline"),
        pytest.param((("targets",), ["amo\rx"]), id="target-inner-carriage-return"),
        pytest.param((("note_type",), " Corpus"), id="padded-note-type"),
        pytest.param(
            (("recipes",), [{"recipe": "corpus_guided_analysis", "budget": b} for b in (1, 2)]),
            id="duplicate-recipe-other-budget",
        ),
        pytest.param((("budgets", "per_run"), True), id="bool-per-run"),
        pytest.param((("limits", "min_tokens"), True), id="bool-limit"),
        pytest.param((("limits", "max_unknown"), False), id="bool-max-unknown"),
        pytest.param((("weights", "coverage"), True), id="bool-weight"),
        pytest.param((("morphology",), {"version": True}), id="bool-morphology-version"),
        pytest.param((("recipes", 0, "note"), "x"), id="extra-in-recipe"),
        pytest.param(
            (("reading",), [{"corpus_id": "vergil-aeneid", "work": "a", "from": "1", "to": "2", "x": 1}]),
            id="extra-in-reading",
        ),
        pytest.param((("limits", "x"), 1), id="extra-in-limits"),
        pytest.param((("weights", "x"), 1), id="extra-in-weights"),
        pytest.param((("morphology",), {"version": 1, "x": 1}), id="extra-in-morphology"),
        pytest.param((("targets",), [""]), id="empty-target"),
        pytest.param((("profile_id",), "Aeneid"), id="profile-id-pattern"),
        pytest.param((("profile_id",), "a"), id="profile-id-short"),
        pytest.param((("deck",), ""), id="empty-deck"),
        pytest.param((("note_type",), ""), id="empty-note-type"),
        pytest.param((("morphology",), {"version": 1, "theme": "neon"}), id="morphology-theme"),
        pytest.param((("morphology",), {"version": 1, "terminology": {"language": "user"}}), id="terminology-version"),
    ],
)
def test_invalid_profiles_exit_3(tmp_path: Path, change: tuple[tuple[str | int, ...], object]) -> None:
    path = write(tmp_path, with_change(*change))
    exit_code, envelope, _ = validate(path)
    assert exit_code == 3
    assert [(d.code, d.path) for d in envelope.diagnostics] == [("invalid_profile", str(path))]
    assert envelope.result is None


@pytest.mark.parametrize("field", ["format", "profile_id", "corpora", "deck", "recipes"])
def test_required_fields(tmp_path: Path, field: str) -> None:
    document = copy.deepcopy(MINIMAL)
    del document[field]
    exit_code, envelope, _ = validate(write(tmp_path, document))
    assert exit_code == 3
    assert [d.code for d in envelope.diagnostics] == ["invalid_profile"]


@pytest.mark.parametrize("tag", ["fr", "de_DE", "de-", " de", 1])
def test_language_error_names_supported_tags(tmp_path: Path, tag: object) -> None:
    _, envelope, _ = validate(write(tmp_path, with_change(("target_language",), tag)))
    assert "German (de) or English (en)" in envelope.diagnostics[0].message


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ('{"format":"latinitas-policy","version":1}', "unsupported_format"),
        ('{"format":"latinitas-corpus-profile","version":2}', "unsupported_version"),
        ("{", "invalid_json"),
        ('{"format":"latinitas-corpus-profile","version":1,"deck":"a","deck":"b"}', "invalid_profile"),
        ("[]", "invalid_profile"),
        ('{"\\ud800":1,"\\ud800":2}', "invalid_profile"),
        ('{"format":"latinitas-corpus-profile","version":1,"budgets":{"\\ud800":1,"\\ud800":2}}', "invalid_profile"),
        (
            '{"format":"latinitas-corpus-profile","version":1,"profile_id":"aa","corpora":["aa"],'
            '"deck":"d","recipes":[{"recipe":"corpus_guided_analysis","budget":1}],"weights":{"target":NaN}}',
            "invalid_profile",
        ),
    ],
)
def test_document_gates(tmp_path: Path, content: str, code: str) -> None:
    path = tmp_path / "profile.json"
    path.write_text(content, encoding="utf-8")
    exit_code, envelope, _ = validate(path)
    assert (exit_code, [d.code for d in envelope.diagnostics]) == (3, [code])


def test_missing_file_is_unreadable(tmp_path: Path) -> None:
    exit_code, envelope, _ = validate(tmp_path / "absent.json")
    assert (exit_code, [d.code for d in envelope.diagnostics]) == (3, ["unreadable"])


def test_validate_is_read_only(tmp_path: Path) -> None:
    path = write(tmp_path, MINIMAL)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    validate(path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["profile.json"]


@pytest.mark.parametrize("given", [["caesar-bg", "vergil-aeneid"], ["vergil-aeneid", "caesar-bg"]])
def test_profile_corpora_accept_exactly_the_listed_corpora(given: list[str]) -> None:
    require_profile_corpora(load_corpus_profile(EXAMPLE), given)


@pytest.mark.parametrize(
    ("given", "named"),
    [
        (["caesar-bg"], "vergil-aeneid"),
        (["caesar-bg", "vergil-aeneid", "cicero-cat"], "cicero-cat"),
        ([], "caesar-bg"),
    ],
)
def test_profile_corpora_mismatch_exits_3(given: list[str], named: str) -> None:
    with pytest.raises(ContractError) as raised:
        require_profile_corpora(load_corpus_profile(EXAMPLE), given)
    assert raised.value.exit_code == 3
    assert raised.value.diagnostic.code == "profile_mismatch"
    assert named in raised.value.diagnostic.message
