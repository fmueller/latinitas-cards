"""Ranked selection through the public CLI over normalized, reviewed corpora."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from corpus_exercises_test import profile
from corpus_facts_test import Fixture, morph, text
from typer.testing import CliRunner

from latinitas_cards.annotation_sidecar import Annotation, read_sidecar
from latinitas_cards.atomic_io import replace_staged, writer_lock
from latinitas_cards.cli import app
from latinitas_cards.corpus_exercises import exercise_candidates
from latinitas_cards.corpus_notes import render_corpus_note

LEXICAL = "corpus_lexical_recognition"
SENTENCE = "corpus_sentence_translation"
RUNNER = CliRunner()


def choose(
    root: Path, facts: Fixture, *, flags: tuple[str, ...] = (), expected_exit: int = 0, **settings: Any
) -> dict[str, Any]:
    configured = profile(**settings)
    path = root / "profile.json"
    path.write_text(configured.model_dump_json())
    result = RUNNER.invoke(
        app,
        [
            "corpus-cards",
            "select",
            str(path),
            "--corpus",
            str(facts.corpus),
            "--selection",
            str(root / "selection.json"),
            "--report",
            str(root / "report.json"),
            *flags,
            "--json",
        ],
    )
    assert result.exit_code == expected_exit, result.output
    return cast(dict[str, Any], json.loads((root / "report.json").read_bytes()))


def lexical(facts: Fixture, index: int, word: str, lemma: str) -> None:
    start = facts.passage(index)["text"].index(word)
    span = (start, start + len(word))
    facts.decide(
        "accepted",
        facts.add(index, "morphology", morph(lemma), span),
        facts.add(index, "translation", text("en", lemma), span),
    )


def test_hand_computed_factors_target_and_sticky_budget(tmp_path: Path) -> None:
    facts = Fixture(tmp_path, ("Pater librum legit.", "Mater filium amat.", "Canis celer currit."))
    lexical(facts, 0, "librum", "līber (1)")
    lexical(facts, 0, "Pater", "pater")
    lexical(facts, 1, "filium", "filius")
    lexical(facts, 2, "Canis", "canis")
    configured: dict[str, Any] = {
        "recipes": [{"recipe": LEXICAL, "budget": 5}],
        "targets": ["liber"],
        "reading": [{"corpus_id": "facts", "work": "text", "from": "p000001", "to": "p000001"}],
        "limits": {"min_tokens": 1, "ideal_tokens": 4, "max_tokens": 10},
        "budgets": {"per_run": 1, "per_work": 4},
    }
    # Reading work name is source-native, not inferred from a path.
    configured["reading"][0]["work"] = facts.passage(0)["work"]
    report = choose(tmp_path, facts, **configured)
    rows = report["candidates"]
    assert rows[0]["target"] == {"form": "librum", "occurrence": 0}
    # 3 words, 2 accepted morphologies: coverage=66; target is the second token.
    assert rows[0]["factors"] == {"reading": 1, "target": 1, "novelty": 1, "length": -1, "unknown": -2, "coverage": 66}
    assert rows[0]["contributions"] == {
        "reading": 40,
        "target": 30,
        "novelty": 20,
        "length": -10,
        "unknown": -40,
        "coverage": 660,
    }
    assert rows[0]["score"] == 700
    view = facts.view()
    chosen = next(
        c for c in exercise_candidates(view, profile(**configured)) if c.latinitas_id == rows[0]["latinitas_id"]
    )
    assert chosen.latinitas_id != min(row["latinitas_id"] for row in rows)
    assert render_corpus_note(chosen, view.passages[chosen.passage_id], profile(**configured))["Latin"] == (
        'Pater <span class="corpus-target">librum</span> legit.'
    )
    assert [row["score"] for row in rows] == [700, 670, 300, 300]
    assert [row["outcome"] for row in rows] == ["selected", "budget:run", "budget:run", "budget:run"]
    assert report["runs"] == 1
    # Kept selections do not consume per_run; same-sentence exclusion follows budgets.
    again = choose(tmp_path, facts, **configured)
    assert [row["outcome"] for row in again["candidates"]] == ["kept", "near_duplicate", "selected", "budget:run"]
    assert again["runs"] == 2
    choose(tmp_path, facts, **configured)
    before = (tmp_path / "selection.json").read_bytes()
    unchanged = choose(tmp_path, facts, **configured)
    assert unchanged["runs"] == 3
    assert (tmp_path / "selection.json").read_bytes() == before
    # All three sticky selections stay, even above the one-new-exercise run budget.
    filtered = choose(tmp_path, facts, **(configured | {"limits": {"max_unknown": 0}}))
    assert sum(row["outcome"] == "kept" for row in filtered["candidates"]) == 3
    assert all(not row["eligible"] and row["factors"] is None for row in filtered["candidates"])
    assert (tmp_path / "selection.json").read_bytes() == before


@pytest.mark.parametrize(
    ("recipe_budget", "work_budget", "expected"), [(1, 1, "budget:recipe"), (3, 1, "budget:work"), (3, 3, "selected")]
)
def test_budget_boundaries(tmp_path: Path, recipe_budget: int, work_budget: int, expected: str) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.", "Mater filium amat."))
    lexical(facts, 0, "librum", "liber")
    lexical(facts, 1, "filium", "filius")
    report = choose(
        tmp_path,
        facts,
        recipes=[{"recipe": LEXICAL, "budget": recipe_budget}],
        budgets={"per_run": 1 if recipe_budget == 1 else 3, "per_work": work_budget},
    )
    assert [row["outcome"] for row in report["candidates"]] == ["selected", expected]
    if expected != "selected":
        # Newly ineligible sticky exercises keep both budget slots occupied.
        facts.retract(
            next(
                row for row in read_sidecar(facts.sidecar) if isinstance(row, Annotation) and row.layer == "morphology"
            )
        )
        rerun = choose(
            tmp_path,
            facts,
            expected_exit=1,
            recipes=[{"recipe": LEXICAL, "budget": recipe_budget}],
            budgets={"per_run": 3, "per_work": work_budget},
        )
        assert sorted(row["outcome"] for row in rerun["candidates"]) == sorted(["kept", expected])


def test_homographs_known_lemmas_and_multiset_exclusions(tmp_path: Path) -> None:
    from known_lemmas_test import _csv, _profile

    from latinitas_cards.manifest import CsvIdentityManifest

    facts = Fixture(
        tmp_path, ("Liber amat amat amat.", "Liber liber liber amat.", "Arbor portat librum.", "Servus librum tenet.")
    )
    lexical(facts, 0, "Liber", "līber (1)")
    lexical(facts, 1, "Liber", "liber (2)")
    lexical(facts, 2, "librum", "liber (1)")
    lexical(facts, 3, "librum", "liber (2)")
    source, known = tmp_path / "known.csv", tmp_path / "known.json"
    _profile(known)
    _csv(source, [("1", "līber"), ("2", "amat"), ("3", "LIBER")])
    Path(f"{source}.latinitas.json").write_text(
        CsvIdentityManifest.scoped(("ID", "Entry", "Forms", "Meaning"), "scope").to_json()
    )
    report = choose(
        tmp_path,
        facts,
        recipes=[{"recipe": LEXICAL, "budget": 10}],
        targets=["liber"],
        weights={"length": 0, "unknown": 0, "coverage": 0},
        flags=("--known-lemmas", str(known), "--known-lemmas-source", str(source)),
    )
    by_passage = {row["passage_id"]: row for row in report["candidates"]}
    # Same set, very different multiplicities: intersection=2, union=6, not a duplicate.
    assert by_passage["facts:p000001"]["outcome"] == by_passage["facts:p000002"]["outcome"] == "selected"
    assert by_passage["facts:p000003"]["outcome"] == by_passage["facts:p000004"]["outcome"] == "per_lemma"
    assert all(row["factors"]["target"] == 1 and row["factors"]["novelty"] == 0 for row in by_passage.values())
    # Accepted lemma is absent on each amat token; surface membership does not make it known.
    assert by_passage["facts:p000001"]["factors"]["unknown"] == -3
    assert report["known_lemmas"] is not None
    assert all(
        row["deck_links"] == [{"identity": "1", "scope": "scope"}, {"identity": "3", "scope": "scope"}]
        for row in by_passage.values()
    )


@pytest.mark.parametrize("while_deselected", [False, True])
def test_context_confirmation_missing_target_and_errors(tmp_path: Path, while_deselected: bool) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.",))
    lexical(facts, 0, "librum", "liber")
    settings: dict[str, Any] = {"recipes": [{"recipe": LEXICAL, "budget": 1}]}
    row = choose(tmp_path, facts, **settings)["candidates"][0]
    identity = row["latinitas_id"]
    if while_deselected:
        deselect(tmp_path, identity)
    before = (tmp_path / "selection.json").read_bytes()
    facts.edit(0, "Puer librum amat.")
    result = RUNNER.invoke(
        app, ["annotations", "segment", str(facts.corpus), "--timestamp", "2026-01-02T00:00:00Z", "--json"]
    )
    assert result.exit_code == 0, result.output
    facts.segmentations = {
        r.anchor.passage_id: r.annotation_id
        for r in read_sidecar(facts.sidecar)
        if isinstance(r, Annotation)
        and r.layer == "segmentation"
        and r.anchor.passage_hash == facts.passage(0)["passage_hash"]
    }
    # A reviewed correction gets a fresh claims file; historical stale decisions are not reused.
    Path(f"{facts.corpus}.claims.json").unlink()
    lexical(facts, 0, "librum", "liber")
    if while_deselected:
        choose(tmp_path, facts, **settings)
        assert (tmp_path / "selection.json").read_bytes() == before
        deselect(tmp_path, identity, restore=True)
        before = (tmp_path / "selection.json").read_bytes()
    corrected = choose(tmp_path, facts, **settings)["candidates"][0]
    assert corrected["latinitas_id"] == identity
    assert corrected["reasons"] == ["context_changed"]
    assert corrected["outcome"] == "kept" and corrected["score"] is None
    assert (tmp_path / "selection.json").read_bytes() == before
    if while_deselected:
        from corpus_output_test import invoke

        mtime = (tmp_path / "selection.json").stat().st_mtime_ns
        preview = invoke(tmp_path, facts, "preview")["result"]["exercises"]
        assert [(e["latinitas_id"], e["reasons"], e["answer"]) for e in preview] == [
            (identity, ["context_changed"], "")
        ]
        invoke(tmp_path, facts, "inspect")
        output = tmp_path / "withheld.csv"
        withheld = invoke(tmp_path, facts, "export", "--output", str(output), code=1)
        assert withheld["diagnostics"][-1]["code"] == "nothing_to_export" and not output.exists()
        assert (tmp_path / "selection.json").stat().st_mtime_ns == mtime
        assert (tmp_path / "selection.json").read_bytes() == before
    confirmed = choose(tmp_path, facts, flags=("--confirm-context", identity), **settings)
    assert confirmed["runs"] == (4 if while_deselected else 2) and confirmed["candidates"][0]["eligible"]
    if while_deselected:
        assert invoke(tmp_path, facts, "preview")["result"]["exercises"][0]["answer"]
        assert invoke(tmp_path, facts, "export", "--output", str(output))["result"]["exported"] == 1
    facts.edit(0, "Puer volumen amat.")
    missing = choose(tmp_path, facts, expected_exit=1, **settings)["candidates"]
    assert [(r["outcome"], r["reasons"]) for r in missing] == [("kept", ["target_missing"])]


def test_locks_aliases_and_failure_preserve_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from latinitas_cards import corpus_selection

    facts = Fixture(tmp_path, ("Puer librum legit.", "Mater filium amat."))
    lexical(facts, 0, "librum", "liber")
    settings: dict[str, Any] = {"recipes": [{"recipe": LEXICAL, "budget": 10}]}
    choose(tmp_path, facts, **settings)
    state, report = tmp_path / "selection.json", tmp_path / "report.json"
    before = {path: path.read_bytes() for path in (state, report)}
    args = [
        "corpus-cards",
        "select",
        str(tmp_path / "profile.json"),
        "--corpus",
        str(facts.corpus),
        "--selection",
        str(state),
        "--report",
        str(report),
        "--json",
    ]
    for output in (state, report):
        with writer_lock(output):
            result = RUNNER.invoke(app, args)
        assert result.exit_code == 4 and '"code":"busy"' in result.output
        assert {path: path.read_bytes() for path in before} == before
    for alias in (
        facts.corpus,
        Path(f"{facts.corpus}.lock"),
        facts.sidecar,
        Path(f"{facts.corpus}.claims.json"),
        Path(f"{facts.sidecar}.lock"),
        state,
    ):
        aliased = args.copy()
        aliased[aliased.index("--report") + 1] = str(alias)
        result = RUNNER.invoke(app, aliased)
        assert result.exit_code == 3 and '"code":"alias"' in result.output
    lexical(facts, 1, "filium", "filius")
    original = replace_staged

    def fail(path: Path, staged: Path) -> None:
        if path == state:
            raise OSError("injected before selection replacement")
        original(path, staged)

    monkeypatch.setattr(corpus_selection, "replace_staged", fail)
    result = RUNNER.invoke(app, args)
    assert result.exit_code != 0
    assert state.read_bytes() == before[state]
    assert not list(tmp_path.glob(".selection.json.*"))
    monkeypatch.setattr(corpus_selection, "replace_staged", original)
    # Even a corrupt orphan report is not state, and cannot advance the run twice.
    report.write_text("NOT STATE")
    assert RUNNER.invoke(app, args).exit_code == 0
    recovered = json.loads(state.read_bytes())
    assert recovered["runs"] == json.loads(before[state])["runs"] + 1
    assert len(recovered["selected"]) == 2


@pytest.mark.parametrize("change", [{"target_language": "de"}, {"target_language": "en-US"}, {"profile_id": "another"}])
def test_exact_selection_binding(tmp_path: Path, change: dict[str, str]) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.",))
    lexical(facts, 0, "librum", "liber")
    choose(tmp_path, facts)
    state = tmp_path / "selection.json"
    before = state.read_bytes()
    (tmp_path / "profile.json").write_text(profile(**change).model_dump_json())
    result = RUNNER.invoke(
        app,
        [
            "corpus-cards",
            "select",
            str(tmp_path / "profile.json"),
            "--corpus",
            str(facts.corpus),
            "--selection",
            str(state),
            "--report",
            str(tmp_path / "report.json"),
            "--json",
        ],
    )
    assert result.exit_code == 4 and '"code":"binding_drift"' in result.output
    assert state.read_bytes() == before


def test_filters_reading_ineligible_order_and_sentence_factors(tmp_path: Path) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.", "Mater filium amat.", "Canis currit", "et latrat."))
    for i in range(4):
        facts.decide("accepted", facts.add(i, "translation", text("en", "whole")))
    lexical(facts, 0, "librum", "liber")
    lexical(facts, 0, "Puer", "puer")
    invalid = [
        facts.add(0, "translation", text("en", "bad"), (1, 3)),
        facts.add(0, "translation", text("en", "bad"), (5, 10)),
    ]
    facts.add(1, "morphology", morph("filius"), (6, 12))  # Open, so no score.
    report = choose(tmp_path, facts, targets=["liber", "puer"], limits={"ideal_tokens": 3})
    sentence = next(r for r in report["candidates"] if r["recipe"] == SENTENCE and r["passage_id"] == "facts:p000001")
    assert sentence["factors"] == {"reading": 0, "target": 2, "novelty": 0, "length": 0, "unknown": -3, "coverage": 66}
    assert sentence["score"] == 660  # 60 - 60 + 660; lexical targets alone get novelty.
    entries = report["candidates"]
    ineligible = [r for r in entries if not r["eligible"] and r["annotation_id"] is None]
    assert [r["latinitas_id"] for r in ineligible] == sorted(r["latinitas_id"] for r in ineligible)
    expected_invalid = sorted(row.annotation_id for row in invalid)
    assert [r["annotation_id"] for r in entries[-len(expected_invalid) :]] == expected_invalid
    assert [r["annotation_id"] for r in entries if r["annotation_id"] is not None] == expected_invalid
    # Fragment wins over length/unknown/reading; length wins over unknown/reading.
    filtered = choose(tmp_path, facts, reading_only=True, limits={"min_tokens": 4, "ideal_tokens": 4, "max_unknown": 0})
    assert {
        r["outcome"]
        for r in filtered["candidates"]
        if r["recipe"] == SENTENCE and r["passage_id"] in {"facts:p000003", "facts:p000004"}
    } == {"filtered:fragment"}
    length_row = next(
        r for r in filtered["candidates"] if r["recipe"] == SENTENCE and r["passage_id"] == "facts:p000002"
    )
    assert length_row["reasons"][0] == "filtered:length"
    unknown = choose(tmp_path, facts, reading_only=True, limits={"max_unknown": 1})
    unknown_row = next(
        r for r in unknown["candidates"] if r["recipe"] == SENTENCE and r["passage_id"] == "facts:p000002"
    )
    assert unknown_row["reasons"][0] == "filtered:unknown"
    reading = choose(tmp_path, facts, reading_only=True)
    reading_row = next(
        r for r in reading["candidates"] if r["recipe"] == SENTENCE and r["passage_id"] == "facts:p000002"
    )
    assert reading_row["reasons"][0] == "filtered:reading"


@pytest.mark.parametrize(("first", "last"), [("p000002", "p000001"), ("absent", "p000002")])
def test_reading_validated_at_use(tmp_path: Path, first: str, last: str) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.", "Mater filium amat."))
    path = tmp_path / "profile.json"
    path.write_text(
        profile(
            reading=[{"corpus_id": "facts", "work": facts.passage(0)["work"], "from": first, "to": last}]
        ).model_dump_json()
    )
    assert RUNNER.invoke(app, ["corpus-cards", "profile", "validate", str(path), "--json"]).exit_code == 0
    result = RUNNER.invoke(
        app,
        [
            "corpus-cards",
            "select",
            str(path),
            "--corpus",
            str(facts.corpus),
            "--selection",
            str(tmp_path / "selection.json"),
            "--report",
            str(tmp_path / "report.json"),
            "--json",
        ],
    )
    assert result.exit_code == 3 and '"code":"invalid_reading"' in result.output
    assert not (tmp_path / "selection.json").exists() and not (tmp_path / "report.json").exists()


def test_subprocess_reproducibility_and_corpus_order(tmp_path: Path) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.",))
    lexical(facts, 0, "librum", "liber")
    other = tmp_path / "other.jsonl"
    source = tmp_path / "other.txt"
    source.write_text("Mater filium amat.\n")
    normalized = RUNNER.invoke(
        app,
        [
            "corpus",
            "normalize",
            str(source),
            "--adapter",
            "text",
            "--corpus-id",
            "other",
            "--output",
            str(other),
            "--json",
        ],
    )
    assert normalized.exit_code == 0, normalized.output
    assert (
        RUNNER.invoke(app, ["annotations", "segment", str(other), "--timestamp", "2026-01-01T00:00:00Z"]).exit_code == 0
    )
    # Reuse the fixture's public annotation/claim helpers for a second candidate-bearing corpus.
    other_facts = object.__new__(Fixture)
    other_facts.corpus, other_facts.sidecar = other, other.with_suffix(".annotations.jsonl")
    other_facts.segmentations = {
        r.anchor.passage_id: r.annotation_id
        for r in read_sidecar(other_facts.sidecar)
        if isinstance(r, Annotation) and r.layer == "segmentation"
    }
    lexical(other_facts, 0, "filium", "filius")
    settings = tmp_path / "profile.json"
    settings.write_text(
        profile(
            corpora=["facts", "other"], recipes=[{"recipe": LEXICAL, "budget": 10}], budgets={"per_run": 1}
        ).model_dump_json()
    )
    outputs = []
    for index, (cwd, tz, locale, seed, paths) in enumerate(
        [
            (tmp_path, "UTC", "C", "1", [facts.corpus, other]),
            (Path("/tmp"), "Europe/Berlin", "C.UTF-8", "42", [other, facts.corpus]),
        ]
    ):
        state, report = tmp_path / f"selection{index}.json", tmp_path / f"report{index}.json"
        args = [
            sys.executable,
            "-m",
            "latinitas_cards",
            "corpus-cards",
            "select",
            str(settings),
            *(arg for path in paths for arg in ("--corpus", str(path))),
            "--selection",
            str(state),
            "--report",
            str(report),
            "--json",
        ]
        result = subprocess.run(
            args,
            cwd=cwd,
            env=os.environ | {"TZ": tz, "LC_ALL": locale, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert [(r["passage_id"], r["outcome"]) for r in json.loads(report.read_bytes())["candidates"]] == [
            ("facts:p000001", "selected"),
            ("other:p000001", "budget:run"),
        ]
        outputs.append((state.read_bytes(), report.read_bytes()))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("second", "outcome"),
    [
        ("Amo amo amo amo.", "near_duplicate"),  # 4/5 exactly (multiplicity differs).
        ("Amo amo amo.", "selected"),  # 3/5, despite identical sets.
    ],
)
def test_near_duplicate_inclusive_multiset_boundary(tmp_path: Path, second: str, outcome: str) -> None:
    facts = Fixture(tmp_path, ("Amo amo amo amo amo.", second))
    for index in range(2):
        facts.decide("accepted", facts.add(index, "translation", text("en", "I love")))
    report = choose(tmp_path, facts, recipes=[{"recipe": SENTENCE, "budget": 10}], weights={"length": 0, "unknown": 0})
    assert [(r["passage_id"], r["outcome"]) for r in report["candidates"]] == [
        ("facts:p000001", "selected"),
        ("facts:p000002", outcome),
    ]


def test_deselected_precedes_budgets_and_never_reselects(tmp_path: Path) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.", "Mater filium amat."))
    lexical(facts, 0, "librum", "liber")
    lexical(facts, 1, "filium", "filius")
    choose(tmp_path, facts, recipes=[{"recipe": LEXICAL, "budget": 10}])
    state = tmp_path / "selection.json"
    value = json.loads(state.read_bytes())
    removed = value["selected"][-1]
    deselect(tmp_path, removed["latinitas_id"])
    report = choose(tmp_path, facts, recipes=[{"recipe": LEXICAL, "budget": 1}], budgets={"per_run": 1, "per_work": 1})
    assert (
        next(r for r in report["candidates"] if r["latinitas_id"] == removed["latinitas_id"])["outcome"] == "deselected"
    )
    assert len(json.loads(state.read_bytes())["selected"]) == 1


def deselect(root: Path, *identities: str, restore: bool = False, reason: str = "Reviewed removal") -> None:
    args = ["corpus-cards", "deselect", str(root / "selection.json"), "--reason", reason, "--json"]
    for identity in identities:
        args += ["--exercise", identity]
    if restore:
        args.append("--restore")
    result = RUNNER.invoke(app, args)
    assert result.exit_code == 0, result.output


def test_deselect_restore_preserves_entries_and_noops(tmp_path: Path) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.", "Mater filium amat."))
    lexical(facts, 0, "librum", "liber")
    lexical(facts, 1, "filium", "filius")
    settings: dict[str, Any] = {"recipes": [{"recipe": LEXICAL, "budget": 2}]}
    choose(tmp_path, facts, **settings)
    state = tmp_path / "selection.json"
    original = json.loads(state.read_bytes())
    identities = [row["latinitas_id"] for row in original["selected"]]
    before, mtime = state.read_bytes(), state.stat().st_mtime_ns
    unknown = RUNNER.invoke(
        app,
        [
            "corpus-cards",
            "deselect",
            str(state),
            "--exercise",
            identities[0],
            "--exercise",
            "absent",
            "--reason",
            "No partial changes",
            "--json",
        ],
    )
    assert unknown.exit_code == 3, unknown.output
    assert json.loads(unknown.stdout)["diagnostics"][0]["code"] == "unknown_exercise"
    assert (state.read_bytes(), state.stat().st_mtime_ns) == (before, mtime)
    deselect(tmp_path, identities[0], reason="First review")
    first_tombstone = json.loads(state.read_bytes())["deselected"][0]
    assert first_tombstone == original["selected"][0] | {"run": 2, "reason": "First review"}
    deselect(tmp_path, *reversed(identities), identities[0], reason="Keep this exact reason.\nReviewed.")
    value = json.loads(state.read_bytes())
    assert value["runs"] == 3 and value["selected"] == []
    assert value["deselected"] == [
        first_tombstone,
        original["selected"][1] | {"run": 3, "reason": "Keep this exact reason.\nReviewed."},
    ]
    before, mtime = state.read_bytes(), state.stat().st_mtime_ns
    deselect(tmp_path, *identities, reason="Must not overwrite the reason")
    choose(tmp_path, facts, **settings)
    assert (state.read_bytes(), state.stat().st_mtime_ns) == (before, mtime)
    result = RUNNER.invoke(
        app,
        [
            "corpus-cards",
            "deselect",
            str(state),
            "--exercise",
            identities[0],
            "--exercise",
            "absent",
            "--reason",
            "No partial updates",
            "--restore",
            "--json",
        ],
    )
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "unknown_exercise"
    assert (state.read_bytes(), state.stat().st_mtime_ns) == (before, mtime)
    deselect(tmp_path, identities[0], restore=True)
    first_restored = json.loads(state.read_bytes())["selected"][0]
    assert first_restored == original["selected"][0] | {"run": 4}
    deselect(tmp_path, *identities, identities[1], restore=True)
    restored = json.loads(state.read_bytes())
    assert restored == original | {"runs": 5, "selected": [first_restored, original["selected"][1] | {"run": 5}]}
    before, mtime = state.read_bytes(), state.stat().st_mtime_ns
    deselect(tmp_path, *identities, restore=True)
    assert (state.read_bytes(), state.stat().st_mtime_ns) == (before, mtime)
    assert all(row["outcome"] == "kept" for row in choose(tmp_path, facts, **settings)["candidates"])


@pytest.mark.parametrize("failure", ["busy", "symlink", "hardlink", "replace", "invalid", "empty_reason"])
def test_deselect_writer_failures_preserve_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    from latinitas_cards import corpus_selection

    facts = Fixture(tmp_path, ("Puer librum legit.",))
    lexical(facts, 0, "librum", "liber")
    identity = choose(tmp_path, facts, recipes=[{"recipe": LEXICAL, "budget": 1}])["candidates"][0]["latinitas_id"]
    state = tmp_path / "selection.json"
    lock = Path(f"{state}.lock")
    expected_code, expected_exit = "unreadable", 3
    if failure in {"symlink", "hardlink"}:
        lock.unlink()
        if failure == "symlink":
            lock.symlink_to(state)
        else:
            os.link(state, lock)
        expected_code = "alias"
    elif failure == "replace":

        def fail_replace(path: Path, staged: Path) -> None:
            raise OSError("Injected before replace")

        monkeypatch.setattr(corpus_selection, "replace_staged", fail_replace)
    elif failure == "invalid":
        state.write_text('{"format":"latinitas-corpus-selection","format":"duplicate"}')
        expected_code = "invalid_input"
    elif failure == "empty_reason":
        expected_code, expected_exit = "usage_error", 2
    else:
        expected_code, expected_exit = "busy", 4
    before, mtime = state.read_bytes(), state.stat().st_mtime_ns
    args = [
        "corpus-cards",
        "deselect",
        str(state),
        "--exercise",
        identity,
        "--reason",
        "" if failure == "empty_reason" else "Reviewed",
        "--json",
    ]
    if failure == "busy":
        with writer_lock(state):
            result = RUNNER.invoke(app, args)
    else:
        result = RUNNER.invoke(app, args)
    assert result.exit_code == expected_exit, result.output
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == expected_code
    assert (state.read_bytes(), state.stat().st_mtime_ns) == (before, mtime)
    assert not list(tmp_path.glob(".selection.json.*"))


def select_args(root: Path, facts: Fixture) -> list[str]:
    return [
        "corpus-cards",
        "select",
        str(root / "profile.json"),
        "--corpus",
        str(facts.corpus),
        "--selection",
        str(root / "selection.json"),
        "--report",
        str(root / "report.json"),
        "--json",
    ]


@pytest.mark.parametrize(
    "malformation", ["duplicate:selected", "duplicate:deselected", "format", "selected", "deselected", "runs"]
)
def test_reject_incomplete_or_ambiguous_persisted_state(tmp_path: Path, malformation: str) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.",))
    lexical(facts, 0, "librum", "liber")
    choose(tmp_path, facts)
    state, report = tmp_path / "selection.json", tmp_path / "report.json"
    value = json.loads(state.read_bytes())
    if malformation.startswith("duplicate:"):
        key = malformation.split(":")[1]
        payload = json.dumps(value)[:-1] + f',"{key}":[]}}'
    else:
        del value[malformation]
        payload = json.dumps(value)
    state.write_text(payload)
    before = {path: path.read_bytes() for path in (state, report)}
    result = RUNNER.invoke(app, select_args(tmp_path, facts))
    assert result.exit_code == 3 and '"code":"invalid_input"' in result.output
    assert {path: path.read_bytes() for path in before} == before


def test_initial_empty_creation_counts_once(tmp_path: Path) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.",))
    assert choose(tmp_path, facts)["runs"] == 1
    state = tmp_path / "selection.json"
    with state.open("rb") as held:
        inode = os.fstat(held.fileno()).st_ino
        before = state.read_bytes()
        assert choose(tmp_path, facts)["runs"] == 1
        assert state.stat().st_ino == inode and state.read_bytes() == before
    lexical(facts, 0, "librum", "liber")
    assert choose(tmp_path, facts)["runs"] == 2
    assert json.loads(state.read_bytes())["selected"][0]["run"] == 2


@pytest.mark.parametrize("option", ["--selection", "--report"])
def test_logical_claims_lock_protected_when_claims_symlinked(tmp_path: Path, option: str) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.",))
    lexical(facts, 0, "librum", "liber")
    choose(tmp_path, facts)
    claims = Path(f"{facts.corpus}.claims.json")
    target = tmp_path / "actual-claims.json"
    claims.rename(target)
    claims.symlink_to(target)
    logical_lock = Path(f"{claims}.lock")
    # Valid state bytes make the selection-output variant independently reachable.
    logical_lock.write_bytes((tmp_path / "selection.json").read_bytes())
    before = logical_lock.read_bytes()
    args = select_args(tmp_path, facts)
    args[args.index(option) + 1] = str(logical_lock)
    with logical_lock.open("rb") as held:
        import fcntl

        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = RUNNER.invoke(app, args)
        assert result.exit_code == 3 and '"code":"alias"' in result.output
        assert logical_lock.stat().st_ino == os.fstat(held.fileno()).st_ino
        assert logical_lock.read_bytes() == before


def test_report_hashes_match_consumed_snapshot_during_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from latinitas_cards import corpus_selection
    from latinitas_cards.canonical import digest_bytes
    from latinitas_cards.corpus_facts import accepted_facts

    facts = Fixture(tmp_path, ("Puer librum legit.", "Mater filium amat."))
    lexical(facts, 0, "librum", "liber")
    before_sidecar = facts.sidecar.read_bytes()
    before_claims = Path(f"{facts.corpus}.claims.json").read_bytes()

    def concurrent_update(corpus: Path, sidecar: Path, claims: Path, *, target_language: str) -> Any:
        captured = accepted_facts(corpus, sidecar, claims, target_language=target_language)
        lexical(facts, 1, "filium", "filius")
        return captured

    monkeypatch.setattr(corpus_selection, "accepted_facts", concurrent_update)
    report = choose(tmp_path, facts, recipes=[{"recipe": LEXICAL, "budget": 10}])
    assert len(report["candidates"]) == 1
    assert report["corpora"]["facts"]["sidecar_digest"] == digest_bytes(before_sidecar)
    assert report["corpora"]["facts"]["claims_digest"] == digest_bytes(before_claims)
    monkeypatch.setattr(corpus_selection, "accepted_facts", accepted_facts)
    fresh = choose(tmp_path, facts, recipes=[{"recipe": LEXICAL, "budget": 10}])
    assert len(fresh["candidates"]) == 2
    assert fresh["corpora"] != report["corpora"]


def test_known_accepted_context_reduces_unknown_count(tmp_path: Path) -> None:
    from known_lemmas_test import _csv, _profile

    from latinitas_cards.manifest import CsvIdentityManifest

    facts = Fixture(tmp_path, ("Puer librum legit.",))
    lexical(facts, 0, "librum", "liber")
    facts.decide("accepted", facts.add(0, "morphology", morph("puer (2)"), (0, 4)))
    source, known = tmp_path / "known.csv", tmp_path / "known.json"
    _profile(known)
    _csv(source, [("1", "puer")])
    Path(f"{source}.latinitas.json").write_text(
        CsvIdentityManifest.scoped(("ID", "Entry", "Forms", "Meaning"), "scope").to_json()
    )
    report = choose(
        tmp_path,
        facts,
        recipes=[{"recipe": LEXICAL, "budget": 10}],
        targets=["liber"],
        limits={"ideal_tokens": 3, "max_unknown": 1},
        flags=("--known-lemmas", str(known), "--known-lemmas-source", str(source)),
    )
    row = next(r for r in report["candidates"] if r["target"]["form"] == "librum")
    assert row["factors"] == {"reading": 0, "target": 1, "novelty": 1, "length": 0, "unknown": -1, "coverage": 66}
    assert row["score"] == 690 and row["eligible"]
    for option in ("--selection", "--report"):
        for protected in (known, source, Path(f"{source}.latinitas.json")):
            args = select_args(tmp_path, facts) + ["--known-lemmas", str(known), "--known-lemmas-source", str(source)]
            args[args.index(option) + 1] = str(protected)
            before = protected.read_bytes()
            result = RUNNER.invoke(app, args)
            assert result.exit_code == 3 and '"code":"alias"' in result.output
            assert protected.read_bytes() == before


def test_signed_weights_guided_target_cap_and_confidence(tmp_path: Path) -> None:
    from latinitas_cards.annotation_sidecar import annotation_id
    from latinitas_cards.canonical import canonical_json

    facts = Fixture(tmp_path, ("Puer librum domi legit.",))
    for word, lemma in [("Puer", "puer"), ("librum", "liber"), ("domi", "domus"), ("legit", "lego")]:
        start = facts.passage(0)["text"].index(word)
        features = {"case": "accusativus"} if word == "librum" else {}
        facts.decide("accepted", facts.add(0, "morphology", morph(lemma, **features), (start, start + len(word))))
    facts.decide("accepted", facts.add(0, "translation", text("en", "The boy reads a book at home.")))
    settings: dict[str, Any] = {
        "recipes": [{"recipe": "corpus_guided_analysis", "budget": 10}, {"recipe": SENTENCE, "budget": 10}],
        "targets": ["domus", "lego", "liber", "puer"],
        "limits": {"ideal_tokens": 5},
        "reading": [{"corpus_id": "facts", "work": facts.passage(0)["work"], "from": "p000001", "to": "p000001"}],
        "weights": {"reading": -2, "target": -3, "novelty": -5, "length": -7, "unknown": -11, "coverage": 1},
    }
    report = choose(tmp_path, facts, **settings)
    sentence, guided = [r for r in report["candidates"] if r["eligible"]]
    assert sentence["recipe"] == SENTENCE and sentence["score"] == 140
    assert sentence["factors"] == {
        "reading": 1,
        "target": 3,
        "novelty": 0,
        "length": -1,
        "unknown": -4,
        "coverage": 100,
    }
    assert sentence["contributions"] == {
        "reading": -2,
        "target": -9,
        "novelty": 0,
        "length": 7,
        "unknown": 44,
        "coverage": 100,
    }
    assert guided["target"] == {"form": "librum", "occurrence": 0} and guided["score"] == 130
    assert guided["factors"] == {"reading": 1, "target": 1, "novelty": 1, "length": -1, "unknown": -3, "coverage": 100}
    assert guided["contributions"] == {
        "reading": -2,
        "target": -3,
        "novelty": -5,
        "length": 7,
        "unknown": 33,
        "coverage": 100,
    }
    # Vary confidence on all accepted facts, re-key and review them; facts/identity stay equal.
    rows = [json.loads(line) for line in facts.sidecar.read_text().splitlines()]
    for row in rows[1:]:
        if row["layer"] != "segmentation":
            row["confidence"] = 7 if row["layer"] == "morphology" else 999
            row["annotation_id"] = annotation_id(row)
    facts.sidecar.write_text("".join(canonical_json(row) + "\n" for row in rows))
    Path(f"{facts.corpus}.claims.json").unlink()
    facts.decide(
        "accepted", *(r for r in read_sidecar(facts.sidecar) if isinstance(r, Annotation) and r.layer != "segmentation")
    )
    (tmp_path / "selection.json").unlink()
    assert choose(tmp_path, facts, **settings)["candidates"] == report["candidates"]


@pytest.mark.parametrize(("second", "expected"), [("amo amo amo amo", "near_duplicate"), ("amo amo amo", "selected")])
def test_phrase_multiset_ignores_surrounding_unit(tmp_path: Path, second: str, expected: str) -> None:
    facts = Fixture(tmp_path, ("Puer amo amo amo amo amo hodie.", f"Mater {second} nunc ridet."))
    for index, phrase in [(0, "amo amo amo amo amo"), (1, second)]:
        start = facts.passage(index)["text"].index(phrase)
        facts.decide("accepted", facts.add(index, "translation", text("en", "I love"), (start, start + len(phrase))))
    report = choose(
        tmp_path,
        facts,
        recipes=[{"recipe": "corpus_phrase_translation", "budget": 10}],
        weights={"length": 0, "unknown": 0},
    )
    assert [r["outcome"] for r in report["candidates"]] == ["selected", expected]


def test_exclusion_precedence_collisions(tmp_path: Path) -> None:
    facts = Fixture(tmp_path, ("Puer librum legit.", "Puer librum legit."))
    for index in range(2):
        lexical(facts, index, "librum", "liber")
    report = choose(tmp_path, facts, recipes=[{"recipe": LEXICAL, "budget": 3}], budgets={"per_run": 1, "per_work": 1})
    assert [r["outcome"] for r in report["candidates"]] == ["selected", "budget:work"]
    report = choose(tmp_path, facts, recipes=[{"recipe": LEXICAL, "budget": 3}])
    assert [r["outcome"] for r in report["candidates"]] == ["kept", "near_duplicate"]  # Both also share a lemma.


def test_zero_score_all_unit_target_and_id_tiebreakers(tmp_path: Path) -> None:
    from corpus_exercises_test import expected_identity

    facts = Fixture(tmp_path, ("Pater librum legit. Mater filium amat.",))
    for word, lemma in [("Pater", "pater"), ("librum", "liber"), ("Mater", "mater"), ("filium", "filius")]:
        start = facts.passage(0)["text"].index(word)
        facts.decide(
            "accepted",
            facts.add(0, "morphology", morph(lemma, case="nominativus"), (start, start + len(word))),
            facts.add(0, "translation", text("en", lemma), (start, start + len(word))),
        )
    for span in [(0, 19), (20, 38), None]:
        facts.decide("accepted", facts.add(0, "translation", text("en", "whole"), span))
    report = choose(
        tmp_path,
        facts,
        recipes=[{"recipe": r, "budget": 20} for r in [LEXICAL, "corpus_guided_analysis", SENTENCE]],
        weights={"reading": 0, "target": 0, "novelty": 0, "length": 0, "unknown": 0, "coverage": 0},
    )
    # IDs computed independently from the identity contract, never the ranking implementation.
    expected = []
    for index, words in enumerate([("pater", "librum"), ("mater", "filium")]):
        unit = {"kind": "sentence", "index": index}
        for word in words:
            expected += sorted(
                expected_identity("facts:p000001", r, "en", unit, {"form": word, "occurrence": 0})[1]
                for r in [LEXICAL, "corpus_guided_analysis"]
            )
        expected.append(expected_identity("facts:p000001", SENTENCE, "en", unit, None)[1])
    expected.append(expected_identity("facts:p000001", SENTENCE, "en", {"kind": "passage"}, None)[1])
    assert [r["latinitas_id"] for r in report["candidates"]] == expected
    assert all(r["score"] == 0 for r in report["candidates"])
