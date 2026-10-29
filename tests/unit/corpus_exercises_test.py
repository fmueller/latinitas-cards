"""Corpus exercise candidates and identities over real normalized, segmented and reviewed corpora."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from corpus_facts_test import STAMP, Fixture, morph, text
from typer.testing import CliRunner

from latinitas_cards.claim_sidecar import sidecar_path
from latinitas_cards.cli import app
from latinitas_cards.corpus_exercises import Candidate, exercise_candidates
from latinitas_cards.corpus_facts import accepted_facts
from latinitas_cards.corpus_profile import CorpusProfile

LINES = (
    "Puer videt puerum, puer ridet.",
    "Mater filium amat. Puer legit.",
    "Canis celeriter currit",
    "et latrat.",
    "Arma et virum cano. Urbem video. Mare video.",
)
RECIPES = (
    "corpus_lexical_recognition",
    "corpus_guided_analysis",
    "corpus_phrase_translation",
    "corpus_sentence_translation",
)


def profile(language: str = "en", **overrides: Any) -> CorpusProfile:
    data: dict[str, Any] = {
        "format": "latinitas-corpus-profile",
        "version": 1,
        "profile_id": "reader-a",
        "corpora": ["facts"],
        "target_language": language,
        "deck": "Latin::A",
        "recipes": [{"recipe": recipe, "budget": 10} for recipe in RECIPES],
    }
    return CorpusProfile.model_validate(data | overrides)


def sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def expected_identity(passage_id: str, recipe: str, language: str, unit: object, target: object) -> tuple[str, str]:
    """Spec derivation written out independently of the production helper."""
    key = "corpus-exercise-v1-" + sha(
        compact({"recipe": recipe, "target_language": language, "unit": unit, "target": target})
    )
    note = compact({"object_key": key, "source_identity": passage_id, "source_scope": "corpus-passage-v1"})
    return key, "latinitas-v2-" + sha(b"latinitas-note-family-v1\0" + note.encode("utf-8"))


def expected_digest(unit_text: str, target_text: str | None) -> str:
    return "sha256:" + sha(compact({"unit_text": unit_text, "target_text": target_text}))


def candidates(facts: Fixture, chosen: CorpusProfile | None = None) -> tuple[Candidate, ...]:
    chosen = chosen or profile()
    view = accepted_facts(
        facts.corpus, facts.sidecar, sidecar_path(facts.corpus), target_language=chosen.target_language
    )
    return exercise_candidates(view, chosen)


def summary(found: tuple[Candidate, ...], passage_id: str) -> list[tuple[Any, ...]]:
    return [
        (c.recipe, c.unit, c.target, c.reasons, c.filtered)
        for c in found
        if c.passage_id == passage_id and c.annotation_id is None
    ]


@pytest.fixture
def facts(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path, LINES)


def test_four_recipes_identity_and_eligibility(facts: Fixture) -> None:
    first = facts.add(0, "morphology", morph("puer"), (0, 4))
    second = facts.add(0, "morphology", morph("puer", case="nominativus"), (19, 23))
    gloss = facts.add(0, "translation", text("en", "boy"), (19, 23))
    whole = facts.add(0, "translation", text("en", "The boy sees the boy, the boy laughs."), (0, 30))
    phrase = facts.add(0, "translation", text("en", "the boy, the boy"), (11, 23))
    trailing = facts.add(0, "translation", text("en", "sees the boy,"), (5, 18))
    twin = facts.add(0, "translation", text("en", "sees the boy"), (5, 18), name="other")
    equal = facts.add(0, "translation", text("en", "The boy sees the boy, the boy laughs"), (0, 29))
    facts.decide("accepted", second, gloss, whole, phrase, trailing, twin, equal)
    found = candidates(facts)
    sentence = {"kind": "sentence", "index": 0}
    occurrence = [{"form": "puer", "occurrence": 0}, {"form": "puer", "occurrence": 1}]
    assert summary(found, "facts:p000001") == [
        ("corpus_lexical_recognition", sentence, occurrence[0], ("open_fact", "missing_fact"), None),
        ("corpus_lexical_recognition", sentence, occurrence[1], (), None),
        ("corpus_guided_analysis", sentence, occurrence[0], ("open_fact",), None),
        ("corpus_guided_analysis", sentence, occurrence[1], (), None),
        ("corpus_phrase_translation", sentence, {"form": "puerum puer", "occurrence": 0}, (), None),
        ("corpus_sentence_translation", sentence, None, (), None),
    ]
    for candidate in found:
        if candidate.annotation_id is None:
            assert (candidate.object_key, candidate.latinitas_id) == expected_identity(
                candidate.passage_id, candidate.recipe, "en", candidate.unit, candidate.target
            )
            assert candidate.card_key == f"{candidate.recipe}:context"
    by_recipe = {c.recipe: c for c in found if c.passage_id == "facts:p000001" and c.annotation_id is None}
    # Literal digests computed with printf and sha256sum from the folded unit and target texts.
    assert by_recipe["corpus_phrase_translation"].context_digest == (
        "sha256:439990f6ad085bc2079e4964bfe29adb7f227357649472141ef21dca8e325d51"
    )
    assert by_recipe["corpus_phrase_translation"].span == (11, 23)
    assert by_recipe["corpus_lexical_recognition"].context_digest == expected_digest(
        "puer videt puerum, puer ridet.", "puer"
    )
    assert by_recipe["corpus_sentence_translation"].context_digest == expected_digest(
        "puer videt puerum, puer ridet.", None
    )
    invalid = [(c.annotation_id, c.reasons, c.unit, c.latinitas_id) for c in found if c.annotation_id is not None]
    assert sorted(invalid) == sorted(
        (row.annotation_id, ("invalid_span",), None, None) for row in (gloss, trailing, twin, equal)
    )
    # The open morphology still yields a candidate, so the report can show it.
    lexical = next(c for c in found if c.target == occurrence[0])
    assert [(f.annotation_id, f.review_status) for f in lexical.resolutions[0].facts] == [(first.annotation_id, "open")]


@pytest.mark.parametrize(
    ("span", "valid"),
    [
        ((11, 23), True),  # "puerum, puer": internal punctuation is ignored in the window
        ((0, 10), True),  # first two words
        ((5, 18), False),  # ends in punctuation
        ((17, 23), False),  # starts at punctuation
        ((0, 29), False),  # every word of the sentence
        ((1, 10), False),  # not on a word-token boundary
        ((0, 4), False),  # one word
    ],
)
def test_phrase_spans_are_qualified(facts: Fixture, span: tuple[int, int], valid: bool) -> None:
    row = facts.add(0, "translation", text("en", "x"), span)
    facts.decide("accepted", row)
    phrases = [c for c in candidates(facts) if c.recipe == "corpus_phrase_translation"]
    assert [(c.reasons, c.annotation_id is None) for c in phrases] == [
        ((), True) if valid else (("invalid_span",), False)
    ]


def test_phrase_crossing_sentences_and_occurrence_counting(tmp_path: Path) -> None:
    facts = Fixture(tmp_path, ("Puer ridet, puer ridet, puer ridet. Puer legit.",))
    rows = [
        facts.add(0, "translation", text("en", "laughs, the boy"), (5, 16)),
        facts.add(0, "translation", text("en", "the boy laughs"), (24, 34)),
        facts.add(0, "translation", text("en", "laughs. The boy"), (29, 40)),
    ]
    facts.decide("accepted", *rows)
    phrases = [c for c in candidates(facts) if c.recipe == "corpus_phrase_translation"]
    # Windows of two words: "puer ridet" occurs at word 0, 2 and 4; "ridet puer" at word 1 and 3.
    assert [(c.target, c.reasons) for c in phrases] == [
        ({"form": "ridet puer", "occurrence": 0}, ()),
        ({"form": "puer ridet", "occurrence": 2}, ()),
        (None, ("invalid_span",)),
    ]


@pytest.mark.parametrize(("span", "valid"), [((0, 44), True), ((5, 52), True), ((0, 52), False)])
def test_phrase_spans_take_at_most_eight_words(tmp_path: Path, span: tuple[int, int], valid: bool) -> None:
    # Words end at 4, 13, 18, 25, 29, 36, 39, 44, 52 and 58: (0, 44) and (5, 52) hold eight, (0, 52) nine.
    facts = Fixture(tmp_path, ("Arma virumque cano Troiae qui primus ab oris Italiam venit.",))
    facts.decide("accepted", facts.add(0, "translation", text("en", "x"), span))
    phrases = [c for c in candidates(facts) if c.recipe == "corpus_phrase_translation"]
    assert [c.reasons for c in phrases] == [() if valid else ("invalid_span",)]


def test_units_fragments_features_and_passage_limits(facts: Fixture) -> None:
    rows = [
        facts.add(1, "morphology", morph("mater"), (0, 5)),
        facts.add(1, "morphology", morph("puer", case="nominativus"), (19, 23)),
        facts.add(1, "translation", text("en", "The mother loves the son."), (0, 18)),
        facts.add(1, "translation", text("en", "The mother loves the son. The boy reads.")),
        facts.add(2, "morphology", morph("curro", "verb", person="tertia", mood="indicativus"), (16, 22)),
        facts.add(2, "translation", text("en", "runs"), (16, 22)),
        facts.add(2, "translation", text("en", "the dog quickly"), (0, 15)),
        facts.add(2, "translation", text("en", "The dog runs quickly")),
        facts.add(0, "translation", text("en", "The boy sees the boy, the boy laughs.")),
    ]
    facts.decide("accepted", *rows)
    found = candidates(facts)
    zero, one = {"kind": "sentence", "index": 0}, {"kind": "sentence", "index": 1}
    mater, puer = {"form": "mater", "occurrence": 0}, {"form": "puer", "occurrence": 0}
    # Guided analysis needs a translation of its sentence; the span-less one serves only the passage.
    assert summary(found, "facts:p000002") == [
        ("corpus_lexical_recognition", zero, mater, ("missing_fact",), None),
        ("corpus_lexical_recognition", one, puer, ("missing_fact",), None),
        ("corpus_guided_analysis", zero, mater, ("no_features",), None),
        ("corpus_guided_analysis", one, puer, ("missing_fact",), None),
        ("corpus_sentence_translation", zero, None, (), None),
        ("corpus_sentence_translation", {"kind": "passage"}, None, (), None),
    ]
    fragment = "filtered:fragment"
    currit = {"form": "currit", "occurrence": 0}
    # Lexical recognition and phrase translation may use fragment units; the other two may not.
    assert summary(found, "facts:p000003") == [
        ("corpus_lexical_recognition", zero, currit, (), None),
        ("corpus_guided_analysis", zero, currit, (), fragment),
        ("corpus_phrase_translation", zero, {"form": "canis celeriter", "occurrence": 0}, (), None),
        ("corpus_sentence_translation", zero, None, (), fragment),
    ]
    # A one-sentence passage's span-less translation serves its sentence; it forms no passage unit.
    assert summary(found, "facts:p000001") == [("corpus_sentence_translation", zero, None, (), None)]
    passage = next(c for c in found if c.unit == {"kind": "passage"})
    assert passage.context_digest == expected_digest("mater filium amat. puer legit.", None)
    limits = {"min_tokens": 3, "ideal_tokens": 4}
    for max_tokens, present in ((5, True), (4, False)):
        bounded = candidates(facts, profile(limits=limits | {"max_tokens": max_tokens}))
        assert ({"kind": "passage"} in [c.unit for c in bounded]) is present


def reviewed(root: Path, lemma: str, gloss: str, phrase: str, sentence: str, status: str | None) -> Fixture:
    root.mkdir()
    facts = Fixture(root, LINES)
    rows = [
        facts.add(0, "morphology", morph(lemma, case="nominativus"), (19, 23)),
        facts.add(0, "translation", text("en", gloss), (19, 23)),
        facts.add(0, "translation", text("de", "Knabe"), (19, 23)),
        facts.add(0, "translation", text("en", phrase), (11, 23)),
        facts.add(0, "translation", text("en", sentence), (0, 30)),
    ]
    if status is not None:
        facts.decide(status, *rows)
    return facts


def ids(found: tuple[Candidate, ...]) -> list[tuple[str, Any, Any, str | None]]:
    return [(c.recipe, c.unit, c.target, c.latinitas_id) for c in found if c.annotation_id is None]


def test_identity_ignores_profile_review_and_values(tmp_path: Path) -> None:
    facts = reviewed(tmp_path / "a", "puer", "boy", "the boy, the boy", "The boy sees the boy.", "accepted")
    before = candidates(facts)
    assert len(ids(before)) == 4 and all(not c.reasons for c in before if c.annotation_id is None)
    other = profile(
        profile_id="reader-b",
        deck="Latin::B",
        targets=["puer"],
        morphology={"theme": "monochrome"},
        recipes=[{"recipe": recipe, "budget": 3} for recipe in reversed(RECIPES)],
    )
    assert ids(candidates(facts, other)) == ids(before)
    assert ids(candidates(facts, profile("en-GB"))) == ids(before)
    german = candidates(facts, profile("de"))
    assert [row[:3] for row in ids(german)] == [row[:3] for row in ids(before)]
    assert not {row[3] for row in ids(german)} & {row[3] for row in ids(before)}
    # The German view serves the German gloss on the same exercise key.
    assert next(c for c in german if c.recipe == "corpus_lexical_recognition").reasons == ()

    values = ("puer (2)", "lad", "a boy, a boy", "A boy laughs.")
    changed = reviewed(tmp_path / "b", *values, "accepted")
    assert ids(candidates(changed)) == ids(before)
    for index, status in enumerate(("withheld", None)):
        unreviewed = candidates(reviewed(tmp_path / f"c{index}", *values, status))
        assert ids(unreviewed) == ids(before)
        assert all(c.reasons for c in unreviewed)


def test_repeated_forms_keep_ids_whatever_the_acceptance_order(facts: Fixture) -> None:
    second = facts.add(0, "morphology", morph("puer", case="nominativus"), (19, 23))
    facts.decide("accepted", second)

    def lexical() -> dict[Any, str | None]:
        return {
            json.dumps(c.target): c.latinitas_id for c in candidates(facts) if c.recipe == "corpus_lexical_recognition"
        }

    target = {"form": "puer", "occurrence": 1}
    expected = expected_identity(
        "facts:p000001", "corpus_lexical_recognition", "en", {"kind": "sentence", "index": 0}, target
    )
    assert lexical() == {json.dumps(target): expected[1]}
    first = facts.add(0, "morphology", morph("puer", case="nominativus"), (0, 4))
    facts.decide("accepted", first)
    first_target = {"form": "puer", "occurrence": 0}
    unit = {"kind": "sentence", "index": 0}
    assert lexical() == {
        json.dumps(target): expected[1],
        json.dumps(first_target): expected_identity(
            "facts:p000001", "corpus_lexical_recognition", "en", unit, first_target
        )[1],
    }


def test_splitting_sentence_zero_changes_the_context_digest_at_index_two(facts: Fixture) -> None:
    sea = facts.add(4, "translation", text("en", "I see the sea."), (33, 44))
    facts.decide("accepted", sea)
    unit = {"kind": "sentence", "index": 2}
    before = next(c for c in candidates(facts) if c.passage_id == "facts:p000005")
    assert (before.unit, before.reasons) == (unit, ())
    assert before.context_digest == expected_digest("mare video.", None)

    facts.edit(4, "Arma et. Virum cano. Urbem video. Mare video.")
    # Without a current segmentation the passage forms no units, so it yields no candidates.
    assert [c for c in candidates(facts) if c.passage_id == "facts:p000005"] == []
    result = CliRunner().invoke(app, ["annotations", "segment", str(facts.corpus), "--timestamp", STAMP, "--json"])
    assert result.exit_code == 0, result.output
    passage = facts.passage(4)
    segmentation = next(
        row["annotation_id"]
        for row in map(json.loads, facts.sidecar.read_text().splitlines()[1:])
        if row.get("layer") == "segmentation" and row["input_hash"] == passage["passage_hash"]
    )
    city = facts.add(4, "translation", text("en", "I see the city."), (21, 33), segmentation=segmentation)
    sidecar_path(facts.corpus).unlink()  # recording refuses while the old decision is stale
    facts.decide("accepted", city)
    after = next(c for c in candidates(facts) if c.passage_id == "facts:p000005")
    assert (after.unit, after.latinitas_id, after.reasons) == (unit, before.latinitas_id, ())
    assert after.context_digest == expected_digest("urbem video.", None) != before.context_digest


def test_target_language_must_match_the_view(facts: Fixture) -> None:
    view = accepted_facts(facts.corpus, facts.sidecar, sidecar_path(facts.corpus), target_language="de")
    with pytest.raises(ValueError, match="target language"):
        exercise_candidates(view, profile("en"))


def test_candidates_import_no_analyzer_or_llm_client() -> None:
    code = (
        "import sys, latinitas_cards.corpus_exercises; "
        "sys.exit(bool({'cltk', 'stanza', 'torch', 'openai', 'anthropic', 'ollama', "
        "'latinitas_cards.cli', 'latinitas_cards.annotation'} & sys.modules.keys()))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False, timeout=30).returncode == 0
