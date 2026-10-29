"""The accepted-fact view resolves per-anchor facts from real corpus, sidecar and claim files."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from latinitas_cards import corpus_facts
from latinitas_cards.annotation_sidecar import Annotation, Retraction, annotation_id, append_records
from latinitas_cards.claim_sidecar import sidecar_path
from latinitas_cards.cli import app
from latinitas_cards.corpus_claims import corpus_claim_fingerprint, join_corpus_decisions, record_corpus_batch
from latinitas_cards.corpus_facts import Morphology, accepted_facts, parse_morphology
from latinitas_cards.corpus_model import latin_fold, passage_hash

RUNNER = CliRunner()
STAMP = "2026-01-01T00:00:00Z"
LINES = ("Arma virumque cano.", "Mater filium amat. Puer legit.", "Canis currit", "et latrat.")


class Fixture:
    """Disposable corpus normalized and segmented through the real CLI."""

    def __init__(self, root: Path, lines: tuple[str, ...] = LINES) -> None:
        source = root / "source.txt"
        source.write_text("\n".join(lines) + "\n")
        self.corpus = root / "corpus.jsonl"
        self.sidecar = root / "corpus.annotations.jsonl"
        for args in (
            [
                "corpus",
                "normalize",
                str(source),
                "--adapter",
                "text",
                "--corpus-id",
                "facts",
                "--output",
                str(self.corpus),
            ],
            ["annotations", "segment", str(self.corpus), "--timestamp", STAMP],
        ):
            result = RUNNER.invoke(app, [*args, "--json"])
            assert result.exit_code == 0, result.output
        self.segmentations = {
            row["anchor"]["passage_id"]: row["annotation_id"]
            for row in map(json.loads, self.sidecar.read_text().splitlines()[1:])
        }

    def passage(self, index: int) -> dict[str, Any]:
        rows = [row for row in map(json.loads, self.corpus.read_text().splitlines()) if row["record"] == "passage"]
        passage: dict[str, Any] = rows[index]
        return passage

    def add(
        self,
        index: int,
        layer: str,
        value: JsonValue,
        span: tuple[int, int] | None = None,
        *,
        bound: bool = True,
        name: str = "fixture",
        supersedes: str | None = None,
        segmentation: str | None = None,
    ) -> Annotation:
        passage = self.passage(index)
        anchor: dict[str, Any] = {"passage_id": passage["passage_id"], "passage_hash": passage["passage_hash"]}
        if span is not None:
            anchor["span"] = list(span)
            if bound:
                anchor["segmentation_id"] = segmentation or self.segmentations[passage["passage_id"]]
        raw: dict[str, Any] = {
            "record": "annotation",
            "layer": layer,
            "anchor": anchor,
            "value": value,
            "input_hash": passage["passage_hash"],
            "producer": {"kind": "agent", "name": name, "version": "1"},
            "created": STAMP,
            "plan": None,
        }
        if supersedes is not None:
            raw["supersedes"] = supersedes
        raw["annotation_id"] = annotation_id(raw)
        record = Annotation.model_validate(raw)
        append_records(self.sidecar, [record], [self.corpus])
        return record

    def retract(self, record: Annotation) -> None:
        raw = record.model_dump(mode="json", exclude={"value", "supersedes", "alternatives", "confidence"})
        raw.update(record="retraction", retracts=record.annotation_id)
        raw["annotation_id"] = annotation_id(raw)
        append_records(self.sidecar, [Retraction.model_validate(raw)], [self.corpus])

    def decide(self, status: str, *records: Annotation) -> None:
        items = [
            {"fingerprint": corpus_claim_fingerprint(row), "status": status, "reviewer": "R", "reason": "Checked"}
            for row in records
        ]
        assert record_corpus_batch(self.corpus, self.sidecar, items) == ()

    def edit(self, index: int, text: str) -> None:
        lines = self.corpus.read_text().splitlines()
        rows = [json.loads(line) for line in lines]
        target = [i for i, row in enumerate(rows) if row["record"] == "passage"][index]
        rows[target].update(text=text, norm=latin_fold(text), passage_hash=passage_hash(text))
        lines[target] = json.dumps(rows[target], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.corpus.write_text("\n".join(lines) + "\n")

    def view(self, language: str = "en") -> Any:
        return accepted_facts(self.corpus, self.sidecar, sidecar_path(self.corpus), target_language=language)


def morph(lemma: str, pos: str = "noun", **features: str) -> JsonValue:
    value: dict[str, JsonValue] = {"tagset": "latinitas-morph/1", "lemma": lemma, "pos": pos}
    if features:
        value["features"] = dict(features)
    return value


def text(language: str, value: str) -> JsonValue:
    return {"language": language, "text": value}


@pytest.fixture
def facts(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def test_review_states_and_lifecycle(facts: Fixture) -> None:
    arma = facts.add(0, "morphology", morph("arma", case="accusativus", number="pluralis"), (0, 4))
    virum = facts.add(0, "morphology", morph("vir"), (5, 13))
    cano = facts.add(0, "morphology", morph("cano", "verb"), (14, 18))
    old = facts.add(1, "morphology", morph("mater"), (0, 5))
    filium = facts.add(1, "morphology", morph("filius"), (6, 12))
    latrat = facts.add(3, "morphology", morph("latro", "verb"), (3, 9))
    barks = facts.add(3, "translation", text("en", "and barks."))
    et_old = facts.add(3, "morphology", morph("et", "conjunction"), (0, 2))
    facts.decide("accepted", arma, old, filium, latrat, barks, et_old)
    facts.decide("withheld", cano)
    new = facts.add(1, "morphology", morph("mater", case="nominativus"), (0, 5), supersedes=old.annotation_id)
    et_new = facts.add(
        3, "morphology", morph("et", "conjunction"), (0, 2), name="redo", supersedes=et_old.annotation_id
    )
    facts.retract(filium)
    facts.retract(et_new)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in facts.corpus.parent.iterdir()}
    view = facts.view()
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in facts.corpus.parent.iterdir()}
    first, second = view.passages["facts:p000001"], view.passages["facts:p000002"]
    resolved = first.morphology(0)
    assert (resolved.reason, resolved.value) == (
        None,
        Morphology("arma", "noun", (("case", "accusativus"), ("number", "pluralis"))),
    )
    assert [(fact.annotation_id, fact.review_status) for fact in resolved.facts] == [(arma.annotation_id, "accepted")]
    assert (first.morphology(1).reason, first.morphology(1).value) == ("open_fact", None)
    assert first.morphology(1).facts[0].annotation_id == virum.annotation_id
    assert first.morphology(2).reason == "withheld_fact"
    assert first.morphology(3).reason == "missing_fact"  # the final punctuation token
    # The accepted decision on the superseded record does not carry to its open replacement.
    assert second.morphology(0).reason == "open_fact"
    assert [fact.annotation_id for fact in second.morphology(0).facts] == [new.annotation_id]
    assert second.morphology(1).reason == "missing_fact"  # retracted after acceptance
    assert view.passages["facts:p000004"].morphology(1).reason is None
    assert view.passages["facts:p000004"].passage_translation().value == "and barks."
    assert view.passages["facts:p000004"].morphology(0).reason == "missing_fact"  # superseded, then retracted

    facts.edit(3, "et latrant.")
    edited = facts.view().passages["facts:p000004"]
    assert edited.segmentation is None and edited.sentences == ()
    assert edited.anchor("morphology", (3, 9)).reason == "stale_fact"
    # Withdrawn records stay missing after an edit; only live records go stale.
    assert edited.anchor("morphology", (0, 2)).reason == "missing_fact"
    assert edited.passage_translation().reason == "stale_fact"


def test_spanned_facts_need_current_segmentation_and_exact_spans(facts: Fixture) -> None:
    old_segmentation = facts.segmentations["facts:p000002"]
    bound = facts.add(1, "translation", text("en", "mother"), (0, 5))
    unbound = facts.add(1, "translation", text("en", "son"), (6, 12), bound=False)
    sentence = facts.add(1, "translation", text("en", "The mother loves the son."), (0, 18))
    passage = facts.add(1, "translation", text("en", "The mother loves the son. The boy reads."))
    near = facts.add(1, "translation", text("en", "the mother loves"), (0, 17))
    facts.decide("accepted", bound, unbound, sentence, passage, near)
    view = facts.view().passages["facts:p000002"]
    assert view.segmentation_id == old_segmentation
    assert [(s.index, s.span, s.tokens, s.fragment) for s in view.sentences] == [
        (0, (0, 18), (0, 4), False),
        (1, (19, 30), (4, 7), False),
    ]
    assert view.translation((0, 5)).value == "mother"
    assert view.translation((6, 12)).reason == "stale_fact"
    assert view.sentence_translation(0).value == "The mother loves the son."
    # Two sentences: the span-less translation is the passage's, not sentence 1's.
    assert view.sentence_translation(1).reason == "missing_fact"
    assert view.passage_translation().value == "The mother loves the son. The boy reads."
    assert view.translation((0, 17)).value == "the mother loves"
    assert view.translation((0, 12)).reason == "missing_fact"

    # A replacement segmentation orphans every fact bound to the old one; span-less facts survive.
    raw = json.loads(facts.sidecar.read_text().splitlines()[2])
    replacement = facts.add(1, "segmentation", raw["value"], supersedes=old_segmentation, name="reseg")
    view = facts.view().passages["facts:p000002"]
    assert view.segmentation_id == replacement.annotation_id
    assert view.translation((0, 5)).reason == "stale_fact"
    assert view.sentence_translation(0).reason == "stale_fact"
    assert view.passage_translation().value == "The mother loves the son. The boy reads."
    rebound = facts.add(1, "translation", text("en", "mother"), (0, 5), segmentation=replacement.annotation_id)
    resolved = facts.view().passages["facts:p000002"].translation((0, 5))
    assert (resolved.reason, [fact.annotation_id for fact in resolved.facts]) == ("open_fact", [rebound.annotation_id])


def test_absent_claims_and_sidecar_mean_no_decisions_or_annotations(facts: Fixture) -> None:
    facts.add(0, "morphology", morph("arma"), (0, 4))
    assert not sidecar_path(facts.corpus).exists()
    assert facts.view().passages["facts:p000001"].morphology(0).reason == "open_fact"
    facts.sidecar.unlink()
    view = facts.view()
    assert view.passages["facts:p000001"].segmentation is None
    assert view.passages["facts:p000001"].anchor("morphology", (0, 4)).reason == "missing_fact"


def test_two_current_segmentations_form_no_units(facts: Fixture) -> None:
    arms = facts.add(0, "translation", text("en", "arms"), (0, 4))
    facts.decide("accepted", arms)
    raw = json.loads(facts.sidecar.read_text().splitlines()[1])
    facts.add(0, "segmentation", raw["value"], name="second")
    view = facts.view().passages["facts:p000001"]
    assert (view.segmentation_id, view.sentences) == (None, ())
    assert view.translation((0, 4)).reason == "stale_fact"


def test_writes_between_reads_never_surface_an_accepted_fact(facts: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    arma = facts.add(0, "morphology", morph("arma"), (0, 4))
    whole = facts.add(1, "translation", text("en", "The mother loves the son. The boy reads."))
    facts.decide("accepted", arma, whole)
    raw = json.loads(facts.sidecar.read_text().splitlines()[3])
    segmentation = facts.segmentations["facts:p000003"]
    join = join_corpus_decisions

    def racing(*args: Path) -> tuple[dict[str, str], list[Any]]:
        result = join(*args)
        facts.retract(arma)
        facts.add(2, "segmentation", raw["value"], supersedes=segmentation, name="reseg")
        facts.edit(1, "Mater filiam amat. Puer legit.")
        return result

    monkeypatch.setattr(corpus_facts, "join_corpus_decisions", racing)
    view = facts.view()
    assert view.passages["facts:p000001"].morphology(0).reason == "missing_fact"
    assert view.passages["facts:p000002"].passage_translation().reason == "stale_fact"
    assert view.passages["facts:p000003"].segmentation_id is None


def test_fragments_are_flagged(facts: Fixture) -> None:
    view = facts.view()
    assert [(s.continued, s.continues, s.fragment) for s in view.passages["facts:p000003"].sentences] == [
        (False, True, True)
    ]
    assert [(s.continued, s.continues, s.fragment) for s in view.passages["facts:p000004"].sentences] == [
        (True, False, True)
    ]
    assert not view.passages["facts:p000001"].sentences[0].fragment


def test_language_filter_runs_before_conflicts(facts: Fixture) -> None:
    rows = [
        facts.add(0, "translation", text("en", "arms"), (0, 4)),
        facts.add(0, "translation", text("de", "Waffen"), (0, 4)),
        facts.add(0, "translation", text("en-GB", "arms"), (0, 4), name="other"),
        facts.add(0, "translation", text("de", "Mann"), (5, 13)),
        facts.add(0, "translation", text("en", "I sing"), (14, 18)),
        facts.add(0, "translation", text("en", "I chant"), (14, 18), name="other"),
    ]
    facts.decide("accepted", *rows)
    english, german = facts.view("en").passages["facts:p000001"], facts.view("de-AT").passages["facts:p000001"]
    assert (english.translation((0, 4)).reason, english.translation((0, 4)).value) == (None, "arms")
    assert [fact.reason for fact in english.translation((0, 4)).facts] == [None, "wrong_language", None]
    assert (german.translation((0, 4)).reason, german.translation((0, 4)).value) == (None, "Waffen")
    assert english.translation((5, 13)).reason == "wrong_language"
    assert german.translation((5, 13)).value == "Mann"
    assert english.translation((14, 18)).reason == "conflicting_facts"
    assert english.translation((14, 18)).value is None
    assert german.translation((14, 18)).reason == "wrong_language"


def test_one_sentence_spanless_and_sentence_translations_share_a_requirement(facts: Fixture) -> None:
    sentence = facts.add(0, "translation", text("en", "Arms and the man I sing."), (0, 19))
    spanless = facts.add(0, "translation", text("en", "I sing of arms and the man."))
    same = facts.add(1, "translation", text("en", "Same."), (0, 18))
    other = facts.add(1, "translation", text("en", "Different."))
    facts.decide("accepted", sentence, spanless, same, other)
    view = facts.view()
    one = view.passages["facts:p000001"].sentence_translation(0)
    assert one.reason == "conflicting_facts"
    assert {fact.annotation_id for fact in one.facts} == {sentence.annotation_id, spanless.annotation_id}
    # A two-sentence passage keeps them apart.
    assert view.passages["facts:p000002"].sentence_translation(0).value == "Same."
    assert view.passages["facts:p000002"].passage_translation().value == "Different."


def test_identical_sentence_and_spanless_translations_count_once(facts: Fixture) -> None:
    rows = (
        facts.add(3, "translation", text("en", "and barks."), (0, 10)),
        facts.add(3, "translation", text("en", "and barks.")),
    )
    facts.decide("accepted", *rows)
    resolved = facts.view().passages["facts:p000004"].sentence_translation(0)
    assert (resolved.reason, resolved.value, len(resolved.facts)) == (None, "and barks.", 2)


def test_morphology_conflicts_unsupported_and_features(facts: Fixture) -> None:
    rows = [
        facts.add(0, "morphology", morph("arma"), (0, 4)),
        facts.add(0, "morphology", morph("armum"), (0, 4), name="other"),
        facts.add(0, "morphology", {"tagset": "ud/2", "lemma": "vir", "pos": "noun"}, (5, 13)),
        facts.add(0, "morphology", morph("cano", "verb", mood="infinitivus"), (14, 18)),
        facts.add(1, "morphology", morph("mater", case="nominativus"), (0, 5)),
        facts.add(1, "morphology", morph("mater", case="nominativus"), (0, 5), name="twin"),
        facts.add(1, "morphology", morph("filius"), (6, 12)),
        facts.add(1, "morphology", morph("mater filium"), (0, 12)),
        facts.add(1, "morphology", morph("punctum"), (17, 18)),
        facts.add(1, "morphology", morph("amo", "verb", mood="indicativus", person="tertia"), (13, 17)),
    ]
    facts.decide("accepted", *rows)
    facts.add(0, "morphology", morph("arma"), (0, 4), name="open")
    first, second = facts.view().passages["facts:p000001"], facts.view().passages["facts:p000002"]
    assert first.morphology(0).reason == "conflicting_facts"
    assert first.morphology(1).reason == "unsupported_morphology"
    assert first.morphology(2).reason == "unsupported_morphology"
    assert second.morphology(0).value == Morphology("mater", "noun", (("case", "nominativus"),))
    assert len(second.morphology(0).facts) == 2
    assert second.analysis(0).value == second.morphology(0).value
    assert (second.morphology(1).reason, second.analysis(1).reason) == (None, "no_features")
    assert second.morphology(2).value == Morphology("amo", "verb", (("mood", "indicativus"), ("person", "tertia")))
    assert second.analysis(2).reason is None
    # Anchors that are not exactly one word token are unsupported wherever they appear.
    assert second.anchor("morphology", (0, 12)).reason == "unsupported_morphology"
    assert second.anchor("morphology", (17, 18)).reason == "unsupported_morphology"
    assert second.analysis(1).facts == second.morphology(1).facts


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (morph("cano", "verb", mood="coniunctivus", tense="praesens"), True),
        (morph("cano", "verb", mood="imperativus"), True),
        (morph("cano", "verb", verbform="infinitivus"), True),
        (morph("cano", "verb", verbform="gerundivum", degree="positivus"), True),
        (morph("melior", "adjective", degree="comparativus", case="nominativus"), True),
        (morph("tempus", tense="futurum exactum"), True),
        (morph("cano", "verb", mood="infinitivus"), False),
        (morph("cano", "verb", mood="participium"), False),
        (morph("cano", "verb", verbform="indicativus"), False),
        (morph("bonus", "adjective", degree="elativus"), False),
        (morph("cano", "verb", aspect="perfectivus"), False),
        (morph("cano", "auxiliary"), False),
        (morph(""), False),
        (morph("e\u0301"), False),
        (morph("\u00e9"), True),
        ({"tagset": "latinitas-morph/2", "lemma": "cano", "pos": "verb"}, False),
        ({"tagset": "latinitas-morph/1", "pos": "verb"}, False),
        ({"tagset": "latinitas-morph/1", "lemma": "cano", "pos": "verb", "gloss": "sing"}, False),
        ({"tagset": "latinitas-morph/1", "lemma": "cano", "pos": "verb", "features": {"case": 1}}, False),
        ({"tagset": "latinitas-morph/1", "lemma": "cano", "pos": "verb", "features": []}, False),
        ({"tagset": "latinitas-morph/1", "lemma": ["cano"], "pos": "verb"}, False),
    ],
)
def test_parse_morphology(value: JsonValue, expected: bool) -> None:
    assert (parse_morphology(value) is not None) is expected


def test_morphology_feature_order_is_canonical() -> None:
    left = parse_morphology(morph("cano", "verb", tense="praesens", mood="indicativus"))
    right = parse_morphology(morph("cano", "verb", mood="indicativus", tense="praesens"))
    assert left == right == Morphology("cano", "verb", (("mood", "indicativus"), ("tense", "praesens")))
    assert parse_morphology(morph("\u00e9")) == Morphology("\u00e9", "noun", ())


def test_view_imports_no_analyzer_or_llm_client() -> None:
    code = (
        "import sys, latinitas_cards.corpus_facts; "
        "sys.exit(bool({'cltk', 'stanza', 'torch', 'openai', 'anthropic', 'ollama'} & sys.modules.keys()))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False, timeout=30).returncode == 0


def test_view_validates_its_own_corpus_read(facts: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    arma = facts.add(0, "morphology", morph("arma"), (0, 4))
    facts.decide("accepted", arma)
    join = join_corpus_decisions

    def corrupting(*args: Path) -> tuple[dict[str, str], list[Any]]:
        result = join(*args)
        facts.corpus.write_text(facts.corpus.read_text().replace('"text":"Arma virumque', '"text":"Bella virumque'))
        return result

    monkeypatch.setattr(corpus_facts, "join_corpus_decisions", corrupting)
    with pytest.raises(ValueError, match="hash_mismatch"):
        facts.view()
