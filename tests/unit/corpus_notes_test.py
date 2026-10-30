"""The corpus-v1 note type rendered from real normalized, segmented and reviewed corpora."""

import csv
import io
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from corpus_exercises_test import candidates, profile
from corpus_facts_test import Fixture, morph, text

from latinitas_cards.corpus_exercises import Candidate
from latinitas_cards.corpus_notes import (
    CORPUS_CARD_CSS,
    CORPUS_STYLE_VERSION,
    CORPUS_TEMPLATES,
    corpus_csv,
    render_corpus_note,
)
from latinitas_cards.corpus_profile import CorpusProfile
from latinitas_cards.destination_state import validate_managed_controls
from latinitas_cards.managed_layouts import known_layouts, layout_version
from latinitas_cards.reference_templates import REFERENCE_CARD_CSS

LINES = ("Marcus amicum videt. Amicus ridet.", "Canis currit", "et latrat.")
FIELDS = (
    "LatinitasID",
    "Latin",
    "Citation",
    "Passage",
    "Language",
    "Source ID",
    "Source Scope",
    "Source Kind",
    "Source Location",
    "Note Schema",
    "Generator",
    "Profile",
    "CorpusLexicalEnabled",
    "CorpusLexicalPrompt",
    "CorpusLexicalAnswer",
    "CorpusAnalysisEnabled",
    "CorpusAnalysisPrompt",
    "CorpusAnalysisAnswer",
    "CorpusPhraseEnabled",
    "CorpusPhrasePrompt",
    "CorpusPhraseAnswer",
    "CorpusSentenceEnabled",
    "CorpusSentencePrompt",
    "CorpusSentenceAnswer",
    "Personal Notes",
)
SLOTS = {
    "corpus_lexical_recognition": "Lexical",
    "corpus_guided_analysis": "Analysis",
    "corpus_phrase_translation": "Phrase",
    "corpus_sentence_translation": "Sentence",
}
# Expected strings are written out here, never read from the string table.
STRINGS = {
    "en": {
        "lexical": "What does the highlighted word mean here?",
        "analysis": "Analyse the highlighted word, then translate the sentence.",
        "phrase": "Translate the highlighted phrase in context.",
        "sentence": "Translate the sentence.",
        "passage": "Translate the passage.",
        "guidance": "Reference translation: other translations can be correct as well.",
        "self": "Self-assessment: compare the meaning, not the exact wording.",
        "features": "Lemma: amicus<br>Case: Accusative<br>Number: Singular",
        "gloss": "friend",
        "sentence_text": "Marcus sees his friend.",
        "phrase_text": "Marcus … the friend",
        "passage_text": "Marcus sees his friend. The friend laughs.",
    },
    "de": {
        "lexical": "Was bedeutet das markierte Wort hier?",
        "analysis": "Bestimme das markierte Wort und übersetze dann den Satz.",
        "phrase": "Übersetze die markierte Wendung im Zusammenhang.",
        "sentence": "Übersetze den Satz.",
        "passage": "Übersetze den Abschnitt.",
        "guidance": "Referenzübersetzung: Andere Übersetzungen können ebenso richtig sein.",
        "self": "Selbsteinschätzung: Vergleiche den Sinn, nicht den genauen Wortlaut.",
        "features": "Lemma: amicus<br>Kasus: Akkusativ<br>Numerus: Singular",
        "gloss": "Freund",
        "sentence_text": "Marcus sieht seinen Freund.",
        "phrase_text": "Marcus … den Freund",
        "passage_text": "Marcus sieht seinen Freund. Der Freund lacht.",
    },
}


@pytest.fixture
def reviewed(tmp_path: Path) -> Fixture:
    facts = Fixture(tmp_path, LINES)
    rows = [facts.add(0, "morphology", morph("amicus", case="accusativus", number="singularis"), (7, 13))]
    for language, words in STRINGS.items():
        rows += [
            facts.add(0, "translation", text(language, words["gloss"]), (7, 13)),
            facts.add(0, "translation", text(language, words["sentence_text"]), (0, 20)),
            facts.add(0, "translation", text(language, words["phrase_text"]), (0, 13)),
            facts.add(0, "translation", text(language, words["passage_text"])),
            facts.add(1, "translation", text(language, "x")),
        ]
    facts.decide("accepted", *rows)
    facts.add(0, "morphology", morph("Marcus"), (0, 6))  # left open: its exercises are ineligible
    return facts


def rendered(facts: Fixture, chosen: CorpusProfile) -> list[tuple[Candidate, dict[str, str]]]:
    view = facts.view(chosen.target_language)
    return [
        (candidate, render_corpus_note(candidate, view.passages[candidate.passage_id], chosen))
        for candidate in candidates(facts, chosen)
        if candidate.annotation_id is None
    ]


def panel(*parts: str, appearance: str = " corpus-light") -> str:
    return f'<div class="corpus-v1{appearance}">' + "".join(parts) + "</div>"


@pytest.mark.parametrize("language", ["en", "de", "en-GB"])
def test_each_recipe_renders_one_enabled_latin_first_slot(reviewed: Fixture, language: str) -> None:
    # A primary subtag of en selects English; the Language field keeps the full tag.
    words = STRINGS[language.split("-")[0]]
    chosen = profile(language, morphology={"terminology": {"version": 1, "language": "user"}})
    notes = rendered(reviewed, chosen)
    eligible = [(c, f) for c, f in notes if not c.reasons and c.filtered is None]
    lexical = '<div class="corpus-gloss">' + words["gloss"] + '</div><div class="corpus-analysis">'
    reference = '<div class="corpus-guidance">' + words["guidance"] + '</div><div class="corpus-translation">'
    self_assessment = '<div class="corpus-self-assessment">' + words["self"] + "</div>"
    target = '<span class="corpus-target">amicum</span>'
    expected = [
        (
            "corpus_lexical_recognition",
            "Marcus " + target + " videt.",
            words["lexical"],
            panel(lexical + words["features"] + "</div>"),
        ),
        (
            "corpus_guided_analysis",
            "Marcus " + target + " videt.",
            words["analysis"],
            panel(
                '<div class="corpus-analysis">' + words["features"] + "</div>",
                reference + words["sentence_text"] + "</div>",
                self_assessment,
            ),
        ),
        (
            "corpus_phrase_translation",
            '<span class="corpus-target">Marcus amicum</span> videt.',
            words["phrase"],
            panel(reference + words["phrase_text"] + "</div>", self_assessment),
        ),
        (
            "corpus_sentence_translation",
            "Marcus amicum videt.",
            words["sentence"],
            panel(reference + words["sentence_text"] + "</div>", self_assessment),
        ),
        (
            "corpus_sentence_translation",
            "Marcus amicum videt. Amicus ridet.",
            words["passage"],
            panel(reference + words["passage_text"] + "</div>", self_assessment),
        ),
    ]
    actual = []
    for candidate, fields in eligible:
        assert tuple(fields) == FIELDS[:-1]
        assert {name: fields[name] for name in FIELDS[2:12]} == {
            "Citation": "p000001 (facts)",
            "Passage": "Marcus amicum videt. Amicus ridet.",
            "Language": language,
            "Source ID": "facts:p000001",
            "Source Scope": "corpus-passage-v1",
            "Source Kind": "corpus",
            "Source Location": candidate.object_key,
            "Note Schema": "corpus-v1",
            "Generator": "corpus-v1",
            "Profile": "reader-a",
        }
        assert fields["LatinitasID"] == candidate.latinitas_id
        label = SLOTS[candidate.recipe]
        enabled = [name for name in FIELDS if name.endswith("Enabled") and fields[name]]
        assert enabled == [f"Corpus{label}Enabled"] and fields[enabled[0]] == "1"
        for other in set(SLOTS.values()) - {label}:
            assert fields[f"Corpus{other}Prompt"] == fields[f"Corpus{other}Answer"] == ""
        actual.append(
            (candidate.recipe, fields["Latin"], fields[f"Corpus{label}Prompt"], fields[f"Corpus{label}Answer"])
        )
        validate_managed_controls(fields)
    assert actual == [
        (recipe, latin, panel(f'<div class="corpus-prompt">{prompt}</div>'), answer)
        for recipe, latin, prompt, answer in expected
    ]
    # Latin terminology, the profile default, keeps Latin labels in either target language.
    latin = next(
        f
        for c, f in rendered(reviewed, profile(language))
        if c.recipe == "corpus_lexical_recognition" and not c.reasons
    )
    assert "Lemma: amicus<br>Casus: accusativus<br>Numerus: singularis" in latin["CorpusLexicalAnswer"]


def test_ineligible_and_fragment_exercises_enable_no_slot(reviewed: Fixture) -> None:
    notes = rendered(reviewed, profile())
    withheld = [(c, f) for c, f in notes if c.reasons or c.filtered]
    # The open "Marcus" morphology and the fragment sentence of passage 2.
    assert sorted((c.recipe, c.passage_id, c.reasons, c.filtered) for c, _ in withheld) == [
        ("corpus_guided_analysis", "facts:p000001", ("open_fact",), None),
        ("corpus_lexical_recognition", "facts:p000001", ("open_fact", "missing_fact"), None),
        ("corpus_sentence_translation", "facts:p000002", (), "filtered:fragment"),
    ]
    for candidate, fields in withheld:
        parts = ("Enabled", "Prompt", "Answer")
        assert all(fields[f"Corpus{label}{part}"] == "" for label in SLOTS.values() for part in parts)
        assert fields["LatinitasID"] == candidate.latinitas_id
    lexical = next(f for c, f in withheld if c.recipe == "corpus_lexical_recognition")
    assert lexical["Latin"] == '<span class="corpus-target">Marcus</span> amicum videt.'
    malformed = reviewed.add(0, "translation", text("en", "Marcus "), (0, 7))
    invalid = next(c for c in candidates(reviewed) if c.annotation_id == malformed.annotation_id)
    with pytest.raises(ValueError, match="identity"):
        render_corpus_note(invalid, reviewed.view().passages[invalid.passage_id], profile())


@pytest.mark.parametrize(
    ("appearance", "classes"), [("light", " corpus-light"), ("dark", " corpus-dark"), ("auto", "")]
)
def test_panels_carry_the_configured_appearance(reviewed: Fixture, appearance: str, classes: str) -> None:
    chosen = profile(morphology={"appearance": appearance})
    for candidate, fields in rendered(reviewed, chosen):
        if not candidate.reasons and candidate.filtered is None:
            label = SLOTS[candidate.recipe]
            for part in ("Prompt", "Answer"):
                assert fields[f"Corpus{label}{part}"].startswith(f'<div class="corpus-v1{classes}">')


def test_hostile_text_and_controls_are_escaped(tmp_path: Path) -> None:
    lines = ("<i>I</i>&\tPuer <b>&amp; ridet \x07 bene.",)
    facts = Fixture(tmp_path, lines, "--citation-template", "caes:book")
    rows = [
        facts.add(0, "morphology", morph("<i>puer</i>", case="nominativus"), (0, 4)),
        facts.add(0, "translation", text("en", 'boy & <b onclick="x">girl</b>\x02'), (0, 4)),
        facts.add(0, "translation", text("en", "<script>alert(1)</script>\x01")),
    ]
    facts.decide("accepted", *rows)
    notes = {c.recipe: f for c, f in rendered(facts, profile()) if not c.reasons}
    assert set(notes) == {"corpus_lexical_recognition", "corpus_guided_analysis", "corpus_sentence_translation"}
    latin = '<span class="corpus-target">Puer</span> &lt;b&gt;&amp;amp; ridet \\x07 bene.'
    for fields in notes.values():
        assert fields["Passage"] == "Puer &lt;b&gt;&amp;amp; ridet \\x07 bene."
        assert fields["Citation"] == "&lt;i&gt;I&lt;/i&gt;&amp; (facts)"
        validate_managed_controls(fields)
        markup = re.sub(r'<(/?)(div|span)( class="[a-z0-9 -]+")?>|<br>', "", "".join(fields.values()))
        assert "<" not in markup and ">" not in markup
    assert notes["corpus_lexical_recognition"]["Latin"] == latin
    assert (
        "boy &amp; &lt;b onclick=&quot;x&quot;&gt;girl&lt;/b&gt;\\x02"
        in (notes["corpus_lexical_recognition"]["CorpusLexicalAnswer"])
    )
    assert "Lemma: &lt;i&gt;puer&lt;/i&gt;" in notes["corpus_guided_analysis"]["CorpusAnalysisAnswer"]
    assert "&lt;script&gt;alert(1)&lt;/script&gt;\\x01" in notes["corpus_sentence_translation"]["CorpusSentenceAnswer"]


def test_csv_follows_the_field_order_without_personal_notes(reviewed: Fixture) -> None:
    chosen = profile("de", note_type="Latinitas Corpus Context v1", deck="Latin::Corpus")
    notes = [f for c, f in rendered(reviewed, chosen) if not c.reasons and c.filtered is None]
    payload = corpus_csv(chosen, notes).decode("utf-8")
    columns = [*FIELDS[:-1], "Tags"]
    header = payload.splitlines()[:6]
    assert header == [
        "#separator:Comma",
        "#html:true",
        "#notetype:Latinitas Corpus Context v1",
        "#deck:Latin::Corpus",
        f"#tags column:{len(columns)}",
        "#columns:" + ",".join(columns),
    ]
    rows = list(csv.reader(io.StringIO("\n".join(payload.splitlines()[6:]))))
    assert rows == [[*note.values(), ""] for note in notes]
    assert "Personal Notes" not in payload


def test_corpus_layout_is_registered_with_its_own_content_fields() -> None:
    layout = layout_version("corpus-v1")
    assert layout.fields == FIELDS
    assert layout.columns == (*FIELDS[:-1], "Tags")
    assert [(s.ordinal, s.template_name, s.semantic_key) for s in layout.slots] == [
        (ordinal, f"Corpus {label}", f"{recipe}:context") for ordinal, (recipe, label) in enumerate(SLOTS.items())
    ]
    assert layout.css == CORPUS_CARD_CSS and CORPUS_STYLE_VERSION == 1
    assert layout.content_fields == {"Latin", "Citation", "Passage"}
    # Existing layouts keep exactly the historical content set and the reference styling.
    historical = frozenset({"Lemma", "Principal Parts", "Meaning", "Generator", "Profile"})
    assert {item.version: (item.content_fields, item.css) for item in known_layouts()} == {
        "3": (historical, REFERENCE_CARD_CSS),
        "phrase-v1": (historical, REFERENCE_CARD_CSS),
        "parsing-v1": (historical, REFERENCE_CARD_CSS),
        "corpus-v1": (frozenset({"Latin", "Citation", "Passage"}), CORPUS_CARD_CSS),
    }


def test_stylesheet_keeps_the_appearance_selectors() -> None:
    # The browser test checks rendering where agent-browser exists; these selectors carry the contract.
    for selector in (
        ".card.nightMode, .card.night_mode {",
        ".corpus-v1.corpus-dark,\n.nightMode .corpus-v1:where(:not(.corpus-light):not(.corpus-dark)),\n"
        ".night_mode .corpus-v1:where(:not(.corpus-light):not(.corpus-dark)) {",
    ):
        assert selector in CORPUS_CARD_CSS


def test_templates_are_guarded_and_latin_first() -> None:
    for template in CORPUS_TEMPLATES:
        slot = template.slot
        guard_open, guard_close = "{{#" + slot.enabled_field + "}}", "{{/" + slot.enabled_field + "}}"
        for markup in (template.front, template.back):
            assert markup.startswith(guard_open) and markup.endswith(guard_close)
        assert template.front.index("{{Latin}}") < template.front.index("{{" + slot.prompt_field + "}}")
        back = template.back
        assert back.index("{{FrontSide}}") < back.index("{{" + slot.answer_field + "}}") < back.index("{{Passage}}")
        assert "{{Citation}}" in back and "{{Personal Notes}}" in back


def card_page(template: Any, fields: dict[str, str], back: bool) -> str:
    front = template.front
    markup = template.back.replace("{{FrontSide}}", front) if back else front
    markup = re.sub(r"\{\{[#/][^}]+\}\}", "", markup)
    for name, value in {**fields, "Personal Notes": "Note: compare videt."}.items():
        markup = markup.replace("{{" + name + "}}", value)
    return (
        '<!doctype html><meta charset="utf-8"><style>'
        + CORPUS_CARD_CSS
        + '</style><body class="card">'
        + markup
        + "</body>"
    )


@pytest.mark.skipif(shutil.which("agent-browser") is None, reason="Chromium/agent-browser is optional")
def test_rendered_cards_follow_the_reference_appearance_contract(reviewed: Fixture, tmp_path: Path) -> None:
    def luminance(rgb: list[int]) -> float:
        channels = [value / 255 for value in rgb]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
        return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    def browser(*arguments: str) -> str:
        return subprocess.check_output(["agent-browser", "--session", "corpus-appearance", *arguments], text=True)

    templates = {template.slot.recipe: template for template in CORPUS_TEMPLATES}
    try:
        for appearance in ("light", "dark", "auto"):
            chosen = profile(morphology={"appearance": appearance})
            note = next(
                f for c, f in rendered(reviewed, chosen) if c.recipe == "corpus_guided_analysis" and not c.reasons
            )
            page = tmp_path / f"{appearance}.html"
            page.write_text(card_page(templates["corpus_guided_analysis"], note, back=True))
            browser("open", page.as_uri())
            for night in ("", "nightMode", "", "night_mode", ""):
                actual = json.loads(
                    browser(
                        "eval",
                        r"""(() => {
                            document.body.className = CLASS;
                            const rgb = value => value.match(/\d+/g).slice(0, 3).map(Number);
                            const back = rgb(getComputedStyle(document.body).backgroundColor);
                            return [...document.querySelectorAll(
                                '.corpus-v1,.corpus-latin,.corpus-target,.corpus-passage,.corpus-citation'
                            )].map(e => {
                                const style = getComputedStyle(e);
                                const own = rgb(style.backgroundColor);
                                const alpha = style.backgroundColor.startsWith('rgba') ? 0 : 1;
                                return [e.className, rgb(style.color), alpha ? own : back];
                            });
                        })()""".replace("CLASS", json.dumps(f"card {night}".strip())),
                    )
                )
                dark_panel = appearance == "dark" or (appearance == "auto" and bool(night))
                for name, foreground, background in actual:
                    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
                    assert (light + 0.05) / (dark + 0.05) >= 4.5, (appearance, night, name)
                    if name.startswith("corpus-v1"):
                        assert (luminance(background) < 0.2) is dark_panel, (appearance, night, name)
                    elif name != "corpus-target":
                        assert (luminance(background) < 0.2) is bool(night), (appearance, night, name)
    finally:
        browser("close")
