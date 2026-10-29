"""The ``corpus-v1`` note type: one note and one enabled card per corpus exercise.

Its fields, guarded templates and stylesheet are a separate contract from the
reference note type, whose fields, templates and styling stay unchanged. Every
asserted value comes from the accepted-fact view; an ineligible exercise renders
no enabled card. CSV carries the fields, never the stylesheet or templates.
"""

import html
from collections.abc import Iterable, Mapping

from .cards import CardSlot
from .corpus_exercises import RECIPE_ORDER, SOURCE_SCOPE, Candidate
from .corpus_facts import Morphology, PassageFacts
from .corpus_profile import CorpusProfile
from .preview_export import serialize_anki_csv
from .profile_setup import encode_unsafe_controls
from .reference_templates import ReferenceCardTemplate
from .terminology import FEATURE_NAMES, card_text, feature_name, feature_value

CORPUS_SCHEMA_VERSION = "corpus-v1"
CORPUS_STYLE_VERSION = 1
CORPUS_SLOTS = tuple(
    CardSlot(
        f"{recipe}:context",
        recipe,
        "context",
        f"Corpus {label}",
        ordinal,
        f"Corpus{label}Enabled",
        f"Corpus{label}Prompt",
        f"Corpus{label}Answer",
    )
    for ordinal, (recipe, label) in enumerate(
        zip(RECIPE_ORDER, ("Lexical", "Analysis", "Phrase", "Sentence"), strict=True)
    )
)
CORPUS_FIELDS = (
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
    *(name for slot in CORPUS_SLOTS for name in (slot.enabled_field, slot.prompt_field, slot.answer_field)),
    "Personal Notes",
)
# Personal Notes is user-owned and never exported; Tags is the CSV transport column.
CORPUS_COLUMNS = (*CORPUS_FIELDS[:-1], "Tags")
# The only fields a managed update may change besides retained card payloads.
CORPUS_CONTENT_FIELDS = frozenset(("Latin", "Citation", "Passage"))


def _template(slot: CardSlot) -> ReferenceCardTemplate:
    guard_open, guard_close = f"{{{{#{slot.enabled_field}}}}}", f"{{{{/{slot.enabled_field}}}}}"
    return ReferenceCardTemplate(
        slot=slot,
        front=f'{guard_open}\n<div class="corpus-latin" lang="la">{{{{Latin}}}}</div>\n'
        f"{{{{{slot.prompt_field}}}}}\n{guard_close}",
        back=f"{guard_open}\n{{{{FrontSide}}}}\n<hr id=answer>\n{{{{{slot.answer_field}}}}}\n"
        '<div class="corpus-passage" lang="la">{{Passage}}</div>\n'
        '<div class="corpus-citation">{{Citation}}</div>\n'
        '{{#Personal Notes}}<div class="corpus-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}\n'
        f"{guard_close}",
    )


CORPUS_TEMPLATES = tuple(_template(slot) for slot in CORPUS_SLOTS)

CORPUS_CARD_CSS = """\
/* Latinitas corpus styling v1 for note type corpus-v1. Manual setup only. */
.card {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 22px;
  line-height: 1.5;
  text-align: center;
  color: #1a1a1a;
  background-color: #ffffff;
}
.card.nightMode, .card.night_mode {
  color: #eeeeee;
  background-color: #2c2c2c;
}

/* Note-level text follows the client; the panels follow the fixed appearance. */
.corpus-latin {
  margin: 0 auto 0.6em;
  max-width: 36em;
  font-size: 26px;
  overflow-wrap: anywhere;
}
.corpus-target {
  font-weight: bold;
  color: #8a3b12;
  text-decoration: underline;
  text-underline-offset: 0.2em;
}
.corpus-passage, .corpus-citation, .corpus-personal-notes {
  margin: 0.8em auto 0;
  max-width: 42em;
  font-size: 18px;
  color: #444444;
  text-align: left;
  overflow-wrap: anywhere;
}
.corpus-passage {
  padding-top: 0.6em;
  border-top: 1px solid #cccccc;
}
.corpus-citation {
  margin-top: 0.3em;
  font-size: 16px;
  font-style: italic;
}
.corpus-personal-notes {
  padding-top: 0.5em;
  border-top: 1px solid #cccccc;
  font-size: 16px;
  font-style: italic;
}
.nightMode .corpus-target, .night_mode .corpus-target {
  color: #f2b38a;
}
.nightMode .corpus-passage, .nightMode .corpus-citation, .nightMode .corpus-personal-notes,
.night_mode .corpus-passage, .night_mode .corpus-citation, .night_mode .corpus-personal-notes {
  color: #dddddd;
  border-color: #666666;
}

.corpus-v1 {
  margin: 0.8em auto 0;
  padding: 0.8em 1em;
  max-width: 42em;
  font-size: 18px;
  text-align: left;
  color: #222222;
  background: #f7f7f4;
  border: 1px solid #bdbdb7;
  border-radius: 0.3em;
  overflow-wrap: anywhere;
}
/* Fixed light and dark ignore the client; auto (no class) follows the night-mode classes. */
.corpus-v1.corpus-dark,
.nightMode .corpus-v1:where(:not(.corpus-light):not(.corpus-dark)),
.night_mode .corpus-v1:where(:not(.corpus-light):not(.corpus-dark)) {
  color: #eeeeee;
  background: #222526;
  border-color: #666666;
}
.corpus-prompt { font-style: italic; }
.corpus-gloss, .corpus-translation { font-size: 22px; font-weight: bold; }
.corpus-analysis { margin: 0.3em 0; }
.corpus-guidance, .corpus-self-assessment { margin-top: 0.4em; font-size: 15px; font-style: italic; }
"""


def render_corpus_note(candidate: Candidate, facts: PassageFacts, profile: CorpusProfile) -> dict[str, str]:
    """Managed ``corpus-v1`` fields in order; only an eligible exercise enables its slot."""
    # Deferred: destination_state imports managed_layouts, which imports this module.
    from .destination_state import validate_managed_controls

    if candidate.unit is None or candidate.latinitas_id is None or candidate.object_key is None:
        raise ValueError("an invalid_span entry has no exercise identity to render")
    passage = facts.passage
    if candidate.passage_id != passage.passage_id:
        raise ValueError("candidate and passage facts differ")
    text = passage.text
    if candidate.unit["kind"] == "passage":
        start, end = 0, len(text)
    else:
        start, end = facts.sentences[int(candidate.unit["index"])].span
    if candidate.span is None:
        latin = _escape(text[start:end])
    else:
        first, last = candidate.span
        latin = (
            _escape(text[start:first])
            + f'<span class="corpus-target">{_escape(text[first:last])}</span>'
            + _escape(text[last:end])
        )
    fields = {
        "LatinitasID": candidate.latinitas_id,
        "Latin": latin,
        "Citation": _escape(f"{passage.citation.display} ({passage.passage_id.partition(':')[0]})"),
        "Passage": _escape(text),
        "Language": profile.target_language,
        "Source ID": passage.passage_id,
        "Source Scope": SOURCE_SCOPE,
        "Source Kind": "corpus",
        "Source Location": candidate.object_key,
        "Note Schema": CORPUS_SCHEMA_VERSION,
        "Generator": CORPUS_SCHEMA_VERSION,
        "Profile": profile.profile_id,
    }
    eligible = not candidate.reasons and candidate.filtered is None
    for slot in CORPUS_SLOTS:
        enabled = eligible and slot.recipe == candidate.recipe
        prompt, answer = _card(candidate, profile) if enabled else ("", "")
        fields |= {slot.enabled_field: "1" if enabled else "", slot.prompt_field: prompt, slot.answer_field: answer}
    validate_managed_controls(fields)
    return fields


def corpus_csv(profile: CorpusProfile, notes: Iterable[Mapping[str, str]]) -> bytes:
    """Initial-creation CSV in spec field order; tags travel in the transport column."""
    return serialize_anki_csv(
        profile.note_type,
        profile.deck,
        CORPUS_COLUMNS,
        (tuple("" if name == "Tags" else note[name] for name in CORPUS_COLUMNS) for note in notes),
    )


def _card(candidate: Candidate, profile: CorpusProfile) -> tuple[str, str]:
    """Localized prompt and answer panels from the candidate's accepted values."""
    language = profile.target_language
    values = [resolution.value for resolution in candidate.resolutions]
    recipe = candidate.recipe
    if recipe == "corpus_lexical_recognition":
        morphology, gloss = values
        key = "corpus_lexical_prompt"
        parts = [_div("corpus-gloss", str(gloss)), _analysis(morphology, profile)]
    else:
        # Every card carrying a reference translation labels it and asks to compare meaning.
        if recipe == "corpus_guided_analysis":
            morphology, translation = values
            key = "corpus_analysis_prompt"
            parts = [_analysis(morphology, profile)]
        else:
            (translation,) = values
            parts = []
            if recipe == "corpus_phrase_translation":
                key = "corpus_phrase_prompt"
            elif candidate.unit is not None and candidate.unit["kind"] == "passage":
                key = "corpus_passage_prompt"
            else:
                key = "corpus_sentence_prompt"
        parts += [
            *_reference(str(translation), language),
            _div("corpus-self-assessment", card_text("self_assessment", language)),
        ]
    appearance = profile.morphology.appearance
    panel = "corpus-v1" + ("" if appearance == "auto" else f" corpus-{appearance}")
    return (
        f'<div class="{panel}">{_div("corpus-prompt", card_text(key, language))}</div>',
        f'<div class="{panel}">{"".join(parts)}</div>',
    )


def _analysis(value: object, profile: CorpusProfile) -> str:
    assert isinstance(value, Morphology)
    settings, language = profile.morphology.terminology, profile.target_language
    features = dict(value.features)
    pairs = [(feature_name("lemma", settings, language), value.lemma)] + [
        (feature_name(name, settings, language), feature_value(name, features[name], settings, language))
        for name in FEATURE_NAMES
        if name in features
    ]
    return f'<div class="corpus-analysis">{"<br>".join(f"{_escape(k)}: {_escape(v)}" for k, v in pairs)}</div>'


def _reference(translation: str, language: str) -> list[str]:
    return [_div("corpus-guidance", card_text("reference_guidance", language)), _div("corpus-translation", translation)]


def _div(name: str, value: str) -> str:
    return f'<div class="{name}">{_escape(value)}</div>'


def _escape(value: str) -> str:
    return html.escape(encode_unsafe_controls(value), quote=True).replace("\n", "<br>")
