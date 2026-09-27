"""Reference note-type publication checks.

External expectations are written as literal data — the note-type field order,
the ten template fronts, the docs-embedded copy, and the synthetic
missing-form eligibility sets — instead of being derived from the production
helpers, so schema, registry, or guidance drift cannot silently re-authorize
itself.
"""

import re
from pathlib import Path

from latinitas_cards.cards import TEMPLATE_REGISTRY, TEMPLATE_REGISTRY_DIGEST
from latinitas_cards.generation import generate_learning_object_notes
from latinitas_cards.notes import AUTHORITATIVE_NOTE_FIELDS, CSV_EXPORT_FIELD_NAMES
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.reference_templates import (
    REFERENCE_CARD_CSS,
    REFERENCE_CARD_TEMPLATES,
    REFERENCE_NOTE_TYPE_FIELDS,
    reference_setup_markdown,
)
from latinitas_cards.sources import CanonicalSourceRecord, SourceProvenance

DOCS_PATH = Path(__file__).parents[2] / "docs" / "reference-note-type.md"
README_PATH = Path(__file__).parents[2] / "README.md"

EXPECTED_NOTE_TYPE_FIELDS = (
    "LatinitasID",
    "Lemma",
    "Principal Parts",
    "Meaning",
    # Tags is deliberately absent: it is the special CSV transport column.
    "Source ID",
    "Source Scope",
    "Source Kind",
    "Source Location",
    "Source Path",
    "Note Schema",
    "Generator",
    "Profile",
    "CompletionPresentEnabled",
    "CompletionPresentPrompt",
    "CompletionPresentAnswer",
    "CompletionInfinitiveEnabled",
    "CompletionInfinitivePrompt",
    "CompletionInfinitiveAnswer",
    "CompletionPerfectEnabled",
    "CompletionPerfectPrompt",
    "CompletionPerfectAnswer",
    "CompletionPPPEnabled",
    "CompletionPPPPrompt",
    "CompletionPPPAnswer",
    "CompletionSupineEnabled",
    "CompletionSupinePrompt",
    "CompletionSupineAnswer",
    "RecognitionPresentEnabled",
    "RecognitionPresentPrompt",
    "RecognitionPresentAnswer",
    "RecognitionInfinitiveEnabled",
    "RecognitionInfinitivePrompt",
    "RecognitionInfinitiveAnswer",
    "RecognitionPerfectEnabled",
    "RecognitionPerfectPrompt",
    "RecognitionPerfectAnswer",
    "RecognitionPPPEnabled",
    "RecognitionPPPPrompt",
    "RecognitionPPPAnswer",
    "RecognitionSupineEnabled",
    "RecognitionSupinePrompt",
    "RecognitionSupineAnswer",
    "Personal Notes",
)

EXPECTED_TEMPLATE_IDENTITIES = (
    ("Completion Present", "principal_part_completion:present_1s", 0),
    ("Completion Infinitive", "principal_part_completion:present_infinitive", 1),
    ("Completion Perfect", "principal_part_completion:perfect_1s", 2),
    ("Completion PPP", "principal_part_completion:perfect_passive_participle", 3),
    ("Completion Supine", "principal_part_completion:supine", 4),
    ("Recognition Present", "principal_part_recognition:present_1s", 5),
    ("Recognition Infinitive", "principal_part_recognition:present_infinitive", 6),
    ("Recognition Perfect", "principal_part_recognition:perfect_1s", 7),
    ("Recognition PPP", "principal_part_recognition:perfect_passive_participle", 8),
    ("Recognition Supine", "principal_part_recognition:supine", 9),
)

EXPECTED_FRONTS = {
    "Completion Present": "{{#CompletionPresentEnabled}}{{CompletionPresentPrompt}}{{/CompletionPresentEnabled}}",
    "Completion Infinitive": (
        "{{#CompletionInfinitiveEnabled}}{{CompletionInfinitivePrompt}}{{/CompletionInfinitiveEnabled}}"
    ),
    "Completion Perfect": "{{#CompletionPerfectEnabled}}{{CompletionPerfectPrompt}}{{/CompletionPerfectEnabled}}",
    "Completion PPP": "{{#CompletionPPPEnabled}}{{CompletionPPPPrompt}}{{/CompletionPPPEnabled}}",
    "Completion Supine": "{{#CompletionSupineEnabled}}{{CompletionSupinePrompt}}{{/CompletionSupineEnabled}}",
    "Recognition Present": "{{#RecognitionPresentEnabled}}{{RecognitionPresentPrompt}}{{/RecognitionPresentEnabled}}",
    "Recognition Infinitive": (
        "{{#RecognitionInfinitiveEnabled}}{{RecognitionInfinitivePrompt}}{{/RecognitionInfinitiveEnabled}}"
    ),
    "Recognition Perfect": "{{#RecognitionPerfectEnabled}}{{RecognitionPerfectPrompt}}{{/RecognitionPerfectEnabled}}",
    "Recognition PPP": "{{#RecognitionPPPEnabled}}{{RecognitionPPPPrompt}}{{/RecognitionPPPEnabled}}",
    "Recognition Supine": "{{#RecognitionSupineEnabled}}{{RecognitionSupinePrompt}}{{/RecognitionSupineEnabled}}",
}

MISSING_FORM_EXAMPLES = (
    (
        "ferō",
        "ferō — ferre — tulī — lātum",
        ("present_1s", "present_infinitive", "perfect_1s", "supine"),
        (
            "Completion Present",
            "Completion Infinitive",
            "Completion Perfect",
            "Completion Supine",
            "Recognition Present",
            "Recognition Infinitive",
            "Recognition Perfect",
            "Recognition Supine",
        ),
    ),
    (
        "ferō",
        "ferō — ferre —  — ",
        ("present_1s", "present_infinitive", "perfect_1s", "perfect_passive_participle"),
        (
            "Completion Present",
            "Completion Infinitive",
            "Recognition Present",
            "Recognition Infinitive",
        ),
    ),
    (
        "amō",
        "amō — amāre — <b></b> — amātum",
        ("present_1s", "present_infinitive", "perfect_1s", "supine"),
        (
            "Completion Present",
            "Completion Infinitive",
            "Completion Supine",
            "Recognition Present",
            "Recognition Infinitive",
            "Recognition Supine",
        ),
    ),
)


def test_reference_note_type_fields_are_the_exact_non_transport_schema_order() -> None:
    assert REFERENCE_NOTE_TYPE_FIELDS == EXPECTED_NOTE_TYPE_FIELDS
    assert REFERENCE_NOTE_TYPE_FIELDS[0] == "LatinitasID"
    assert REFERENCE_NOTE_TYPE_FIELDS[-1] == "Personal Notes"
    assert "Tags" not in REFERENCE_NOTE_TYPE_FIELDS
    transport_fields = tuple(field.name for field in AUTHORITATIVE_NOTE_FIELDS if field.ownership == "transport")
    assert transport_fields == ("Tags",)


def test_reference_note_type_fields_align_with_the_exported_csv_columns() -> None:
    # The exported CSV keeps Tags at column 5 and never offers Personal Notes;
    # the note type keeps Personal Notes and has no Tags field.
    assert CSV_EXPORT_FIELD_NAMES[4] == "Tags"
    assert CSV_EXPORT_FIELD_NAMES[:4] + CSV_EXPORT_FIELD_NAMES[5:] + ("Personal Notes",) == EXPECTED_NOTE_TYPE_FIELDS
    assert "Personal Notes" not in CSV_EXPORT_FIELD_NAMES
    assert "Tags" not in EXPECTED_NOTE_TYPE_FIELDS


def test_one_wholly_guarded_template_per_frozen_registry_slot() -> None:
    identities = tuple(
        (template.slot.template_name, template.slot.semantic_key, template.slot.ordinal)
        for template in REFERENCE_CARD_TEMPLATES
    )
    assert identities == EXPECTED_TEMPLATE_IDENTITIES
    assert len(REFERENCE_CARD_TEMPLATES) == len(TEMPLATE_REGISTRY) == 10

    for template in REFERENCE_CARD_TEMPLATES:
        name = template.slot.template_name
        enabled = name.replace(" ", "") + "Enabled"
        assert template.front == EXPECTED_FRONTS[name]
        assert re.fullmatch(r"\{\{#\w+Enabled\}\}\{\{\w+Prompt\}\}\{\{/\w+Enabled\}\}", template.front)
        # The whole back is guarded by the same per-card eligibility field and
        # renders the managed answer, shared context, and shared Personal Notes.
        assert template.back.startswith(f"{{{{#{enabled}}}}}")
        assert template.back.endswith(f"{{{{/{enabled}}}}}")
        for referenced in re.findall(r"\{\{[#/]?([\w ]+)\}\}", template.back):
            assert referenced in EXPECTED_NOTE_TYPE_FIELDS
        assert "{{#Meaning}}" in template.back and "{{/Meaning}}" in template.back
        assert "{{#Personal Notes}}" in template.back and "{{/Personal Notes}}" in template.back
        assert template.slot.prompt_field not in template.back


def test_back_css_classes_exist_in_the_shared_styling() -> None:
    for style_class in ("latinitas-answer", "latinitas-context", "latinitas-personal-notes"):
        assert f".{style_class}" in REFERENCE_CARD_CSS
    assert ".latinitas-prompt" not in REFERENCE_CARD_CSS


def test_docs_publish_the_generated_reference_setup_verbatim() -> None:
    docs = DOCS_PATH.read_text(encoding="utf-8")
    begin_marker = "<!-- latinitas-reference-setup begin -->\n"
    begin = docs.index(begin_marker) + len(begin_marker)
    end = docs.index("<!-- latinitas-reference-setup end -->")
    published = docs[begin:end].strip("\n")
    generated = reference_setup_markdown().strip("\n")
    assert published == generated
    for version_evidence in ("3", "v1", TEMPLATE_REGISTRY_DIGEST):
        assert version_evidence in generated
    for field_name in EXPECTED_NOTE_TYPE_FIELDS:
        assert f"\n{field_name}\n" in generated
    for front in EXPECTED_FRONTS.values():
        assert front in generated


def _eligible_templates(lemma: str, forms: str, roles: tuple[str, ...]) -> tuple[str, ...]:
    profile = DeckProfile.default(
        note_type="Latin vocabulary",
        lexical_entry_field="Lemma",
        principal_parts_field="Principal parts",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="Source ID"),
        principal_part_roles=roles,
        separators=(" — ",),
        selected_recipes=("principal_part_completion", "principal_part_recognition"),
    )
    record = CanonicalSourceRecord(
        source_kind="csv",
        note_type=profile.note_type,
        fields={
            "Source ID": "entry-1",
            profile.fields.lexical_entry_field: lemma,
            profile.fields.principal_parts_field: forms,
            "German gloss": "tragen",
        },
        provenance=SourceProvenance(source_path=Path("fixture.csv"), location="row 2", row_number=2),
        source_identity="entry-1",
    )
    result = generate_learning_object_notes((record,), profile, source_scope="scope-reference")
    assert len(result.notes) == 1
    note = result.notes[0]
    return tuple(card.slot.template_name for card in note.cards if card.eligible)


def test_missing_form_examples_match_generated_eligibility_and_the_published_table() -> None:
    docs = DOCS_PATH.read_text(encoding="utf-8")
    for lemma, forms, roles, expected in MISSING_FORM_EXAMPLES:
        eligible = _eligible_templates(lemma, forms, roles)
        assert eligible == expected
        published_rows = [line for line in docs.splitlines() if line.startswith(f"| `{forms}` |")]
        assert len(published_rows) == 1, f"expected exactly one docs example row for {forms!r}"
        assert ", ".join(roles) in published_rows[0]
        assert ", ".join(expected) in published_rows[0]


def test_readme_documents_the_reference_setup_guide() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    assert "[Reference note type and safe import](docs/reference-note-type.md)" in readme
