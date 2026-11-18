"""Installed authority records, changed only by a reviewed application update.

Representability is not evidence applicability or an enabled automatic scope.
T188 and T191 supply the finite applicable rules and recipe target enumeration.
"""

POLICY_REGISTRY = (
    '{"policy":{"checker_sha256":null,"comparison_sha256":null,"criteria_sha256":null,'
    '"format":"latinitas-automatic-policy","id":"v050-disabled","revision":"1","scope":[], '
    '"support_registry_sha256":null,"version":1,"versions_sha256":null},'
    '"state":"disabled","authorization":null}',
)

AUTHORIZATION_HISTORY: tuple[str, ...] = ()

SUPPORT_REGISTRY = {
    "version": "v050-support/1",
    "entries": (
        {
            "recipe": "form_translation",
            "materials": ("source-form/1",),
            "roles": ("other", "present_1s", "present_infinitive", "perfect_1s"),
            "target": ("meaning",),
        },
        {
            "recipe": "form_recognition",
            "materials": ("source-form/1",),
            "roles": ("other", "present_1s", "perfect_1s"),
            "target": ("features",),
        },
        {"recipe": "ppp_translation", "materials": ("confirmed-ppp/1",), "roles": ("ppp",), "target": ("meaning",)},
        {"recipe": "ppp_recognition", "materials": ("confirmed-ppp/1",), "roles": ("ppp",), "target": ("features",)},
        {
            "recipe": "phrase_translation",
            "materials": ("preposition-noun/1", "noun-genitive/1", "verb-noun/1", "adjective-noun/1"),
            "roles": ("other",),
            "target": ("meaning",),
        },
        {
            "recipe": "phrase_recognition",
            "materials": ("preposition-noun/1", "noun-genitive/1", "verb-noun/1", "adjective-noun/1"),
            "roles": ("other",),
            "target": ("features",),
        },
        {
            "recipe": "verb_production",
            "materials": ("source-form/1",),
            "roles": ("present_1s", "present_infinitive", "perfect_1s", "ppp"),
            "target": ("person", "number", "tense", "mood", "voice"),
        },
        {
            "recipe": "noun_declension",
            "materials": ("source-form/1",),
            "roles": ("other",),
            "target": ("case", "number"),
        },
        {
            "recipe": "adjective_declension",
            "materials": ("source-form/1", "confirmed-ppp/1"),
            "roles": ("other", "ppp"),
            "target": ("case", "number", "gender"),
        },
        {
            "recipe": "form_cloze",
            "materials": ("target-cloze/1",),
            "roles": ("other", "present_1s", "perfect_1s", "ppp"),
            "target": ("features", "context"),
        },
        {
            "recipe": "phrase_form_cloze",
            "materials": ("target-cloze/1",),
            "roles": ("other",),
            "target": ("features", "context"),
        },
        {"recipe": "rule_application", "materials": ("curated/1",), "roles": ("other",), "target": ("rule",)},
        {"recipe": "mnemonic_recall", "materials": ("curated/1",), "roles": ("other",), "target": ("mnemonic",)},
        {"recipe": "formation_rule_recall", "materials": ("curated/1",), "roles": ("other",), "target": ("rule",)},
        {"recipe": "component_explanation", "materials": ("curated/1",), "roles": ("other",), "target": ("rule",)},
    ),
    "features": "phrase_inputs.Features/1",
    "source_cell_rule": "confirmed-selected-source-cells/1",
    "evidence_applicability": "source-bound-applicable-rule-or-library/1",
    "ppp_role": "explicit confirmed ppp with role evidence; never supine or unspecified fourth part",
    "cloze": "explicit supported target and non-meaning distinction required",
    "unsupported": "missing mapping, sense, role, target or applicable evidence withholds",
}
