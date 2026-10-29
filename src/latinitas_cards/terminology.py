"""Shared, exact-match presentation vocabulary, independent of claim review."""

from typing import Literal

from .profile import TerminologySettings

type Language = Literal["la", "de", "en"]
type Labels = dict[Language, str]

# Each canonical value maps to Latin, German and English. Free-text reviews are not translated.
FEATURE_NAMES: dict[str, Labels] = {
    "lemma": {"la": "Lemma", "de": "Lemma", "en": "Lemma"},
    "case": {"la": "Casus", "de": "Kasus", "en": "Case"},
    "number": {"la": "Numerus", "de": "Numerus", "en": "Number"},
    "person": {"la": "Persona", "de": "Person", "en": "Person"},
    "tense": {"la": "Tempus", "de": "Tempus", "en": "Tense"},
    "mood": {"la": "Modus", "de": "Modus", "en": "Mood"},
    "voice": {"la": "Vox", "de": "Genus verbi", "en": "Voice"},
    "gender": {"la": "Genus", "de": "Genus", "en": "Gender"},
    "degree": {"la": "Gradus", "de": "Steigerungsstufe", "en": "Degree"},
    "verbform": {"la": "Forma verbi", "de": "Verbform", "en": "Verb form"},
}
FEATURE_VALUES: dict[str, dict[str, Labels]] = {
    "case": {
        "nominativus": {"la": "nominativus", "de": "Nominativ", "en": "Nominative"},
        "genetivus": {"la": "genetivus", "de": "Genitiv", "en": "Genitive"},
        "dativus": {"la": "dativus", "de": "Dativ", "en": "Dative"},
        "accusativus": {"la": "accusativus", "de": "Akkusativ", "en": "Accusative"},
        "ablativus": {"la": "ablativus", "de": "Ablativ", "en": "Ablative"},
        "vocativus": {"la": "vocativus", "de": "Vokativ", "en": "Vocative"},
        "locativus": {"la": "locativus", "de": "Lokativ", "en": "Locative"},
    },
    "number": {
        "singularis": {"la": "singularis", "de": "Singular", "en": "Singular"},
        "pluralis": {"la": "pluralis", "de": "Plural", "en": "Plural"},
    },
    "person": {
        "prima": {"la": "prima", "de": "1. Person", "en": "1st person"},
        "secunda": {"la": "secunda", "de": "2. Person", "en": "2nd person"},
        "tertia": {"la": "tertia", "de": "3. Person", "en": "3rd person"},
    },
    "tense": {
        "praesens": {"la": "praesens", "de": "Präsens", "en": "Present"},
        "imperfectum": {"la": "imperfectum", "de": "Imperfekt", "en": "Imperfect"},
        "futurum": {"la": "futurum", "de": "Futur I", "en": "Future"},
        "perfectum": {"la": "perfectum", "de": "Perfekt", "en": "Perfect"},
        "plusquamperfectum": {"la": "plusquamperfectum", "de": "Plusquamperfekt", "en": "Pluperfect"},
        "futurum exactum": {"la": "futurum exactum", "de": "Futur II", "en": "Future perfect"},
    },
    "mood": {
        "indicativus": {"la": "indicativus", "de": "Indikativ", "en": "Indicative"},
        "coniunctivus": {"la": "coniunctivus", "de": "Konjunktiv", "en": "Subjunctive"},
        "imperativus": {"la": "imperativus", "de": "Imperativ", "en": "Imperative"},
        "infinitivus": {"la": "infinitivus", "de": "Infinitiv", "en": "Infinitive"},
        "participium": {"la": "participium", "de": "Partizip", "en": "Participle"},
    },
    "voice": {
        "activa": {"la": "activa", "de": "Aktiv", "en": "Active"},
        "passiva": {"la": "passiva", "de": "Passiv", "en": "Passive"},
    },
    "gender": {
        "masculinum": {"la": "masculinum", "de": "Maskulinum", "en": "Masculine"},
        "femininum": {"la": "femininum", "de": "Femininum", "en": "Feminine"},
        "neutrum": {"la": "neutrum", "de": "Neutrum", "en": "Neuter"},
    },
    "degree": {
        "positivus": {"la": "positivus", "de": "Positiv", "en": "Positive"},
        "comparativus": {"la": "comparativus", "de": "Komparativ", "en": "Comparative"},
        "superlativus": {"la": "superlativus", "de": "Superlativ", "en": "Superlative"},
    },
    "verbform": {
        "infinitivus": {"la": "infinitivus", "de": "Infinitiv", "en": "Infinitive"},
        "participium": {"la": "participium", "de": "Partizip", "en": "Participle"},
        "gerundium": {"la": "gerundium", "de": "Gerundium", "en": "Gerund"},
        "gerundivum": {"la": "gerundivum", "de": "Gerundivum", "en": "Gerundive"},
        "supinum": {"la": "supinum", "de": "Supinum", "en": "Supine"},
    },
}
ROLE_LABELS: dict[str, Labels] = {
    "present_1s": {
        "la": "Praesens, persona prima singularis",
        "de": "Präsens, 1. Person Singular",
        "en": "Present, 1st person singular",
    },
    "present_infinitive": {"la": "Infinitivus praesentis", "de": "Infinitiv", "en": "Present infinitive"},
    "perfect_1s": {
        "la": "Perfectum, persona prima singularis",
        "de": "Perfekt, 1. Person Singular",
        "en": "Perfect, 1st person singular",
    },
    "perfect_passive_participle": {
        "la": "Participium perfecti passivi (PPP)",
        "de": "Partizip Perfekt Passiv (PPP)",
        "en": "Perfect passive participle (PPP)",
    },
    "supine": {"la": "Supinum", "de": "Supinum", "en": "Supine"},
}


type CardLanguage = Literal["de", "en"]

# Fixed generated-card text keyed by (string_key, language). German entries are the historical
# strings, byte for byte. Latin-first prompts such as "Formam resolve." are not translated.
CARD_STRINGS: dict[str, dict[CardLanguage, str]] = {
    "absent_value": {"de": "— (Nicht vorhanden)", "en": "— (Not present)"},
    "withheld_value": {"de": "— (Zurückgehalten)", "en": "— (Withheld)"},
    "analysis_withheld": {
        "de": "Analyse zurückgehalten; einzelne Belege prüfen.",
        "en": "Analysis withheld; check the individual evidence.",
    },
    "related_stems": {"de": "Verwandte Stämme: ", "en": "Related stems: "},
    "other_stems_withheld": {"de": "Weitere Stämme: Analyse zurückgehalten.", "en": "Other stems: analysis withheld."},
    "compare_principal_parts": {"de": "Stammformen vergleichen", "en": "Compare principal parts"},
    "role_absent": {
        "de": "Nicht vorhanden; Quelle oder Rollenprofil prüfen.",
        "en": "Not present; check the source or role profile.",
    },
    "role_unresolved": {
        "de": "Unresolved source evidence; target withheld. Alternativen und Hinweise prüfen.",
        "en": "Unresolved source evidence; target withheld. Check alternatives and notes.",
    },
    "role_review_required": {
        "de": "Linguistic review required; target withheld. Einzelne Rollenbelege ausdrücklich prüfen.",
        "en": "Linguistic review required; target withheld. Explicitly check each role's evidence.",
    },
    "role_source_form": {
        "de": "Quellform im bestätigten Rollenprofil; Analyse benötigt gesonderte Prüfung.",
        "en": "Source form in the confirmed role profile; analysis needs separate review.",
    },
    "blank_unresolved": {"de": "— (unresolved source evidence)", "en": "— (unresolved source evidence)"},
    "blank_review_required": {"de": "— (linguistic review required)", "en": "— (linguistic review required)"},
    "part_unresolved": {
        "de": "(unresolved source evidence; target withheld)",
        "en": "(unresolved source evidence; target withheld)",
    },
    "part_review_required": {
        "de": "(linguistic review required; target withheld)",
        "en": "(linguistic review required; target withheld)",
    },
    "phrase_translation_prompt": {
        "de": "Übersetze und vergleiche anschließend mit der Referenz.",
        "en": "Translate, then compare with the reference.",
    },
    "phrase_recognition_prompt": {
        "de": "Bestimme {form} [{target}]: {distinction}.",
        "en": "Identify {form} [{target}]: {distinction}.",
    },
    "analysis": {"de": "Analyse", "en": "Analysis"},
    "reading": {"de": "Lesart {index}: ", "en": "Reading {index}: "},
    "self_assessment": {
        "de": "Selbsteinschätzung: Vergleiche den Sinn, nicht den genauen Wortlaut.",
        "en": "Self-assessment: compare the meaning, not the exact wording.",
    },
    "reviewed_reference": {"de": "Reviewed reference", "en": "Reviewed reference"},
    "proposed_reference": {"de": "Proposed reference — not asserted", "en": "Proposed reference — not asserted"},
    "withheld": {"de": "Withheld: ", "en": "Withheld: "},
}


def card_language(language_tag: str) -> CardLanguage:
    """A primary subtag of ``en`` selects English; any other renders German, exactly as before."""

    return "en" if language_tag.lower().split("-")[0] == "en" else "de"


def card_text(key: str, language_tag: str, **values: object) -> str:
    text = CARD_STRINGS[key][card_language(language_tag)]
    return text.format(**values) if values else text


def terminology_language(settings: TerminologySettings, language_tag: str) -> Language:
    """Latin mode ignores the tag; otherwise the card language applies."""

    return "la" if settings.language == "latin" else card_language(language_tag)


def feature_name(feature: str, settings: TerminologySettings, language_tag: str) -> str:
    labels = FEATURE_NAMES.get(feature)
    return labels[terminology_language(settings, language_tag)] if labels else feature


def feature_value(feature: str, value: str, settings: TerminologySettings, language_tag: str) -> str:
    labels = FEATURE_VALUES.get(feature, {}).get(value)
    return labels[terminology_language(settings, language_tag)] if labels else value


def role_label(role: str, settings: TerminologySettings, language_tag: str) -> str:
    labels = ROLE_LABELS.get(role)
    return labels[terminology_language(settings, language_tag)] if labels else role


def terminology_report(settings: TerminologySettings, language_tag: str, override: str | None = None) -> str:
    language = terminology_language(settings, language_tag)
    return (
        f"Effective terminology v{settings.version}: {settings.language} ({language}); override: {override or 'none'}"
    )
