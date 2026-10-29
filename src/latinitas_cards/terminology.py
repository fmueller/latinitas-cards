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


def terminology_language(settings: TerminologySettings, language_tag: str) -> Language:
    """Latin mode ignores the tag; a primary subtag of ``en`` selects English, any other German."""

    if settings.language == "latin":
        return "la"
    return "en" if language_tag.lower().split("-")[0] == "en" else "de"


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
