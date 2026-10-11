"""Shared, exact-match presentation vocabulary, independent of claim review."""

from .profile import TerminologySettings

# Each canonical value maps to Latin and German. Free-text reviews are not translated.
FEATURE_NAMES = {
    "lemma": ("Lemma", "Lemma"),
    "case": ("Casus", "Kasus"),
    "number": ("Numerus", "Numerus"),
    "person": ("Persona", "Person"),
    "tense": ("Tempus", "Tempus"),
    "mood": ("Modus", "Modus"),
    "voice": ("Vox", "Genus verbi"),
    "gender": ("Genus", "Genus"),
}
FEATURE_VALUES = {
    "case": {
        "nominativus": ("nominativus", "Nominativ"),
        "genetivus": ("genetivus", "Genitiv"),
        "dativus": ("dativus", "Dativ"),
        "accusativus": ("accusativus", "Akkusativ"),
        "ablativus": ("ablativus", "Ablativ"),
        "vocativus": ("vocativus", "Vokativ"),
        "locativus": ("locativus", "Lokativ"),
    },
    "number": {"singularis": ("singularis", "Singular"), "pluralis": ("pluralis", "Plural")},
    "person": {
        "prima": ("prima", "1. Person"),
        "secunda": ("secunda", "2. Person"),
        "tertia": ("tertia", "3. Person"),
    },
    "tense": {
        "praesens": ("praesens", "Präsens"),
        "imperfectum": ("imperfectum", "Imperfekt"),
        "futurum": ("futurum", "Futur I"),
        "perfectum": ("perfectum", "Perfekt"),
        "plusquamperfectum": ("plusquamperfectum", "Plusquamperfekt"),
        "futurum exactum": ("futurum exactum", "Futur II"),
    },
    "mood": {
        "indicativus": ("indicativus", "Indikativ"),
        "coniunctivus": ("coniunctivus", "Konjunktiv"),
        "imperativus": ("imperativus", "Imperativ"),
        "infinitivus": ("infinitivus", "Infinitiv"),
        "participium": ("participium", "Partizip"),
    },
    "voice": {"activa": ("activa", "Aktiv"), "passiva": ("passiva", "Passiv")},
    "gender": {
        "masculinum": ("masculinum", "Maskulinum"),
        "femininum": ("femininum", "Femininum"),
        "neutrum": ("neutrum", "Neutrum"),
    },
}
ROLE_LABELS = {
    "present_1s": ("Praesens, persona prima singularis", "Präsens, 1. Person Singular"),
    "present_infinitive": ("Infinitivus praesentis", "Infinitiv"),
    "perfect_1s": ("Perfectum, persona prima singularis", "Perfekt, 1. Person Singular"),
    "perfect_passive_participle": ("Participium perfecti passivi (PPP)", "Partizip Perfekt Passiv (PPP)"),
    "supine": ("Supinum", "Supinum"),
}


def feature_name(feature: str, settings: TerminologySettings) -> str:
    labels = FEATURE_NAMES.get(feature)
    return labels[settings.language == "user"] if labels else feature


def feature_value(feature: str, value: str, settings: TerminologySettings) -> str:
    labels = FEATURE_VALUES.get(feature, {}).get(value)
    return labels[settings.language == "user"] if labels else value


def role_label(role: str, settings: TerminologySettings) -> str:
    labels = ROLE_LABELS.get(role)
    return labels[settings.language == "user"] if labels else role


def terminology_report(settings: TerminologySettings, override: str | None = None) -> str:
    language = "la" if settings.language == "latin" else "de"
    return (
        f"Effective terminology v{settings.version}: {settings.language} ({language}); override: {override or 'none'}"
    )
