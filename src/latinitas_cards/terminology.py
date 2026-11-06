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

# Fixed generated-card text keyed by (string_key, language).
# Latin-first prompts such as "Formam resolve." are not translated.
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
        "de": "Quellbeleg ungeklärt; Zielform zurückgehalten. Alternativen und Hinweise prüfen.",
        "en": "Unresolved source evidence; target withheld. Check alternatives and notes.",
    },
    "role_review_required": {
        "de": "Sprachliche Prüfung erforderlich; Zielform zurückgehalten. Einzelne Rollenbelege ausdrücklich prüfen.",
        "en": "Linguistic review required; target withheld. Explicitly check each role's evidence.",
    },
    "role_source_form": {
        "de": "Quellform im bestätigten Rollenprofil; Analyse benötigt gesonderte Prüfung.",
        "en": "Source form in the confirmed role profile; analysis needs separate review.",
    },
    "blank_unresolved": {"de": "— (Quellbeleg ungeklärt)", "en": "— (unresolved source evidence)"},
    "blank_review_required": {"de": "— (sprachliche Prüfung erforderlich)", "en": "— (linguistic review required)"},
    "part_unresolved": {
        "de": "(Quellbeleg ungeklärt; Zielform zurückgehalten)",
        "en": "(unresolved source evidence; target withheld)",
    },
    "part_review_required": {
        "de": "(sprachliche Prüfung erforderlich; Zielform zurückgehalten)",
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
    "reviewed_reference": {"de": "Geprüfte Referenz", "en": "Reviewed reference"},
    "proposed_reference": {
        "de": "Vorgeschlagene Referenz — nicht bestätigt",
        "en": "Proposed reference — not asserted",
    },
    "withheld": {"de": "Zurückgehalten: ", "en": "Withheld: "},
    "claim_stale": {
        "de": "Veraltete oder abweichende Belegprüfung; erneute Prüfung erforderlich",
        "en": "Stale or different claim review; renewed review required",
    },
    "claim_unsupported": {
        "de": "Nicht gestützter Beleg; stützende Nachweise erforderlich",
        "en": "Unsupported claim; supporting evidence required",
    },
    "claim_uncalibrated": {
        "de": "Keine anwendbare Kalibrierung; ausdrückliche Belegprüfung erforderlich",
        "en": "No applicable calibration; explicit claim review required",
    },
    "phrase_unresolved_distinction": {
        "de": "Die geprüfte Unterscheidung ist in den angegebenen kontextuellen Lesarten ungeklärt.",
        "en": "The tested distinction is unresolved in the supplied contextual readings.",
    },
    "phrase_reference_language_mismatch": {
        "de": "Referenzsprache {reference!r} weicht von der Kartensprache {language!r} ab",
        "en": "Reference language {reference!r} differs from card language {language!r}",
    },
    "phrase_combination": {"de": "Kombination", "en": "combination"},
    "phrase_answer": {"de": "Antwort", "en": "answer"},
    "phrase_invalid_reference": {
        "de": "Ungültige verfasste Referenz; technische Details im Diagnosebericht.",
        "en": "invalid authored reference: {error}",
    },
    "corpus_lexical_prompt": {
        "de": "Was bedeutet das markierte Wort hier?",
        "en": "What does the highlighted word mean here?",
    },
    "corpus_analysis_prompt": {
        "de": "Bestimme das markierte Wort und übersetze dann den Satz.",
        "en": "Analyse the highlighted word, then translate the sentence.",
    },
    "corpus_phrase_prompt": {
        "de": "Übersetze die markierte Wendung im Zusammenhang.",
        "en": "Translate the highlighted phrase in context.",
    },
    "corpus_sentence_prompt": {"de": "Übersetze den Satz.", "en": "Translate the sentence."},
    "corpus_passage_prompt": {"de": "Übersetze den Abschnitt.", "en": "Translate the passage."},
    "citation_book": {"de": "Buch", "en": "Book"},
    "citation_chapter": {"de": "Kapitel", "en": "Chapter"},
    "citation_section": {"de": "Abschnitt", "en": "Section"},
    "citation_line": {"de": "Vers", "en": "Line"},
    "citation_verse": {"de": "Vers", "en": "Verse"},
    "citation_passage": {"de": "Abschnitt", "en": "Passage"},
    "citation_unknown_author": {"de": "Autor nicht angegeben", "en": "Author not supplied"},
    "citation_caesar_author": {"de": "Julius Caesar", "en": "Julius Caesar"},
    "citation_caesar_work": {"de": "Der Gallische Krieg", "en": "The Gallic War"},
    "citation_virgil_author": {"de": "Vergil", "en": "Virgil"},
    "citation_virgil_work": {"de": "Aeneis", "en": "Aeneid"},
    "citation_provenance": {"de": "Quellennachweis und Grenzen", "en": "Source provenance and limitations"},
    "citation_technical": {"de": "Technische Details", "en": "Technical details"},
    "citation_not_supplied": {"de": "Nicht angegeben", "en": "Not supplied"},
    "citation_field_source_file": {"de": "Quelldatei", "en": "Source file"},
    "citation_field_work": {"de": "Werk", "en": "Work"},
    "citation_field_edition": {"de": "Ausgabe", "en": "Edition"},
    "citation_field_edition_id": {"de": "Ausgabenkennung (CTS)", "en": "Edition identifier (CTS)"},
    "citation_field_passage": {"de": "Stelle", "en": "Passage"},
    "citation_field_source_url": {"de": "Quell-URL (angegeben)", "en": "Source URL (supplied)"},
    "citation_field_revision": {"de": "Revision (angegeben)", "en": "Revision (supplied)"},
    "citation_field_digest": {"de": "Prüfsumme der Quelldatei (SHA256)", "en": "Source-file digest (SHA256)"},
    "citation_field_license": {"de": "Quelllizenz (angegeben)", "en": "Source license (supplied)"},
    "citation_missing_author": {"de": "Autor fehlt", "en": "Author missing"},
    "citation_missing_work_title": {
        "de": "Werktitel fehlt; lokale Werk-ID angezeigt",
        "en": "Work title missing; local work ID shown",
    },
    "citation_missing_edition": {"de": "Ausgabenbeschreibung fehlt", "en": "Edition description missing"},
    "citation_missing_edition_urn": {
        "de": "Keine belegte CTS-Ausgabenkennung",
        "en": "No source-attested CTS edition identity",
    },
    "citation_missing_source_url": {"de": "Quell-URL fehlt", "en": "Source URL missing"},
    "citation_missing_revision": {"de": "Quellrevision fehlt", "en": "Source revision missing"},
    "citation_missing_source_license": {
        "de": "Quelllizenz fehlt; keine Rechtefreigabe",
        "en": "Source license missing; no rights clearance",
    },
    "citation_no_verified_passage_link": {
        "de": "Kein verifizierter Online-Stellenlink; nicht verlinkt",
        "en": "No verified online passage link; not linked",
    },
    "citation_resolver_revision": {
        "de": "Online-Ausgabe und Stelle verifiziert, nicht die Revision der Quelldatei",
        "en": "Online edition and passage verified, not the source-file revision",
    },
    "reference_guidance": {
        "de": "Referenzübersetzung: Andere Übersetzungen können ebenso richtig sein.",
        "en": "Reference translation: other translations can be correct as well.",
    },
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
