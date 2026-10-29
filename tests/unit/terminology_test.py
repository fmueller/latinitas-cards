import pytest

from latinitas_cards.profile import TerminologySettings
from latinitas_cards.terminology import (
    CARD_STRINGS,
    FEATURE_NAMES,
    FEATURE_VALUES,
    ROLE_LABELS,
    Labels,
    card_text,
    feature_name,
    feature_value,
    role_label,
    terminology_report,
)

LATIN = TerminologySettings(language="latin")
USER = TerminologySettings(language="user")


def _tables() -> list[tuple[str, Labels]]:
    tables = [(f"feature {key}", labels) for key, labels in FEATURE_NAMES.items()]
    tables += [(f"role {key}", labels) for key, labels in ROLE_LABELS.items()]
    tables += [
        (f"value {feature}.{key}", labels)
        for feature, values in FEATURE_VALUES.items()
        for key, labels in values.items()
    ]
    return tables


def test_every_terminology_key_has_latin_german_and_english() -> None:
    for name, labels in _tables():
        assert set(labels) == {"la", "de", "en"}, name
        assert all(text.strip() for text in labels.values()), name
    assert set(FEATURE_VALUES) | {"lemma"} == set(FEATURE_NAMES)


def test_degree_and_verbform_are_controlled_in_all_three_languages() -> None:
    assert set(FEATURE_VALUES["degree"]) == {"positivus", "comparativus", "superlativus"}
    assert set(FEATURE_VALUES["verbform"]) == {"infinitivus", "participium", "gerundium", "gerundivum", "supinum"}
    assert feature_value("degree", "comparativus", USER, "en") == "Comparative"
    assert feature_value("degree", "comparativus", USER, "de") == "Komparativ"
    assert feature_value("verbform", "gerundivum", USER, "en-GB") == "Gerundive"
    assert feature_value("verbform", "gerundivum", LATIN, "en") == "gerundivum"


@pytest.mark.parametrize(
    ("settings", "tag", "expected"),
    [
        (USER, "en", ("Case", "Genitive", "Perfect passive participle (PPP)")),
        (USER, "EN-us", ("Case", "Genitive", "Perfect passive participle (PPP)")),
        (USER, "de-AT", ("Kasus", "Genitiv", "Partizip Perfekt Passiv (PPP)")),
        # Other subtags render the German table, exactly as before English existed.
        (USER, "fr", ("Kasus", "Genitiv", "Partizip Perfekt Passiv (PPP)")),
        (LATIN, "en", ("Casus", "genetivus", "Participium perfecti passivi (PPP)")),
        (LATIN, "fr", ("Casus", "genetivus", "Participium perfecti passivi (PPP)")),
    ],
)
def test_user_terminology_follows_the_primary_subtag(
    settings: TerminologySettings, tag: str, expected: tuple[str, str, str]
) -> None:
    assert (
        feature_name("case", settings, tag),
        feature_value("case", "genetivus", settings, tag),
        role_label("perfect_passive_participle", settings, tag),
    ) == expected


def test_unknown_keys_render_verbatim() -> None:
    assert feature_name("aspect", USER, "en") == "aspect"
    assert feature_value("case", "instrumentalis", USER, "en") == "instrumentalis"
    assert role_label("future_participle", USER, "en") == "future_participle"


@pytest.mark.parametrize(
    ("settings", "tag", "effective"),
    [(USER, "en-GB", "user (en)"), (USER, "de", "user (de)"), (LATIN, "en", "latin (la)"), (LATIN, "fr", "latin (la)")],
)
def test_terminology_report_names_the_effective_language(
    settings: TerminologySettings, tag: str, effective: str
) -> None:
    assert terminology_report(settings, tag, "user") == f"Effective terminology v1: {effective}; override: user"


def test_every_fixed_card_string_has_german_and_english() -> None:
    assert CARD_STRINGS
    for key, texts in CARD_STRINGS.items():
        assert set(texts) == {"de", "en"}, key
        assert all(text.strip() for text in texts.values()), key


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("en", "Compare principal parts"),
        ("EN-gb", "Compare principal parts"),
        ("de", "Stammformen vergleichen"),
        ("de-CH", "Stammformen vergleichen"),
        # Any other primary subtag keeps the historical German text.
        ("fr", "Stammformen vergleichen"),
        ("eng", "Stammformen vergleichen"),
    ],
)
def test_card_text_selects_english_only_for_primary_subtag_en(tag: str, expected: str) -> None:
    assert card_text("compare_principal_parts", tag) == expected
