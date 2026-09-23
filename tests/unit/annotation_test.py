from types import SimpleNamespace

from latinitas_cards.annotation import WordAnalysis, analyze_cltk_word, format_ud_features


def _feature(key: str, value: str) -> SimpleNamespace:
    return SimpleNamespace(key=key, value=value)


def _word(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "lemma": "amo",
        "upos": SimpleNamespace(tag="VERB"),
        "xpos": None,
        "features": SimpleNamespace(features=[_feature("Mood", "Ind"), _feature("Aspect", "Imp")]),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_format_ud_features_sorts_keys_and_joins_multiple_values() -> None:
    features = SimpleNamespace(
        features=[
            _feature("InflClass", "LatX"),
            _feature("Case", "Nom"),
            _feature("InflClass", "IndEurO"),
        ]
    )

    assert format_ud_features(features) == "Case=Nom|InflClass=IndEurO,LatX"


def test_format_ud_features_handles_missing_features() -> None:
    assert format_ud_features(None) == ""
    assert format_ud_features(SimpleNamespace(features=[])) == ""


def test_analyze_cltk_word_maps_cltk2_objects_to_strings() -> None:
    assert analyze_cltk_word(_word()) == WordAnalysis(
        lemma="amo",
        upos="VERB",
        xpos="",
        morph_features="Aspect=Imp|Mood=Ind",
    )


def test_analyze_cltk_word_tolerates_missing_annotations() -> None:
    word = _word(lemma=None, upos=None, features=None)

    assert analyze_cltk_word(word) == WordAnalysis(lemma="", upos="", xpos="", morph_features="")


def test_analyze_cltk_word_keeps_xpos_when_present() -> None:
    assert analyze_cltk_word(_word(xpos="N3|modA")).xpos == "N3|modA"
