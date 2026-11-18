"""Reference-backed finite cells, independent of the implementation's endings.

Allen & Greenough §§37–40, 110, 184 (DCC, Ayer 2014) give these
contrasts. These expectations are examples from the paradigms, not a second
invocation of the derivation algorithm.
"""

import pytest

from latinitas_cards.form_recipes import derive_form
from latinitas_cards.phrase_inputs import Features


@pytest.mark.parametrize(
    ("pos", "anchors", "target", "expected", "stem"),
    [
        (
            "verb",
            ("amō", "amāre"),
            dict(person="2", number="singular", tense="present", mood="indicative", voice="active"),
            "amās",
            "amā",
        ),
        (
            "verb",
            ("amō", "amāre"),
            dict(person="3", number="plural", tense="present", mood="indicative", voice="active"),
            "amant",
            "amā",
        ),
        (
            "noun",
            ("puella", "puellae"),
            dict(case="accusative", number="singular", gender="feminine"),
            "puellam",
            "puellā",
        ),
        (
            "noun",
            ("puella", "puellae"),
            dict(case="genitive", number="plural", gender="feminine"),
            "puellārum",
            "puellā",
        ),
        (
            "adjective",
            ("bonus", "bona", "bonum"),
            dict(case="accusative", number="singular", gender="feminine"),
            "bonam",
            "bon",
        ),
        (
            "adjective",
            ("bonus", "bona", "bonum"),
            dict(case="nominative", number="plural", gender="neuter"),
            "bona",
            "bon",
        ),
    ],
)
def test_reference_contrasts(
    pos: str, anchors: tuple[str, ...], target: dict[str, str], expected: str, stem: str
) -> None:
    result = derive_form(pos, anchors, Features.model_validate(target))
    assert result.surface == expected
    assert result.stem == stem
    assert result.evidence.reference.startswith("https://dcc.dickinson.edu/grammar/latin/")


@pytest.mark.parametrize(
    ("pos", "anchors", "target"),
    [
        ("verb", ("amō", "amāre"), dict(person="2", number="singular")),
        (
            "verb",
            ("moneō", "monēre"),
            dict(person="2", number="singular", tense="present", mood="indicative", voice="active"),
        ),
        (
            "verb",
            ("amō", "monēre"),
            dict(person="2", number="singular", tense="present", mood="indicative", voice="active"),
        ),
        ("noun", ("puella", "puellī"), dict(case="genitive", number="plural", gender="feminine")),
        ("adjective", ("bonus", "bona", "bonum"), dict(case="accusative", number="singular")),
        ("adjective", ("bonus", "mala", "bonum"), dict(case="accusative", number="singular", gender="feminine")),
    ],
)
def test_inapplicable_and_underspecified_cells_never_guess(
    pos: str, anchors: tuple[str, ...], target: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="unsupported|incomplete"):
        derive_form(pos, anchors, Features.model_validate(target))
