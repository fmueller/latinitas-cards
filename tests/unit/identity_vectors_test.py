"""Versioned, independently specified identity input/output vectors.

The expected values below were computed once with an independent shell
implementation of the documented derivation (canonical JSON payload plus the
domain-separated SHA-256 seed) and are pinned as literals.  They exist to
detect algorithm drift: the production helper must keep reproducing exactly
these identifiers.  Never regenerate these expectations by calling the
production helper.
"""

from __future__ import annotations

import pytest

from latinitas_cards.corpus_exercises import exercise_identity
from latinitas_cards.corpus_profile import CorpusRecipe
from latinitas_cards.identity import (
    IdentityError,
    derive_anki_guid_from_latinitas_id,
    derive_latinitas_id,
)

IDENTITY_VECTORS_VERSION = "note-family-v1"

VECTOR_SCOPE_A = "scope-0123456789abcdef0123456789abcdef"
VECTOR_SCOPE_B = "scope-fedcba9876543210fedcba9876543210"

VECTORS: tuple[tuple[str, str, str | None, str], ...] = (
    (
        "fixture-guid-001",
        "lexeme-1",
        None,
        "latinitas-v2-abd3d6b01dba87e51245a3142b10095407f20c6f49d6a9a1654003efe1084e6d",
    ),
    (
        "fixture-guid-002",
        "lexeme-1",
        None,
        "latinitas-v2-78a4f3389f56143bfe3eb9f8b3d499b28594229c51a3ad4d849a33ee810c70dd",
    ),
    (
        "csv-source-000001",
        "lexeme-1",
        VECTOR_SCOPE_A,
        "latinitas-v2-3187e72b75cf36d831af45246f481b179b42387f21c6913eede414e6d72fa9a0",
    ),
    (
        "csv-source-000001",
        "lexeme-1",
        VECTOR_SCOPE_B,
        "latinitas-v2-0afbfed14739abba9f0ceebb3a35c8221f4418fa43c74b1fec009d4cac2a7f8e",
    ),
    (
        "csv-source-000002",
        "lexeme-1",
        VECTOR_SCOPE_A,
        "latinitas-v2-c1d3712ea77ac1965599c621b99a62d3b5f8558444c4740d31f1b8c300717473",
    ),
)


# Corpus exercises: (passage_id, recipe, target_language, unit, target, object_key, latinitas_id).
# object_key = "corpus-exercise-v1-" + sha256 of the canonical JSON of {recipe, target_language, unit,
# target}, then the note-family derivation with source_scope "corpus-passage-v1"; computed with
# printf and sha256sum, never with the production helper.
CORPUS_VECTORS: tuple[
    tuple[str, CorpusRecipe, str, dict[str, str | int], dict[str, str | int] | None, str, str], ...
] = (
    (
        "vergil-aeneid:1.1",
        "corpus_lexical_recognition",
        "de",
        {"kind": "sentence", "index": 0},
        {"form": "puer", "occurrence": 1},
        "corpus-exercise-v1-e33ab85bb4d783cdb0fb1d5c82cb5a52b970d2cad738d8850a0346461d68ddaf",
        "latinitas-v2-46fc62fe1652d51b8f59034b7436c73a575d8e3c2f3423c0fbe3382dc9dd1769",
    ),
    (
        "vergil-aeneid:1.1",
        "corpus_lexical_recognition",
        "en",
        {"kind": "sentence", "index": 0},
        {"form": "puer", "occurrence": 1},
        "corpus-exercise-v1-7c0ae482564c9ad7d2c36914a644ca40ceb5022ac4049250bbb0773c96135873",
        "latinitas-v2-e651d78231c8e2e45ad16becaf69f8f36d60e072e8b347f3028f8d5ea468a668",
    ),
    (
        "vergil-aeneid:1.1",
        "corpus_guided_analysis",
        "de",
        {"kind": "sentence", "index": 1},
        {"form": "virumque", "occurrence": 0},
        "corpus-exercise-v1-4ee85d19d9e31443930e29c94511f4ab66abd9614053801f3bc736cf1539cffd",
        "latinitas-v2-0be385849ce16ec87c1c7228ee437ad93739d865f93e620929ec455f02da8d8f",
    ),
    (
        "vergil-aeneid:1.1",
        "corpus_guided_analysis",
        "en",
        {"kind": "sentence", "index": 1},
        {"form": "virumque", "occurrence": 0},
        "corpus-exercise-v1-1a5d48f1f2bb990822d279934a0aaeacf6409ec0985cad86858b6fa69909d140",
        "latinitas-v2-d8c5caf05ce3ad9df9ace99d362d39c122742f23c09cb29dcc45949babdad7bf",
    ),
    (
        "vergil-aeneid:1.2",
        "corpus_phrase_translation",
        "de",
        {"kind": "sentence", "index": 0},
        {"form": "puerum puer", "occurrence": 0},
        "corpus-exercise-v1-c78a0ee520faa0034c5584b759d3ec2e82d293391914fd32576fb881d4592686",
        "latinitas-v2-89648fbbe96f9b3f8faa310d3177d04d7e158be005364022b0dfad78de98bb4e",
    ),
    (
        "vergil-aeneid:1.2",
        "corpus_phrase_translation",
        "en",
        {"kind": "sentence", "index": 0},
        {"form": "puerum puer", "occurrence": 0},
        "corpus-exercise-v1-f1cf4ccfae18fc8e15462b9b9a48d1cbea75190ea54e65cd1d562dbd1926d07d",
        "latinitas-v2-fb508484db5f71f2316e52ffb65b998c9a7481ae164807e6b4328c31935b64a8",
    ),
    (
        "vergil-aeneid:1.2",
        "corpus_sentence_translation",
        "de",
        {"kind": "passage"},
        None,
        "corpus-exercise-v1-3e1981ed7199b6bbb5d455b883eaf4064cdc9ba7084f6f1fbebb862b773f8f10",
        "latinitas-v2-efb7bcd2bafa0d00505fe53655623a3b54a560d2da78f71b3a03ab6d2aef661d",
    ),
    (
        "vergil-aeneid:1.2",
        "corpus_sentence_translation",
        "en",
        {"kind": "passage"},
        None,
        "corpus-exercise-v1-64bd99807754512a7f576680d9bef0a0785be13460aa6c3c8ae77c403cee4be0",
        "latinitas-v2-3cf2cb42c66d39e0d5bf13689895a5b3eb199207ceb8649582553782a9e01520",
    ),
    # A folded form outside ASCII (Greek in a Latin text) hashes as raw UTF-8, not \u escapes.
    (
        "vergil-aeneid:1.1",
        "corpus_lexical_recognition",
        "en",
        {"kind": "sentence", "index": 0},
        {"form": "\u03bb\u03bf\u03b3\u03bf\u03c3", "occurrence": 0},
        "corpus-exercise-v1-f6cbbc30c405762d9107a73b2ed3e48745886d50fedd123b15c6ca2f9037ac47",
        "latinitas-v2-d355a69ab15a9fba61b4306c27212bc662fb723dca4345d307db142c279eab5f",
    ),
)


def test_corpus_exercise_vectors_reproduce_exactly() -> None:
    for passage_id, recipe, language, unit, target, object_key, latinitas_id in CORPUS_VECTORS:
        assert exercise_identity(passage_id, recipe, language, unit, target) == (object_key, latinitas_id)
    # The target language is identity input as its lowercased primary subtag only.
    passage_id, recipe, _, unit, target, object_key, latinitas_id = CORPUS_VECTORS[1]
    assert exercise_identity(passage_id, recipe, "EN-GB", unit, target) == (object_key, latinitas_id)
    assert len({vector[6] for vector in CORPUS_VECTORS}) == len(CORPUS_VECTORS)


def test_versioned_identity_vectors_reproduce_exactly() -> None:
    assert IDENTITY_VECTORS_VERSION == "note-family-v1"
    for source_identity, object_key, source_scope, expected in VECTORS:
        assert derive_latinitas_id(source_identity, object_key, source_scope=source_scope) == expected


def test_versioned_vectors_pin_scope_separation_for_colliding_local_ids() -> None:
    scoped_a = VECTORS[2][3]
    scoped_b = VECTORS[3][3]
    assert scoped_a != scoped_b
    assert not {scoped_a, scoped_b} & {vectors[3] for vectors in (VECTORS[0], VECTORS[1])}


def test_versioned_vectors_pin_transport_guid_derivation() -> None:
    assert (
        derive_anki_guid_from_latinitas_id(VECTORS[0][3])
        == "anki-guid-v1-19733201cc9d9b1c7f81a0ba58af83d3deedbc9b9c7695e12dc03a213cca4939"
    )


def test_identity_rejects_empty_parts() -> None:
    with pytest.raises(IdentityError, match="source_identity"):
        derive_latinitas_id("", "lexeme-1")
    with pytest.raises(IdentityError, match="object_key"):
        derive_latinitas_id("source-17", " ")
    with pytest.raises(IdentityError, match="source_scope"):
        derive_latinitas_id("csv-source-000001", "lexeme-1", source_scope="")
