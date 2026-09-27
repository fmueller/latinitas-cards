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
