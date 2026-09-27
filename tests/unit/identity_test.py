from pathlib import Path

import pytest

from latinitas_cards.identity import (
    IdentityError,
    ResolvedSourceIdentity,
    derive_anki_guid_from_latinitas_id,
    derive_card_semantic_key,
    derive_latinitas_id,
    resolve_source_identities,
    resolve_source_identity,
)
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.sources import CanonicalSourceRecord, SourceProvenance


def _record(
    *,
    source_kind: str = "csv",
    source_identity: str | None = None,
    note_guid: str | None = None,
) -> CanonicalSourceRecord:
    return CanonicalSourceRecord(
        source_kind=source_kind,  # type: ignore[arg-type]
        note_type="Vocabulary",
        fields={"Lemma": "amo", "Principal parts": "amo — amare"},
        provenance=SourceProvenance(source_path=Path("deck.csv"), location="row 2", note_id=41),
        source_identity=source_identity,
        note_guid=note_guid,
    )


def _profile(strategy: SourceIdentityConfig) -> DeckProfile:
    return DeckProfile.default(
        note_type="Vocabulary",
        lexical_entry_field="Lemma",
        principal_parts_field="Principal parts",
        source_identity=strategy,
    )


def test_note_identity_uses_only_source_object_key_and_scope() -> None:
    base = derive_latinitas_id("source-17", "lexeme-1")

    assert base == derive_latinitas_id("source-17", "lexeme-1")
    assert base != derive_latinitas_id("source-18", "lexeme-1")
    assert base != derive_latinitas_id("source-17", "lexeme-2")

    scoped = derive_latinitas_id("csv-source-000001", "lexeme-1", source_scope="scope-alpha")
    assert scoped != derive_latinitas_id("csv-source-000001", "lexeme-1", source_scope="scope-beta")
    assert scoped != base


def test_note_identity_is_a_stable_note_family_id_with_transport_guid() -> None:
    identity = derive_latinitas_id("source-17", "lexeme-1")

    assert identity.startswith("latinitas-v2-")
    guid = derive_anki_guid_from_latinitas_id(identity)
    assert guid.startswith("anki-guid-v1-")
    assert guid == derive_anki_guid_from_latinitas_id(identity)


def test_card_semantic_keys_are_separate_from_note_identity() -> None:
    completion = derive_card_semantic_key("principal_part_completion", "perfect_1s")
    recognition = derive_card_semantic_key("principal_part_recognition", "perfect_1s")

    assert completion == "principal_part_completion:perfect_1s"
    assert completion != recognition
    assert derive_card_semantic_key("principal_part_completion", "supine") != completion
    assert "latinitas" not in completion


def test_card_semantic_key_rejects_empty_or_delimiter_parts() -> None:
    with pytest.raises(IdentityError, match="recipe_identity"):
        derive_card_semantic_key("", "perfect_1s")
    with pytest.raises(IdentityError, match="semantic_role"):
        derive_card_semantic_key("principal_part_completion", " ")
    with pytest.raises(IdentityError, match="delimiter"):
        derive_card_semantic_key("principal:part", "perfect_1s")
    with pytest.raises(IdentityError, match="delimiter"):
        derive_card_semantic_key("principal_part_completion", "perfect:1s")


def test_source_identity_resolution_requires_explicit_csv_or_manifest_identity() -> None:
    csv_record = _record()

    with pytest.raises(IdentityError, match="stable source identity"):
        resolve_source_identity(csv_record, _profile(SourceIdentityConfig(strategy="note_guid")))

    with pytest.raises(IdentityError, match="source identity"):
        resolve_source_identity(
            csv_record,
            _profile(SourceIdentityConfig(strategy="source_id_field", field="Source ID")),
            source_scope="scope-alpha",
        )

    assert resolve_source_identity(
        csv_record,
        _profile(SourceIdentityConfig(strategy="manifest")),
        manifest_identity="manifest-17",
        source_scope="scope-alpha",
    ) == ResolvedSourceIdentity(value="manifest-17", scope="scope-alpha")


def test_csv_source_identities_require_a_persisted_scope() -> None:
    csv_record = _record(source_identity="L001")
    profile = _profile(SourceIdentityConfig(strategy="source_id_field", field="Source ID"))

    with pytest.raises(IdentityError, match="source scope"):
        resolve_source_identity(csv_record, profile)

    resolved = resolve_source_identity(csv_record, profile, source_scope="scope-alpha")
    assert resolved == ResolvedSourceIdentity(value="L001", scope="scope-alpha")
    assert derive_latinitas_id(resolved.value, "lexeme-1", source_scope=resolved.scope) == derive_latinitas_id(
        "L001", "lexeme-1", source_scope="scope-alpha"
    )


def test_native_guid_identity_never_uses_local_anki_note_id_or_scope() -> None:
    record = _record(source_kind="apkg", source_identity="native-guid", note_guid="native-guid")

    resolved = resolve_source_identity(record, _profile(SourceIdentityConfig(strategy="note_guid")))

    assert resolved == ResolvedSourceIdentity(value="native-guid", scope=None)
    assert derive_latinitas_id(resolved.value, "lexeme-1") == derive_latinitas_id("native-guid", "lexeme-1")


def test_source_identity_resolution_rejects_duplicate_stable_ids() -> None:
    records = (_record(source_identity="same-id"), _record(source_identity="same-id"))

    with pytest.raises(IdentityError, match="unique"):
        resolve_source_identities(
            records,
            _profile(SourceIdentityConfig(strategy="source_id_field", field="ID")),
            source_scope="scope-alpha",
        )
