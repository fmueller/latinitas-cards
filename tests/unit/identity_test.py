from pathlib import Path

import pytest

from latinitas_cards.identity import (
    IdentityError,
    derive_anki_guid,
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


def test_identity_uses_only_immutable_source_recipe_and_semantic_key() -> None:
    identity = derive_latinitas_id("source-17", "principal_part_completion", "perfect_1s")

    assert identity == derive_latinitas_id("source-17", "principal_part_completion", "perfect_1s")
    assert identity != derive_latinitas_id("source-17", "principal_part_recognition", "perfect_1s")
    assert identity != derive_latinitas_id("source-17", "principal_part_completion", "supine")

    guid = derive_anki_guid("source-17", "principal_part_completion", "perfect_1s")
    assert guid == derive_anki_guid("source-17", "principal_part_completion", "perfect_1s")
    assert guid != derive_anki_guid("source-17", "principal_part_completion", "supine")


def test_identity_rejects_empty_logical_parts() -> None:
    with pytest.raises(IdentityError, match="source_identity"):
        derive_latinitas_id("", "recipe", "exercise")

    with pytest.raises(IdentityError, match="exercise_key"):
        derive_latinitas_id("source", "recipe", " ")


def test_source_identity_resolution_requires_explicit_csv_or_manifest_identity() -> None:
    csv_record = _record()

    with pytest.raises(IdentityError, match="stable source identity"):
        resolve_source_identity(csv_record, _profile(SourceIdentityConfig(strategy="note_guid")))

    with pytest.raises(IdentityError, match="source identity"):
        resolve_source_identity(
            csv_record,
            _profile(SourceIdentityConfig(strategy="source_id_field", field="Source ID")),
        )

    assert (
        resolve_source_identity(
            csv_record,
            _profile(SourceIdentityConfig(strategy="manifest")),
            manifest_identity="manifest-17",
        )
        == "manifest-17"
    )


def test_source_identity_resolution_uses_native_guid_not_local_anki_note_id() -> None:
    record = _record(source_kind="apkg", source_identity="native-guid", note_guid="native-guid")

    assert resolve_source_identity(record, _profile(SourceIdentityConfig(strategy="note_guid"))) == "native-guid"


def test_source_identity_resolution_rejects_duplicate_stable_ids() -> None:
    records = (_record(source_identity="same-id"), _record(source_identity="same-id"))

    with pytest.raises(IdentityError, match="unique"):
        resolve_source_identities(records, _profile(SourceIdentityConfig(strategy="source_id_field", field="ID")))
