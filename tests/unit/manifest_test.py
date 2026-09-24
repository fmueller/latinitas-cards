import os
from dataclasses import replace
from pathlib import Path

import pytest

from latinitas_cards.manifest import CsvIdentityManifest, ManifestError, reconcile_csv_manifest
from latinitas_cards.sources import CanonicalSourceRecord, read_csv_records


def _records(tmp_path: Path, name: str, rows: list[tuple[str, str]]) -> tuple[CanonicalSourceRecord, ...]:
    source = tmp_path / name
    body = "Lemma,Gloss\n" + "".join(f"{lemma},{gloss}\n" for lemma, gloss in rows)
    source.write_text(body, encoding="utf-8")
    return read_csv_records(source)


def test_idless_csv_manifest_allocates_only_with_explicit_approval_and_reconciles_reordering(
    tmp_path: Path,
) -> None:
    first = _records(tmp_path, "first.csv", [("amo", "lieben"), ("dico", "sagen")])

    pending = reconcile_csv_manifest(first, None)
    assert pending.identities_by_row == {}
    assert {item.kind for item in pending.reviews} == {"unmatched"}

    allocated = reconcile_csv_manifest(first, None, approved_allocations={0, 1})
    identities = allocated.identities_by_row
    assert len(identities) == 2
    assert allocated.reviews == ()

    reordered = _records(tmp_path, "reordered.csv", [("dico", "sagen"), ("amo", "lieben")])
    reconciled = reconcile_csv_manifest(reordered, allocated.manifest)

    assert reconciled.identities_by_row == {0: identities[1], 1: identities[0]}
    assert reconciled.reviews == ()


def test_manifest_marks_edited_rows_for_explicit_identity_reuse(tmp_path: Path) -> None:
    original = _records(tmp_path, "original.csv", [("amo", "lieben")])
    allocated = reconcile_csv_manifest(original, None, approved_allocations={0})
    identity = allocated.identities_by_row[0]

    edited = _records(tmp_path, "edited.csv", [("amo", "lieben / lieben")])
    pending = reconcile_csv_manifest(edited, allocated.manifest)

    assert pending.identities_by_row == {}
    assert any(item.kind == "edited" and item.row_index == 0 for item in pending.reviews)

    approved = reconcile_csv_manifest(edited, allocated.manifest, approved_reuse={0: identity})
    assert approved.identities_by_row == {0: identity}
    assert not any(item.kind == "edited" for item in approved.reviews)
    assert approved.manifest.entries[0].fingerprint != allocated.manifest.entries[0].fingerprint


def test_exact_duplicates_never_silently_share_or_reassign_an_identity(tmp_path: Path) -> None:
    original = _records(tmp_path, "original.csv", [("amo", "lieben")])
    allocated = reconcile_csv_manifest(original, None, approved_allocations={0})
    existing_identity = allocated.identities_by_row[0]

    duplicate_rows = _records(tmp_path, "duplicates.csv", [("amo", "lieben"), ("amo", "lieben")])
    pending = reconcile_csv_manifest(duplicate_rows, allocated.manifest)

    assert pending.identities_by_row == {}
    assert [item.kind for item in pending.reviews].count("duplicate") == 2

    approved = reconcile_csv_manifest(
        duplicate_rows,
        allocated.manifest,
        approved_reuse={0: existing_identity},
        approved_allocations={1},
    )
    assert approved.identities_by_row[0] == existing_identity
    assert approved.identities_by_row[1] != existing_identity
    assert approved.reviews == ()


def test_insertion_and_removal_are_reviewed_and_stale_snapshot_is_explicit(tmp_path: Path) -> None:
    original = _records(tmp_path, "original.csv", [("amo", "lieben"), ("dico", "sagen")])
    allocated = reconcile_csv_manifest(original, None, approved_allocations={0, 1})

    inserted = _records(
        tmp_path,
        "inserted.csv",
        [("amo", "lieben"), ("video", "sehen"), ("dico", "sagen")],
    )
    insertion = reconcile_csv_manifest(inserted, allocated.manifest)
    assert any(item.kind == "unmatched" and item.row_index == 1 for item in insertion.reviews)
    assert any(item.kind == "stale_manifest" for item in insertion.reviews)

    removed = _records(tmp_path, "removed.csv", [("amo", "lieben")])
    removal = reconcile_csv_manifest(removed, allocated.manifest)
    assert any(
        item.kind == "removed" and item.source_identity == allocated.identities_by_row[1] for item in removal.reviews
    )

    stale = replace(allocated.manifest, source_digest="not-the-current-snapshot")
    stale_result = reconcile_csv_manifest(original, stale)
    assert any(item.kind == "stale_manifest" for item in stale_result.reviews)
    assert stale_result.identities_by_row == {}
    assert not stale_result.complete

    approved = reconcile_csv_manifest(
        original,
        stale,
        approved_reuse={0: allocated.identities_by_row[0], 1: allocated.identities_by_row[1]},
    )
    assert approved.identities_by_row == allocated.identities_by_row
    assert approved.complete


def test_manifest_identity_tampering_blocks_automatic_reuse(tmp_path: Path) -> None:
    records = _records(tmp_path, "source.csv", [("amo", "lieben")])
    allocated = reconcile_csv_manifest(records, None, approved_allocations={0})
    tampered_entry = replace(allocated.manifest.entries[0], source_identity="substituted-id")
    tampered = replace(allocated.manifest, entries=(tampered_entry,))

    result = reconcile_csv_manifest(records, tampered)

    assert result.identities_by_row == {}
    assert any(item.kind == "stale_manifest" for item in result.reviews)
    assert not result.complete


def test_unresolved_removal_review_survives_manifest_save_and_reload(tmp_path: Path) -> None:
    original = _records(tmp_path, "original.csv", [("amo", "lieben"), ("dico", "sagen")])
    allocated = reconcile_csv_manifest(original, None, approved_allocations={0, 1})
    removed = _records(tmp_path, "removed.csv", [("amo", "lieben")])
    pending = reconcile_csv_manifest(removed, allocated.manifest)
    path = tmp_path / "source.csv.latinitas.json"

    pending.manifest.save(path)
    reloaded = reconcile_csv_manifest(removed, CsvIdentityManifest.load(path))

    assert any(item.kind == "removed" for item in reloaded.reviews)


def test_approved_removal_becomes_a_tombstone_without_reappearing_as_review(tmp_path: Path) -> None:
    original = _records(tmp_path, "original.csv", [("amo", "lieben"), ("dico", "sagen")])
    allocated = reconcile_csv_manifest(original, None, approved_allocations={0, 1})
    removed = _records(tmp_path, "removed.csv", [("amo", "lieben")])
    removed_identity = allocated.identities_by_row[1]

    approved = reconcile_csv_manifest(removed, allocated.manifest, approved_removals={removed_identity})

    assert approved.complete
    removed_entry = next(entry for entry in approved.manifest.entries if entry.source_identity == removed_identity)
    assert removed_entry.active is False
    assert not any(item.source_identity == removed_identity for item in approved.reviews)


def test_removal_approval_cannot_tombstone_an_identity_assigned_to_a_current_row(tmp_path: Path) -> None:
    records = _records(tmp_path, "source.csv", [("amo", "lieben")])
    allocated = reconcile_csv_manifest(records, None, approved_allocations={0})
    identity = allocated.identities_by_row[0]

    with pytest.raises(ManifestError, match="conflicts"):
        reconcile_csv_manifest(records, allocated.manifest, approved_removals={identity})


def test_manifest_save_replaces_atomically_without_destroying_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path, "source.csv", [("amo", "lieben")])
    manifest = reconcile_csv_manifest(records, None, approved_allocations={0}).manifest
    path = tmp_path / "source.csv.latinitas.json"
    path.write_text("existing manifest", encoding="utf-8")

    def fail_replace(_source: str, _destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        manifest.save(path)

    assert path.read_text(encoding="utf-8") == "existing manifest"


def test_manifest_round_trip_is_human_readable_and_preserves_tombstones(tmp_path: Path) -> None:
    records = _records(tmp_path, "source.csv", [("amo", "lieben")])
    allocated = reconcile_csv_manifest(records, None, approved_allocations={0})
    path = tmp_path / "source.csv.latinitas.json"

    allocated.manifest.save(path)
    loaded = CsvIdentityManifest.load(path)

    assert loaded == allocated.manifest
    assert '"entries"' in path.read_text(encoding="utf-8")
