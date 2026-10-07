import csv
import errno
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from latinitas_cards.checkpoint import PriorExportCheckpoint
from latinitas_cards.identity import derive_latinitas_id
from latinitas_cards.legacy_transition import LegacyTransitionError
from latinitas_cards.manifest import CsvIdentityManifest, allocate_source_scope, reconcile_csv_manifest
from latinitas_cards.notes import TAGS_CSV_COLUMN
from latinitas_cards.preview_export import (
    PrincipalPartExportError,
    PrincipalPartExportResult,
    deterministic_csv_bytes,
    prepare_principal_part_export,
    write_principal_part_csv,
)
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig, load_profile, resolve_profile
from latinitas_cards.sources import read_csv_records

FIXTURE = Path(__file__).parents[1] / "fixtures" / "representative-university-latin.apkg"

EXPECTED_EXPORT_COLUMNS = (
    "LatinitasID",
    "Lemma",
    "Principal Parts",
    "Meaning",
    "Tags",
    "Source ID",
    "Source Scope",
    "Source Kind",
    "Source Location",
    "Source Path",
    "Note Schema",
    "Generator",
    "Profile",
    "CompletionPresentEnabled",
    "CompletionPresentPrompt",
    "CompletionPresentAnswer",
    "CompletionInfinitiveEnabled",
    "CompletionInfinitivePrompt",
    "CompletionInfinitiveAnswer",
    "CompletionPerfectEnabled",
    "CompletionPerfectPrompt",
    "CompletionPerfectAnswer",
    "CompletionPPPEnabled",
    "CompletionPPPPrompt",
    "CompletionPPPAnswer",
    "CompletionSupineEnabled",
    "CompletionSupinePrompt",
    "CompletionSupineAnswer",
    "RecognitionPresentEnabled",
    "RecognitionPresentPrompt",
    "RecognitionPresentAnswer",
    "RecognitionInfinitiveEnabled",
    "RecognitionInfinitivePrompt",
    "RecognitionInfinitiveAnswer",
    "RecognitionPerfectEnabled",
    "RecognitionPerfectPrompt",
    "RecognitionPerfectAnswer",
    "RecognitionPPPEnabled",
    "RecognitionPPPPrompt",
    "RecognitionPPPAnswer",
    "RecognitionSupineEnabled",
    "RecognitionSupinePrompt",
    "RecognitionSupineAnswer",
)


def _profile(
    *,
    source_identity: SourceIdentityConfig | None = None,
    tags: tuple[str, ...] = ("latinitas", "principal-parts"),
) -> DeckProfile:
    return DeckProfile.default(
        note_type="Latin Vocabulary",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        source_identity=source_identity or SourceIdentityConfig(strategy="source_id_field", field="Stable ID"),
        principal_part_roles=("present_infinitive", "present_1s", "perfect_1s", "supine"),
        separators=(",",),
        generated_note_type="Latinitas Principal Parts",
        target_deck="Latin::Latinitas::Review",
        tags=tags,
        selected_recipes=("principal_part_recognition", "principal_part_completion"),
    )


def _representative_profile() -> DeckProfile:
    return DeckProfile.default(
        note_type="Representative Latin Vocabulary",
        lexical_entry_field="Entry",
        principal_parts_field="Construction hints",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="note_guid"),
        principal_part_roles=(
            "present_infinitive",
            "present_1s",
            "perfect_1s",
            "perfect_passive_participle",
        ),
        separators=(",",),
        generated_note_type="Latinitas Principal Parts",
        target_deck="Latin::Latinitas::Review",
        tags=("latinitas", "provenance"),
        selected_recipes=("principal_part_completion", "principal_part_recognition"),
    )


def _write_source(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as source:
        writer = csv.writer(source, lineterminator="\n")
        writer.writerow(("Stable ID", "Lemma", "Forms", "German gloss"))
        writer.writerows(rows)


def _parse_export(payload: bytes) -> tuple[list[str], list[list[str]], str]:
    text = payload.decode("utf-8")
    lines = text.splitlines(keepends=True)
    metadata = "".join(lines[:6])
    header = lines[5].removeprefix("#columns:").removesuffix("\n").split(",")
    reader = csv.reader(io.StringIO("".join(lines[6:])), lineterminator="\n")
    return header, list(reader), metadata


def _write_tagged_package(package_path: Path, member_name: str, *, tags_by_guid: dict[str, str] | None = None) -> None:
    work = package_path.with_suffix("")
    work.mkdir(exist_ok=True)
    database_path = work / "collection.anki2"
    database_path.unlink(missing_ok=True)
    con = sqlite3.connect(database_path)
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, tags TEXT, flds TEXT)")
    con.execute("CREATE TABLE notetypes (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE fields (ntid INTEGER, ord INTEGER, name TEXT)")
    con.execute("INSERT INTO notetypes (id, name) VALUES (10, 'Latin Vocabulary')")
    con.executemany(
        "INSERT INTO fields (ntid, ord, name) VALUES (?, ?, ?)",
        ((10, 0, "Lemma"), (10, 1, "Forms"), (10, 2, "German gloss")),
    )
    default_tags = {
        "guid-a": "latin verb::irregular Vokabeln-Übung",
        "guid-b": "grammar grammar latinitas",
        "guid-c": "",
    }
    tags = {**default_tags, **(tags_by_guid or {})}
    con.executemany(
        "INSERT INTO notes (id, guid, mid, tags, flds) VALUES (?, ?, ?, ?, ?)",
        (
            (1, "guid-a", 10, tags["guid-a"], "dīcō\x1fdīcere, dīcō, dīxī, dictum\x1fsagen"),
            (2, "guid-b", 10, tags["guid-b"], "amāre\x1famāre, amō, amāvī, amātum\x1flieben"),
            (3, "guid-c", 10, tags["guid-c"], "ferō\x1fferre, ferō, tulī, lātum\x1ftragen"),
        ),
    )
    con.commit()
    con.close()
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.write(database_path, member_name)


def _tagged_package_profile() -> DeckProfile:
    return DeckProfile.default(
        note_type="Latin Vocabulary",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="note_guid"),
        principal_part_roles=("present_infinitive", "present_1s", "perfect_1s", "supine"),
        separators=(",",),
        generated_note_type="Latinitas Principal Parts",
        target_deck="Latin::Latinitas::Review",
        tags=("latinitas", "latin"),
        selected_recipes=("principal_part_completion", "principal_part_recognition"),
    )


def _manifest_profile() -> DeckProfile:
    return _profile(source_identity=SourceIdentityConfig(strategy="manifest"))


def _committed_scoped_export(
    tmp_path: Path,
    *,
    rows: list[tuple[str, str, str, str]] | None = None,
    profile: DeckProfile | None = None,
    source_name: str = "source.csv",
    output_name: str = "generated.csv",
    approve: bool = True,
) -> tuple[PrincipalPartExportResult, Path, Path, Path]:
    source = tmp_path / source_name
    resolved_profile = profile or _profile()
    _write_source(source, rows or [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    state = Path(f"{source}.latinitas.json")
    result = prepare_principal_part_export(
        source,
        resolved_profile,
        manifest_path=state,
        approve_new_scope=approve,
    )
    output = tmp_path / output_name
    write_principal_part_csv(result, output)
    return result, source, state, output


def test_preview_result_reports_representative_objects_and_structured_counts(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(
        source,
        [
            ("entry-β", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen\n& prüfen"),
            ("entry-ambiguous", "ferō", "ferre, ferō, tulī", "tragen"),
            ("entry-unsupported", "videō", "vidēre; videō; vīdī; vīsum", "sehen"),
        ],
    )

    result = prepare_principal_part_export(
        source,
        _profile(),
        manifest_path=state,
        approve_new_scope=True,
    )

    assert result.generated_count == 1
    assert result.skipped_count == 2
    assert result.ambiguous_count == 2
    assert result.generation.notes[0].provenance.source_path is None
    assert result.generation.notes[0].provenance.source_identity == "entry-β"
    assert any(skip.code == "unmarked_omission" and skip.status == "ambiguous" for skip in result.generation.skips)
    assert any(skip.code == "separator_mismatch" for skip in result.generation.skips)


def test_read_only_preview_requests_scope_confirmation_without_minting_ids_or_writing_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])

    pending = prepare_principal_part_export(source, _profile(), manifest_path=state)

    assert pending.scope_pending
    assert pending.source_scope is None
    assert pending.generation.notes == ()
    assert any(review.kind == "scope_confirmation" for review in pending.manifest_reviews)
    assert not state.exists()
    with pytest.raises(PrincipalPartExportError, match="scope"):
        deterministic_csv_bytes(pending)
    assert not state.exists()


def test_unapproved_export_writes_neither_output_nor_scope_state(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    output = tmp_path / "generated.csv"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])

    pending = prepare_principal_part_export(source, _profile(), manifest_path=state)

    with pytest.raises(PrincipalPartExportError, match="scope"):
        write_principal_part_csv(pending, output)
    assert not output.exists()
    assert not state.exists()


def test_first_approved_export_commits_scope_assignments_and_output_together(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    output = tmp_path / "generated.csv"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    profile = _profile()

    approved = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approve_new_scope=True,
    )

    assert not approved.scope_pending
    assert approved.source_scope
    assert approved.source_scope.startswith("scope-")
    assert not state.exists()
    write_principal_part_csv(approved, output)

    assert output.exists()
    persisted = CsvIdentityManifest.load(state)
    assert persisted.source_scope == approved.source_scope
    assert persisted.schema_version == 2

    retry = prepare_principal_part_export(source, profile, manifest_path=state)
    assert not retry.scope_pending
    assert retry.source_scope == approved.source_scope
    assert deterministic_csv_bytes(retry) == deterministic_csv_bytes(approved)
    assert {note.latinitas_id for note in retry.generation.notes} == {
        note.latinitas_id for note in approved.generation.notes
    }


def test_independent_sources_with_identical_content_and_local_ids_never_share_ids(tmp_path: Path) -> None:
    amo_dir = tmp_path / "amo"
    fero_dir = tmp_path / "fero"
    amo_dir.mkdir()
    fero_dir.mkdir()
    identical_rows = [
        ("ignored-a", "ferō", "ferre, ferō, tulī, lātum", "tragen"),
        ("ignored-b", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen"),
    ]

    amo_ids: dict[str, str] = {}
    fero_ids: dict[str, str] = {}
    for directory, ids in ((amo_dir, amo_ids), (fero_dir, fero_ids)):
        source = directory / "source.csv"
        state = Path(f"{source}.latinitas.json")
        _write_source(source, identical_rows)
        result = prepare_principal_part_export(
            source,
            _manifest_profile(),
            manifest_path=state,
            approved_allocations={0, 1},
            approve_new_scope=True,
        )
        write_principal_part_csv(result, directory / "generated.csv")
        persisted = CsvIdentityManifest.load(state)
        assert persisted.source_scope
        ids.update(
            {
                entry.source_identity: note.latinitas_id
                for entry, note in zip(
                    sorted(persisted.entries, key=lambda item: item.source_identity),
                    sorted(result.generation.notes, key=lambda note: note.provenance.source_identity or ""),
                    strict=True,
                )
            }
        )

    assert set(amo_ids) == set(fero_ids) == {"csv-source-000001", "csv-source-000002"}
    assert amo_ids != fero_ids
    assert len([*amo_ids.values(), *fero_ids.values()]) == len({*amo_ids.values(), *fero_ids.values()})


def test_moving_or_copying_persisted_state_keeps_the_same_scope_and_ids(tmp_path: Path) -> None:
    result, source, state, _output = _committed_scoped_export(tmp_path)

    moved = tmp_path / "moved"
    moved.mkdir()
    moved_source = moved / "renamed.csv"
    moved_state = Path(f"{moved_source}.latinitas.json")
    shutil.copyfile(source, moved_source)
    shutil.copyfile(state, moved_state)

    copied = prepare_principal_part_export(moved_source, result.profile, manifest_path=moved_state)

    assert copied.source_scope == result.source_scope
    assert {note.latinitas_id for note in copied.generation.notes} == {
        note.latinitas_id for note in result.generation.notes
    }


def test_explicit_csv_ids_are_scoped_and_require_bootstrap(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(source, [("L001", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])

    pending = prepare_principal_part_export(source, _profile(), manifest_path=state)
    assert pending.scope_pending
    assert pending.generation.notes == ()
    assert not state.exists()

    scoped = prepare_principal_part_export(
        source,
        _profile(),
        manifest_path=state,
        approve_new_scope=True,
    )
    note = scoped.generation.notes[0]
    assert note.provenance.source_identity == "L001"
    assert note.provenance.source_scope == scoped.source_scope
    assert note.latinitas_id != derive_latinitas_id("L001", note.object_key)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_source = other_dir / "source.csv"
    _write_source(other_source, [("L001", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    other = prepare_principal_part_export(
        other_source,
        _profile(),
        manifest_path=Path(f"{other_source}.latinitas.json"),
        approve_new_scope=True,
    )
    assert other.generation.notes[0].latinitas_id != note.latinitas_id


def test_legacy_unscoped_manifest_requires_explicit_fresh_start(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    legacy_state = {
        "schema_version": 1,
        "source_columns": ["Stable ID", "Lemma", "Forms", "German gloss"],
        "source_digest": "",
        "mapping_digest": "",
        "next_source_number": 2,
        "entries": [
            {
                "source_identity": "csv-source-000001",
                "fingerprint": "legacy-fingerprint",
                "last_row_index": 0,
                "active": True,
            }
        ],
    }
    state.write_text(
        json.dumps(legacy_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    blocked = prepare_principal_part_export(source, _manifest_profile(), manifest_path=state)
    assert blocked.scope_pending
    assert blocked.generation.notes == ()
    assert any("unscoped" in review.message or "fresh start" in review.message for review in blocked.manifest_reviews)

    fresh = prepare_principal_part_export(
        source,
        _manifest_profile(),
        manifest_path=state,
        approved_reuse={0: "csv-source-000001"},
        approve_new_scope=True,
    )
    assert fresh.source_scope
    assert fresh.manifest_reviews == ()
    write_principal_part_csv(fresh, tmp_path / "generated.csv")
    persisted = CsvIdentityManifest.load(state)
    assert persisted.source_scope == fresh.source_scope


def test_prepare_rejects_a_generated_note_type_declared_legacy_before_any_work(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    profile = _profile()

    with pytest.raises(LegacyTransitionError) as rejected:
        prepare_principal_part_export(
            source,
            profile,
            legacy_note_types=(profile.generated_note_type,),
        )

    assert "legacy note model" in str(rejected.value)
    assert not Path(f"{source}.latinitas.json").exists()
    assert not Path(f"{source}.latinitas-cards.json").exists()

    with pytest.raises(LegacyTransitionError):
        prepare_principal_part_export(
            tmp_path / "missing.csv",
            profile,
            legacy_note_types=(profile.generated_note_type,),
        )


def test_prepare_allows_export_when_declared_legacy_types_do_not_match(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    profile = _profile()

    result = prepare_principal_part_export(
        source,
        profile,
        legacy_note_types=("Latinitas Legacy Exercise", "Legacy Split"),
        approve_new_scope=True,
    )

    assert not result.scope_pending
    assert result.source_scope


def test_padded_generated_note_type_profiles_cannot_bypass_the_declared_legacy_guard(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    raw = _profile().to_machine_readable()
    raw["generated_note_type"] = "Latinitas Principal Parts  "
    profile_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(LegacyTransitionError):
        prepare_principal_part_export(
            source,
            load_profile(profile_path),
            legacy_note_types=("Latinitas Principal Parts",),
        )

    with pytest.raises(LegacyTransitionError):
        prepare_principal_part_export(
            source,
            resolve_profile(load_profile(profile_path), {"generated_note_type": " Latinitas Principal Parts "}),
            legacy_note_types=("Latinitas Principal Parts",),
        )


def test_legacy_unscoped_manifest_migration_preserves_matching_row_assignments(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    output = tmp_path / "generated.csv"
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    legacy = reconcile_csv_manifest(
        read_csv_records(source),
        None,
        approved_allocations={0},
    ).manifest
    assert legacy.schema_version == 1 and legacy.source_scope == ""
    state.write_text(legacy.to_json(), encoding="utf-8")

    migrated = prepare_principal_part_export(
        source,
        _manifest_profile(),
        manifest_path=state,
        approve_new_scope=True,
    )

    assert not migrated.scope_pending
    assert migrated.manifest_reviews == ()
    assert {note.provenance.source_identity for note in migrated.generation.notes} == {"csv-source-000001"}
    write_principal_part_csv(migrated, output)
    persisted = CsvIdentityManifest.load(state)
    assert persisted.source_scope == migrated.source_scope
    assert {entry.source_identity for entry in persisted.entries} == {"csv-source-000001"}


def test_anki_source_tags_reach_the_object_note_and_the_serialized_tags_column(tmp_path: Path) -> None:
    for package_name in ("tagged-package.apkg", "tagged-package.colpkg"):
        package = tmp_path / package_name
        _write_tagged_package(package, "collection.anki2")
        original_digest = hashlib.sha256(package.read_bytes()).digest()
        profile = _tagged_package_profile()

        first = deterministic_csv_bytes(prepare_principal_part_export(package, profile, approve_fresh_import=True))
        second = deterministic_csv_bytes(prepare_principal_part_export(package, profile, approve_fresh_import=True))

        assert first == second
        assert hashlib.sha256(package.read_bytes()).digest() == original_digest
        header, rows, _metadata = _parse_export(first)
        assert header == list(EXPECTED_EXPORT_COLUMNS)
        tags_index = header.index("Tags")
        assert tags_index + 1 == TAGS_CSV_COLUMN == 5
        expected_tags_by_source = {
            "guid-a": "latin verb::irregular Vokabeln-Übung latinitas",
            "guid-b": "grammar latinitas latin",
            "guid-c": "latinitas latin",
        }
        tags_by_source: dict[str, str] = {}
        for row in rows:
            source_id = row[header.index("Source ID")]
            tags_by_source[source_id] = row[tags_index]
        assert tags_by_source == expected_tags_by_source
        assert len(rows) == 3
        assert all(row[header.index("Source Scope")] == "" for row in rows)


def test_untagged_anki_source_keeps_configured_tags_only_in_serialized_output(tmp_path: Path) -> None:
    package = tmp_path / "untagged.apkg"
    _write_tagged_package(package, "collection.anki2", tags_by_guid={"guid-a": "", "guid-b": "", "guid-c": ""})
    profile = _tagged_package_profile()

    payload = deterministic_csv_bytes(prepare_principal_part_export(package, profile, approve_fresh_import=True))

    _header, rows, _metadata = _parse_export(payload)
    assert {row[4] for row in rows} == {"latinitas latin"}


def test_invalid_source_tags_block_export_with_actionable_skip(tmp_path: Path) -> None:
    package = tmp_path / "invalid-tags.apkg"
    _write_tagged_package(package, "collection.anki2", tags_by_guid={"guid-a": "latin bad\x9btag"})
    profile = _tagged_package_profile()

    result = prepare_principal_part_export(package, profile, approve_fresh_import=True)

    skip_codes = {skip.code for skip in result.generation.skips}
    assert "invalid_source_tags" in skip_codes
    invalid_skip = next(skip for skip in result.generation.skips if skip.code == "invalid_source_tags")
    assert invalid_skip.source_identity == "guid-a"
    assert invalid_skip.source_location == "note 1"
    assert "bad" not in invalid_skip.message
    assert {note.provenance.source_identity for note in result.generation.notes} == {"guid-b", "guid-c"}
    _header, rows, _metadata = _parse_export(deterministic_csv_bytes(result))
    assert {row[4] for row in rows} == {"grammar latinitas latin", "latinitas latin"}


def test_markup_unsafe_inherited_tags_never_reach_the_serialized_tags_column(tmp_path: Path) -> None:
    package = tmp_path / "hostile-tags.apkg"
    _write_tagged_package(
        package,
        "collection.anki2",
        tags_by_guid={"guid-a": "<img/src=x/onerror=alert(1)> latin"},
    )
    profile = _tagged_package_profile()

    result = prepare_principal_part_export(package, profile, approve_fresh_import=True)
    payload = deterministic_csv_bytes(result).decode("utf-8")

    assert {note.provenance.source_identity for note in result.generation.notes} == {"guid-b", "guid-c"}
    hostile_skip = next(skip for skip in result.generation.skips if skip.code == "invalid_source_tags")
    assert hostile_skip.source_identity == "guid-a"
    assert "onerror" not in hostile_skip.message
    assert "<img" not in payload
    assert "onerror" not in payload


def test_csv_export_is_utf8_deterministic_and_uses_anki_import_metadata(tmp_path: Path) -> None:
    profile = _profile(tags=("latinitas", "provenance-β"))
    result, source, state, _output = _committed_scoped_export(tmp_path, profile=profile)

    first = deterministic_csv_bytes(result)
    repeat = deterministic_csv_bytes(prepare_principal_part_export(source, profile, manifest_path=state))

    assert first == repeat
    assert first.startswith(b"#separator:Comma\n")
    assert "#html:true\n" in first.decode("utf-8")
    assert "#notetype:Latinitas Principal Parts\n" in first.decode("utf-8")
    assert "#deck:Latin::Latinitas::Review\n" in first.decode("utf-8")
    assert "#tags column:5\n" in first.decode("utf-8")
    assert f"#columns:{','.join(EXPECTED_EXPORT_COLUMNS)}\n" in first.decode("utf-8")
    header, rows, metadata = _parse_export(first)
    assert metadata.startswith("#separator:Comma\n#html:true\n")
    assert header == list(EXPECTED_EXPORT_COLUMNS)
    assert len(rows) == 1
    assert rows[0][0].startswith("latinitas-v2-")
    assert rows[0][4] == "latinitas provenance-β"
    assert rows[0][header.index("Source ID")] == "entry-1"
    assert rows[0][header.index("Source Scope")] == result.source_scope
    assert rows[0][header.index("Source Kind")] == "csv"
    assert rows[0][header.index("Note Schema")]
    assert rows[0][header.index("Generator")]
    assert rows[0][header.index("Profile")].startswith("profile-sha256:")
    assert "sagen" in first.decode("utf-8")
    assert "source.csv" not in first.decode("utf-8")


def test_csv_export_omits_the_user_owned_personal_notes_column(tmp_path: Path) -> None:
    result, _source, _state, _output = _committed_scoped_export(tmp_path)

    payload = deterministic_csv_bytes(result)
    text = payload.decode("utf-8")
    header, rows, _ = _parse_export(payload)

    assert "Personal Notes" not in text
    assert header[-1] == "RecognitionSupineAnswer"
    assert header[13] == "CompletionPresentEnabled"
    assert len(header) == 43
    assert all(len(row) == len(header) for row in rows)


def test_csv_export_escapes_source_identity_for_html_import(tmp_path: Path) -> None:
    result, _source, _state, _output = _committed_scoped_export(
        tmp_path,
        rows=[("<img src=x onerror=alert(1)>", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")],
    )

    text = deterministic_csv_bytes(result).decode("utf-8")

    assert "&lt;img src=x onerror=alert(1)&gt;" in text
    assert "<img src=x onerror=alert(1)>" not in text


def test_csv_export_renders_source_html_as_safe_readable_knowledge(tmp_path: Path) -> None:
    result, _source, state, _output = _committed_scoped_export(
        tmp_path,
        rows=[
            (
                "entry-html",
                "<i>dīcō</i>",
                "dīcere, dīcō, dīxī, dictum",
                "erste Bedeutung<div>zweite <b>Bedeutung</b>&lt;br&gt;dritte</div>"
                "<script>alert(1)</script><img src=x onerror=alert(2)>",
            )
        ],
    )
    source_before = _source.read_bytes()

    first = deterministic_csv_bytes(result)
    repeat = deterministic_csv_bytes(prepare_principal_part_export(_source, result.profile, manifest_path=state))

    assert first == repeat
    text = first.decode("utf-8")
    header, rows, _metadata = _parse_export(first)
    lemma_index = header.index("Lemma")
    parts_index = header.index("Principal Parts")
    meaning_index = header.index("Meaning")

    assert rows[0][lemma_index] == "dīcō"
    assert "<strong>Präsens, 1. Person Singular:</strong> dīcō" in rows[0][parts_index]
    assert rows[0][meaning_index] == "erste Bedeutung<br>zweite Bedeutung<br>dritte"
    assert "&lt;div&gt;" not in text
    assert "&lt;b&gt;" not in text
    assert "&lt;br&gt;" not in text
    assert "&lt;i&gt;" not in text
    assert "alert(1)" not in text
    assert "alert(2)" not in text
    assert "<script" not in text
    assert "<img" not in text
    assert _source.read_bytes() == source_before


def test_csv_export_encodes_unsafe_controls_without_removing_newlines(tmp_path: Path) -> None:
    result, _source, _state, _output = _committed_scoped_export(
        tmp_path,
        rows=[("entry-\x1b[31m\x9b", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen\x00\nprüfen")],
    )

    text = deterministic_csv_bytes(result).decode("utf-8")

    assert "\x1b" not in text
    assert "\x00" not in text
    assert "\x9b" not in text
    assert "\\x1b[31m\\x9b" in text
    assert "\\x00" in text
    assert "sagen" in text and "prüfen" in text


def _unsupported_role_profile() -> DeckProfile:
    return _profile().apply_overrides({"principal_parts": {"roles": ["stem_a", "stem_b"], "separators": [" — "]}})


def test_export_reports_source_entries_objects_cards_and_zero_eligible_notes_separately(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(
        source,
        [
            ("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen"),
            ("entry-2", "sum", "sum, esse, fuī, ", "sein"),
            ("entry-3", "ferō", "ferre, ferō, tulī", "tragen"),
        ],
    )

    result = prepare_principal_part_export(
        source,
        _profile(),
        manifest_path=state,
        approve_new_scope=True,
    )

    assert result.source_entry_count == 3
    assert result.object_count == 2
    assert result.exported_note_count == 2
    assert result.card_count == 12
    assert result.zero_card_note_count == 0
    assert result.skipped_count == 1
    assert result.generation.generated_warning_count == 2
    assert any(skip.code == "omitted_principal_part" for skip in result.generation.skips)
    assert any(skip.code == "unmarked_omission" for skip in result.generation.skips)


def test_csv_export_omits_zero_eligible_notes_instead_of_blank_native_cards(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(
        source,
        [
            ("entry-1", "ferō", "ferō — ferre", "tragen"),
            ("entry-2", "sum", "sum — esse", "sein"),
        ],
    )

    result = prepare_principal_part_export(
        source,
        _unsupported_role_profile(),
        manifest_path=state,
        approve_new_scope=True,
    )

    assert result.source_entry_count == 2
    assert result.object_count == 2
    assert result.zero_card_note_count == 2
    assert result.exported_note_count == 0
    assert result.card_count == 0
    assert {note.provenance.source_identity for note in result.zero_card_notes} == {"entry-1", "entry-2"}
    header, rows, _metadata = _parse_export(deterministic_csv_bytes(result))
    assert rows == []
    assert header == list(EXPECTED_EXPORT_COLUMNS)


def test_exported_rows_always_carry_at_least_one_enabled_card_guard(tmp_path: Path) -> None:
    result, _source, _state, _output = _committed_scoped_export(
        tmp_path,
        rows=[("entry-1", "sum", "sum, esse, fuī, ", "sein")],
    )

    header, rows, _metadata = _parse_export(deterministic_csv_bytes(result))

    enabled_values = [rows[0][index] for index, name in enumerate(header) if name.endswith("Enabled")]
    assert enabled_values.count("1") == 6
    assert set(enabled_values) == {"1", ""}


@pytest.mark.parametrize("suffix", [".apkg", ".colpkg"])
def test_package_preview_and_export_leave_source_bytes_unchanged(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"representative{suffix}"
    output = tmp_path / "generated.csv"
    shutil.copyfile(FIXTURE, source)
    source_before = source.read_bytes()

    result = prepare_principal_part_export(source, _representative_profile(), approve_fresh_import=True)
    assert result.generated_count > 0
    assert not result.scope_pending
    assert source.read_bytes() == source_before

    write_principal_part_csv(result, output)

    assert output.exists()
    assert source.read_bytes() == source_before


def test_managed_text_and_tags_updates_keep_one_row_per_object_identity(tmp_path: Path) -> None:
    result, source, state, _output = _committed_scoped_export(tmp_path, profile=_profile(tags=("old",)))

    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen; aussprechen")])
    revised = prepare_principal_part_export(source, _profile(tags=("new", "html-v2")), manifest_path=state)

    original_ids = {note.latinitas_id for note in result.generation.notes}
    revised_ids = [note.latinitas_id for note in revised.generation.notes]
    assert set(revised_ids) == original_ids
    assert len(revised_ids) == len(set(revised_ids)) == 1
    assert deterministic_csv_bytes(result) != deterministic_csv_bytes(revised)


def test_idless_csv_requires_scope_then_explicit_allocation_and_reuses_ids_after_reordering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    profile = _manifest_profile()
    _write_source(
        source,
        [
            ("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen"),
            ("ignored-b", "ferō", "ferre, ferō, tulī, lātum", "tragen"),
        ],
    )

    scope_pending = prepare_principal_part_export(source, profile, manifest_path=state)
    assert scope_pending.scope_pending
    assert scope_pending.generation.notes == ()

    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0, 1},
        approve_new_scope=True,
    )
    output = tmp_path / "generated.csv"
    write_principal_part_csv(allocated, output)
    assert state.exists()
    allocated_ids = {note.provenance.source_identity: note.latinitas_id for note in allocated.generation.notes}

    _write_source(
        source,
        [
            ("ignored-b", "ferō", "ferre, ferō, tulī, lātum", "tragen"),
            ("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen"),
        ],
    )
    reordered = prepare_principal_part_export(source, profile, manifest_path=state)
    reordered_ids = {note.provenance.source_identity: note.latinitas_id for note in reordered.generation.notes}
    assert reordered.manifest_reviews == ()
    assert reordered.source_scope == allocated.source_scope
    assert reordered_ids == allocated_ids


def test_manifest_approvals_are_rejected_without_a_scope_bootstrap(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    profile = _manifest_profile()
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])

    with pytest.raises(PrincipalPartExportError, match="scope"):
        prepare_principal_part_export(
            source,
            profile,
            manifest_path=state,
            approved_allocations={0},
        )


@pytest.mark.parametrize(
    "approval_kwargs",
    [
        {"approved_reuse": {0: "L001"}},
        {"approved_allocations": {0}},
        {"approved_removals": ("L001",)},
    ],
)
def test_row_approvals_are_rejected_for_explicit_csv_id_sources(
    tmp_path: Path,
    approval_kwargs: dict[str, object],
) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(source, [("L001", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    committed = prepare_principal_part_export(source, _profile(), manifest_path=state, approve_new_scope=True)
    write_principal_part_csv(committed, tmp_path / "bootstrap.csv")

    with pytest.raises(PrincipalPartExportError, match="manifest identity review"):
        prepare_principal_part_export(
            source,
            _profile(),
            manifest_path=state,
            **approval_kwargs,  # type: ignore[arg-type]
        )


def test_scope_bootstrap_error_names_the_cli_flag(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])

    with pytest.raises(PrincipalPartExportError, match="--approve-scope"):
        prepare_principal_part_export(source, _manifest_profile(), manifest_path=state, approved_allocations={0})


@pytest.mark.parametrize("manifest_exists", [False, True])
def test_manifest_commit_failure_restores_existing_or_absent_output_pair(
    tmp_path: Path,
    manifest_exists: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    profile = _manifest_profile()
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])

    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0},
        approve_new_scope=True,
    )
    if manifest_exists:
        write_principal_part_csv(allocated, output)
        output_before = output.read_bytes()
        state_before = state.read_bytes()
        allocated = prepare_principal_part_export(source, profile, manifest_path=state)
    else:
        output.write_bytes(b"prior output")
        output_before = output.read_bytes()
        state_before = None

    original_replace = os.replace
    failed = False

    def fail_manifest_once(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal failed
        if not failed and not isinstance(destination_name, bytes) and Path(destination_name) == state:
            failed = True
            raise OSError("simulated manifest commit failure")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", fail_manifest_once)

    with pytest.raises(PrincipalPartExportError, match="No output or committed state was changed"):
        write_principal_part_csv(allocated, output)

    assert failed
    assert output.read_bytes() == output_before
    if state_before is None:
        assert not state.exists()
    else:
        assert state.read_bytes() == state_before


def test_manifest_rollback_failure_without_backups_reports_affected_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    profile = _manifest_profile()
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0},
        approve_new_scope=True,
    )

    original_replace = os.replace
    manifest_failed = False

    def fail_manifest_once(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal manifest_failed
        if not manifest_failed and not isinstance(destination_name, bytes) and Path(destination_name) == state:
            manifest_failed = True
            raise OSError("simulated manifest commit failure")
        original_replace(source_name, destination_name)

    original_unlink = Path.unlink

    def fail_output_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == output:
            raise OSError("simulated output rollback failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", fail_manifest_once)
    monkeypatch.setattr(Path, "unlink", fail_output_unlink)

    with pytest.raises(PrincipalPartExportError) as error:
        write_principal_part_csv(allocated, output)

    message = str(error.value)
    assert manifest_failed
    assert output.exists()
    assert not state.exists()
    assert "Recovery is required" in message
    assert str(output) in message
    assert "No output or committed state was changed" not in message


def test_manifest_rollback_failure_retains_backup_for_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    profile = _manifest_profile()
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0},
        approve_new_scope=True,
    )
    output.write_bytes(b"prior output")

    original_replace = os.replace
    manifest_failed = False
    rollback_failed = False

    def fail_commit_and_rollback(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal manifest_failed, rollback_failed
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        source_text = os.fsdecode(source_name)
        if destination == state and not manifest_failed:
            manifest_failed = True
            raise OSError("simulated manifest commit failure")
        if manifest_failed and destination == output and ".backup." in source_text:
            rollback_failed = True
            raise OSError("simulated rollback failure")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", fail_commit_and_rollback)

    with pytest.raises(PrincipalPartExportError, match="Recovery is required"):
        write_principal_part_csv(allocated, output)

    assert manifest_failed
    assert rollback_failed
    assert list(tmp_path.glob(".generated.csv.backup.*"))


def test_manifest_rollback_failure_reports_manifest_destination_and_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    profile = _manifest_profile()
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0},
        approve_new_scope=True,
    )
    write_principal_part_csv(allocated, output)
    output_before = output.read_bytes()
    allocated = prepare_principal_part_export(source, profile, manifest_path=state)

    original_replace = os.replace
    manifest_failed = False
    rollback_failed = False

    def fail_manifest_commit_and_rollback(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal manifest_failed, rollback_failed
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        source_text = os.fsdecode(source_name)
        if destination == state and not manifest_failed:
            manifest_failed = True
            raise OSError("simulated manifest commit failure")
        if destination == state and manifest_failed and ".backup." in source_text:
            rollback_failed = True
            raise OSError("simulated manifest rollback failure")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", fail_manifest_commit_and_rollback)

    with pytest.raises(PrincipalPartExportError) as error:
        write_principal_part_csv(allocated, output)

    backup_paths = list(tmp_path.glob(f".{state.name}.backup.*"))
    message = str(error.value)
    assert manifest_failed
    assert rollback_failed
    assert output.read_bytes() == output_before
    assert not state.exists()
    assert backup_paths
    assert str(state) in message
    assert str(backup_paths[0]) in message
    assert "Recovery is required" in message


def test_state_committed_by_another_process_after_prepare_blocks_the_export(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    output = tmp_path / "generated.csv"
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    profile = _manifest_profile()
    approved = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0},
        approve_new_scope=True,
    )

    competing_scope = allocate_source_scope()
    competing = CsvIdentityManifest.scoped(("Stable ID", "Lemma", "Forms", "German gloss"), competing_scope)
    state.write_text(competing.to_json(), encoding="utf-8")
    competing_bytes = state.read_bytes()

    with pytest.raises(PrincipalPartExportError, match="identity state changed"):
        write_principal_part_csv(approved, output)

    assert not output.exists()
    assert state.read_bytes() == competing_bytes
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.backup.*")) == []


def test_state_removed_after_prepare_blocks_the_export(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    state = Path(f"{source}.latinitas.json")
    output = tmp_path / "generated.csv"
    profile = _manifest_profile()
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    committed = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0},
        approve_new_scope=True,
    )
    write_principal_part_csv(committed, output)
    refreshed = prepare_principal_part_export(source, profile, manifest_path=state)
    state.unlink()

    with pytest.raises(PrincipalPartExportError, match="identity state changed"):
        write_principal_part_csv(refreshed, tmp_path / "regenerated.csv")

    assert not (tmp_path / "regenerated.csv").exists()


def _is_staged_commit(source_text: str) -> bool:
    return source_text.endswith(".tmp") and ".backup." not in source_text


@pytest.mark.parametrize("fail_at", [1, 2, 3])
@pytest.mark.parametrize("existing", [False, True])
def test_early_staging_failure_preserves_legacy_csv_manifest_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int, existing: bool
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    prepared = prepare_principal_part_export(
        source, _manifest_profile(), manifest_path=state, approved_allocations={0}, approve_new_scope=True
    )
    if existing:
        # Prepare first so invalid binary originals exercise recovery, not state parsing.
        for index, path in enumerate((output, state, Path(f"{source}.latinitas-cards.json"))):
            path.write_bytes(bytes([index, 255, 0, 128]) + b"original\r\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    calls = 0
    original_mkstemp = tempfile.mkstemp

    def fail_creation(*, prefix: str, suffix: str, dir: Path) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(errno.ENOSPC, "injected legacy staging failure")
        return original_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    monkeypatch.setattr("latinitas_cards.preview_export.tempfile.mkstemp", fail_creation)
    with pytest.raises(PrincipalPartExportError, match="No output or committed state was changed"):
        write_principal_part_csv(prepared, output)
    assert calls == fail_at
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_keyboard_interrupt_after_backup_moves_restores_prior_pair_and_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    profile = _manifest_profile()
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0},
        approve_new_scope=True,
    )
    write_principal_part_csv(allocated, output)
    output_before = output.read_bytes()
    state_before = state.read_bytes()
    checkpoint_before = Path(f"{source}.latinitas-cards.json").read_bytes()
    source_before = source.read_bytes()
    second = prepare_principal_part_export(source, profile, manifest_path=state)

    original_replace = os.replace

    def interrupt_output_commit(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        source_text = os.fsdecode(source_name)
        if destination == output and _is_staged_commit(source_text):
            raise KeyboardInterrupt("simulated interruption before output replacement")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", interrupt_output_commit)

    with pytest.raises(KeyboardInterrupt):
        write_principal_part_csv(second, output)

    checkpoint = Path(f"{source}.latinitas-cards.json")
    assert output.read_bytes() == output_before
    assert state.read_bytes() == state_before
    assert checkpoint.read_bytes() == checkpoint_before
    assert source.read_bytes() == source_before
    assert list(tmp_path.glob(".*.tmp")) == []

    retry = prepare_principal_part_export(source, profile, manifest_path=state)
    assert retry.manifest_reviews == ()
    assert retry.source_scope == allocated.source_scope
    assert {(note.provenance.source_identity, note.latinitas_id) for note in retry.generation.notes} == {
        (note.provenance.source_identity, note.latinitas_id) for note in allocated.generation.notes
    }


def test_keyboard_interrupt_during_first_scoped_export_keeps_scope_state_uncommitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    approved = prepare_principal_part_export(source, _profile(), manifest_path=state, approve_new_scope=True)

    original_replace = os.replace

    def interrupt_output_commit(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        source_text = os.fsdecode(source_name)
        if destination == output and _is_staged_commit(source_text):
            raise KeyboardInterrupt("simulated interruption before output replacement")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", interrupt_output_commit)

    with pytest.raises(KeyboardInterrupt):
        write_principal_part_csv(approved, output)

    assert not output.exists()
    assert not state.exists()
    assert not Path(f"{source}.latinitas-cards.json").exists()
    assert list(tmp_path.glob(".*.tmp")) == []

    monkeypatch.undo()
    retry_pending = prepare_principal_part_export(source, _profile(), manifest_path=state)
    assert retry_pending.scope_pending
    assert retry_pending.generation.notes == ()

    retried = prepare_principal_part_export(source, _profile(), manifest_path=state, approve_new_scope=True)
    write_principal_part_csv(retried, output)
    persisted = CsvIdentityManifest.load(state)
    assert persisted.source_scope == retried.source_scope
    checkpoint = PriorExportCheckpoint.load(Path(f"{source}.latinitas-cards.json"))
    assert checkpoint.source_scope == retried.source_scope
    assert deterministic_csv_bytes(retried).startswith(b"#separator:Comma\n")


def _prepare_committed_pair_for_interruption(
    tmp_path: Path,
) -> tuple[PrincipalPartExportResult, Path, Path, Path, bytes, bytes, bytes]:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    profile = _manifest_profile()
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0},
        approve_new_scope=True,
    )
    write_principal_part_csv(allocated, output)
    second = prepare_principal_part_export(source, profile, manifest_path=state)
    return (
        second,
        source,
        output,
        state,
        output.read_bytes(),
        state.read_bytes(),
        source.read_bytes(),
    )


@pytest.mark.parametrize("perform", [False, True], ids=["before-move", "after-move"])
@pytest.mark.parametrize(
    ("prior_output", "prior_manifest", "injection"),
    [
        pytest.param(prior_output, prior_manifest, injection, id=f"{output_id}-{manifest_id}-{injection}")
        for output_id, prior_output in [("output-present", b"#separator:Comma\nprior\n"), ("output-absent", None)]
        for manifest_id, prior_manifest in [("manifest-present", b'{"prior": "manifest"}\n'), ("manifest-absent", None)]
        for injection in ["backup-output", "backup-manifest", "commit-output", "commit-manifest"]
        if not (injection == "backup-output" and prior_output is None)
        and not (injection == "backup-manifest" and prior_manifest is None)
    ],
)
def test_keyboard_interrupt_around_each_destructive_move_restores_the_prior_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_output: bytes | None,
    prior_manifest: bytes | None,
    injection: str,
    perform: bool,
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    profile = _manifest_profile()
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approved_allocations={0},
        approve_new_scope=True,
    )
    if prior_output is not None:
        output.write_bytes(prior_output)
    if prior_manifest is not None:
        state.write_bytes(prior_manifest)
    source_before = source.read_bytes()
    phase, _, target_name = injection.partition("-")
    target = output if target_name == "output" else state

    original_replace = os.replace
    hits = 0

    def interrupting_replace(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal hits
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        source_text = os.fsdecode(source_name)
        if phase == "backup":
            matches = Path(source_text) == target
        else:
            matches = destination == target and _is_staged_commit(source_text)
        if matches:
            hits += 1
            if perform:
                original_replace(source_name, destination_name)
            raise KeyboardInterrupt("simulated interruption")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", interrupting_replace)

    with pytest.raises(KeyboardInterrupt):
        write_principal_part_csv(allocated, output)

    assert hits == 1
    if prior_output is None:
        assert not output.exists()
    else:
        assert output.read_bytes() == prior_output
    if prior_manifest is None:
        assert not state.exists()
    else:
        assert state.read_bytes() == prior_manifest
    assert source.read_bytes() == source_before
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("perform", [False, True], ids=["before-move", "after-move"])
@pytest.mark.parametrize(
    ("prior_state", "injection"),
    [
        pytest.param("committed", "backup-checkpoint", id="state-present-backup-checkpoint"),
        pytest.param("committed", "commit-checkpoint", id="state-present-commit-checkpoint"),
        pytest.param("fresh", "commit-checkpoint", id="state-absent-commit-checkpoint"),
    ],
)
def test_keyboard_interrupt_around_checkpoint_moves_restores_the_prior_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_state: str,
    injection: str,
    perform: bool,
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    state = Path(f"{source}.latinitas.json")
    checkpoint_file = Path(f"{source}.latinitas-cards.json")
    profile = _profile()
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    if prior_state == "committed":
        committed = prepare_principal_part_export(source, profile, manifest_path=state, approve_new_scope=True)
        write_principal_part_csv(committed, output)
        prepared = prepare_principal_part_export(source, profile, manifest_path=state)
        checkpoint_before = checkpoint_file.read_bytes()
        output_before = output.read_bytes()
        state_before = state.read_bytes()
    else:
        prepared = prepare_principal_part_export(source, profile, manifest_path=state, approve_new_scope=True)
        checkpoint_before = b""
        output_before = b""
        state_before = b""
    source_before = source.read_bytes()

    original_replace = os.replace
    hits = 0

    def interrupting_replace(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal hits
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        source_text = os.fsdecode(source_name)
        if injection == "backup-checkpoint":
            matches = Path(source_text) == checkpoint_file
        else:
            matches = destination == checkpoint_file and _is_staged_commit(source_text)
        if matches:
            hits += 1
            if perform:
                original_replace(source_name, destination_name)
            raise KeyboardInterrupt("simulated interruption")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", interrupting_replace)

    with pytest.raises(KeyboardInterrupt):
        write_principal_part_csv(prepared, output)

    assert hits == 1
    if prior_state == "committed":
        assert checkpoint_file.read_bytes() == checkpoint_before
        assert output.read_bytes() == output_before
        assert state.read_bytes() == state_before
    else:
        assert not checkpoint_file.exists()
        assert not state.exists()
        assert not output.exists()
    assert source.read_bytes() == source_before
    assert list(tmp_path.glob(".*.tmp")) == []

    monkeypatch.undo()
    retry = prepare_principal_part_export(
        source,
        profile,
        manifest_path=state,
        approve_new_scope=prior_state == "fresh",
    )
    write_principal_part_csv(retry, output)
    advanced = PriorExportCheckpoint.load(checkpoint_file)
    assert advanced.source_scope == retry.source_scope
    assert advanced.objects


def test_keyboard_interrupt_during_recovery_retains_both_backups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second, source, output, state, output_before, state_before, source_before = (
        _prepare_committed_pair_for_interruption(tmp_path)
    )
    checkpoint_file = Path(f"{source}.latinitas-cards.json")
    checkpoint_before = checkpoint_file.read_bytes()

    original_replace = os.replace
    interrupted_commit = False

    def interrupt_commit_and_recovery(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal interrupted_commit
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        source_text = os.fsdecode(source_name)
        if destination == output and _is_staged_commit(source_text):
            interrupted_commit = True
            raise KeyboardInterrupt("simulated interruption before output replacement")
        if interrupted_commit and destination == output and ".backup." in source_text:
            raise KeyboardInterrupt("simulated interruption during recovery")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", interrupt_commit_and_recovery)

    with pytest.raises(KeyboardInterrupt) as error:
        write_principal_part_csv(second, output)

    assert error.value.args == ("simulated interruption during recovery",)
    assert not output.exists()
    assert not state.exists()
    assert not checkpoint_file.exists()
    output_backups = list(tmp_path.glob(f".{output.name}.backup.*"))
    state_backups = list(tmp_path.glob(f".{state.name}.backup.*"))
    checkpoint_backups = list(tmp_path.glob(f".{checkpoint_file.name}.backup.*"))
    assert len(output_backups) == 1 and output_backups[0].read_bytes() == output_before
    assert len(state_backups) == 1 and state_backups[0].read_bytes() == state_before
    assert len(checkpoint_backups) == 1 and checkpoint_backups[0].read_bytes() == checkpoint_before
    assert source.read_bytes() == source_before


def test_keyboard_interrupt_with_failed_restoration_retains_backup_and_reports_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second, source, output, state, output_before, state_before, source_before = (
        _prepare_committed_pair_for_interruption(tmp_path)
    )

    original_replace = os.replace
    interrupted_commit = False

    def interrupt_commit_fail_restoration(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal interrupted_commit
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        source_text = os.fsdecode(source_name)
        if destination == output and _is_staged_commit(source_text):
            interrupted_commit = True
            raise KeyboardInterrupt("simulated interruption before output replacement")
        if interrupted_commit and destination == output and ".backup." in source_text:
            raise OSError("simulated restoration failure")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", interrupt_commit_fail_restoration)

    with pytest.raises(KeyboardInterrupt) as error:
        write_principal_part_csv(second, output)

    message = str(error.value)
    assert "Recovery is required" in message
    assert str(output) in message
    output_backups = list(tmp_path.glob(f".{output.name}.backup.*"))
    assert len(output_backups) == 1 and output_backups[0].read_bytes() == output_before
    assert str(output_backups[0]) in message
    assert not output.exists()
    assert state.read_bytes() == state_before
    assert source.read_bytes() == source_before


def test_base_exception_cancellation_propagates_after_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second, source, output, state, output_before, state_before, source_before = (
        _prepare_committed_pair_for_interruption(tmp_path)
    )

    original_replace = os.replace

    def cancel_output_commit(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        if destination == output and _is_staged_commit(os.fsdecode(source_name)):
            raise SystemExit("simulated cancellation before output replacement")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", cancel_output_commit)

    with pytest.raises(SystemExit):
        write_principal_part_csv(second, output)

    assert output.read_bytes() == output_before
    assert state.read_bytes() == state_before
    assert source.read_bytes() == source_before
    assert list(tmp_path.glob(".*.tmp")) == []


def test_base_exception_with_failed_restoration_notes_recovery_and_retains_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second, source, output, state, output_before, state_before, source_before = (
        _prepare_committed_pair_for_interruption(tmp_path)
    )

    original_replace = os.replace
    interrupted_commit = False

    def cancel_commit_fail_restoration(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal interrupted_commit
        destination = Path(destination_name) if not isinstance(destination_name, bytes) else None
        source_text = os.fsdecode(source_name)
        if destination == output and _is_staged_commit(source_text):
            interrupted_commit = True
            raise SystemExit("simulated cancellation before output replacement")
        if interrupted_commit and destination == output and ".backup." in source_text:
            raise OSError("simulated restoration failure")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", cancel_commit_fail_restoration)

    # A mutant can turn cancellation into KeyboardInterrupt. Capture it so the
    # wrong exception fails an assertion instead of aborting pytest's runner.
    with pytest.raises((SystemExit, KeyboardInterrupt)) as error:
        write_principal_part_csv(second, output)

    assert isinstance(error.value, SystemExit)
    notes = getattr(error.value, "__notes__", ())
    assert any("Recovery is required" in note and str(output) in note for note in notes)
    output_backups = list(tmp_path.glob(f".{output.name}.backup.*"))
    assert len(output_backups) == 1 and output_backups[0].read_bytes() == output_before
    assert any(str(output_backups[0]) in note for note in notes)
    assert not output.exists()
    assert state.read_bytes() == state_before
    assert source.read_bytes() == source_before


def test_export_rejects_non_regular_output_and_manifest_destinations(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    result = prepare_principal_part_export(
        source,
        _profile(),
        manifest_path=Path(f"{source}.latinitas.json"),
        approve_new_scope=True,
    )
    output_directory = tmp_path / "output-directory"
    output_directory.mkdir()

    with pytest.raises(PrincipalPartExportError, match="regular file"):
        write_principal_part_csv(result, output_directory)
    assert output_directory.is_dir()

    manifest = tmp_path / "manifest.json"
    manifest_result = prepare_principal_part_export(
        source,
        _manifest_profile(),
        manifest_path=manifest,
        approved_allocations={0},
        approve_new_scope=True,
    )
    manifest_directory = tmp_path / "manifest-directory"
    manifest_directory.mkdir()
    invalid_manifest_result = replace(manifest_result, manifest_path=manifest_directory)

    with pytest.raises(PrincipalPartExportError, match="regular file"):
        write_principal_part_csv(invalid_manifest_result, tmp_path / "generated.csv")
    assert manifest_directory.is_dir()


def test_export_never_overwrites_prepared_profile_when_override_differs(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_a = tmp_path / "profile-a.json"
    profile_b = tmp_path / "profile-b.json"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    _profile().save(profile_a)
    _profile(tags=("other",)).save(profile_b)
    result = prepare_principal_part_export(
        source,
        _profile(),
        profile_path=profile_a,
        manifest_path=Path(f"{source}.latinitas.json"),
        approve_new_scope=True,
    )
    profile_before = profile_a.read_bytes()

    with pytest.raises(PrincipalPartExportError, match="profile path"):
        write_principal_part_csv(result, profile_a, profile_path=profile_b)

    assert profile_a.read_bytes() == profile_before


@pytest.mark.parametrize("tag", ["bad tag", "\tbad\n"])
def test_profile_rejects_anki_tags_containing_whitespace(tag: str) -> None:
    with pytest.raises(ValueError, match="whitespace"):
        _profile(tags=(tag,))


def test_export_rejects_input_profile_and_manifest_aliases_without_overwriting(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    profile = _profile()
    profile.save(profile_path)
    result = prepare_principal_part_export(
        source,
        profile,
        profile_path=profile_path,
        manifest_path=Path(f"{source}.latinitas.json"),
        approve_new_scope=True,
    )
    source_before = source.read_bytes()
    profile_before = profile_path.read_bytes()

    with pytest.raises(PrincipalPartExportError, match="must not overwrite"):
        write_principal_part_csv(result, source, profile_path=profile_path)
    with pytest.raises(PrincipalPartExportError, match="must not overwrite"):
        write_principal_part_csv(result, profile_path, profile_path=profile_path)

    assert source.read_bytes() == source_before
    assert profile_path.read_bytes() == profile_before


@pytest.mark.parametrize("recipe", ["principal_part_completion", "principal_part_recognition"])
def test_preview_and_csv_retain_review_evidence_without_unconditional_targets(
    tmp_path: Path,
    recipe: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from html import unescape

    from latinitas_cards.commands.principal_parts import render_principal_part_preview
    from latinitas_cards.profile_setup import build_representative_examples

    source = tmp_path / "source.csv"
    alternatives = "amāre|amare, amō|amo, amāvī, amātum"
    conflict = "dīcere, dīcō, dīxī, dictum<br>supine (not PPP)"
    _write_source(source, [("a", "amāre", alternatives, "lieben"), ("b", "dīcere", conflict, "sagen")])
    profile = _profile().apply_overrides(
        {
            "principal_parts": {
                "roles": ["present_infinitive", "present_1s", "perfect_1s", "perfect_passive_participle"],
                "pipe_alternatives": True,
                "trailing_poet_hint": True,
            },
            "selected_recipes": [recipe],
        }
    )
    examples = build_representative_examples(read_csv_records(source), profile)
    assert examples[0].structural_status == "success"
    assert "Unresolved evidence" in examples[0].structural_message
    assert examples[1].structural_status == "unsupported"
    assert "review" in examples[1].structural_message
    result = prepare_principal_part_export(source, profile, approve_new_scope=True)
    assert (result.source_entry_count, result.generated_count, result.skipped_count, result.ambiguous_count) == (
        2,
        1,
        1,
        1,
    )
    assert result.generation.generated_warning_count == 1
    header, rows, _ = _parse_export(deterministic_csv_bytes(result))
    assert len(rows) == (0 if recipe == "principal_part_completion" else 1)
    fields = dict(result.generation.notes[0].to_anki_fields())
    if rows:
        assert dict(zip(header, rows[0], strict=True))["Principal Parts"] == fields["Principal Parts"]
    evidence_text = fields["Principal Parts"].split("</span>")[0].split(">", 1)[1]
    evidence = json.loads(unescape(evidence_text))
    assert evidence["raw"] == alternatives
    assert evidence["candidates"] == [["amāre", "amare"], ["amō", "amo"], ["amāvī"], ["amātum"]]
    assert evidence["rules"] == [
        "literal comma",
        "confirmed pipe alternatives within each slot",
        "trim",
        "preserve alternative order",
    ]
    assert result.card_count == (0 if recipe == "principal_part_completion" else 1)
    assert fields["CompletionInfinitiveEnabled"] == fields["RecognitionInfinitiveEnabled"] == ""
    assert fields["CompletionPresentEnabled"] == fields["RecognitionPresentEnabled"] == ""
    render_principal_part_preview(result, limit=2)
    output = capsys.readouterr().out
    assert "Generated entries: 1/2" in output
    assert "Wholly skipped entries: 1/2" in output
    assert "Generated entries with warnings: 1 (overlaps generated)" in output
    assert "Raw extraction evidence:" in output
    assert "supine (not PPP)" in output


def test_manifest_snapshot_and_removal_reviews_are_not_ambiguous_current_entries(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    first = ("a", "dīcere", "dīcere, dīcō, dīxī, dictum", "sagen")
    second = ("b", "amāre", "amāre, amō, amāvī, amātum", "lieben")
    _write_source(source, [first, second])
    profile = _profile(source_identity=SourceIdentityConfig(strategy="manifest"))
    initial = prepare_principal_part_export(
        source, profile, approve_new_scope=True, approve_fresh_import=True, approved_allocations=(0, 1)
    )
    write_principal_part_csv(initial, tmp_path / "initial.csv")
    _write_source(source, [first])
    reviewed = prepare_principal_part_export(source, profile)
    assert {review.kind for review in reviewed.manifest_reviews} == {"removed", "stale_manifest"}
    assert reviewed.source_entry_count == reviewed.generated_count == 1
    assert reviewed.skipped_count == 0
    assert reviewed.ambiguous_count == 1
    assert [skip.code for skip in reviewed.generation.skips] == ["linguistic_review_required"]
