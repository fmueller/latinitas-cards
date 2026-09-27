"""Prior-export eligibility checkpoint contract for conditional sibling cards.

The checkpoint is the retained evidence that makes regenerating a scoped CSV
source safe: it records what was actually exported, never what a destination
is assumed to contain.  Eligibility loss, recipe disablement, or shared-field
degradation withholds whole note rows instead of clearing fronts; absent or
parse-failed objects keep their last safe evidence.
"""

import json
import shutil
from pathlib import Path

import pytest

from latinitas_cards.cards import TEMPLATE_REGISTRY_DIGEST
from latinitas_cards.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    NOTE_FAMILY,
    CheckpointError,
    PriorExportCheckpoint,
)
from latinitas_cards.identity import derive_latinitas_id
from latinitas_cards.notes import NOTE_SCHEMA_VERSION
from latinitas_cards.preview_export import (
    PrincipalPartExportError,
    PrincipalPartExportResult,
    deterministic_csv_bytes,
    prepare_principal_part_export,
    write_principal_part_csv,
)
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig


def _profile(
    *,
    roles: tuple[str, ...] = ("present_infinitive", "present_1s", "perfect_1s", "supine"),
    recipes: tuple[str, ...] = ("principal_part_completion", "principal_part_recognition"),
) -> DeckProfile:
    return DeckProfile.default(
        note_type="CSV source",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="Stable ID"),
        principal_part_roles=roles,
        separators=(" — ",),
        selected_recipes=recipes,
        tags=("latinitas",),
    )


def _write_source(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    lines = ["Stable ID,Lemma,Forms,German gloss"]
    lines += [",".join('"' + value.replace('"', '""') + '"' for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export(
    source: Path,
    profile: DeckProfile,
    *,
    approve_fresh_import: bool = False,
) -> PrincipalPartExportResult:
    return prepare_principal_part_export(
        source,
        profile,
        manifest_path=Path(f"{source}.latinitas.json"),
        approve_new_scope=True,
        approve_fresh_import=approve_fresh_import,
    )


def _checkpoint_path(source: Path) -> Path:
    return Path(f"{source}.latinitas-cards.json")


def _committed(source: Path, profile: DeckProfile, rows: list[tuple[str, str, str, str]]) -> PrincipalPartExportResult:
    _write_source(source, rows)
    result = _export(source, profile)
    write_principal_part_csv(result, source.parent / "generated.csv")
    return result


def test_first_approved_export_commits_checkpoint_bound_to_scope_schema_and_registry(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    profile = _profile()
    rows = [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")]
    _write_source(source, rows)
    result = _export(source, profile)

    assert not _checkpoint_path(source).exists()
    write_principal_part_csv(result, output)

    checkpoint = PriorExportCheckpoint.load(_checkpoint_path(source))
    assert checkpoint.checkpoint_version == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint.source_scope == result.source_scope
    assert checkpoint.note_family == NOTE_FAMILY
    assert checkpoint.note_schema == NOTE_SCHEMA_VERSION
    assert checkpoint.template_registry_digest == TEMPLATE_REGISTRY_DIGEST
    assert checkpoint.export_fingerprint.startswith("export-sha256:")
    note = result.generation.notes[0]
    assert checkpoint.objects == (
        (
            note.latinitas_id,
            (
                "principal_part_completion:present_1s",
                "principal_part_completion:present_infinitive",
                "principal_part_completion:perfect_1s",
                "principal_part_completion:supine",
                "principal_part_recognition:present_1s",
                "principal_part_recognition:present_infinitive",
                "principal_part_recognition:perfect_1s",
                "principal_part_recognition:supine",
            ),
        ),
    )


def test_repeat_export_is_deterministic_and_leaves_checkpoint_stable(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    rows = [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")]
    _committed(source, profile, rows)
    checkpoint_before = _checkpoint_path(source).read_bytes()
    output_before = (tmp_path / "generated.csv").read_bytes()

    repeat = _export(source, profile)
    write_principal_part_csv(repeat, tmp_path / "generated.csv")

    assert (tmp_path / "generated.csv").read_bytes() == output_before
    assert _checkpoint_path(source).read_bytes() == checkpoint_before
    assert repeat.exported_note_count == 1
    assert repeat.card_count == 8
    assert not repeat.card_eligibility_reviews


def test_read_only_preview_never_advances_or_creates_checkpoint_state(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    _committed(source, profile, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    checkpoint_before = _checkpoint_path(source).read_bytes()

    previewed = prepare_principal_part_export(
        source,
        profile,
        manifest_path=Path(f"{source}.latinitas.json"),
    )

    assert previewed.exported_note_count == 1
    assert _checkpoint_path(source).read_bytes() == checkpoint_before


def test_eligibility_loss_withholds_the_whole_note_row_and_retains_last_safe_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    profile = _profile()
    first = _committed(source, profile, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    note_id = first.generation.notes[0].latinitas_id
    prior_objects = PriorExportCheckpoint.load(_checkpoint_path(source)).objects

    _write_source(source, [("entry-1", "dīcō", "dīcere — dīcō —  — dictum", "sagen")])
    degraded = _export(source, profile)

    assert degraded.object_count == 1
    assert degraded.zero_card_note_count == 0
    assert degraded.exported_note_count == 0
    assert degraded.card_count == 0
    withheld = next(review for review in degraded.card_eligibility_reviews if review.kind == "withheld")
    assert withheld.latinitas_id == note_id
    assert set(withheld.card_keys) == {
        "principal_part_completion:perfect_1s",
        "principal_part_recognition:perfect_1s",
    }
    assert "review" in withheld.message.lower()

    write_principal_part_csv(degraded, output)
    text = output.read_text(encoding="utf-8")
    assert text.count("\n") == 6
    assert source.name not in text.split("\n")[6:]
    retained = PriorExportCheckpoint.load(_checkpoint_path(source))
    assert retained.objects == prior_objects


def test_recipe_deselection_withholds_previously_exported_cards(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    recognition_only = _profile(recipes=("principal_part_recognition",))
    _committed(source, profile, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])

    deselected = _export(source, recognition_only)

    assert deselected.exported_note_count == 0
    withheld = next(review for review in deselected.card_eligibility_reviews if review.kind == "withheld")
    assert set(withheld.card_keys) == {f"principal_part_completion:{role}" for role in profile.principal_parts.roles}


def test_wording_and_gloss_updates_still_export_and_advance_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    _committed(source, profile, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    prior_objects = PriorExportCheckpoint.load(_checkpoint_path(source)).objects

    _write_source(source, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen; sprechen")])
    revised = _export(source, profile)

    assert revised.exported_note_count == 1
    assert not revised.card_eligibility_reviews
    write_principal_part_csv(revised, tmp_path / "generated.csv")
    assert PriorExportCheckpoint.load(_checkpoint_path(source)).objects == prior_objects


def test_absent_and_parse_failed_objects_retain_their_prior_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    first = _committed(
        source,
        profile,
        [
            ("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen"),
            ("entry-2", "ferō", "ferre — ferō — tulī — lātum", "tragen"),
        ],
    )
    prior_objects = PriorExportCheckpoint.load(_checkpoint_path(source)).objects
    assert len(prior_objects) == 2

    _write_source(source, [("entry-2", "ferō", "ferre: ferō: tulī: lātum", "tragen")])
    broken = _export(source, profile)

    assert broken.object_count == 0
    absent_identities = {review.latinitas_id for review in broken.card_eligibility_reviews if review.kind == "absent"}
    assert absent_identities == {note.latinitas_id for note in first.generation.notes}
    assert all("disappeared" not in review.message.lower() for review in broken.card_eligibility_reviews)

    write_principal_part_csv(broken, tmp_path / "generated.csv")
    assert PriorExportCheckpoint.load(_checkpoint_path(source)).objects == prior_objects


def test_zero_eligible_note_with_prior_evidence_is_withheld_not_silently_omitted(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    unsupported_roles = _profile(roles=("stem_a", "stem_b"))
    first = _committed(source, profile, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    prior_objects = PriorExportCheckpoint.load(_checkpoint_path(source)).objects

    _write_source(source, [("entry-1", "dīcō", "dīcere — dīcō", "sagen")])
    degraded = _export(source, unsupported_roles)

    note = degraded.zero_card_notes[0]
    assert note.latinitas_id == first.generation.notes[0].latinitas_id
    assert degraded.exported_note_count == 0
    withheld = next(review for review in degraded.card_eligibility_reviews if review.kind == "withheld")
    assert withheld.latinitas_id == note.latinitas_id
    write_principal_part_csv(degraded, tmp_path / "generated.csv")
    retained = PriorExportCheckpoint.load(_checkpoint_path(source))
    assert retained.objects == prior_objects
    assert dict(retained.objects)[note.latinitas_id]


def test_added_supported_cards_preserve_surviving_keys_and_note_id(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    recognition_only = _profile(recipes=("principal_part_recognition",))
    both = _profile()
    first = _committed(source, recognition_only, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    prior_keys = dict(PriorExportCheckpoint.load(_checkpoint_path(source)).objects)[
        first.generation.notes[0].latinitas_id
    ]

    enriched = _export(source, both)

    assert enriched.exported_note_count == 1
    assert not enriched.card_eligibility_reviews
    assert enriched.generation.notes[0].latinitas_id == first.generation.notes[0].latinitas_id
    assert set(prior_keys) <= set(enriched.generation.notes[0].card_keys)
    write_principal_part_csv(enriched, tmp_path / "generated.csv")
    advanced = dict(PriorExportCheckpoint.load(_checkpoint_path(source)).objects)
    assert set(advanced[first.generation.notes[0].latinitas_id]) == set(enriched.generation.notes[0].card_keys)


@pytest.mark.parametrize(
    "damage", ["delete", "corrupt", "corrupt-non-utf8", "rebind-scope", "rebind-registry", "rebind-schema"]
)
def test_missing_corrupt_or_incompatible_checkpoint_requires_explicit_fresh_import(
    tmp_path: Path,
    damage: str,
) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    _committed(source, profile, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    checkpoint_file = _checkpoint_path(source)
    if damage == "delete":
        checkpoint_file.unlink()
    elif damage == "corrupt":
        checkpoint_file.write_text("{not json", encoding="utf-8")
    elif damage == "corrupt-non-utf8":
        checkpoint_file.write_bytes(b'\xff\xfe{"source_scope": "scope-x"}')
    else:
        payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if damage == "rebind-scope":
            payload["source_scope"] = "scope-other"
        elif damage == "rebind-registry":
            payload["template_registry_digest"] = "card-registry-sha256:other"
        else:
            payload["note_schema"] = "0"
        checkpoint_file.write_text(json.dumps(payload), encoding="utf-8")

    gated = _export(source, profile)

    assert gated.checkpoint_pending
    assert any(review.kind == "checkpoint_confirmation" for review in gated.card_eligibility_reviews)
    with pytest.raises(PrincipalPartExportError, match="fresh"):
        deterministic_csv_bytes(gated)
    with pytest.raises(PrincipalPartExportError, match="fresh"):
        write_principal_part_csv(gated, tmp_path / "generated.csv")

    fresh = _export(source, profile, approve_fresh_import=True)

    assert fresh.checkpoint_pending
    assert fresh.fresh_import_approved
    assert not fresh.card_eligibility_reviews
    assert fresh.exported_note_count == 1
    write_principal_part_csv(fresh, tmp_path / "generated.csv")
    restored = PriorExportCheckpoint.load(checkpoint_file)
    assert restored.source_scope == fresh.source_scope
    assert restored.template_registry_digest == TEMPLATE_REGISTRY_DIGEST


def test_checkpoint_serialization_round_trip_and_validation(tmp_path: Path) -> None:
    note_id = derive_latinitas_id("entry-1", "lexeme-1", source_scope="scope-x")
    checkpoint = PriorExportCheckpoint.scoped(
        source_scope="scope-x",
        note_schema=NOTE_SCHEMA_VERSION,
        template_registry_digest=TEMPLATE_REGISTRY_DIGEST,
        objects=((note_id, ("principal_part_completion:supine",)),),
        export_fingerprint="export-sha256:abc",
    )

    round_trip = PriorExportCheckpoint.from_json(checkpoint.to_json())

    assert round_trip == checkpoint
    assert round_trip.object_keys == {note_id: ("principal_part_completion:supine",)}
    with pytest.raises(CheckpointError):
        PriorExportCheckpoint.from_json("{broken")
    with pytest.raises(CheckpointError):
        PriorExportCheckpoint.from_json(json.dumps({"unexpected": True}))
    with pytest.raises(CheckpointError):
        tampered = json.loads(checkpoint.to_json())
        tampered["checkpoint_version"] = 99
        PriorExportCheckpoint.from_mapping(tampered)


def _write_package(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    import sqlite3
    import zipfile

    work = path.with_suffix("")
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
    con.execute(
        "INSERT INTO notes (id, guid, mid, tags, flds) VALUES (1, ?, 10, '', ?)",
        (rows[0][0], "\x1f".join(rows[0][1:])),
    )
    con.commit()
    con.close()
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(database_path, "collection.anki2")


def _package_profile(
    *,
    recipes: tuple[str, ...] = ("principal_part_completion", "principal_part_recognition"),
) -> DeckProfile:
    return DeckProfile.default(
        note_type="Latin Vocabulary",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        source_identity=SourceIdentityConfig(strategy="note_guid"),
        principal_part_roles=("present_infinitive", "present_1s", "perfect_1s", "supine"),
        separators=(" — ",),
        selected_recipes=recipes,
        tags=("latinitas",),
    )


def test_package_sources_persist_checkpoint_and_withhold_deselected_cards(tmp_path: Path) -> None:
    from latinitas_cards.checkpoint import GLOBAL_SOURCE_SCOPE

    source = tmp_path / "deck.apkg"
    output = tmp_path / "generated.csv"
    _write_package(source, [("guid-a", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    first = prepare_principal_part_export(source, _package_profile(), approve_fresh_import=True)
    assert first.exported_note_count == 1
    assert first.card_count == 8
    write_principal_part_csv(first, output)

    checkpoint_file = Path(f"{source}.latinitas-cards.json")
    committed = PriorExportCheckpoint.load(checkpoint_file)
    assert committed.source_scope == GLOBAL_SOURCE_SCOPE
    assert len(committed.objects) == 1

    deselected = prepare_principal_part_export(source, _package_profile(recipes=("principal_part_recognition",)))

    assert deselected.exported_note_count == 0
    withheld = next(review for review in deselected.card_eligibility_reviews if review.kind == "withheld")
    assert "principal_part_completion:perfect_1s" in withheld.card_keys
    header, rows, _metadata = _parse_output(deterministic_csv_bytes(deselected))
    assert rows == []
    write_principal_part_csv(deselected, output)
    retained = PriorExportCheckpoint.load(checkpoint_file)
    assert retained.objects == committed.objects


def test_missing_package_checkpoint_requires_explicit_fresh_import(tmp_path: Path) -> None:
    source = tmp_path / "deck.apkg"
    output = tmp_path / "generated.csv"
    _write_package(source, [("guid-a", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    first = prepare_principal_part_export(source, _package_profile())
    assert first.checkpoint_pending
    assert not first.fresh_import_approved
    with pytest.raises(PrincipalPartExportError, match="fresh"):
        write_principal_part_csv(first, output)

    approved = prepare_principal_part_export(source, _package_profile(), approve_fresh_import=True)
    assert approved.fresh_import_approved
    assert approved.exported_note_count == 1
    write_principal_part_csv(approved, output)
    checkpoint_file = Path(f"{source}.latinitas-cards.json")
    assert PriorExportCheckpoint.load(checkpoint_file).objects


def test_renamed_package_source_with_eligibility_loss_stays_review_only(tmp_path: Path) -> None:
    source = tmp_path / "deck.apkg"
    renamed = tmp_path / "deck (1).apkg"
    output = tmp_path / "generated.csv"
    _write_package(source, [("guid-a", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    approved = prepare_principal_part_export(source, _package_profile(), approve_fresh_import=True)
    write_principal_part_csv(approved, output)
    prior_objects = PriorExportCheckpoint.load(Path(f"{source}.latinitas-cards.json")).objects
    shutil.copyfile(source, renamed)

    degraded = prepare_principal_part_export(renamed, _package_profile(recipes=("principal_part_recognition",)))

    assert degraded.checkpoint_pending
    assert not degraded.fresh_import_approved
    confirmation = next(
        review for review in degraded.card_eligibility_reviews if review.kind == "checkpoint_confirmation"
    )
    assert "fresh import" in confirmation.message
    with pytest.raises(PrincipalPartExportError, match="fresh"):
        deterministic_csv_bytes(degraded)
    with pytest.raises(PrincipalPartExportError, match="fresh"):
        write_principal_part_csv(degraded, output)
    assert PriorExportCheckpoint.load(Path(f"{source}.latinitas-cards.json")).objects == prior_objects
    assert not Path(f"{renamed}.latinitas-cards.json").exists()


def test_corrupt_package_checkpoint_requires_explicit_fresh_import(tmp_path: Path) -> None:
    source = tmp_path / "deck.apkg"
    output = tmp_path / "generated.csv"
    _write_package(source, [("guid-a", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    first = prepare_principal_part_export(source, _package_profile(), approve_fresh_import=True)
    write_principal_part_csv(first, output)
    Path(f"{source}.latinitas-cards.json").write_text("{damaged", encoding="utf-8")

    gated = prepare_principal_part_export(source, _package_profile())

    assert gated.checkpoint_pending
    with pytest.raises(PrincipalPartExportError, match="fresh"):
        deterministic_csv_bytes(gated)

    fresh = prepare_principal_part_export(source, _package_profile(), approve_fresh_import=True)
    assert fresh.exported_note_count == 1
    write_principal_part_csv(fresh, output)
    assert PriorExportCheckpoint.load(Path(f"{source}.latinitas-cards.json")).objects


def test_package_profile_stored_at_the_checkpoint_sidecar_path_is_rejected_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "deck.apkg"
    _write_package(source, [("guid-a", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    profile_path = Path(f"{source}.latinitas-cards.json")
    _package_profile().save(profile_path)
    profile_before = profile_path.read_bytes()

    with pytest.raises(PrincipalPartExportError, match="paths must be distinct"):
        prepare_principal_part_export(source, _package_profile(), profile_path=profile_path)

    assert profile_path.read_bytes() == profile_before


def test_checkpoint_path_override_commits_evidence_at_the_explicit_location(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    checkpoint_override = tmp_path / "retained" / "cards.json"
    checkpoint_override.parent.mkdir()
    profile = _profile()
    _write_source(source, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    first = prepare_principal_part_export(
        source,
        profile,
        manifest_path=Path(f"{source}.latinitas.json"),
        approve_new_scope=True,
        checkpoint_path=checkpoint_override,
    )

    assert first.checkpoint_path == checkpoint_override
    assert not checkpoint_override.exists()
    write_principal_part_csv(first, output)

    committed = PriorExportCheckpoint.load(checkpoint_override)
    assert committed.source_scope == first.source_scope
    assert committed.objects
    assert not _checkpoint_path(source).exists()

    repeat = prepare_principal_part_export(
        source,
        profile,
        manifest_path=Path(f"{source}.latinitas.json"),
        checkpoint_path=checkpoint_override,
    )
    assert repeat.loaded_checkpoint == committed


def test_export_never_overwrites_the_checkpoint_evidence_file(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    _write_source(source, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    result = _export(source, profile)
    checkpoint_file = _checkpoint_path(source)

    with pytest.raises(PrincipalPartExportError, match="must not overwrite"):
        write_principal_part_csv(result, checkpoint_file)

    assert not checkpoint_file.exists()


def _parse_output(payload: bytes) -> tuple[list[str], list[list[str]], str]:
    import csv
    import io

    text = payload.decode("utf-8")
    lines = text.splitlines(keepends=True)
    metadata = "".join(lines[:6])
    header = lines[5].removeprefix("#columns:").removesuffix("\n").split(",")
    reader = csv.reader(io.StringIO("".join(lines[6:])), lineterminator="\n")
    return header, list(reader), metadata


def test_checkpoint_committed_by_another_process_after_prepare_blocks_the_export(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    _committed(source, profile, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    prepared = _export(source, profile)
    checkpoint_file = _checkpoint_path(source)
    concurrent = PriorExportCheckpoint.load(checkpoint_file)
    tampered = PriorExportCheckpoint.scoped(
        source_scope=concurrent.source_scope,
        objects=(),
        export_fingerprint="export-sha256:other",
    )
    checkpoint_file.write_text(tampered.to_json(), encoding="utf-8")

    with pytest.raises(PrincipalPartExportError, match="checkpoint changed since preparation"):
        write_principal_part_csv(prepared, tmp_path / "blocked.csv")

    assert PriorExportCheckpoint.load(checkpoint_file) == tampered


def test_checkpoint_removed_after_prepare_blocks_the_export(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    _committed(source, profile, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    prepared = _export(source, profile)
    _checkpoint_path(source).unlink()

    with pytest.raises(PrincipalPartExportError, match="disappeared since preparation"):
        write_principal_part_csv(prepared, tmp_path / "blocked.csv")

    assert not _checkpoint_path(source).exists()
    assert not (tmp_path / "blocked.csv").exists()


def test_fresh_import_rejects_a_valid_checkpoint_committed_after_prepare(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    _committed(source, profile, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    checkpoint_file = _checkpoint_path(source)
    valid_evidence = checkpoint_file.read_bytes()
    checkpoint_file.write_text("{damaged", encoding="utf-8")
    fresh = _export(source, profile, approve_fresh_import=True)
    checkpoint_file.write_bytes(valid_evidence)

    with pytest.raises(PrincipalPartExportError, match="checkpoint changed since preparation"):
        write_principal_part_csv(fresh, tmp_path / "blocked.csv")

    assert checkpoint_file.read_bytes() == valid_evidence


def test_incompatible_checkpoint_committed_after_prepare_blocks_the_export(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    _write_source(source, [("entry-1", "dīcō", "dīcere — dīcō — dīxī — dictum", "sagen")])
    prepared = _export(source, profile)
    foreign = PriorExportCheckpoint.scoped(source_scope="scope-other", objects=())
    checkpoint_file = _checkpoint_path(source)
    checkpoint_file.write_text(foreign.to_json(), encoding="utf-8")

    with pytest.raises(PrincipalPartExportError, match="checkpoint changed since preparation"):
        write_principal_part_csv(prepared, tmp_path / "blocked.csv")

    assert PriorExportCheckpoint.load(checkpoint_file) == foreign
    assert not (tmp_path / "blocked.csv").exists()


def test_multiline_glosses_and_parts_render_single_line_prompts_with_br(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile()
    _write_source(source, [("entry-1", "dīcō", "dīcere — dīcō — dīxī<br>poet. — dictum", "sagen\nprüfen")])
    result = _export(source, profile)

    prompts = [card.prompt for card in result.generation.notes[0].cards if card.eligible]
    assert prompts
    assert all("\n" not in prompt for prompt in prompts)
    assert any("dīxī<br>poet." in prompt for prompt in prompts)
    assert any("sagen<br>prüfen" in prompt for prompt in prompts)
    payload = deterministic_csv_bytes(result)
    data = payload.decode("utf-8").split("#columns")[1].split("\n", 1)[1]
    assert "\n" not in data.rstrip("\n")
