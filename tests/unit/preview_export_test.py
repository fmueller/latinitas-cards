import csv
import io
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from latinitas_cards.preview_export import (
    PrincipalPartExportError,
    deterministic_csv_bytes,
    prepare_principal_part_export,
    write_principal_part_csv,
)
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "representative-university-latin.apkg"


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


def test_preview_result_reports_representative_notes_and_structured_counts(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(
        source,
        [
            ("entry-β", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen\n& prüfen"),
            ("entry-ambiguous", "ferō", "ferre, ferō, tulī", "tragen"),
            ("entry-unsupported", "videō", "vidēre; videō; vīdī; vīsum", "sehen"),
        ],
    )

    result = prepare_principal_part_export(source, _profile())

    assert result.generated_count == 8
    assert result.skipped_count == 2
    assert result.ambiguous_count == 1
    assert result.generation.notes[0].provenance.source_path is None
    assert result.generation.notes[0].provenance.source_identity == "entry-β"
    assert any(skip.code == "unmarked_omission" and skip.status == "ambiguous" for skip in result.generation.skips)
    assert any(skip.code == "separator_mismatch" for skip in result.generation.skips)


def test_csv_export_is_utf8_deterministic_and_uses_anki_import_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile(tags=("latinitas", "provenance-β"))
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen\n& prüfen")])

    first = deterministic_csv_bytes(prepare_principal_part_export(source, profile))
    second = deterministic_csv_bytes(prepare_principal_part_export(source, profile))

    assert first == second
    assert first.startswith(b"#separator:Comma\n")
    assert "#html:true\n" in first.decode("utf-8")
    assert "#notetype:Latinitas Principal Parts\n" in first.decode("utf-8")
    assert "#deck:Latin::Latinitas::Review\n" in first.decode("utf-8")
    assert "#tags column:4\n" in first.decode("utf-8")
    assert (
        "#columns:LatinitasID,Prompt,Answer,Tags,Source ID,Source Kind,Source Location,Source Path,Recipe,"
        "Exercise Key,Recipe Version,Personal Notes\n" in first.decode("utf-8")
    )
    header, rows, metadata = _parse_export(first)
    assert metadata.startswith("#separator:Comma\n#html:true\n")
    assert header == [
        "LatinitasID",
        "Prompt",
        "Answer",
        "Tags",
        "Source ID",
        "Source Kind",
        "Source Location",
        "Source Path",
        "Recipe",
        "Exercise Key",
        "Recipe Version",
        "Personal Notes",
    ]
    assert len(rows) == 8
    assert rows[0][0].startswith("latinitas-v1-")
    assert rows[0][3] == "latinitas provenance-β"
    assert "sagen" in first.decode("utf-8")
    assert "source.csv" not in first.decode("utf-8")


def test_csv_export_escapes_source_identity_for_html_import(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(
        source,
        [("<img src=x onerror=alert(1)>", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")],
    )

    payload = deterministic_csv_bytes(prepare_principal_part_export(source, _profile()))
    text = payload.decode("utf-8")

    assert "&lt;img src=x onerror=alert(1)&gt;" in text
    assert "<img src=x onerror=alert(1)>" not in text


def test_csv_export_preserves_completion_and_recognition_html_section_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])

    payload = deterministic_csv_bytes(prepare_principal_part_export(source, _profile()))
    header, rows, _ = _parse_export(payload)
    recipe_index = header.index("Recipe")
    exercise_index = header.index("Exercise Key")
    prompt_index = header.index("Prompt")
    answer_index = header.index("Answer")

    completion = next(
        row for row in rows if row[recipe_index] == "principal_part_completion" and row[exercise_index] == "perfect_1s"
    )
    recognition = next(
        row for row in rows if row[recipe_index] == "principal_part_recognition" and row[exercise_index] == "perfect_1s"
    )

    assert completion[prompt_index].startswith(
        "<div>Ergänze die fehlende Stammform.</div><div><strong>Stammformen</strong></div><div>"
    )
    assert completion[prompt_index].endswith("</div><div><strong>Bedeutung:</strong> sagen</div>")
    assert completion[answer_index] == (
        "<div><strong>Fehlende Stammform:</strong> dīxī</div>"
        "<div><strong>Rolle:</strong> Perfekt, 1. Person Singular</div>"
    )
    assert recognition[answer_index].startswith(
        "<div><strong>Lemma:</strong> dīcō</div><div><strong>Stammformen</strong></div><div>"
    )
    assert recognition[answer_index].endswith(
        "</div><div><strong>Rolle:</strong> Perfekt, 1. Person Singular</div>"
        "<div><strong>Bedeutung:</strong> sagen</div>"
    )


def test_csv_export_renders_source_html_as_safe_readable_text_in_both_recipes(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(
        source,
        [
            (
                "entry-html",
                "<i>dīcō</i>",
                "dīcere, dīcō, dīxī, dictum",
                "erste Bedeutung<div>zweite <b>Bedeutung</b>&lt;br&gt;dritte</div>"
                "<script>alert(1)</script><img src=x onerror=alert(2)>",
            )
        ],
    )
    source_before = source.read_bytes()

    first = deterministic_csv_bytes(prepare_principal_part_export(source, _profile()))
    second = deterministic_csv_bytes(prepare_principal_part_export(source, _profile()))

    assert first == second
    text = first.decode("utf-8")
    header, rows, _metadata = _parse_export(first)
    recipe_index = header.index("Recipe")
    exercise_index = header.index("Exercise Key")
    prompt_index = header.index("Prompt")
    answer_index = header.index("Answer")
    expected_meaning = "<div><strong>Bedeutung:</strong> erste Bedeutung<br>zweite Bedeutung<br>dritte</div>"
    completion = next(
        row for row in rows if row[recipe_index] == "principal_part_completion" and row[exercise_index] == "perfect_1s"
    )
    recognition = next(
        row for row in rows if row[recipe_index] == "principal_part_recognition" and row[exercise_index] == "supine"
    )

    assert completion[prompt_index].endswith(expected_meaning)
    assert recognition[answer_index].endswith(expected_meaning)
    assert recognition[prompt_index] == "Welche Stammform ist „dictum“?"
    assert "<div><strong>Lemma:</strong> dīcō</div>" in recognition[answer_index]
    assert "&lt;div&gt;" not in text
    assert "&lt;b&gt;" not in text
    assert "&lt;br&gt;" not in text
    assert "&lt;i&gt;" not in text
    assert "alert(1)" not in text
    assert "alert(2)" not in text
    assert "<script" not in text
    assert "<img" not in text
    assert source.read_bytes() == source_before


def test_csv_export_encodes_unsafe_controls_without_removing_newlines(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(
        source,
        [("entry-\x1b[31m\x9b", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen\x00\nprüfen")],
    )

    text = deterministic_csv_bytes(prepare_principal_part_export(source, _profile())).decode("utf-8")

    assert "\x1b" not in text
    assert "\x00" not in text
    assert "\x9b" not in text
    assert "\\x1b[31m\\x9b" in text
    assert "\\x00" in text
    assert "sagen" in text and "prüfen" in text


@pytest.mark.parametrize("suffix", [".apkg", ".colpkg"])
def test_package_preview_and_export_leave_source_bytes_unchanged(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"representative{suffix}"
    output = tmp_path / "generated.csv"
    shutil.copyfile(FIXTURE, source)
    source_before = source.read_bytes()

    result = prepare_principal_part_export(source, _representative_profile())
    assert result.generated_count > 0
    assert source.read_bytes() == source_before

    write_principal_part_csv(result, output)

    assert output.exists()
    assert source.read_bytes() == source_before


def test_managed_text_and_tags_updates_keep_one_logical_row_per_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    original = prepare_principal_part_export(source, _profile(tags=("old",)))

    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen; aussprechen")])
    revised = prepare_principal_part_export(source, _profile(tags=("new", "html-v2")))

    original_ids = {note.latinitas_id for note in original.generation.notes}
    revised_ids = [note.latinitas_id for note in revised.generation.notes]
    assert set(revised_ids) == original_ids
    assert len(revised_ids) == len(set(revised_ids))
    assert deterministic_csv_bytes(original) != deterministic_csv_bytes(revised)


def test_idless_csv_requires_explicit_allocation_and_reuses_ids_after_reordering(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile = _profile(source_identity=SourceIdentityConfig(strategy="manifest"))
    _write_source(
        source,
        [
            ("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen"),
            ("ignored-b", "ferō", "ferre, ferō, tulī, lātum", "tragen"),
        ],
    )
    manifest = Path(f"{source}.latinitas.json")

    pending = prepare_principal_part_export(source, profile, manifest_path=manifest)
    assert pending.generated_count == 0
    assert pending.manifest_reviews
    with pytest.raises(PrincipalPartExportError, match="explicit identity review"):
        write_principal_part_csv(pending, tmp_path / "blocked.csv")
    assert not (tmp_path / "blocked.csv").exists()
    assert not manifest.exists()

    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=manifest,
        approved_allocations={0, 1},
    )
    output = tmp_path / "generated.csv"
    write_principal_part_csv(allocated, output)
    assert manifest.exists()
    allocated_ids = {
        note.provenance.source_identity: note.latinitas_id
        for note in allocated.generation.notes
        if note.recipe.exercise_key == "present_infinitive"
    }

    _write_source(
        source,
        [
            ("ignored-b", "ferō", "ferre, ferō, tulī, lātum", "tragen"),
            ("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen"),
        ],
    )
    reordered = prepare_principal_part_export(source, profile, manifest_path=manifest)
    reordered_ids = {
        note.provenance.source_identity: note.latinitas_id
        for note in reordered.generation.notes
        if note.recipe.exercise_key == "present_infinitive"
    }
    assert reordered.manifest_reviews == ()
    assert reordered_ids == allocated_ids


@pytest.mark.parametrize("manifest_exists", [False, True])
def test_manifest_commit_failure_restores_existing_or_absent_output_pair(
    tmp_path: Path,
    manifest_exists: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    manifest = Path(f"{source}.latinitas.json")
    profile = _profile(source_identity=SourceIdentityConfig(strategy="manifest"))
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])

    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=manifest,
        approved_allocations={0},
    )
    if manifest_exists:
        write_principal_part_csv(allocated, output)
        output_before = output.read_bytes()
        manifest_before = manifest.read_bytes()
    else:
        output.write_bytes(b"prior output")
        output_before = output.read_bytes()
        manifest_before = None

    original_replace = os.replace
    failed = False

    def fail_manifest_once(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal failed
        if not failed and not isinstance(destination_name, bytes) and Path(destination_name) == manifest:
            failed = True
            raise OSError("simulated manifest commit failure")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", fail_manifest_once)

    with pytest.raises(PrincipalPartExportError, match="No output or manifest was changed"):
        write_principal_part_csv(allocated, output)

    assert failed
    assert output.read_bytes() == output_before
    if manifest_before is None:
        assert not manifest.exists()
    else:
        assert manifest.read_bytes() == manifest_before


def test_manifest_rollback_failure_without_backups_reports_affected_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    manifest = Path(f"{source}.latinitas.json")
    profile = _profile(source_identity=SourceIdentityConfig(strategy="manifest"))
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=manifest,
        approved_allocations={0},
    )

    original_replace = os.replace
    manifest_failed = False

    def fail_manifest_once(
        source_name: str | bytes | os.PathLike[str],
        destination_name: str | bytes | os.PathLike[str],
    ) -> None:
        nonlocal manifest_failed
        if not manifest_failed and not isinstance(destination_name, bytes) and Path(destination_name) == manifest:
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
    assert not manifest.exists()
    assert "Recovery is required" in message
    assert str(output) in message
    assert "No output or manifest was changed" not in message


def test_manifest_rollback_failure_retains_backup_for_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "generated.csv"
    manifest = Path(f"{source}.latinitas.json")
    profile = _profile(source_identity=SourceIdentityConfig(strategy="manifest"))
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=manifest,
        approved_allocations={0},
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
        if destination == manifest and not manifest_failed:
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
    manifest = Path(f"{source}.latinitas.json")
    profile = _profile(source_identity=SourceIdentityConfig(strategy="manifest"))
    _write_source(source, [("ignored-a", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    allocated = prepare_principal_part_export(
        source,
        profile,
        manifest_path=manifest,
        approved_allocations={0},
    )
    write_principal_part_csv(allocated, output)
    output_before = output.read_bytes()

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
        if destination == manifest and not manifest_failed:
            manifest_failed = True
            raise OSError("simulated manifest commit failure")
        if destination == manifest and manifest_failed and ".backup." in source_text:
            rollback_failed = True
            raise OSError("simulated manifest rollback failure")
        original_replace(source_name, destination_name)

    monkeypatch.setattr("latinitas_cards.preview_export.os.replace", fail_manifest_commit_and_rollback)

    with pytest.raises(PrincipalPartExportError) as error:
        write_principal_part_csv(allocated, output)

    backup_paths = list(tmp_path.glob(f".{manifest.name}.backup.*"))
    message = str(error.value)
    assert manifest_failed
    assert rollback_failed
    assert output.read_bytes() == output_before
    assert not manifest.exists()
    assert backup_paths
    assert str(manifest) in message
    assert str(backup_paths[0]) in message
    assert "Recovery is required" in message


def test_export_rejects_non_regular_output_and_manifest_destinations(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(source, [("entry-1", "dīcō", "dīcere, dīcō, dīxī, dictum", "sagen")])
    result = prepare_principal_part_export(source, _profile())
    output_directory = tmp_path / "output-directory"
    output_directory.mkdir()

    with pytest.raises(PrincipalPartExportError, match="regular file"):
        write_principal_part_csv(result, output_directory)
    assert output_directory.is_dir()

    manifest = tmp_path / "manifest.json"
    manifest_result = prepare_principal_part_export(
        source,
        _profile(source_identity=SourceIdentityConfig(strategy="manifest")),
        manifest_path=manifest,
        approved_allocations={0},
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
    result = prepare_principal_part_export(source, _profile(), profile_path=profile_a)
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
    result = prepare_principal_part_export(source, profile, profile_path=profile_path)
    source_before = source.read_bytes()
    profile_before = profile_path.read_bytes()

    with pytest.raises(PrincipalPartExportError, match="must not overwrite"):
        write_principal_part_csv(result, source, profile_path=profile_path)
    with pytest.raises(PrincipalPartExportError, match="must not overwrite"):
        write_principal_part_csv(result, profile_path, profile_path=profile_path)

    assert source.read_bytes() == source_before
    assert profile_path.read_bytes() == profile_before
