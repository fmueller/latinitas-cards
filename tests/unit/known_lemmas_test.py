import csv
import hashlib
import json
import sqlite3
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from latinitas_cards.identity import ResolvedSourceIdentity
from latinitas_cards.known_lemmas import load_known_lemmas
from latinitas_cards.manifest import CsvIdentityManifest, reconcile_csv_manifest
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.sources import CanonicalSourceError, read_source_records


def _profile(path: Path, strategy: str = "source_id_field") -> None:
    DeckProfile.default(
        note_type="Vocabulary",
        lexical_entry_field="Entry",
        principal_parts_field="Forms",
        meaning_field="Meaning",
        source_identity=SourceIdentityConfig.model_validate(
            {"strategy": strategy, **({"field": "ID"} if strategy == "source_id_field" else {})}
        ),
        principal_part_roles=("present_1s", "present_infinitive", "perfect_1s", "supine"),
        separators=(",",),
    ).save(path)


ROWS = [
    ("z", '<script>wrong</script><b title="x > y">“DĪCŌ,</b> dīcere”'),
    ("a", "dīcō, dīcere"),
    ("b", "(amō),amāre"),
    ("c", "... <b>videō</b>"),
    ("d", "<style>wrong</style><!-- none -->"),
    ("e", "[JŪS]<br>iūris"),
]


def _csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ID", "Entry", "Forms", "Meaning"))
        writer.writerows((identity, entry, "", "") for identity, entry in rows)


def _package(path: Path, rows: Sequence[tuple[str, str]] = ROWS) -> None:
    database = path.with_suffix(".sqlite")
    with sqlite3.connect(database) as con:
        con.execute("CREATE TABLE col (id INTEGER PRIMARY KEY, models TEXT)")
        con.execute(
            "INSERT INTO col VALUES (1, ?)",
            (
                json.dumps(
                    {"1": {"name": "Vocabulary", "flds": [{"name": name} for name in ("Entry", "Forms", "Meaning")]}}
                ),
            ),
        )
        con.execute("CREATE TABLE notes (id INTEGER, guid TEXT, mid INTEGER, tags TEXT, flds TEXT)")
        con.executemany(
            "INSERT INTO notes VALUES (?, ?, 1, '', ?)",
            [(index, identity, entry + "\x1f\x1f") for index, (identity, entry) in enumerate(rows)],
        )
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(database, "collection.anki2")
    database.unlink()


@pytest.mark.parametrize("suffix", [".apkg", ".colpkg"])
def test_known_lemmas_match_csv_and_packages_without_writes(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / "deck.csv"
    profile = tmp_path / "profile.json"
    _profile(profile)
    _csv(source, ROWS)
    state = Path(f"{source}.latinitas.json")
    state.write_text(CsvIdentityManifest.scoped(("ID", "Entry", "Forms", "Meaning"), "scope-A").to_json())
    package = tmp_path / f"deck{suffix}"
    package_profile = tmp_path / "package.json"
    _profile(package_profile, "note_guid")
    _package(package)
    paths = (source, profile, state, package, package_profile)
    before = {path: path.read_bytes() for path in paths}
    csv_result = load_known_lemmas(profile, source)
    package_result = load_known_lemmas(package_profile, package)
    expected = {"dico", "amo),amare", "ius"}
    assert set(csv_result.lemmas) == set(package_result.lemmas) == expected
    assert csv_result.lemmas["dico"] == (ResolvedSourceIdentity("z", "scope-A"), ResolvedSourceIdentity("a", "scope-A"))
    assert package_result.lemmas["dico"] == (ResolvedSourceIdentity("a"), ResolvedSourceIdentity("z"))
    assert [skip.location for skip in csv_result.skipped] == ["row 5", "row 6"]
    assert len(package_result.skipped) == 2
    for result, source_path, profile_path in (
        (csv_result, source, profile),
        (package_result, package, package_profile),
    ):
        assert result.source_digest == "sha256:" + hashlib.sha256(before[source_path]).hexdigest()
        assert result.profile_digest == "sha256:" + hashlib.sha256(before[profile_path]).hexdigest()
    assert {path: path.read_bytes() for path in paths} == before
    assert set(tmp_path.iterdir()) == set(paths)


def test_manifest_reuses_identity_after_reordering_and_skips_unresolved_rows(tmp_path: Path) -> None:
    source = tmp_path / "deck.csv"
    profile = tmp_path / "profile.json"
    _profile(profile, "manifest")
    _csv(source, [("1", "amō"), ("2", "videō")])
    records = read_source_records(source)
    manifest = reconcile_csv_manifest(
        records,
        CsvIdentityManifest.scoped(tuple(records[0].fields), "persisted"),
        approved_allocations={0: "original-amo", 1: "original-video"},
    ).manifest
    state = tmp_path / "custom-state.json"
    state.write_text(manifest.to_json())
    _csv(source, [("2", "videō"), ("3", "dīcō"), ("1", "amō")])
    before = {path: path.read_bytes() for path in (source, profile, state)}
    result = load_known_lemmas(profile, source, manifest_path=state)
    assert result.lemmas == {
        "video": (ResolvedSourceIdentity("original-video", "persisted"),),
        "amo": (ResolvedSourceIdentity("original-amo", "persisted"),),
    }
    assert [skip.location for skip in result.skipped] == ["row 3"]
    assert "identity" in result.skipped[0].message
    assert {path: path.read_bytes() for path in before} == before


def test_missing_scope_fails_without_creating_state(tmp_path: Path) -> None:
    source = tmp_path / "deck.csv"
    profile = tmp_path / "profile.json"
    _profile(profile)
    _csv(source, [("1", "amō")])
    before = {path: path.read_bytes() for path in (source, profile)}
    with pytest.raises(ValueError, match="scope"):
        load_known_lemmas(profile, source)
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("bad_row", ["bad,dīcō\n", "bad,dīcō,,,extra\n"])
def test_unreadable_csv_rows_do_not_shift_manifest_identities(tmp_path: Path, bad_row: str) -> None:
    source, profile = tmp_path / "deck.csv", tmp_path / "profile.json"
    _profile(profile, "manifest")
    _csv(source, [("z", "amō"), ("a", "videō")])
    records = read_source_records(source)
    state = Path(f"{source}.latinitas.json")
    state.write_text(
        reconcile_csv_manifest(
            records,
            CsvIdentityManifest.scoped(tuple(records[0].fields), "scope"),
            approved_allocations={0: "amo-id", 1: "video-id"},
        ).manifest.to_json()
    )
    source.write_text("ID,Entry,Forms,Meaning\na,videō,,\n" + bad_row + "z,amō,,\n")
    before = {path: path.read_bytes() for path in (source, profile, state)}
    with pytest.raises(CanonicalSourceError, match="number of fields"):
        read_source_records(source)
    result = load_known_lemmas(profile, source)
    assert result.lemmas == {
        "video": (ResolvedSourceIdentity("video-id", "scope"),),
        "amo": (ResolvedSourceIdentity("amo-id", "scope"),),
    }
    assert [(skip.location, bool(skip.message)) for skip in result.skipped] == [("row 3", True)]
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("suffix", [".apkg", ".colpkg"])
def test_unreadable_package_note_is_reported_without_losing_valid_notes(tmp_path: Path, suffix: str) -> None:
    source, profile = tmp_path / f"deck{suffix}", tmp_path / "profile.json"
    _profile(profile, "note_guid")
    _package(source, [("z", "amō"), ("", "dīcō"), ("a", "videō")])
    before = {path: path.read_bytes() for path in (source, profile)}
    with pytest.raises(CanonicalSourceError, match="GUID"):
        read_source_records(source)
    result = load_known_lemmas(profile, source)
    assert result.lemmas == {"amo": (ResolvedSourceIdentity("z"),), "video": (ResolvedSourceIdentity("a"),)}
    assert [(skip.location, bool(skip.message)) for skip in result.skipped] == [("note 1", True)]
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_missing_identity_and_lexical_text_are_reported_individually(tmp_path: Path) -> None:
    source, profile = tmp_path / "deck.csv", tmp_path / "profile.json"
    _profile(profile)
    _csv(source, [("a", "amō"), ("", "dīcō"), ("c", "<style>hidden</style>"), ("d", "videō")])
    state = Path(f"{source}.latinitas.json")
    state.write_text(CsvIdentityManifest.scoped(("ID", "Entry", "Forms", "Meaning"), "scope").to_json())
    result = load_known_lemmas(profile, source)
    assert set(result.lemmas) == {"amo", "video"}
    assert [skip.location for skip in result.skipped] == ["row 3", "row 4"]
    assert "identity" in result.skipped[0].message
    assert "lexical" in result.skipped[1].message
