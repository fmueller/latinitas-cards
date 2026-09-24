import json
from pathlib import Path
from typing import cast

import click
import pytest
from click.testing import CliRunner, Result
from typer.main import get_command as _typer_get_command

from latinitas_cards.cli import app
from latinitas_cards.profile import DeckProfile
from latinitas_cards.profile_setup import apply_profile_overrides, propose_profile, propose_profile_from_inspection
from latinitas_cards.sources import CanonicalSourceRecord, SourceInspection, SourceProvenance

FIXTURE = Path(__file__).parents[1] / "fixtures" / "representative-university-latin.apkg"


def _command(runner: CliRunner, arguments: list[str]) -> Result:
    return runner.invoke(cast(click.Command, _typer_get_command(app)), ["setup", *arguments])


def _write_csv(path: Path) -> None:
    path.write_text(
        "Stable ID,Lemma,Forms,German gloss,Reference\n"
        'entry-1,dīcō,"dīcere,dīcō,dīxī,dictum",sagen,synthetic-1\n'
        'entry-2,ferō,"ferre,ferō,tulī,lātum",,synthetic-2\n',
        encoding="utf-8",
    )


def _write_duplicate_id_csv(path: Path) -> None:
    path.write_text(
        "Stable ID,Lemma,Forms,German gloss,Reference\n"
        'entry-1,dīcō,"dīcere,dīcō,dīxī,dictum",sagen,synthetic-1\n'
        'entry-1,ferō,"ferre,ferō,tulī,lātum",,synthetic-2\n',
        encoding="utf-8",
    )


def _record(note_type: str, identity: str, entry: str) -> CanonicalSourceRecord:
    return CanonicalSourceRecord(
        source_kind="apkg",
        note_type=note_type,
        fields={
            "Entry": entry,
            "Forms": "ferre, ferō, tulī, lātum",
            "German gloss": "tragen",
        },
        provenance=SourceProvenance(source_path=Path("multi.apkg"), location=f"note {identity}"),
        source_identity=identity,
        note_guid=identity,
    )


def test_representative_apkg_proposal_uses_approved_roles_and_shows_uncertainty() -> None:
    proposal = propose_profile(FIXTURE)

    assert proposal.profile.note_type == "Representative Latin Vocabulary"
    assert proposal.profile.fields.lexical_entry_field == "Entry"
    assert proposal.profile.fields.principal_parts_field == "Construction hints"
    assert proposal.profile.fields.meaning_field == "German gloss"
    assert proposal.profile.principal_parts.roles == (
        "present_infinitive",
        "present_1s",
        "perfect_1s",
        "perfect_passive_participle",
    )
    assert proposal.profile.principal_parts.separators == (",",)
    assert proposal.profile.selected_recipes == (
        "principal_part_completion",
        "principal_part_recognition",
    )
    assert any("eligibility" in uncertainty.lower() for uncertainty in proposal.uncertainties)
    assert any(example.structural_status == "success" for example in proposal.examples)


def test_setup_confirmed_csv_profile_is_saved_without_mutating_source_and_reports_json(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "project" / "profile.json"
    _write_csv(source)
    source_before = source.read_bytes()

    result = _command(
        CliRunner(),
        [
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--non-interactive",
            "--confirm",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "saved"
    assert payload["profile"]["source_identity"] == {"strategy": "source_id_field", "field": "Stable ID"}
    assert payload["effective_profile"]["language_tag"] == "de"
    assert profile_path.exists()
    assert DeckProfile.from_json(profile_path.read_text(encoding="utf-8")).fields.lexical_entry_field == "Lemma"
    assert source.read_bytes() == source_before


def test_setup_explicit_corrections_are_persisted_and_cancelled_setup_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    corrected_path = tmp_path / "corrected.json"
    cancelled_path = tmp_path / "cancelled.json"
    _write_csv(source)
    source_before = source.read_bytes()

    corrected = _command(
        CliRunner(),
        [
            "--input",
            str(source),
            "--profile",
            str(corrected_path),
            "--principal-parts-field",
            "Forms",
            "--without-meaning-field",
            "--generated-note-type",
            "Corrected Principal Parts",
            "--non-interactive",
            "--confirm",
        ],
    )

    assert corrected.exit_code == 0
    saved = DeckProfile.from_json(corrected_path.read_text(encoding="utf-8"))
    assert saved.fields.principal_parts_field == "Forms"
    assert saved.fields.meaning_field is None
    assert saved.generated_note_type == "Corrected Principal Parts"
    assert source.read_bytes() == source_before

    cancelled = _command(
        CliRunner(),
        [
            "--input",
            str(source),
            "--profile",
            str(cancelled_path),
            "--non-interactive",
        ],
    )

    assert cancelled.exit_code == 0
    assert "cancelled" in cancelled.stdout.lower()
    assert not cancelled_path.exists()
    assert source.read_bytes() == source_before


def test_setup_interactive_flow_shows_examples_uncertainty_and_requires_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "interactive.json"
    _write_csv(source)
    prompts = "\n" * 11 + "y\n"

    result = CliRunner().invoke(
        cast(click.Command, _typer_get_command(app)),
        ["setup", "--input", str(source), "--profile", str(profile_path)],
        input=prompts,
    )

    assert result.exit_code == 0
    assert "Representative examples:" in result.output
    assert "Uncertainty:" in result.output
    assert "Source identity: source_id_field (field: Stable ID)" in result.output
    assert "Generated-content language tag: de" in result.output
    assert profile_path.exists()
    assert DeckProfile.from_json(profile_path.read_text(encoding="utf-8")).fields.meaning_field is None


def test_correcting_note_type_rebuilds_examples_and_counts_from_all_note_types() -> None:
    records = (_record("Alpha", "alpha-guid", "alpha"), _record("Beta", "beta-guid", "beta"))
    inspection = SourceInspection(
        records=records,
        note_types=("Alpha", "Beta"),
        fields_by_note_type={"Alpha": ("Entry", "Forms", "German gloss"), "Beta": ("Entry", "Forms", "German gloss")},
    )

    proposal = propose_profile_from_inspection(inspection, Path("multi.apkg"), note_type="Alpha")
    corrected = apply_profile_overrides(proposal, {"note_type": "Beta"})

    assert corrected.record_count == 1
    assert corrected.fields == ("Entry", "Forms", "German gloss")
    assert [example.lexical_entry for example in corrected.examples] == ["beta"]


def test_setup_rejects_duplicate_source_ids_on_save_and_reuse(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_duplicate_id_csv(source)

    rejected = _command(
        CliRunner(),
        [
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--source-id-field",
            "Stable ID",
            "--non-interactive",
            "--confirm",
        ],
    )

    assert rejected.exit_code != 0
    assert "unique" in rejected.output.lower()
    assert not profile_path.exists()

    valid_source = tmp_path / "valid.csv"
    _write_csv(valid_source)
    initial = _command(
        CliRunner(),
        ["--input", str(valid_source), "--profile", str(profile_path), "--non-interactive", "--confirm"],
    )
    _write_duplicate_id_csv(valid_source)
    reused = _command(
        CliRunner(),
        ["--input", str(valid_source), "--profile", str(profile_path), "--non-interactive"],
    )

    assert initial.exit_code == 0
    assert reused.exit_code != 0
    assert "unique" in reused.output.lower()


def test_json_mode_requires_non_interactive_execution(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "profile.json"
    _write_csv(source)

    result = _command(CliRunner(), ["--input", str(source), "--profile", str(profile_path), "--json"])

    assert result.exit_code != 0
    assert "--json" in result.output
    assert not profile_path.exists()


def test_setup_rejects_profile_path_that_aliases_source(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_csv(source)
    before = source.read_bytes()

    result = _command(
        CliRunner(),
        [
            "--input",
            str(source),
            "--profile",
            str(source),
            "--reconfigure",
            "--non-interactive",
            "--confirm",
        ],
    )

    assert result.exit_code != 0
    assert "source" in result.output.lower() and "profile" in result.output.lower()
    assert source.read_bytes() == before


def test_setup_bounds_and_escapes_untrusted_example_values(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.csv"
    source.write_text(
        "Stable ID,Lemma,Forms,German gloss,Reference\n"
        'entry-1,"amo\nINJECT\x1b[31m\u202eRTL","amare, amo, amavi, amatum",sagen,synthetic-1\n',
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.json"

    result = _command(
        CliRunner(),
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive"],
    )

    assert result.exit_code == 0
    assert "\x1b[31m" not in result.output
    assert "\u202e" not in result.output
    assert "\\n" in result.output
    assert "\\u202e" in result.output


def test_setup_reports_profile_filesystem_errors_without_traceback(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    blocked_parent = tmp_path / "blocked"
    _write_csv(source)
    blocked_parent.write_text("not a directory", encoding="utf-8")

    result = _command(
        CliRunner(),
        [
            "--input",
            str(source),
            "--profile",
            str(blocked_parent / "profile.json"),
            "--non-interactive",
            "--confirm",
        ],
    )

    assert result.exit_code != 0
    assert "could not save profile" in result.output.lower()
    assert "traceback" not in result.output.lower()


def test_setup_reuses_confirmed_profiles_for_csv_and_apkg_without_repeating_setup(tmp_path: Path) -> None:
    runner = CliRunner()
    csv_source = tmp_path / "source.csv"
    csv_profile = tmp_path / "csv-profile.json"
    _write_csv(csv_source)

    initial_csv = _command(
        runner,
        [
            "--input",
            str(csv_source),
            "--profile",
            str(csv_profile),
            "--non-interactive",
            "--confirm",
        ],
    )
    reused_csv = _command(
        runner,
        ["--input", str(csv_source), "--profile", str(csv_profile), "--non-interactive", "--json"],
    )

    assert initial_csv.exit_code == 0
    assert reused_csv.exit_code == 0
    assert json.loads(reused_csv.stdout)["status"] == "reused"
    assert json.loads(reused_csv.stdout)["status"] != "proposal"

    apkg_profile = tmp_path / "apkg-profile.json"
    initial_apkg = _command(
        runner,
        [
            "--input",
            str(FIXTURE),
            "--profile",
            str(apkg_profile),
            "--non-interactive",
            "--confirm",
        ],
    )
    reused_apkg = _command(
        runner,
        ["--input", str(FIXTURE), "--profile", str(apkg_profile), "--non-interactive", "--json"],
    )

    assert initial_apkg.exit_code == 0
    assert reused_apkg.exit_code == 0
    assert json.loads(reused_apkg.stdout)["status"] == "reused"


def test_setup_reuse_reports_only_the_saved_note_type_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "multi.apkg"
    profile_path = tmp_path / "profile.json"
    source.write_bytes(b"synthetic package placeholder")
    DeckProfile.default(
        note_type="Alpha",
        lexical_entry_field="Entry",
        principal_parts_field="Forms",
        meaning_field="German gloss",
        principal_part_roles=("present_1s", "present_infinitive", "perfect_1s", "supine"),
        separators=(",",),
    ).save(profile_path)
    inspection = SourceInspection(
        records=(_record("Alpha", "alpha-guid", "alpha"), _record("Beta", "beta-guid", "beta")),
        note_types=("Alpha", "Beta"),
        fields_by_note_type={"Alpha": ("Entry", "Forms", "German gloss"), "Beta": ("Entry", "Forms", "German gloss")},
    )
    monkeypatch.setattr("latinitas_cards.setup_flow.inspect_source", lambda _: inspection)

    result = _command(
        CliRunner(),
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive"],
    )

    assert result.exit_code == 0
    assert "inspected 1 record(s)" in result.stdout
    assert "inspected 2 record(s)" not in result.stdout


def test_setup_rejects_unsupported_saved_profile_schema_without_overwriting_it(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "unsupported.json"
    _write_csv(source)
    profile_path.write_text('{"schema_version": 99}\n', encoding="utf-8")
    before = profile_path.read_bytes()

    result = _command(
        CliRunner(),
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive"],
    )

    assert result.exit_code != 0
    assert "Unsupported profile schema version 99" in result.output
    assert profile_path.read_bytes() == before
