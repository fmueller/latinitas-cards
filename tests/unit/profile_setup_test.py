import json
import unicodedata
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


def _write_lesson_code_csv(path: Path) -> None:
    path.write_text(
        "LatinumB_Lektion,Latein,LatinumB_Formen\n"
        '12,amō,"amāre, amō, amāvī, amātum"\n'
        '13,ferō,"ferre, ferō, tulī, lātum"\n',
        encoding="utf-8",
    )


def _write_renamed_lesson_code_csv(path: Path) -> None:
    path.write_text(
        'Spalte A,Spalte B,Spalte C\n12,amō,"amāre, amō, amāvī, amātum"\n13,ferō,"ferre, ferō, tulī, lātum"\n',
        encoding="utf-8",
    )


def _write_misleading_name_csv(path: Path) -> None:
    path.write_text(
        "Latein Vokabel,Lektion Nr,Formen der Wörter\n"
        '12,amō,"amāre, amō, amāvī, amātum"\n'
        '13,ferō,"ferre, ferō, tulī, lātum"\n',
        encoding="utf-8",
    )


def _write_tied_lexical_csv(path: Path) -> None:
    path.write_text(
        'Wort A,Wort B,Formen\namō,amō,"amāre, amō, amāvī, amātum"\nferō,ferō,"ferre, ferō, tulī, lātum"\n',
        encoding="utf-8",
    )


def _write_sparse_lexical_csv(path: Path) -> None:
    path.write_text(
        "Lektion,Wort,Formen\n"
        '7,amō,"amāre, amō, amāvī, amātum"\n'
        '8,,"ferre, ferō, tulī, lātum"\n'
        '9,,"tollere, tollō, sustulī, lātum"\n',
        encoding="utf-8",
    )


def test_lesson_code_values_do_not_outrank_latin_content_for_lexical_entry(tmp_path: Path) -> None:
    source = tmp_path / "lesson-codes.csv"
    _write_lesson_code_csv(source)

    proposal = propose_profile(source)

    assert proposal.profile.fields.lexical_entry_field == "Latein"
    assert proposal.profile.fields.principal_parts_field == "LatinumB_Formen"
    assert proposal.field_choices_required == ()

    by_field = {candidate.field: candidate for candidate in proposal.field_candidates}
    lexical = by_field["Latein"]
    assert lexical.role == "lexical_entry"
    assert "amō" in lexical.sample_values and "ferō" in lexical.sample_values
    assert any("match a principal-part form" in reason for reason in lexical.evidence)

    lesson = by_field["LatinumB_Lektion"]
    assert lesson.role == "unmapped"
    assert any("numeric" in reason for reason in lesson.evidence)
    assert any("0/2" in reason for reason in lesson.evidence)
    assert lesson.sample_values == ("12", "13")


def test_equivalent_renamed_fixture_selects_the_same_fields(tmp_path: Path) -> None:
    source = tmp_path / "renamed.csv"
    _write_renamed_lesson_code_csv(source)

    proposal = propose_profile(source)

    assert proposal.profile.fields.lexical_entry_field == "Spalte B"
    assert proposal.profile.fields.principal_parts_field == "Spalte C"
    assert proposal.field_choices_required == ()


def test_misleading_field_names_lose_to_content_evidence(tmp_path: Path) -> None:
    source = tmp_path / "misleading.csv"
    _write_misleading_name_csv(source)

    proposal = propose_profile(source)

    assert proposal.profile.fields.lexical_entry_field == "Lektion Nr"
    assert proposal.profile.fields.principal_parts_field == "Formen der Wörter"
    assert proposal.field_choices_required == ()


def test_tied_lexical_evidence_requires_explicit_choice_before_saving(tmp_path: Path) -> None:
    source = tmp_path / "tied.csv"
    _write_tied_lexical_csv(source)

    proposal = propose_profile(source)

    assert proposal.profile.fields.lexical_entry_field in {"Wort A", "Wort B"}
    assert len(proposal.field_choices_required) == 1
    required = proposal.field_choices_required[0]
    assert required.role == "lexical_entry"
    assert required.candidates == ("Wort A", "Wort B")
    assert "tie" in required.reason.lower()


def test_tied_lexical_evidence_blocks_confirmed_save_and_explicit_choice_saves_and_reloads(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    source = tmp_path / "tied.csv"
    profile_path = tmp_path / "tied-profile.json"
    _write_tied_lexical_csv(source)

    rejected = _command(
        runner,
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive", "--confirm"],
    )
    assert rejected.exit_code != 0
    assert "explicit" in rejected.output.lower()
    assert not profile_path.exists()

    cancelled = _command(
        runner,
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive"],
    )
    assert cancelled.exit_code == 0
    assert "cancelled" in cancelled.stdout.lower()
    assert "field choice required" in cancelled.stdout.lower()
    assert not profile_path.exists()

    cancelled_json = _command(
        runner,
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive", "--json"],
    )
    assert cancelled_json.exit_code == 0
    cancelled_payload = json.loads(cancelled_json.stdout)
    assert cancelled_payload["status"] == "cancelled"
    assert [choice["role"] for choice in cancelled_payload["field_choices_required"]] == ["lexical_entry"]
    assert not profile_path.exists()

    saved = _command(
        runner,
        [
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--lexical-entry-field",
            "Wort B",
            "--non-interactive",
            "--confirm",
        ],
    )
    assert saved.exit_code == 0
    persisted = DeckProfile.from_json(profile_path.read_text(encoding="utf-8"))
    assert persisted.fields.lexical_entry_field == "Wort B"

    reused = _command(
        runner,
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive", "--json"],
    )
    assert reused.exit_code == 0
    assert json.loads(reused.stdout)["status"] == "reused"


def test_sparse_lexical_evidence_requires_explicit_choice_before_saving(tmp_path: Path) -> None:
    runner = CliRunner()
    source = tmp_path / "sparse.csv"
    profile_path = tmp_path / "sparse-profile.json"
    _write_sparse_lexical_csv(source)

    proposal = propose_profile(source)
    assert len(proposal.field_choices_required) == 1
    required = proposal.field_choices_required[0]
    assert required.role == "lexical_entry"
    assert "sparse" in required.reason.lower()

    rejected = _command(
        runner,
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive", "--confirm"],
    )
    assert rejected.exit_code != 0
    assert "explicit" in rejected.output.lower()
    assert not profile_path.exists()

    saved = _command(
        runner,
        [
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--lexical-entry-field",
            "Wort",
            "--non-interactive",
            "--confirm",
        ],
    )
    assert saved.exit_code == 0
    assert DeckProfile.from_json(profile_path.read_text(encoding="utf-8")).fields.lexical_entry_field == "Wort"


def test_interactive_tied_setup_requires_typing_an_explicit_field_choice(tmp_path: Path) -> None:
    source = tmp_path / "tied.csv"
    profile_path = tmp_path / "interactive-tied.json"
    _write_tied_lexical_csv(source)

    saved = CliRunner().invoke(
        cast(click.Command, _typer_get_command(app)),
        ["setup", "--input", str(source), "--profile", str(profile_path)],
        input="Wort A\n" + "\n" * 11 + "y\n",
    )

    assert saved.exit_code == 0
    assert "Field choice required" in saved.output
    assert DeckProfile.from_json(profile_path.read_text(encoding="utf-8")).fields.lexical_entry_field == "Wort A"

    empty_choice_path = tmp_path / "interactive-empty.json"
    rejected = CliRunner().invoke(
        cast(click.Command, _typer_get_command(app)),
        ["setup", "--input", str(source), "--profile", str(empty_choice_path)],
        input="\n" + "\n" * 11 + "y\n",
    )

    assert rejected.exit_code != 0
    assert "explicit" in rejected.output.lower()
    assert not empty_choice_path.exists()


def test_interactive_rejected_final_confirmation_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "rejected.json"
    _write_csv(source)

    result = CliRunner().invoke(
        cast(click.Command, _typer_get_command(app)),
        ["setup", "--input", str(source), "--profile", str(profile_path)],
        input="\n" * 11 + "n\n",
    )

    assert result.exit_code == 0
    assert "cancelled" in result.output.lower()
    assert not profile_path.exists()


def test_proposal_output_shows_candidate_values_and_reasons(tmp_path: Path) -> None:
    source = tmp_path / "lesson-codes.csv"
    profile_path = tmp_path / "reasons.json"
    _write_lesson_code_csv(source)

    result = _command(CliRunner(), ["--input", str(source), "--profile", str(profile_path), "--non-interactive"])

    assert result.exit_code == 0
    assert "Field-candidate evidence:" in result.stdout
    assert "Latein -> lexical_entry" in result.stdout
    assert "LatinumB_Lektion -> unmapped" in result.stdout
    assert "match a principal-part form" in result.stdout
    assert "'12'" in result.stdout


def _write_tied_principal_csv(path: Path) -> None:
    path.write_text(
        "Wort,Formen A,Formen B\n"
        'amō,"amāre, amō, amāvī, amātum","amāre, amō, amāvī, amātum"\n'
        'ferō,"ferre, ferō, tulī, lātum","ferre, ferō, tulī, lātum"\n',
        encoding="utf-8",
    )


def test_tied_principal_evidence_requires_explicit_choice_and_flag_remedy_saves(tmp_path: Path) -> None:
    runner = CliRunner()
    source = tmp_path / "tied-principal.csv"
    profile_path = tmp_path / "tied-principal-profile.json"
    _write_tied_principal_csv(source)

    proposal = propose_profile(source)
    assert proposal.profile.fields.principal_parts_field in {"Formen A", "Formen B"}
    assert [required.role for required in proposal.field_choices_required] == ["principal_parts"]
    assert "principal-part structure" in proposal.field_choices_required[0].reason
    assert proposal.field_choices_required[0].candidates == ("Formen A", "Formen B")

    rejected = _command(
        runner,
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive", "--confirm"],
    )
    assert rejected.exit_code != 0
    assert "explicit" in rejected.output.lower()
    assert not profile_path.exists()

    saved = _command(
        runner,
        [
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--principal-parts-field",
            "Formen A",
            "--non-interactive",
            "--confirm",
        ],
    )
    assert saved.exit_code == 0
    assert DeckProfile.from_json(profile_path.read_text(encoding="utf-8")).fields.principal_parts_field == "Formen A"


def test_missing_principal_structure_requires_explicit_principal_choice(tmp_path: Path) -> None:
    source = tmp_path / "no-structure.csv"
    source.write_text("Eintrag,Notiz\namo,kurze Notiz\n", encoding="utf-8")
    profile_path = tmp_path / "no-structure.json"

    proposal = propose_profile(source)

    assert [required.role for required in proposal.field_choices_required] == ["principal_parts"]
    assert "separator pattern" in proposal.field_choices_required[0].reason

    result = _command(
        CliRunner(),
        ["--input", str(source), "--profile", str(profile_path), "--non-interactive", "--confirm"],
    )
    assert result.exit_code != 0
    assert "explicit" in result.output.lower()
    assert not profile_path.exists()


def test_interactive_principal_choice_prompt_rejects_empty_and_unknown_answers(tmp_path: Path) -> None:
    source = tmp_path / "tied-principal.csv"
    profile_path = tmp_path / "interactive-principal.json"
    _write_tied_principal_csv(source)

    saved = CliRunner().invoke(
        cast(click.Command, _typer_get_command(app)),
        ["setup", "--input", str(source), "--profile", str(profile_path)],
        input="FormenA\nFormen A\n" + "\n" * 11 + "y\n",
    )

    assert saved.exit_code == 0
    assert "Unknown field" in saved.output
    assert DeckProfile.from_json(profile_path.read_text(encoding="utf-8")).fields.principal_parts_field == "Formen A"

    rejected_path = tmp_path / "interactive-principal-empty.json"
    rejected = CliRunner().invoke(
        cast(click.Command, _typer_get_command(app)),
        ["setup", "--input", str(source), "--profile", str(rejected_path)],
        input="\n" + "\n" * 11 + "y\n",
    )

    assert rejected.exit_code != 0
    assert "explicit" in rejected.output.lower()
    assert not rejected_path.exists()


def test_bogus_explicit_field_override_reports_clean_validation_error(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    profile_path = tmp_path / "bogus.json"
    _write_csv(source)

    result = _command(
        CliRunner(),
        [
            "--input",
            str(source),
            "--profile",
            str(profile_path),
            "--principal-parts-field",
            "Bogus",
            "--non-interactive",
            "--confirm",
        ],
    )

    assert result.exit_code == 2
    assert not isinstance(result.exception, StopIteration)
    assert "configured principal-parts field is missing from the source" in result.output
    assert not profile_path.exists()


def _record_with_fields(note_type: str, identity: str, fields: dict[str, str]) -> CanonicalSourceRecord:
    return CanonicalSourceRecord(
        source_kind="apkg",
        note_type=note_type,
        fields=fields,
        provenance=SourceProvenance(source_path=Path("multi.apkg"), location=f"note {identity}"),
        source_identity=identity,
        note_guid=identity,
    )


def test_note_type_override_with_disjoint_fields_rebuilds_evidence_safely() -> None:
    inspection = SourceInspection(
        records=(
            _record_with_fields("Alpha", "alpha-guid", {"Latein": "amō", "Forms": "amāre, amō, amāvī, amātum"}),
            _record_with_fields("Beta", "beta-guid", {"Lemma": "ferō", "Forms": "ferre, ferō, tulī, lātum"}),
        ),
        note_types=("Alpha", "Beta"),
        fields_by_note_type={
            "Alpha": ("Latein", "Forms"),
            "Beta": ("Lemma", "Forms"),
        },
    )

    proposal = propose_profile_from_inspection(inspection, Path("multi.apkg"), note_type="Alpha")
    corrected = apply_profile_overrides(proposal, {"note_type": "Beta"})

    assert corrected.record_count == 1
    assert corrected.fields == ("Forms", "Lemma")

    empty = apply_profile_overrides(proposal, {"note_type": "Gamma"})

    assert empty.record_count == 0
    assert empty.fields == ()


def test_nfd_decomposed_values_still_rank_by_content(tmp_path: Path) -> None:
    source = tmp_path / "nfd.csv"
    source.write_text(
        "Lektion,Latein,Formen\n"
        f'12,{unicodedata.normalize("NFD", "amō")},"amāre, amō, amāvī, amātum"\n'
        f'13,{unicodedata.normalize("NFD", "ferō")},"ferre, ferō, tulī, lātum"\n',
        encoding="utf-8",
    )

    proposal = propose_profile(source)

    assert proposal.profile.fields.lexical_entry_field == "Latein"
    assert proposal.field_choices_required == ()
