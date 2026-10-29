import json
import os
from pathlib import Path
from typing import Any

import pytest

from latinitas_cards.profile import (
    DEFAULT_GENERATED_NOTE_TYPE,
    DEFAULT_LANGUAGE_TAG,
    DEFAULT_SELECTED_RECIPES,
    DEFAULT_TAGS,
    DEFAULT_TARGET_DECK,
    DeckProfile,
    FieldOverrides,
    PrincipalPartOverrides,
    ProfileOverrides,
    ProfileValidationError,
    SourceIdentityConfig,
    UnsupportedProfileSchemaError,
    resolve_profile,
    validate_profile,
)


def _profile() -> DeckProfile:
    return DeckProfile.default(
        note_type="Latin vocabulary",
        lexical_entry_field="Lemma",
        principal_parts_field="Principal parts",
        meaning_field="German",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="Source ID"),
        principal_part_roles=("present_1s", "present_infinitive", "perfect_1s", "supine"),
        separators=(" — ", "; ", " / "),
    )


def test_profile_round_trips_through_human_and_machine_readable_json() -> None:
    profile = _profile()

    machine = profile.to_machine_readable()
    human = profile.to_human_readable()

    assert machine["schema_version"] == 2
    assert machine["fields"]["lexical_entry_field"] == "Lemma"
    assert '"schema_version": 2' in human
    assert DeckProfile.from_human_readable(human) == profile


def test_explicit_legacy_terminology_upgrade_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from latinitas_cards.cli import app

    legacy = _profile().to_machine_readable()
    legacy["morphology"].pop("terminology")
    source = tmp_path / "legacy.json"
    target = tmp_path / "upgraded.json"
    source.write_text(json.dumps(legacy))
    with pytest.raises(ProfileValidationError, match="explicit terminology upgrade"):
        DeckProfile.from_mapping(legacy)
    result = CliRunner().invoke(app, ["terminology", "upgrade", str(source), str(target)])
    assert result.exit_code == 0, result.output
    upgraded = json.loads(target.read_text())
    assert upgraded["morphology"]["terminology"] == {"version": 1, "language": "latin"}
    assert DeckProfile.from_mapping(upgraded).morphology.terminology.language == "latin"
    assert json.loads(source.read_text()) == legacy
    assert CliRunner().invoke(app, ["terminology", "upgrade", str(source), str(target)]).exit_code == 2


@pytest.mark.parametrize("setting", [{"version": 2}, {"version": True}, {"version": 1, "language": "english"}, {}])
def test_terminology_unknown_or_unversioned_settings_fail(setting: dict[str, object]) -> None:
    data = _profile().to_machine_readable()
    data["morphology"]["terminology"] = setting
    with pytest.raises(ProfileValidationError):
        DeckProfile.from_mapping(data)


@pytest.mark.parametrize("tag", ["fr", "la", "deu", "eng-US"])
def test_terminology_user_language_fails_closed_naming_supported_tags(tag: str) -> None:
    with pytest.raises(ProfileValidationError, match=r"Unsupported terminology user language.*\(de\).*\(en\)"):
        _profile().apply_overrides({"language_tag": tag, "morphology": {"terminology": {"language": "user"}}})


@pytest.mark.parametrize("tag", ["de", "de-CH", "en", "EN-gb"])
def test_terminology_user_language_accepts_german_and_english(tag: str) -> None:
    profile = _profile().apply_overrides({"language_tag": tag, "morphology": {"terminology": {"language": "user"}}})
    assert (profile.language_tag, profile.morphology.terminology.language) == (tag, "user")


def test_other_language_tags_keep_validating_with_latin_terminology() -> None:
    profile = _profile().apply_overrides({"language_tag": "fr", "morphology": {"terminology": {"language": "latin"}}})
    assert DeckProfile.from_mapping(profile.to_machine_readable()).language_tag == "fr"


def test_profile_defaults_are_explicit_and_language_tags_are_not_implicit() -> None:
    profile = DeckProfile.default(
        note_type="Vocabulary",
        lexical_entry_field="Entry",
        principal_parts_field="Forms",
    )

    assert profile.language_tag == DEFAULT_LANGUAGE_TAG == "de"
    assert profile.generated_note_type == DEFAULT_GENERATED_NOTE_TYPE
    assert profile.target_deck == DEFAULT_TARGET_DECK
    assert profile.tags == DEFAULT_TAGS
    assert profile.selected_recipes == DEFAULT_SELECTED_RECIPES

    english_profile = DeckProfile.default(
        note_type="Vocabulary",
        lexical_entry_field="Entry",
        principal_parts_field="Forms",
        language_tag="en-US",
    )
    assert english_profile.language_tag == "en-US"


def test_profile_save_uses_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile()
    destination = tmp_path / "profile.json"
    replacements: list[tuple[str, str]] = []
    original_replace = os.replace

    def record_replace(source: Any, target: Any) -> None:
        replacements.append((str(source), str(target)))
        original_replace(source, target)

    monkeypatch.setattr("latinitas_cards.profile.os.replace", record_replace)

    profile.save(destination)

    assert replacements
    assert str(replacements[-1][1]) == str(destination)
    assert DeckProfile.from_json(destination.read_text(encoding="utf-8")) == profile


def test_effective_profile_applies_explicit_overrides_deterministically() -> None:
    profile = _profile()
    overrides = ProfileOverrides(
        fields=FieldOverrides(principal_parts_field="Forms v2"),
        principal_parts=PrincipalPartOverrides(separators=(" / ",)),
        language_tag="de-AT",
        target_deck="Latin::Review",
        selected_recipes=("principal_part_recognition",),
    )

    effective_once = resolve_profile(profile, overrides)
    effective_twice = resolve_profile(profile, overrides)

    assert effective_once == effective_twice
    assert effective_once.fields.principal_parts_field == "Forms v2"
    assert effective_once.principal_parts.separators == (" / ",)
    assert effective_once.language_tag == "de-AT"
    assert effective_once.target_deck == "Latin::Review"
    assert effective_once.selected_recipes == ("principal_part_recognition",)
    assert profile.fields.principal_parts_field == "Principal parts"


def test_effective_profile_can_switch_from_field_identity_to_note_guid() -> None:
    profile = _profile()

    effective = resolve_profile(profile, {"source_identity": {"strategy": "note_guid"}})

    assert effective.source_identity == SourceIdentityConfig(strategy="note_guid")


def test_null_source_identity_strategy_override_is_a_no_op() -> None:
    profile = _profile()

    effective = resolve_profile(profile, {"source_identity": {"strategy": None}})

    assert effective == profile


def test_validation_reports_missing_values() -> None:
    with pytest.raises(ProfileValidationError) as raised:
        DeckProfile.from_mapping({"schema_version": 1})

    paths = {issue.path for issue in raised.value.issues}
    assert {"source_identity", "note_type", "fields", "principal_parts"} <= paths


def test_validation_reports_incompatible_and_contradictory_values() -> None:
    payload = _profile().to_machine_readable()
    payload["fields"]["principal_parts_field"] = "Lemma"
    payload["principal_parts"]["separators"] = [" | ", " ; "]

    with pytest.raises(ProfileValidationError) as raised:
        DeckProfile.from_mapping(payload)

    message = str(raised.value)
    assert "must be different" in message
    assert "separator" in message


def test_unsupported_schema_versions_fail_clearly() -> None:
    payload = _profile().to_machine_readable()
    payload["schema_version"] = 99

    with pytest.raises(UnsupportedProfileSchemaError, match="Unsupported profile schema version 99"):
        DeckProfile.from_mapping(payload)


def test_profiles_reject_credentials_without_echoing_secret_values() -> None:
    payload = _profile().to_machine_readable()
    payload["credentials"] = {"token": "do-not-leak"}

    with pytest.raises(ProfileValidationError) as raised:
        DeckProfile.from_mapping(payload)

    assert "credential" in str(raised.value).lower()
    assert "do-not-leak" not in str(raised.value)


def test_invalid_recipe_values_are_not_echoed_in_profile_errors() -> None:
    secret = "SENTINEL_SECRET_VALUE"
    profile_payload = _profile().to_machine_readable()
    profile_payload["selected_recipes"] = [secret]

    with pytest.raises(ProfileValidationError) as profile_error:
        DeckProfile.from_mapping(profile_payload)
    with pytest.raises(ProfileValidationError) as override_error:
        ProfileOverrides.from_mapping({"selected_recipes": [secret]})

    assert secret not in str(profile_error.value)
    assert secret not in str(override_error.value)


def test_validate_profile_returns_issues_without_deck_specific_assumptions() -> None:
    issues = validate_profile({"schema_version": 1, "note_type": "Any note type"})

    assert issues
    assert any(issue.path == "source_identity" for issue in issues)
