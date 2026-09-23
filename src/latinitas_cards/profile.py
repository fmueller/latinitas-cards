"""Versioned, adapter-independent deck profile domain models.

Profiles are stored as indented UTF-8 JSON. JSON keeps the project file both
human-readable and machine-readable without coupling the domain contract to a
prompting flow or a source adapter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CURRENT_PROFILE_SCHEMA_VERSION = 1
SUPPORTED_PROFILE_SCHEMA_VERSIONS = (CURRENT_PROFILE_SCHEMA_VERSION,)

DEFAULT_LANGUAGE_TAG = "de"
DEFAULT_GENERATED_NOTE_TYPE = "Latinitas Principal Parts"
DEFAULT_TARGET_DECK = "Latin::Latinitas"
DEFAULT_TAGS = ("latinitas",)
DEFAULT_PRINCIPAL_PART_ROLES = ("present_1s", "present_infinitive", "perfect_1s", "supine")
DEFAULT_SEPARATORS = (" — ",)
DEFAULT_SELECTED_RECIPES = ("principal_part_completion", "principal_part_recognition")

ProfileSchemaVersion = Literal[1]
SourceIdentityStrategy = Literal["note_guid", "source_id_field", "manifest"]
RecipeName = Literal["principal_part_completion", "principal_part_recognition"]

_LANGUAGE_TAG_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_CREDENTIAL_KEY_PARTS = (
    "access_key",
    "api_key",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_RECIPE_ALIASES = {
    "completion": "principal_part_completion",
    "principal_part_completion": "principal_part_completion",
    "principal_parts_completion": "principal_part_completion",
    "principal_part_recognition": "principal_part_recognition",
    "principal_parts_recognition": "principal_part_recognition",
    "recognition": "principal_part_recognition",
}


def _validated_language_tag(value: str) -> str:
    if not _LANGUAGE_TAG_RE.fullmatch(value):
        raise ValueError("language_tag must be a valid language tag")
    return value


def _validated_tags(value: tuple[str, ...]) -> tuple[str, ...]:
    tags = tuple(tag.strip() for tag in value)
    if any(not tag for tag in tags):
        raise ValueError("tags must not contain empty values")
    if len(set(tags)) != len(tags):
        raise ValueError("tags must be distinct")
    return tags


@dataclass(frozen=True)
class ProfileIssue:
    """A safe, structured profile validation diagnostic."""

    path: str
    code: str
    message: str


class ProfileValidationError(ValueError):
    """Raised when profile data cannot satisfy the domain contract."""

    def __init__(self, issues: Sequence[ProfileIssue]) -> None:
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)
        super().__init__(f"Profile validation failed: {details}")

    @classmethod
    def from_pydantic(cls, error: ValidationError) -> ProfileValidationError:
        issues: list[ProfileIssue] = []
        for item in error.errors():
            location = item.get("loc", ())
            path = ".".join(str(part) for part in location) or "profile"
            message = str(item.get("msg", "invalid value"))
            error_type = str(item.get("type", "invalid"))
            if error_type == "missing":
                code = "missing"
            elif "must be different" in message or "must be distinct" in message:
                code = "contradictory"
            elif error_type.startswith(("literal", "string", "too_short", "too_long")):
                code = "incompatible"
            else:
                code = "invalid"
            issues.append(ProfileIssue(path=path, code=code, message=message))
        return cls(issues)


class UnsupportedProfileSchemaError(ProfileValidationError):
    """Raised when a profile uses a schema version this release cannot read."""

    def __init__(self, version: object) -> None:
        self.version = version
        super().__init__(
            (
                ProfileIssue(
                    path="schema_version",
                    code="unsupported_schema_version",
                    message=(
                        f"Unsupported profile schema version {version!r}; "
                        f"supported versions are {SUPPORTED_PROFILE_SCHEMA_VERSIONS}."
                    ),
                ),
            )
        )


class _ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SourceIdentityConfig(_ProfileModel):
    """How a source record's immutable identity is obtained by an adapter."""

    strategy: SourceIdentityStrategy = "note_guid"
    field: str | None = Field(default=None, min_length=1)

    @field_validator("strategy", mode="before")
    @classmethod
    def _normalise_strategy(cls, value: object) -> object:
        aliases = {
            "source_guid": "note_guid",
            "csv_source_id": "source_id_field",
            "source_id": "source_id_field",
        }
        if isinstance(value, str):
            return aliases.get(value, value)
        return value

    @model_validator(mode="after")
    def _validate_field_for_strategy(self) -> SourceIdentityConfig:
        if self.strategy == "source_id_field" and self.field is None:
            raise ValueError("source_identity.field is required for the source_id_field strategy")
        if self.strategy != "source_id_field" and self.field is not None:
            raise ValueError("source_identity.field is only valid for the source_id_field strategy")
        return self


class SourceIdentityOverrides(_ProfileModel):
    """Partial source identity values used by explicit configuration overrides."""

    strategy: SourceIdentityStrategy | None = None
    field: str | None = Field(default=None, min_length=1)

    @field_validator("strategy", mode="before")
    @classmethod
    def _normalise_strategy(cls, value: object) -> object:
        return SourceIdentityConfig._normalise_strategy(value)


class ProfileFields(_ProfileModel):
    """Named source fields required by profile-driven generation."""

    lexical_entry_field: str = Field(min_length=1)
    principal_parts_field: str = Field(min_length=1)
    meaning_field: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_distinct_fields(self) -> ProfileFields:
        if self.lexical_entry_field == self.principal_parts_field:
            raise ValueError("lexical_entry_field and principal_parts_field must be different")
        if self.meaning_field in {self.lexical_entry_field, self.principal_parts_field}:
            raise ValueError("meaning_field must be different from the source mapping fields")
        return self


class FieldOverrides(_ProfileModel):
    """Partial source field mapping used by explicit configuration overrides."""

    lexical_entry_field: str | None = Field(default=None, min_length=1)
    principal_parts_field: str | None = Field(default=None, min_length=1)
    meaning_field: str | None = Field(default=None, min_length=1)


class PrincipalPartLayout(_ProfileModel):
    """Ordered semantic roles and separators for a principal-part field."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)

    roles: tuple[str, ...] = Field(min_length=2)
    separators: tuple[str, ...] = Field(min_length=1)

    @field_validator("roles")
    @classmethod
    def _validate_roles(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        roles = tuple(role.strip() for role in value)
        if any(not role for role in roles):
            raise ValueError("principal-part roles must not be empty")
        if len(set(roles)) != len(roles):
            raise ValueError("principal-part roles must be distinct")
        return roles

    @field_validator("separators")
    @classmethod
    def _validate_separators(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not separator.strip() for separator in value):
            raise ValueError("principal-part separators must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_separator_count(self) -> PrincipalPartLayout:
        expected = len(self.roles) - 1
        if len(self.separators) not in {1, expected}:
            raise ValueError(
                f"principal-part separators must contain one reusable separator or exactly {expected} separators"
            )
        return self


class PrincipalPartOverrides(_ProfileModel):
    """Partial principal-part layout used by explicit configuration overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)

    roles: tuple[str, ...] | None = Field(default=None, min_length=2)
    separators: tuple[str, ...] | None = Field(default=None, min_length=1)


def _canonicalise_recipes(value: tuple[str, ...]) -> tuple[str, ...]:
    recipes: list[str] = []
    for recipe in value:
        key = recipe.strip().lower().replace("-", "_").replace(" ", "_")
        canonical = _RECIPE_ALIASES.get(key)
        if canonical is None:
            raise ValueError("unsupported selected recipe")
        if canonical in recipes:
            raise ValueError("selected recipes must be distinct")
        recipes.append(canonical)
    if not recipes:
        raise ValueError("selected_recipes must not be empty")
    return tuple(recipes)


class ProfileOverrides(_ProfileModel):
    """Explicit, optional values that replace profile values for one run."""

    source_identity: SourceIdentityOverrides | None = None
    note_type: str | None = Field(default=None, min_length=1)
    fields: FieldOverrides | None = None
    principal_parts: PrincipalPartOverrides | None = None
    language_tag: str | None = Field(default=None, min_length=1)
    generated_note_type: str | None = Field(default=None, min_length=1)
    target_deck: str | None = Field(default=None, min_length=1)
    tags: tuple[str, ...] | None = Field(default=None, min_length=1)
    selected_recipes: tuple[str, ...] | None = Field(default=None, min_length=1)

    @field_validator("language_tag")
    @classmethod
    def _validate_language_tag(cls, value: str | None) -> str | None:
        return None if value is None else _validated_language_tag(value)

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        return None if value is None else _validated_tags(value)

    @field_validator("selected_recipes")
    @classmethod
    def _validate_selected_recipes(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        return None if value is None else _canonicalise_recipes(value)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ProfileOverrides:
        _reject_credential_keys(data)
        try:
            return cls.model_validate(dict(data))
        except ValidationError as error:
            raise ProfileValidationError.from_pydantic(error) from error


class DeckProfile(_ProfileModel):
    """The versioned, adapter-independent contract for one source deck."""

    schema_version: ProfileSchemaVersion
    source_identity: SourceIdentityConfig
    note_type: str = Field(min_length=1)
    fields: ProfileFields
    principal_parts: PrincipalPartLayout
    language_tag: str = DEFAULT_LANGUAGE_TAG
    generated_note_type: str = Field(default=DEFAULT_GENERATED_NOTE_TYPE, min_length=1)
    target_deck: str = Field(default=DEFAULT_TARGET_DECK, min_length=1)
    tags: tuple[str, ...] = Field(default=DEFAULT_TAGS, min_length=1)
    selected_recipes: tuple[str, ...] = Field(default=DEFAULT_SELECTED_RECIPES, min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        if value not in SUPPORTED_PROFILE_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported profile schema version {value}")
        return value

    @field_validator("language_tag")
    @classmethod
    def _validate_language_tag(cls, value: str) -> str:
        return _validated_language_tag(value)

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_tags(value)

    @field_validator("selected_recipes")
    @classmethod
    def _validate_selected_recipes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonicalise_recipes(value)

    @model_validator(mode="after")
    def _validate_source_identity_field(self) -> DeckProfile:
        if self.source_identity.strategy == "source_id_field":
            mapped_fields = {
                self.fields.lexical_entry_field,
                self.fields.principal_parts_field,
                self.fields.meaning_field,
            }
            if self.source_identity.field in mapped_fields:
                raise ValueError("source identity field must be distinct from mapped content fields")
        return self

    @classmethod
    def default(
        cls,
        *,
        note_type: str,
        lexical_entry_field: str,
        principal_parts_field: str,
        meaning_field: str | None = None,
        source_identity: SourceIdentityConfig | None = None,
        principal_part_roles: Sequence[str] = DEFAULT_PRINCIPAL_PART_ROLES,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        language_tag: str = DEFAULT_LANGUAGE_TAG,
        generated_note_type: str = DEFAULT_GENERATED_NOTE_TYPE,
        target_deck: str = DEFAULT_TARGET_DECK,
        tags: Sequence[str] = DEFAULT_TAGS,
        selected_recipes: Sequence[str] = DEFAULT_SELECTED_RECIPES,
    ) -> DeckProfile:
        """Build a valid profile with only source-specific mappings required."""

        return cls.from_mapping(
            {
                "schema_version": CURRENT_PROFILE_SCHEMA_VERSION,
                "source_identity": source_identity or SourceIdentityConfig(),
                "note_type": note_type,
                "fields": {
                    "lexical_entry_field": lexical_entry_field,
                    "principal_parts_field": principal_parts_field,
                    "meaning_field": meaning_field,
                },
                "principal_parts": {
                    "roles": tuple(principal_part_roles),
                    "separators": tuple(separators),
                },
                "language_tag": language_tag,
                "generated_note_type": generated_note_type,
                "target_deck": target_deck,
                "tags": tuple(tags),
                "selected_recipes": tuple(selected_recipes),
            }
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DeckProfile:
        """Load and validate one profile mapping without exposing secret values."""

        _reject_credential_keys(data)
        if "schema_version" not in data:
            raise ProfileValidationError((ProfileIssue("schema_version", "missing", "schema_version is required"),))
        version = data["schema_version"]
        if type(version) is int and version not in SUPPORTED_PROFILE_SCHEMA_VERSIONS:
            raise UnsupportedProfileSchemaError(version)
        try:
            return cls.model_validate(dict(data))
        except ValidationError as error:
            raise ProfileValidationError.from_pydantic(error) from error

    @classmethod
    def from_json(cls, value: str) -> DeckProfile:
        try:
            decoded: object = json.loads(value)
        except json.JSONDecodeError as error:
            raise ProfileValidationError(
                (ProfileIssue("profile", "invalid_json", "profile is not valid JSON"),)
            ) from error
        if not isinstance(decoded, Mapping):
            raise ProfileValidationError(
                (ProfileIssue("profile", "incompatible", "profile JSON must contain an object"),)
            )
        return cls.from_mapping(cast(Mapping[str, Any], decoded))

    from_human_readable = from_json

    def to_machine_readable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    to_dict = to_machine_readable

    def to_human_readable(self) -> str:
        return json.dumps(self.to_machine_readable(), ensure_ascii=False, indent=2) + "\n"

    to_json = to_human_readable

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_human_readable(), encoding="utf-8")

    def apply_overrides(self, overrides: ProfileOverrides | Mapping[str, Any]) -> DeckProfile:
        """Return the deterministic effective profile for one explicit override set."""

        parsed = overrides if isinstance(overrides, ProfileOverrides) else ProfileOverrides.from_mapping(overrides)
        values = self.model_dump(mode="python")
        changes = parsed.model_dump(exclude_none=True, mode="python")
        for nested in ("source_identity", "fields", "principal_parts"):
            nested_changes = changes.pop(nested, None)
            if nested_changes is not None:
                if nested == "source_identity":
                    source_identity = parsed.source_identity
                    if source_identity is not None:
                        strategy_changed = (
                            source_identity.strategy is not None and "strategy" in source_identity.model_fields_set
                        )
                        if strategy_changed and source_identity.strategy != "source_id_field":
                            nested_changes["field"] = None
                values[nested].update(nested_changes)
        values.update(changes)
        return DeckProfile.from_mapping(cast(Mapping[str, Any], values))


def _reject_credential_keys(data: Mapping[str, Any]) -> None:
    path = _find_credential_key(data)
    if path is not None:
        raise ProfileValidationError(
            (
                ProfileIssue(
                    path=path,
                    code="credentials_forbidden",
                    message="credential fields are not permitted in profiles",
                ),
            )
        )


def _find_credential_key(value: object, path: str = "") -> str | None:
    if not isinstance(value, Mapping):
        return None
    for raw_key, nested in value.items():
        key = str(raw_key)
        normalised = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        child_path = f"{path}.{key}" if path else key
        if any(part in normalised.split("_") for part in _CREDENTIAL_KEY_PARTS):
            return child_path
        found = _find_credential_key(nested, child_path)
        if found is not None:
            return found
    return None


def resolve_profile(
    profile: DeckProfile | Mapping[str, Any],
    overrides: ProfileOverrides | Mapping[str, Any] | None = None,
) -> DeckProfile:
    """Load a profile and apply explicit overrides without mutating the source profile."""

    loaded = profile if isinstance(profile, DeckProfile) else DeckProfile.from_mapping(profile)
    return loaded if overrides is None else loaded.apply_overrides(overrides)


def load_profile(path: str | Path) -> DeckProfile:
    return DeckProfile.from_json(Path(path).read_text(encoding="utf-8"))


def load_effective_profile(
    path: str | Path,
    overrides: ProfileOverrides | Mapping[str, Any] | None = None,
) -> DeckProfile:
    return resolve_profile(load_profile(path), overrides)


def validate_profile(data: Mapping[str, Any]) -> tuple[ProfileIssue, ...]:
    """Return structured diagnostics instead of raising for validation-only callers."""

    try:
        DeckProfile.from_mapping(data)
    except ProfileValidationError as error:
        return error.issues
    return ()
