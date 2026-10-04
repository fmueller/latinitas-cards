"""Typed, corpus-independent authored JSONL input with whole-file diagnostics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from .profile import tag_character_violation


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("must be a nonempty string")
    return value


class Provenance(BaseModel):
    """Opaque labels and citations; no normalization or corpus lookup."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    document: str
    section: str
    reference: str | None = None

    _required_labels = field_validator("document", "section")(_nonempty)


class _AuthoredItem(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    schema_version: Literal[1]
    key: str
    provenance: Provenance
    status: Literal["include", "skip"]
    tags: tuple[str, ...] = ()
    language_tag: str
    # Set by the loader, never accepted from the import file.
    line_number: int = Field(default=0, exclude=True)

    _required_key = field_validator("key")(_nonempty)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    @field_validator("language_tag")
    @classmethod
    def _language(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*", value):
            raise ValueError("language_tag must be a valid language tag")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _tags(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list) or any(
            not isinstance(tag, str) or not tag or tag_character_violation(tag) for tag in value
        ):
            raise ValueError("tags must be an array of nonempty strings without whitespace or control characters")
        return tuple(value)


class VocabItem(_AuthoredItem):
    kind: Literal["vocab"]
    lemma: str
    meaning: str
    dictionary_form: str | None = None

    _required_content = field_validator("lemma", "meaning")(_nonempty)


class FormItem(_AuthoredItem):
    kind: Literal["form"]
    text_form: str
    base_form: str
    analysis: str
    translation: str
    context: str | None = None

    _required_content = field_validator("text_form", "base_form", "analysis", "translation")(_nonempty)


class QaItem(_AuthoredItem):
    kind: Literal["qa"]
    question: str
    answer: str

    _required_content = field_validator("question", "answer")(_nonempty)


AuthoredItem = VocabItem | FormItem | QaItem
_ITEM_ADAPTER: TypeAdapter[AuthoredItem] = TypeAdapter(Annotated[AuthoredItem, Field(discriminator="kind")])


@dataclass(frozen=True)
class ImportIssue:
    line_number: int
    field: str
    assumption: str

    def __str__(self) -> str:
        return f"line {self.line_number}, {self.field}: {self.assumption}"


class AuthoredImportError(ValueError):
    def __init__(self, errors: tuple[ImportIssue, ...]) -> None:
        self.errors = errors
        super().__init__("Authored import validation failed: " + "; ".join(map(str, errors)))


@dataclass(frozen=True)
class AuthoredImportResult:
    """Partial items are diagnostic only. Selection/export must call require_valid."""

    diagnostic_items: tuple[AuthoredItem, ...]
    errors: tuple[ImportIssue, ...]

    def require_valid(self) -> tuple[AuthoredItem, ...]:
        if self.errors:
            raise AuthoredImportError(self.errors)
        return self.diagnostic_items


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def load_authored_import(path: Path) -> AuthoredImportResult:
    """Validate every physical line, including skipped items, without selection.

    Read failures propagate as OSError. Recoverable encoding, JSON and schema
    errors are aggregated; later lines still contribute diagnostic items.
    """
    items: list[AuthoredItem] = []
    errors: list[ImportIssue] = []
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                errors.append(ImportIssue(line_number, "encoding", "must be valid UTF-8"))
                continue
            try:
                data = json.loads(text, object_pairs_hook=_unique_object)
            except ValueError as error:
                errors.append(ImportIssue(line_number, "JSON", f"must be valid JSON with unique object keys: {error}"))
                continue
            if not isinstance(data, dict):
                errors.append(ImportIssue(line_number, "item", "must be a JSON object"))
                continue
            if "line_number" in data:
                errors.append(ImportIssue(line_number, "line_number", "is loader metadata, not an input field"))
            try:
                item = _ITEM_ADAPTER.validate_python(data)
            except ValidationError as error:
                for issue in error.errors(include_input=False, include_url=False):
                    field = ".".join(map(str, issue["loc"])) or "kind"
                    errors.append(ImportIssue(line_number, field, issue["msg"]))
            else:
                items.append(item.model_copy(update={"line_number": line_number}))
    return AuthoredImportResult(tuple(items), tuple(errors))
