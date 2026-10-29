"""Strict latinitas-corpus-profile v1: corpora, recipes, budgets and ranking inputs.

The profile holds no file paths and never enters claim fingerprints. Reading ranges are
checked against actual corpora only where they are used.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, ValidationError, ValidatorFunctionWrapHandler, field_validator, model_validator

from .canonical import canonical_json, digest
from .cli_contract import Code, ContractError
from .corpus_model import CorpusId, CorpusModel, LanguageTag, VersionOne, latin_fold
from .profile import MorphologySettings

type CorpusRecipe = Literal[
    "corpus_lexical_recognition",
    "corpus_guided_analysis",
    "corpus_phrase_translation",
    "corpus_sentence_translation",
]
type Budget = Annotated[int, Field(ge=1, le=1000)]
type TokenCount = Annotated[int, Field(ge=1, le=200)]
type Weight = Annotated[int, Field(ge=-100, le=100)]
# Non-blank, without surrounding whitespace, CR or LF; rejected, never stripped.
type Name = Annotated[str, Field(pattern=r"^\S(?:[^\r\n]*\S)?$")]


class RecipeBudget(CorpusModel):
    recipe: CorpusRecipe
    budget: Budget


class Budgets(CorpusModel):
    per_run: Budget = 50
    per_work: Budget = 200


class ReadingRange(CorpusModel):
    corpus_id: CorpusId
    work: str
    from_: str = Field(alias="from")
    to: str


class Limits(CorpusModel):
    min_tokens: TokenCount = 3
    ideal_tokens: TokenCount = 12
    max_tokens: TokenCount = 30
    max_unknown: Annotated[int, Field(ge=0, le=200)] | None = None

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if not self.min_tokens <= self.ideal_tokens <= self.max_tokens:
            raise ValueError("limits require min_tokens <= ideal_tokens <= max_tokens")
        return self


class Weights(CorpusModel):
    reading: Weight = 40
    target: Weight = 30
    novelty: Weight = 20
    length: Weight = 10
    unknown: Weight = 20
    coverage: Weight = 10


class CorpusProfile(VersionOne):
    format: Literal["latinitas-corpus-profile"]
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    corpora: list[CorpusId] = Field(min_length=1)
    target_language: LanguageTag = "de"
    note_type: Name = "Latinitas Corpus Context v1"
    deck: Name
    recipes: list[RecipeBudget] = Field(min_length=1)
    budgets: Budgets = Field(default_factory=Budgets)
    reading: list[ReadingRange] = Field(default_factory=list)
    reading_only: bool = False
    targets: list[Name] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    weights: Weights = Field(default_factory=Weights)
    morphology: MorphologySettings = Field(default_factory=MorphologySettings)

    @field_validator("target_language", mode="wrap")
    @classmethod
    def supported_language(cls, value: object, handler: ValidatorFunctionWrapHandler) -> str:
        message = "Unsupported target language; German (de) or English (en) is required"
        try:
            tag: str = handler(value)
        except ValidationError as error:
            raise ValueError(message) from error
        if tag.lower().split("-")[0] not in {"de", "en"}:
            raise ValueError(message)
        return tag

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.corpora != sorted(set(self.corpora)):
            raise ValueError("corpora must be sorted and unique")
        recipes = [entry.recipe for entry in self.recipes]
        if len(recipes) != len(set(recipes)):
            raise ValueError("each recipe may appear at most once")
        if any(entry.corpus_id not in self.corpora for entry in self.reading):
            raise ValueError("reading corpus_id must be listed in corpora")
        folded = [latin_fold(target) for target in self.targets]
        if folded != sorted(set(folded)):
            raise ValueError("targets must be sorted and unique after latin-fold/1")
        return self


def load_corpus_profile(path: Path) -> CorpusProfile:
    """Gate the document envelope, then validate strictly without normalizing input."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(Code.INVALID_PROFILE, f"Duplicate JSON key: {key!r}", path=str(path))
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error
    try:
        data = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ContractError(Code.INVALID_JSON, str(error), path=str(path)) from error
    if isinstance(data, dict):
        if "format" in data and data["format"] != "latinitas-corpus-profile":
            raise ContractError(Code.UNSUPPORTED_FORMAT, "Expected latinitas-corpus-profile", path=str(path))
        if type(data.get("version")) is int and data["version"] != 1:
            raise ContractError(Code.UNSUPPORTED_VERSION, "Expected version 1", path=str(path))
    try:
        canonical_json(data)
        return CorpusProfile.model_validate(data)
    except (ValueError, RecursionError) as error:
        raise ContractError(Code.INVALID_PROFILE, str(error), path=str(path)) from error


def profile_digest(profile: CorpusProfile) -> str:
    """Structured digest of the validated profile, with defaults applied."""
    return digest(profile.model_dump(mode="json"))


def require_profile_corpora(profile: CorpusProfile, corpus_ids: Iterable[str]) -> None:
    """Reject corpora the profile does not list and listed corpora that are missing."""
    given = set(corpus_ids)
    unlisted = sorted(given - set(profile.corpora))
    missing = sorted(set(profile.corpora) - given)
    problems = []
    if unlisted:
        problems.append(f"unlisted corpora: {', '.join(unlisted)}")
    if missing:
        problems.append(f"missing corpora: {', '.join(missing)}")
    if problems:
        raise ContractError(Code.PROFILE_MISMATCH, f"Profile {profile.profile_id}: {'; '.join(problems)}")
