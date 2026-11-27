"""Approved opt-in profile and deterministic recipe resolution, without execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    model_serializer,
    model_validator,
)

from .acceptance_policy import SupportRegistry, lookup_policy
from .acceptance_records import Hash, Record, Text, canonical, decode_json
from .acceptance_registry import SUPPORT_REGISTRY
from .hosted_accounting import EstimatedAccounting
from .providers import Endpoint

RECIPES = tuple(e.recipe for e in SupportRegistry.read_json(canonical(SUPPORT_REGISTRY)).entries)
DEFAULT_RECIPES = ("form_translation", "ppp_translation", "phrase_translation")
PRODUCTION_RECIPES = ("verb_production", "noun_declension", "adjective_declension")
GRAMMAR_STUDY = (*PRODUCTION_RECIPES, "formation_rule_recall", "component_explanation", "mnemonic_recall")


class RecipeOverrides(Record):
    include: tuple[Text, ...]
    exclude: tuple[Text, ...]

    @model_validator(mode="after")
    def choices(self) -> Self:
        if set(self.include) & set(self.exclude) or not set((*self.include, *self.exclude)) <= set(RECIPES):
            raise ValueError("conflicting or unknown recipe override")
        if len(set(self.include)) != len(self.include) or len(set(self.exclude)) != len(self.exclude):
            raise ValueError("duplicate recipe override")
        return self


def resolve_recipes(preset: str | None, *overrides: RecipeOverrides) -> tuple[str, ...]:
    if preset not in (None, "grammar-study"):
        raise ValueError("unknown preset")
    result = list(GRAMMAR_STUDY if preset else DEFAULT_RECIPES)
    for override in overrides:
        result = [recipe for recipe in result if recipe not in override.exclude]
        result.extend(recipe for recipe in override.include if recipe not in result)
    return tuple(result)


class FileReference(Record):
    path: Text
    sha256: Hash


class InputReference(FileReference):
    kind: Literal["deck", "phrases", "corpus", "curated"]


class Selection(Record):
    patterns: tuple[Text, ...]
    entries: tuple[Text, ...]
    variants_per_binding: int = Field(gt=0, le=32)
    output_limit: int | None = Field(gt=0)


class Processing(Record):
    max_source_units: int | None = Field(default=None, gt=0)
    max_calls: int = Field(default=200, gt=0)
    max_tokens: int = Field(default=200000, gt=0)
    timeout_seconds: int = Field(default=60, gt=0, le=60)
    retries: int = Field(default=1, ge=0, le=1)
    max_output_tokens: int = Field(default=2048, gt=0, le=8192)

    @model_serializer(mode="wrap")
    def historical_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result: dict[str, object] = handler(self)
        if self.max_output_tokens == 2048:
            result.pop("max_output_tokens", None)
        return result

    @classmethod
    def read_json(cls, data: bytes | str) -> Self:
        return cls.model_validate_json(canonical(decode_json(data)))

    @model_validator(mode="before")
    @classmethod
    def json_fields(cls, value: object, info: ValidationInfo) -> object:
        # Preserve historical profile bytes/digests when the allowance is unchanged.
        if info.mode == "json" and isinstance(value, dict):
            value = {"max_output_tokens": 2048, **value}
        if info.mode == "json" and (not isinstance(value, dict) or set(value) != set(cls.model_fields)):
            raise ValueError("record requires exactly its versioned keys")
        return value


class EstimatedProcessing(Processing):
    accounting: EstimatedAccounting
    max_calls: int = Field(default=200, gt=0, le=200)
    max_tokens: int = Field(default=200000, gt=0, le=200000)
    retries: Literal[0] = 0


class Providers(Record):
    proposal: Endpoint | None
    counter_check: Endpoint | None

    @model_validator(mode="after")
    def paired(self) -> Self:
        if (self.proposal is None) != (self.counter_check is None):
            raise ValueError("both provider endpoints or neither are required")
        return self


class Policy(Record):
    id: Text
    version: Text
    sha256: Hash


class Layout(Record):
    id: Text
    version: int = Field(gt=0)
    template_sha256: Hash


class ProfileSnapshot(Record):
    """Exact profile data for reporting; does not establish installed policy authority."""

    format: Literal["latinitas-pipeline-profile"]
    version: Literal[1, 2]
    deck_profile: FileReference
    inputs: tuple[InputReference, ...] = Field(min_length=1)
    language: Literal["de", "en"]
    preset: Literal["grammar-study"] | None
    recipe_overrides: RecipeOverrides
    recipes: tuple[Text, ...]
    selection: Selection
    processing: SerializeAsAny[Processing | EstimatedProcessing]
    providers: Providers
    policy: Policy
    layout: Layout

    @model_validator(mode="after")
    def effective(self) -> Self:
        if (self.version == 2) != isinstance(self.processing, EstimatedProcessing):
            raise ValueError("estimated accounting requires profile version 2")
        if self.recipes != resolve_recipes(self.preset, self.recipe_overrides):
            raise ValueError("saved recipes differ from resolved preset/overrides")
        return self

    @property
    def accounting(self) -> EstimatedAccounting | None:
        return self.processing.accounting if isinstance(self.processing, EstimatedProcessing) else None


class PipelineProfile(ProfileSnapshot):
    @model_validator(mode="after")
    def installed_policy(self) -> Self:
        lookup_policy(self.policy.id, self.policy.version, self.policy.sha256)
        return self

    def overridden(self, overrides: RecipeOverrides) -> PipelineProfile:
        """Merge CLI choices into saved overrides without mutating the source."""
        include = tuple(r for r in self.recipe_overrides.include if r not in overrides.exclude)
        exclude = tuple(r for r in self.recipe_overrides.exclude if r not in overrides.include)
        merged = RecipeOverrides(
            include=tuple(dict.fromkeys((*include, *overrides.include))),
            exclude=tuple(dict.fromkeys((*exclude, *overrides.exclude))),
        )
        return PipelineProfile.read_json(
            canonical(
                {
                    **self.model_dump(),
                    "recipe_overrides": merged,
                    "recipes": resolve_recipes(self.preset, merged),
                }
            )
        )


def load_pipeline_profile(path: Path) -> PipelineProfile:
    return PipelineProfile.read_json(path.read_bytes())
