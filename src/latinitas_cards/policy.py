"""Versioned sidecar-write policy; never a linguistic acceptance decision."""

import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from .canonical import digest_bytes
from .cli_contract import Code, ContractError

type Operation = Literal["add", "supersede", "retract"]
type Layer = Literal["segmentation", "morphology", "translation", "correction"]
type ProducerKind = Literal["rule", "analyzer", "agent", "llm", "human"]
type Action = Literal["auto", "review", "deny"]


class PolicyRule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operation: Operation | Literal["*"]
    layer: Layer | Literal["*"]
    producer_kind: ProducerKind | Literal["*"]
    producer: str
    action: Action

    @model_validator(mode="after")
    def auto_only_add(self) -> Self:
        if self.action == "auto" and self.operation != "add":
            raise ValueError("auto requires an explicit add operation")
        return self


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    format: Literal["latinitas-policy"]
    version: Literal[1]
    rules: list[PolicyRule]
    receipt_actor: Literal["human", "any"] = "human"

    @field_validator("version", mode="before")
    @classmethod
    def strict_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("version must be a strict integer")
        return value


def load_policy(path: Path) -> Policy:
    """Load a policy with stable diagnostics shared by all future consumers."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(Code.INVALID_POLICY, f"Duplicate JSON key: {key}", path=str(path))
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error
    try:
        data = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as error:
        raise ContractError(Code.INVALID_JSON, str(error), path=str(path)) from error
    if isinstance(data, dict):
        if "format" in data and data["format"] != "latinitas-policy":
            raise ContractError(Code.UNSUPPORTED_FORMAT, "Expected latinitas-policy", path=str(path))
        if type(data.get("version")) is int and data["version"] != 1:
            raise ContractError(Code.UNSUPPORTED_VERSION, "Expected version 1", path=str(path))
    try:
        return Policy.model_validate(data)
    except ValidationError as error:
        raise ContractError(Code.INVALID_POLICY, str(error), path=str(path)) from error


def policy_digest(path: Path | None) -> str | None:
    """Hash exact file bytes, not a parsed or normalized policy."""
    if path is None:
        return None
    try:
        return digest_bytes(path.read_bytes())
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error


class Route(NamedTuple):
    action: Action
    reasons: tuple[str, ...]


def route(
    operation: Operation,
    layer: Layer,
    producer_kind: ProducerKind,
    producer_name: str,
    destructive: bool,
    conflicting: bool,
    policy: Policy | None,
) -> Route:
    """First match wins; risk facts can demote auto, never override deny."""
    action: Action = "auto" if policy is None and operation == "add" else "review"
    reasons = ["no_policy" if policy is None else "default_review"]
    if policy is not None:
        for index, rule in enumerate(policy.rules):
            if (
                rule.operation in ("*", operation)
                and rule.layer in ("*", layer)
                and rule.producer_kind in ("*", producer_kind)
                and fnmatchcase(producer_name, rule.producer)
            ):
                action = rule.action
                reasons = [f"rule:{index}"]
                break
    if destructive:
        reasons.append("destructive")
    if conflicting:
        reasons.append("conflicting")
    if (destructive or conflicting) and action == "auto":
        action = "review"
    return Route(action, tuple(reasons))
