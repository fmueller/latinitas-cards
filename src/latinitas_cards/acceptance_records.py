"""Strict v1 acceptance records; integrity is separate from export eligibility."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal, Self, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .canonical import canonical_json
from .claim_review import ClaimKind, Evidence

type Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
type Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type Timestamp = Annotated[str, Field(pattern=r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z$")]
type Method = Literal["library", "rule", "blind_model", "semantic_comparison", "human"]


def canonical(value: object) -> bytes:
    def encode(item: object) -> object:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        if isinstance(item, Evidence):
            return {"source": item.source, "reference": item.reference, "version": item.version, "text": item.text}
        raise TypeError(f"not canonical JSON: {type(item).__name__}")

    return canonical_json(json.loads(json.dumps(value, default=encode, allow_nan=False))).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def json_fields(cls, value: object, info: ValidationInfo) -> object:
        if info.mode != "json":
            return value
        if not isinstance(value, dict) or set(value) != set(cls.model_fields):
            raise ValueError("record requires exactly its versioned keys")
        # Before-model validation exposes JSON arrays as Python lists; preserve
        # strict scalar validation while restoring the declared tuple shape.
        return {
            key: tuple(item)
            if isinstance(item, list) and get_origin(cls.model_fields[key].annotation) is tuple
            else item
            for key, item in value.items()
        }

    @classmethod
    def read_json(cls, data: bytes | str) -> Self:
        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        raw = json.loads(data, object_pairs_hook=unique)
        if not isinstance(raw, dict) or set(raw) != set(cls.model_fields):
            raise ValueError("record requires exactly its versioned keys")
        return cls.model_validate_json(canonical(raw))

    @field_validator("version", mode="before", check_fields=False)
    @classmethod
    def integer_version(cls, value: object) -> object:
        if "format" in cls.model_fields and type(value) is not int:
            raise ValueError("unsupported_version: version must be integer 1")
        return value

    @field_validator("created_at", "started_at", "finished_at", "approved_at", mode="after", check_fields=False)
    @classmethod
    def timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @model_validator(mode="after")
    def unique_arrays(self) -> Self:
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, tuple):
                keys = [getattr(item, "id", item) for item in value]
                if len({canonical(key) for key in keys}) != len(keys):
                    raise ValueError(f"duplicate {name}")
        return self


class Binding(Record):
    source_scope: Text
    source_identity: Text
    source_sha256: Hash
    extraction_sha256: Hash
    mapping_sha256: Hash
    task_sha256: Hash
    language: Literal["de", "en"]
    config_sha256: Hash


class Versions(Record):
    producer: Text
    analyzer: Text
    proposal_prompt: Text | None
    counter_prompt: Text | None
    checker: Text
    comparison: Text
    policy: Text
    policy_sha256: Hash
    proposal_model: Text | None
    counter_model: Text | None

    @model_validator(mode="after")
    def model_path(self) -> Self:
        fields = (self.proposal_prompt, self.counter_prompt, self.proposal_model, self.counter_model)
        if any(v is not None for v in fields) and any(v is None for v in fields):
            raise ValueError("model path needs both immutable prompts and deployments")
        return self


class Assertion(Record):
    id: Text
    kind: ClaimKind
    value: Text
    selected_sense: Text | None
    alternatives: tuple[Text, ...]
    evidence: tuple[Evidence, ...]

    @model_validator(mode="after")
    def sense(self) -> Self:
        if (self.kind == "translation") != (self.selected_sense is not None):
            raise ValueError("only translation requires a selected sense")
        return self


class Check(Record):
    id: Text
    assertion_ids: tuple[Text, ...] = Field(min_length=1)
    method: Method
    version: Text
    result: Literal["pass", "fail", "unresolved", "not_applicable"]
    evidence: tuple[Evidence, ...]
    alternatives: tuple[Text, ...]
    objections: tuple[Text, ...]
    input_sha256: Hash

    @model_validator(mode="after")
    def passing(self) -> Self:
        if self.result == "pass" and (self.objections or not self.evidence):
            raise ValueError("pass requires evidence and no objections")
        return self


class CheckEnvelope(Record):
    binding: Binding
    versions: Versions
    method: Method
    checked_assertions: tuple[Assertion, ...]
    prerequisite_output_sha256s: tuple[Hash, ...]
    reference_constraints: tuple[Evidence, ...]


class Human(Record):
    reviewer: Text
    reason: Text
    reviewed_sha256: Hash


class Acceptance(Record):
    format: Literal["latinitas-acceptance"] = "latinitas-acceptance"
    version: Literal[1] = 1
    decision_id: Hash
    binding: Binding
    versions: Versions
    assertions: tuple[Assertion, ...]
    checks: tuple[Check, ...]
    origin: Literal["human", "automatic", "none"]
    state: Literal["accepted", "withheld", "failed", "stale"]
    human: Human | None
    stage_ids: tuple[Text, ...]
    reason: Text
    created_at: Timestamp

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if digest(self.model_dump(exclude={"decision_id"})) != self.decision_id:
            raise ValueError("decision digest mismatch")
        if (self.origin == "human") != (self.human is not None):
            raise ValueError("human origin requires a distinct attestation")
        ids = {a.id for a in self.assertions}
        if any(not set(c.assertion_ids) <= ids for c in self.checks):
            raise ValueError("dangling assertion ID")
        if self.human and self.human.reviewed_sha256 != digest(
            {"binding": self.binding, "assertions": self.assertions}
        ):
            raise ValueError("human review binding mismatch")
        if self.state == "accepted":
            covered = {i for c in self.checks if c.result == "pass" for i in c.assertion_ids}
            if (
                self.origin == "none"
                or not ids
                or covered != ids
                or not self.stage_ids
                or any(not a.evidence for a in self.assertions)
                or any(c.result != "pass" for c in self.checks)
            ):
                raise ValueError("accepted decision lacks complete passing evidence")
        return self

    @classmethod
    def create(cls, **values: object) -> Acceptance:
        data = {"format": "latinitas-acceptance", "version": 1, **values}
        data["decision_id"] = digest(data)
        return cls.model_validate_json(canonical(data))


type StageName = Literal["ingest", "analyze", "propose", "counter_check", "decide", "select", "render", "export"]


class Stage(Record):
    format: Literal["latinitas-stage"] = "latinitas-stage"
    version: Literal[1] = 1
    run_id: Text
    stage_id: Text
    stage: StageName
    binding: Binding
    versions: Versions
    prerequisite_ids: tuple[Text, ...]
    state: Literal["pending", "running", "succeeded", "failed", "cancelled", "unknown", "deferred"]
    input_sha256: Hash
    output_sha256: Hash | None
    artifact: Text | None
    call_ids: tuple[Text, ...]
    reason: Text | None
    started_at: Timestamp | None
    finished_at: Timestamp | None

    @model_validator(mode="after")
    def outcome(self) -> Self:
        if self.state == "succeeded":
            if self.output_sha256 is None or self.artifact is None or self.finished_at is None:
                raise ValueError("success requires finished artifact")
        elif self.output_sha256 is not None or self.artifact is not None:
            raise ValueError("only successful stages have output")
        return self

    def reuse_key(self, prerequisite_output_sha256s: tuple[str, ...]) -> str:
        if len(prerequisite_output_sha256s) != len(self.prerequisite_ids):
            raise ValueError("missing prerequisite output commitments")
        return digest(
            {
                "stage": self.stage,
                "input_sha256": self.input_sha256,
                "binding": self.binding,
                "versions": self.versions,
                "prerequisite_output_sha256s": prerequisite_output_sha256s,
            }
        )
