"""Version-one annotation records and canonical sidecar identity."""

import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    JsonValue,
    RootModel,
    TypeAdapter,
    ValidationError,
    model_serializer,
    model_validator,
)

from .atomic_io import atomic_replace, reject_aliases, writer_lock
from .canonical import canonical_json, digest
from .cli_contract import Code, ContractError
from .corpus_model import CorpusModel, CorpusPassage, LanguageTag, Sha256, VersionOne


def utc_timestamp(value: str) -> str:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value):
        raise ValueError("expected YYYY-MM-DDTHH:MM:SSZ")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return value


def ordered_span(value: list[int]) -> list[int]:
    if value[1] <= value[0]:
        raise ValueError("span must have start < end")
    return value


type Span = Annotated[
    list[Annotated[int, Field(ge=0)]], Field(min_length=2, max_length=2), AfterValidator(ordered_span)
]
type Created = Annotated[str, AfterValidator(utc_timestamp)]
type Layer = Literal["segmentation", "morphology", "translation", "correction"]


def annotation_id(record: dict[str, Any]) -> str:
    return digest({key: value for key, value in record.items() if key not in {"annotation_id", "created", "plan"}})


class AnnotationHeader(VersionOne):
    format: Literal["latinitas-annotations"]
    record: Literal["header"]
    version: Literal[1]


class Producer(CorpusModel):
    kind: Literal["rule", "analyzer", "agent", "llm", "human"]
    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    version: str = Field(min_length=1)


class OptionalFields(CorpusModel):
    @model_validator(mode="before")
    @classmethod
    def no_explicit_nulls(cls, value: Any) -> Any:
        if isinstance(value, dict) and any(item is None and key != "plan" for key, item in value.items()):
            raise ValueError("optional fields must be omitted, not null")
        return value

    @model_serializer(mode="wrap")
    def omit_absent(self, handler: Any) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None or key == "plan"}


class Anchor(OptionalFields):
    passage_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}:(?:[A-Za-z0-9._-]|%[0-9A-F]{2})+$", max_length=256)
    passage_hash: Sha256
    segmentation_id: Sha256 | None = None
    span: Span | None = None


class Translation(CorpusModel):
    language: LanguageTag
    text: str = Field(min_length=1)


class Correction(CorpusModel):
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def nfc(self) -> Self:
        if not unicodedata.is_normalized("NFC", self.text):
            raise ValueError("correction text must be NFC")
        return self


class SegmentationToken(CorpusModel):
    kind: Literal["word", "punct"]
    span: Span


class SegmentationSentence(CorpusModel):
    continued: bool
    continues: bool
    span: Span
    tokens: Span


class Segmentation(CorpusModel):
    sentences: list[SegmentationSentence]
    tokenizer: Literal["latin-tok/1"]
    tokens: list[SegmentationToken]

    @model_validator(mode="after")
    def partition(self) -> Self:
        previous = 0
        for token in self.tokens:
            if token.span[0] < previous:
                raise ValueError("token spans overlap or are unordered")
            previous = token.span[1]
        index = 0
        for sentence in self.sentences:
            start, end = sentence.tokens
            if start != index or end > len(self.tokens):
                raise ValueError("sentences must partition tokens")
            if sentence.span != [self.tokens[start].span[0], self.tokens[end - 1].span[1]]:
                raise ValueError("sentence span must cover its tokens")
            index = end
        if index != len(self.tokens):
            raise ValueError("sentences must cover all tokens")
        return self


def validate_value(layer: Layer, value: JsonValue) -> None:
    canonical_json(value)
    if layer == "morphology":
        if not isinstance(value, dict) or not isinstance(value.get("tagset"), str) or not value["tagset"]:
            raise ValueError("morphology requires a non-empty tagset")
    elif layer == "translation":
        Translation.model_validate(value)
    elif layer == "correction":
        Correction.model_validate(value)
    else:
        Segmentation.model_validate(value)


class AnnotationRecord(OptionalFields):
    annotation_id: Sha256
    layer: Layer
    anchor: Anchor
    producer: Producer
    input_hash: Sha256
    created: Created
    plan: Sha256 | None

    @model_validator(mode="after")
    def identity(self) -> Self:
        if self.input_hash != self.anchor.passage_hash:
            raise ValueError("input_hash must match anchor.passage_hash")
        raw = self.model_dump(mode="json")
        if annotation_id(raw) != self.annotation_id:
            raise ValueError("annotation_id does not recompute")
        return self


class Annotation(AnnotationRecord):
    record: Literal["annotation"]
    value: JsonValue
    alternatives: list[JsonValue] | None = None
    confidence: Annotated[int, Field(ge=0, le=1000)] | None = None
    supersedes: Sha256 | None = None

    @model_validator(mode="after")
    def layer_value(self) -> Self:
        validate_value(self.layer, self.value)
        for value in self.alternatives or []:
            validate_value(self.layer, value)
        if self.layer == "correction" and self.anchor.span is None:
            raise ValueError("correction requires a span")
        return self


class Retraction(AnnotationRecord):
    record: Literal["retraction"]
    retracts: Sha256

    def validate_target(self, target: Annotation | None) -> None:
        if target is None or (self.layer, self.anchor, self.input_hash) != (
            target.layer,
            target.anchor,
            target.input_hash,
        ):
            raise ValueError("retraction must copy the retracted annotation binding")


type SidecarRecord = Annotated[AnnotationHeader | Annotation | Retraction, Field(discriminator="record")]


class AnnotationSchema(RootModel[SidecarRecord]):
    """Schema for each line of the sidecar, including its strict header."""


RECORD_ADAPTER: TypeAdapter[SidecarRecord] = TypeAdapter(SidecarRecord)


def validate_anchor(
    record: Annotation | Retraction, passage: CorpusPassage, segmentation: Annotation | None = None
) -> None:
    """Check code-point bounds and, when supplied, the referenced token boundaries."""
    anchor = record.anchor
    if anchor.passage_id != passage.passage_id or anchor.passage_hash != passage.passage_hash:
        raise ValueError("anchor does not bind this passage")
    if anchor.span is not None and anchor.span[1] > len(passage.text):
        raise ValueError("anchor span exceeds passage text")
    if isinstance(record, Annotation) and record.layer == "segmentation":
        value = Segmentation.model_validate(record.value)
        if value.tokens and value.tokens[-1].span[1] > len(passage.text):
            raise ValueError("segmentation exceeds passage text")
    if segmentation is not None:
        if segmentation.layer != "segmentation" or anchor.segmentation_id != segmentation.annotation_id:
            raise ValueError("anchor must reference a segmentation")
        if segmentation.anchor.passage_id != anchor.passage_id or segmentation.input_hash != anchor.passage_hash:
            raise ValueError("segmentation binds a different passage")
        value = Segmentation.model_validate(segmentation.value)
        boundaries = {offset for token in value.tokens for offset in token.span}
        if anchor.span is not None and not all(offset in boundaries for offset in anchor.span):
            raise ValueError("anchor span must end on token boundaries")


def read_sidecar(path: Path) -> Iterator[SidecarRecord]:
    """Stream strict records, gating the envelope before parsing annotations."""
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            seen = False
            annotations: dict[str, Annotation] = {}
            for number, line in enumerate(stream, 1):
                seen = True
                try:
                    raw = json.loads(line)
                except ValueError as error:
                    raise ContractError(Code.INVALID_JSON, f"Line {number}: {error}", path=str(path)) from error
                if number == 1 and isinstance(raw, dict):
                    if raw.get("format") != "latinitas-annotations":
                        raise ContractError(Code.UNSUPPORTED_FORMAT, "Unknown annotation format", path=str(path))
                    if type(raw.get("version")) is not int or raw["version"] > 1:
                        raise ContractError(Code.UNSUPPORTED_VERSION, "Unsupported annotation version", path=str(path))
                try:
                    record = RECORD_ADAPTER.validate_python(raw)
                    if line != canonical_json(raw) + "\n":
                        raise ValueError("sidecar lines must be canonical JSON plus newline")
                    if (number == 1) != isinstance(record, AnnotationHeader):
                        raise ValueError("only the first line must be a header")
                    if isinstance(record, Retraction):
                        record.validate_target(annotations.get(record.retracts))
                    elif isinstance(record, Annotation):
                        annotations[record.annotation_id] = record
                    yield record
                except (ValidationError, ValueError) as error:
                    raise ContractError(Code.INVALID_SIDECAR, f"Line {number}: {error}", path=str(path)) from error
            if not seen:
                raise ContractError(Code.MISSING_HEADER, "Annotation stream is empty", path=str(path))
    except (OSError, UnicodeError) as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error


def resolve_created(flag: str | None, env: Mapping[str, str]) -> str:
    """Resolve reproducible provenance, never silently substituting a wall clock."""
    value = flag if flag is not None else env.get("SOURCE_DATE_EPOCH")
    try:
        if flag is not None:
            return utc_timestamp(flag)
        if value is not None and re.fullmatch(r"[0-9]+", value):
            instant = datetime.fromtimestamp(int(value), UTC)
            return utc_timestamp(instant.isoformat(timespec="seconds").replace("+00:00", "Z"))
    except (ValueError, OverflowError, OSError):
        pass
    raise ContractError(
        Code.TIMESTAMP_REQUIRED, f"Expected UTC --timestamp or non-negative SOURCE_DATE_EPOCH; got {value!r}"
    )


def append_records(
    path: Path,
    records: Iterable[Annotation | Retraction]
    | Callable[[list[Annotation | Retraction]], Iterable[Annotation | Retraction]],
    protected: Iterable[Path],
) -> int:
    """Select and append while holding the writer lock, including no-op retries."""
    try:
        protected = tuple(protected)
        lock = Path(f"{path.resolve()}.lock")
        reject_aliases(path, [*protected, lock, Path(f"{path}.lock")])
        reject_aliases(lock, protected)
        with writer_lock(path):
            reject_aliases(path, [*protected, lock])
            existing: list[SidecarRecord] = (
                list(read_sidecar(path))
                if path.exists()
                else [AnnotationHeader(format="latinitas-annotations", record="header", version=1)]
            )
            current = [record for record in existing if isinstance(record, (Annotation, Retraction))]
            known = {record.annotation_id for record in current}
            annotations = {record.annotation_id: record for record in current if isinstance(record, Annotation)}
            additions: list[Annotation | Retraction] = []
            for record in records(current) if callable(records) else records:
                if isinstance(record, Retraction):
                    try:
                        record.validate_target(annotations.get(record.retracts))
                    except ValueError as error:
                        raise ContractError(Code.INVALID_SIDECAR, str(error), path=str(path)) from error
                if record.annotation_id not in known:
                    additions.append(record)
                    known.add(record.annotation_id)
                    if isinstance(record, Annotation):
                        annotations[record.annotation_id] = record
            if additions or not path.exists():
                data = "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in [*existing, *additions])
                atomic_replace(path, data.encode("utf-8"))
            return len(additions)
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error
