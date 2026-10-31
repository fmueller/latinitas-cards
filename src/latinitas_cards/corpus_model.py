"""Version-one corpus records and frozen text/identity primitives."""

import re
import unicodedata
from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)

from .canonical import digest, digest_bytes

type CorpusId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")]
type NativeRef = Annotated[str, Field(pattern=r"^(?:[A-Za-z0-9._-]|%[0-9A-F]{2})+$")]
type Sha256 = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
type PositiveOrder = Annotated[int, Field(ge=1)]

CTS_EDITION_PATTERN = r"^urn:cts:[A-Za-z0-9_-]+:[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$"

_SAFE_REF_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")

# RFC 5646 well-formed syntax, including private-use and grandfathered tags.
_GRANDFATHERED = (
    "en-GB-oed|i-ami|i-bnn|i-default|i-enochian|i-hak|i-klingon|i-lux|i-mingo|i-navajo|i-pwn|i-tao|i-tay|i-tsu|"
    "sgn-BE-FR|sgn-BE-NL|sgn-CH-DE|art-lojban|cel-gaulish|no-bok|no-nyn|zh-guoyu|zh-hakka|zh-min|zh-min-nan|zh-xiang"
)
_LANGUAGE_PATTERN = (
    r"^(?:(?:[A-Za-z]{2,3}(?:-[A-Za-z]{3}){0,3}|[A-Za-z]{4}|[A-Za-z]{5,8})"
    r"(?:-[A-Za-z]{4})?(?:-(?:[A-Za-z]{2}|[0-9]{3}))?(?:-(?:[A-Za-z0-9]{5,8}|[0-9][A-Za-z0-9]{3}))*"
    r"(?:-[0-9A-WY-Za-wy-z](?:-[A-Za-z0-9]{2,8})+)*(?:-[xX](?:-[A-Za-z0-9]{1,8})+)?"
    r"|[xX](?:-[A-Za-z0-9]{1,8})+|"
    + re.sub(r"[A-Za-z]", lambda match: f"[{match[0].lower()}{match[0].upper()}]", _GRANDFATHERED)
    + r")$"
)


def _unique_language_subtags(value: str) -> str:
    subtags = value.lower().split("-")
    if subtags[0] == "x":
        return value
    variants: set[str] = set()
    singletons: set[str] = set()
    in_extension = False
    for subtag in subtags[1:]:
        if subtag == "x":
            break
        if len(subtag) == 1:
            if subtag in singletons:
                raise ValueError("language tag repeats an extension singleton")
            singletons.add(subtag)
            in_extension = True
        elif not in_extension and (len(subtag) >= 5 or (len(subtag) == 4 and subtag[0].isdigit())):
            if subtag in variants:
                raise ValueError("language tag repeats a variant")
            variants.add(subtag)
    return value


type LanguageTag = Annotated[str, Field(pattern=_LANGUAGE_PATTERN), AfterValidator(_unique_language_subtags)]


def encode_ref_component(value: str) -> str:
    return "".join(
        chr(byte) if byte in _SAFE_REF_BYTES else f"%{byte:02X}"
        for byte in unicodedata.normalize("NFC", value).encode("utf-8")
    )


def canonical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def passage_hash(text: str) -> str:
    return digest_bytes(canonical_text(text).encode("utf-8"))


def latin_fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFC", text))
    stripped = "".join(char for char in decomposed if not unicodedata.category(char).startswith("M"))
    return stripped.casefold().replace("æ", "ae").replace("œ", "oe").replace("j", "i")


class CorpusModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", serialize_by_alias=True)


class VersionOne(CorpusModel):
    version: Literal[1]

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("version must be an integer")
        return value


class CorpusDescriptor(VersionOne):
    format: Literal["latinitas-corpus-descriptor"]
    edition: str | None = None
    publisher: str | None = None
    url: str | None = None
    revision: str | None = None
    license: str | None = None
    title: str | None = None
    period: str | None = None
    register_: str | None = Field(default=None, alias="register")

    @model_serializer(mode="wrap")
    def omit_absent_revision(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return {key: value for key, value in handler(self).items() if key != "revision" or value is not None}


class CorpusSource(CorpusModel):
    file_name: str = Field(min_length=1)
    sha256: Sha256
    bytes: int = Field(ge=0)

    @field_validator("file_name")
    @classmethod
    def basename(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("source file_name must be a basename")
        return value


class CorpusHeader(VersionOne):
    format: Literal["latinitas-corpus"]
    record: Literal["header"] = "header"
    corpus_id: CorpusId
    language: LanguageTag = "la"
    text_canon: Literal[1] = 1
    norm_rules: Literal["latin-fold/1"] = "latin-fold/1"
    adapter: Literal["usfx/1", "text/1", "tei/1"]
    source: CorpusSource
    edition: str | None = None
    publisher: str | None = None
    url: str | None = None
    revision: str | None = None
    license: str | None = None

    @model_serializer(mode="wrap")
    def omit_absent_revision(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return {key: value for key, value in handler(self).items() if key != "revision" or value is not None}

    @field_validator("text_canon", mode="before")
    @classmethod
    def integer_canon(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("text_canon must be an integer")
        return value


class CorpusMetadata(CorpusModel):
    record: Literal["corpus"] = "corpus"
    corpus_id: CorpusId
    title: str | None = None
    period: str | None = None
    register_: str | None = Field(default=None, alias="register")


class CorpusWork(CorpusModel):
    record: Literal["work"] = "work"
    work: str = Field(min_length=1)
    title: str | None = None
    order: PositiveOrder
    author: str | None = None
    edition_urn: Annotated[str, Field(pattern=CTS_EDITION_PATTERN)] | None = None

    @model_serializer(mode="wrap")
    def omit_absent_bibliography(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return {
            key: value
            for key, value in handler(self).items()
            if key not in {"author", "edition_urn"} or value is not None
        }


class CitationPart(CorpusModel):
    unit: str = Field(min_length=1)
    value: str


class Citation(CorpusModel):
    scheme: str = Field(min_length=1)
    parts: list[CitationPart] = Field(min_length=1)
    display: str

    @model_validator(mode="after")
    def consistent_display(self) -> Self:
        units = [part.unit for part in self.parts]
        values = [part.value for part in self.parts]
        if self.scheme == "usfx-bcv":
            if units != ["book", "chapter", "verse"]:
                raise ValueError("usfx-bcv requires book, chapter and verse parts")
            expected = f"{values[0]} {values[1]}:{values[2]}"
        elif self.scheme == "ordinal":
            if units != ["passage"]:
                raise ValueError("ordinal requires a passage part")
            expected = values[0]
        else:
            expected = ".".join(values)
        if self.display != expected:
            raise ValueError("citation display does not match its raw parts")
        return self


class CorpusTranslation(CorpusModel):
    language: LanguageTag
    text: str
    provenance: str


class CorpusNote(CorpusModel):
    kind: Literal["f", "x", "fe", "note", "variant"]
    text: str


class CorpusHeading(CorpusModel):
    text: str


class CorpusMarkup(CorpusModel):
    element: str
    attributes: dict[str, str]
    span: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def ordered_span(self) -> Self:
        if self.span[1] < self.span[0]:
            raise ValueError("span end precedes start")
        return self


def canonical_markup(raw: str, entries: list[CorpusMarkup]) -> str:
    """Map raw span intervals through NFC and whitespace collapse with sparse offsets.

    Match NFD origins in canonical combining order on both sides of NFC. This
    preserves offsets even when combining marks reorder across XML boundaries.
    A span touching part of a composed character includes that whole code point.
    Only markup boundaries get offset storage; combining runs use packed indices.
    """

    def origins(text: str) -> Iterator[int]:
        marks: dict[int, array[int]] = {}
        for index, char in enumerate(text):
            for part in unicodedata.normalize("NFD", char):
                combining = unicodedata.combining(part)
                if not combining:
                    for category in sorted(marks):
                        yield from marks[category]
                    marks.clear()
                    yield index
                else:
                    if combining not in marks:
                        marks[combining] = array("I")
                    marks[combining].append(index)
        for category in sorted(marks):
            yield from marks[category]

    for entry in entries:
        start, end = entry.span
        while start < end and raw[start].isspace():
            start += 1
        while end > start and raw[end - 1].isspace():
            end -= 1
        entry.span = [start, end]
    boundaries = sorted({0, len(raw), *(point for entry in entries for point in entry.span)})
    starts = [len(raw)] * (len(boundaries) - 1)
    ends = [0] * (len(boundaries) - 1)
    nfc = unicodedata.normalize("NFC", raw)
    removed = 0
    pending = False
    maximum_target = -1
    for source, target in zip(origins(raw), origins(nfc), strict=True):
        whitespace = nfc[target].isspace()
        if target > maximum_target:
            if whitespace:
                removed += 1
                pending = pending or target >= removed
            elif pending:
                removed -= 1
                pending = False
            maximum_target = target
        # Reordered marks can revisit a preceding NFC character, but cannot
        # cross whitespace, so its removed-whitespace count remains unchanged.
        start = target - removed + int(whitespace)
        end = start + int(not whitespace)
        segment = bisect_right(boundaries, source) - 1
        starts[segment] = min(starts[segment], start)
        ends[segment] = max(ends[segment], end)
    for entry in entries:
        left, right = (bisect_left(boundaries, point) for point in entry.span)
        if left < right:
            entry.span = [min(starts[left:right]), max(ends[left:right])]
        else:
            point = ends[left - 1] if left else 0
            entry.span = [point, point]
    return canonical_text(nfc)


class CorpusPassage(CorpusModel):
    record: Literal["passage"] = "passage"
    passage_id: str
    work: str = Field(min_length=1)
    native_ref: NativeRef
    order: PositiveOrder
    citation: Citation
    text: str
    norm: str
    passage_hash: Sha256
    translations: list[CorpusTranslation] | None = None
    notes: list[CorpusNote] | None = None
    headings: list[CorpusHeading] | None = None
    source_markup: list[CorpusMarkup] | None = None
    source_extra: list[CorpusMarkup] | None = None

    @field_validator("source_markup", "source_extra")
    @classmethod
    def bounded_spans(cls, value: list[CorpusMarkup] | None, info: ValidationInfo) -> list[CorpusMarkup] | None:
        text = info.data.get("text")
        if isinstance(text, str) and value and any(entry.span[1] > len(text) for entry in value):
            raise ValueError("markup span exceeds canonical text length")
        return value

    @model_validator(mode="after")
    def identity(self) -> Self:
        corpus_id, separator, ref = self.passage_id.partition(":")
        if (
            not separator
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", corpus_id)
            or ref != self.native_ref
            or not unicodedata.is_normalized("NFC", self.passage_id)
            or len(self.passage_id.encode("utf-8")) > 256
        ):
            raise ValueError("passage_id must be corpus_id:native_ref, NFC, at most 256 UTF-8 bytes")
        return self


type CorpusRecord = Annotated[CorpusHeader | CorpusMetadata | CorpusWork | CorpusPassage, Field(discriminator="record")]


class CorpusRecordSchema(RootModel[CorpusRecord]):
    """Schema root for a single record in the v1 JSONL stream."""


def record_digest(record: CorpusRecord) -> str:
    return digest(record.model_dump(mode="json"))
