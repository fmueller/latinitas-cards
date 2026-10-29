"""Legacy USFX projection for the experimental Vulgate commands."""

import re
import unicodedata
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from xml.parsers import expat

import pandas as pd
from pydantic import ValidationError

from .cli_contract import Code, ContractError, Diagnostic
from .corpus_model import (
    Citation,
    CitationPart,
    CorpusDescriptor,
    CorpusHeader,
    CorpusHeading,
    CorpusMarkup,
    CorpusMetadata,
    CorpusNote,
    CorpusPassage,
    CorpusRecord,
    CorpusWork,
    canonical_text,
    encode_ref_component,
    latin_fold,
    passage_hash,
)
from .corpus_validation import validate_records
from .xml_reader import XmlEventReader


def normalize_latin(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.replace("æ", "ae").replace("œ", "oe")
    s = s.replace("j", "i")
    return s


def legacy_projection(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    current_book: str | None = None
    current_chapter: int | None = None
    current_verse: int | None = None
    buffer: list[str] = []

    number_re = re.compile(r"\d+")

    def norm_space(s: str) -> str:
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def flush() -> None:
        nonlocal current_verse, buffer
        if current_book and current_chapter is not None and current_verse is not None:
            txt = norm_space(" ".join(part for part in buffer if part and part.strip()))
            if txt:
                rows.append(
                    {
                        "book": current_book,
                        "chapter": int(current_chapter),
                        "verse": int(current_verse),
                        "text": txt,
                    }
                )
        buffer = []
        current_verse = None

    # Coalesce parser chunks into ElementTree-style .text/.tail fragments before
    # joining with spaces. Closing </v> does not end a legacy verse; <ve> does,
    # after its initial text but before any children.
    initial_tag: str | None = None
    text: list[str] = []
    try:
        for kind, value, attrs in XmlEventReader(path, namespaces=True):
            if kind == "text":
                text.append(value)
                continue
            if current_verse is not None and initial_tag not in ("book", "b", "c", "chapter"):
                buffer.append("".join(text))
            text = []
            if initial_tag == "ve":
                flush()
            initial_tag = None
            if kind != "start":
                continue
            tag = value.lower()
            initial_tag = tag
            if tag in ("book", "b"):
                flush()
                current_book = attrs.get("id") or attrs.get("code") or attrs.get("n") or current_book
            elif tag in ("c", "chapter"):
                flush()
                m = number_re.search(attrs.get("id") or attrs.get("n") or "")
                current_chapter = int(m.group()) if m else current_chapter
            elif tag in ("v", "verse"):
                flush()
                m = number_re.search(attrs.get("id") or attrs.get("n") or "")
                current_verse = int(m.group()) if m else 0
    except ContractError as error:
        if error.diagnostic.code != Code.INVALID_XML:
            raise
        cause = error.__cause__
        if isinstance(cause, expat.ExpatError):
            raise ValueError(f"{expat.ErrorString(cause.code)}: line {cause.lineno}, column {cause.offset}") from error
        raise ValueError(error.diagnostic.message) from error

    # End of document: flush any open verse
    flush()

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed zero verses from USFX; check that <v/> ... text ... <ve/> structure is present.")

    df["text_norm"] = df["text"].apply(normalize_latin)
    return df.sort_values(["book", "chapter", "verse"]).reset_index(drop=True)


def _canonical_markup(raw: str, entries: list[CorpusMarkup]) -> str:
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


_SUPPORTED = {"usfx", "book", "h", "c", "v", "ve", "p", "q", "wj", "w", "f", "x", "fe", "s"}


class _StrictBuilder:
    """Strict source structure and passage metadata, independent of legacy semantics."""

    def __init__(self, corpus_id: str) -> None:
        self.corpus_id = corpus_id
        self.stack: list[str] = []
        self.book: str | None = None
        self.chapter: str | None = None
        self.verse: str | None = None
        self.buffer: list[str] = []
        self.title: list[str] = []
        self.trailing_ve = False
        self.works: dict[str, CorpusWork] = {}
        self.passages: list[CorpusPassage] = []
        self.refs: set[str] = set()
        self.raw_length = 0
        self.markup: list[CorpusMarkup] = []
        self.extra: list[CorpusMarkup] = []
        self.open_markup: list[tuple[int, CorpusMarkup]] = []
        self.notes: list[CorpusNote] = []
        self.headings: list[CorpusHeading] = []
        self.pending_notes: list[CorpusNote] = []
        self.pending_headings: list[CorpusHeading] = []
        self.capture: str | None = None
        self.capture_depth = 0
        self.capture_text: list[str] = []

    def invalid(self, message: str) -> None:
        raise ContractError(Code.INVALID_STRUCTURE, message)

    def finish_verse(self) -> None:
        assert self.book is not None and self.chapter is not None and self.verse is not None
        if self.open_markup:
            self.invalid("Markup crosses a verse boundary")
        raw = "".join(self.buffer)
        text = _canonical_markup(raw, [*self.markup, *self.extra]) if self.markup or self.extra else canonical_text(raw)
        if not text:
            raise ContractError(Code.EMPTY_PASSAGE, "Verse has no canonical text")
        values = [self.book, self.chapter, self.verse]
        ref = ".".join(encode_ref_component(value) for value in values)
        if ref in self.refs:
            raise ContractError(Code.DUPLICATE_SOURCE_REF, f"Duplicate source reference: {ref}")
        self.refs.add(ref)
        self.passages.append(
            CorpusPassage(
                passage_id=f"{self.corpus_id}:{ref}",
                work=self.book,
                native_ref=ref,
                order=len(self.passages) + 1,
                citation=Citation(
                    scheme="usfx-bcv",
                    parts=[
                        CitationPart(unit=unit, value=value)
                        for unit, value in zip(["book", "chapter", "verse"], values, strict=True)
                    ],
                    display=f"{self.book} {self.chapter}:{self.verse}",
                ),
                text=text,
                norm=latin_fold(text),
                passage_hash=passage_hash(text),
                notes=self.notes or None,
                headings=self.headings or None,
                source_markup=self.markup or None,
                source_extra=self.extra or None,
            )
        )
        self.verse = None
        self.buffer = []
        self.raw_length = 0
        self.notes, self.headings, self.markup, self.extra = [], [], [], []

    def finish_work(self) -> None:
        if self.pending_notes or self.pending_headings:
            self.invalid("Note or heading has no next passage in its work")

    def event(self, kind: str, name: str, attrs: dict[str, str]) -> None:
        parent = self.stack[-1] if self.stack else None
        if self.capture is not None:
            if kind == "text":
                self.capture_text.append(name)
            elif kind == "start":
                self.stack.append(name)
            else:
                if len(self.stack) == self.capture_depth:
                    text = canonical_text("".join(self.capture_text))
                    if self.capture == "s":
                        self.pending_headings.append(CorpusHeading(text=text))
                    else:
                        note = CorpusNote.model_validate({"kind": self.capture, "text": text})
                        (self.notes if self.verse is not None else self.pending_notes).append(note)
                    self.capture = None
                self.stack.pop()
            return
        if kind == "text":
            if parent == "h":
                self.title.append(name)
            elif self.verse is not None:
                self.buffer.append(name)
                self.raw_length += len(name)
            elif name.strip():
                self.invalid("Non-whitespace text outside a verse")
            return
        if kind == "start":
            if name in {"f", "x", "fe", "s"}:
                if self.book is None:
                    self.invalid(f"{name} outside a book")
                self.capture, self.capture_text = name, []
                self.stack.append(name)
                self.capture_depth = len(self.stack)
                return
            if name in {"p", "q", "wj", "w"} or name not in _SUPPORTED:
                if self.book is None:
                    self.invalid(f"{name} outside a book")
                if name not in {"p", "q"}:
                    if self.verse is None:
                        self.invalid("Inline markup outside a verse")
                    entry = CorpusMarkup(element=name, attributes=attrs, span=[self.raw_length, self.raw_length])
                    (self.markup if name in {"wj", "w"} else self.extra).append(entry)
                    self.open_markup.append((len(self.stack) + 1, entry))
                self.stack.append(name)
                return
            if name == "usfx":
                if parent is not None:
                    self.invalid("usfx must be the root")
            else:
                if parent not in {"usfx", "book", "p", "q"}:
                    self.invalid(f"Invalid parent for {name}")
                if name in {"book", "c", "v", "h"} and self.verse is not None:
                    self.invalid("Unterminated milestone verse")
                if name != "book" and self.book is None:
                    self.invalid(f"{name} outside a book")
                if name in {"book", "c", "v"}:
                    value = unicodedata.normalize("NFC", attrs.get("id", attrs.get("n", "")))
                    if not value or (name == "book" and "id" not in attrs):
                        self.invalid(f"Missing {name} identity")
                    self.trailing_ve = False
                    if name == "book":
                        if parent != "usfx":
                            self.invalid("Nested book")
                        self.finish_work()
                        self.book, self.chapter = value, None
                        if value not in self.works:
                            self.works[value] = CorpusWork(work=value, order=len(self.works) + 1)
                    elif name == "c":
                        self.chapter = value
                    else:
                        if self.chapter is None:
                            self.invalid("Verse outside a chapter")
                        self.verse, self.buffer = value, []
                        self.notes, self.pending_notes = self.pending_notes, []
                        self.headings, self.pending_headings = self.pending_headings, []
                elif name == "h":
                    self.title = []
                    self.trailing_ve = False
                elif name == "ve":
                    if self.verse is not None:
                        self.finish_verse()
                    elif not self.trailing_ve:
                        self.invalid("ve without a verse")
                    self.trailing_ve = False
            self.stack.append(name)
            return
        if self.open_markup and self.open_markup[-1][0] == len(self.stack):
            _, entry = self.open_markup.pop()
            entry.span[1] = self.raw_length
        if name == "v" and kind == "end":
            self.finish_verse()
            self.trailing_ve = True
        elif name == "h":
            assert self.book is not None
            self.works[self.book].title = canonical_text("".join(self.title))
        elif name == "book" and kind == "end":
            if self.verse is not None:
                self.invalid("Unterminated milestone verse")
            self.finish_work()
            self.book, self.chapter = None, None
            self.trailing_ve = False
        elif name == "usfx":
            if self.verse is not None:
                self.invalid("Unterminated milestone verse")
            self.finish_work()
        self.stack.pop()


def build_usfx(
    source: Path,
    corpus_id: str,
    language: str,
    descriptor: CorpusDescriptor | None,
    *,
    allow_unknown: bool = False,
    diagnostics: list[Diagnostic] | None = None,
) -> list[CorpusRecord]:
    """Fully validate before the caller can open an output for replacement."""
    reader = XmlEventReader(source)
    builder = _StrictBuilder(corpus_id)
    unknown: Counter[str] = Counter()
    failure: ContractError | None = None
    capture_depth = 0
    descriptor = descriptor or CorpusDescriptor(format="latinitas-corpus-descriptor", version=1)
    try:
        for kind, name, attrs in reader:
            if kind == "start":
                if capture_depth:
                    capture_depth += 1
                elif name in {"f", "x", "fe", "s"}:
                    capture_depth = 1
                elif name not in _SUPPORTED:
                    unknown[name] += 1
            elif kind != "text" and capture_depth:
                capture_depth -= 1
            if failure is None:
                try:
                    builder.event(kind, name, attrs)
                except ContractError as error:
                    failure = error
        if unknown:
            message = "Unknown elements: " + ", ".join(f"{name}={count}" for name, count in sorted(unknown.items()))
            if not allow_unknown:
                raise ContractError(Code.UNKNOWN_ELEMENT, message)
            if diagnostics is not None:
                diagnostics.append(Diagnostic(code=Code.UNKNOWN_ELEMENT, severity="warning", message=message))
        if failure:
            raise failure
        if not builder.works:
            raise ContractError(Code.INVALID_STRUCTURE, "USFX must contain at least one work")
        assert reader.source is not None
        records: list[CorpusRecord] = [
            CorpusHeader(
                format="latinitas-corpus",
                version=1,
                corpus_id=corpus_id,
                language=language,
                adapter="usfx/1",
                source=reader.source,
                edition=descriptor.edition,
                publisher=descriptor.publisher,
                url=descriptor.url,
                license=descriptor.license,
            ),
            CorpusMetadata(
                corpus_id=corpus_id, title=descriptor.title, period=descriptor.period, register=descriptor.register_
            ),
            *builder.works.values(),
            *builder.passages,
        ]
        _, diagnostics = validate_records(records, path=str(source))
        if diagnostics:
            raise ContractError(Code.INVALID_CORPUS, "; ".join(d.message for d in diagnostics))
        return records
    except ValidationError as error:
        raise ContractError(Code.INVALID_CORPUS, str(error)) from error
