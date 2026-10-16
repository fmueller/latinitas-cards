"""Legacy USFX projection for the experimental Vulgate commands."""

import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, deque
from collections.abc import Iterator
from pathlib import Path
from xml.parsers import expat

import pandas as pd
from pydantic import ValidationError

from .cli_contract import Code, ContractError
from .corpus_model import (
    Citation,
    CitationPart,
    CorpusDescriptor,
    CorpusHeader,
    CorpusMetadata,
    CorpusPassage,
    CorpusRecord,
    CorpusSource,
    CorpusWork,
    canonical_text,
    encode_ref_component,
    latin_fold,
    passage_hash,
)
from .corpus_validation import validate_records


def normalize_latin(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.replace("æ", "ae").replace("œ", "oe")
    s = s.replace("j", "i")
    return s


def legacy_projection(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()

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

    # We need to walk in document order and grab text residing outside tags too.
    # xml.etree doesn't give easy "between siblings" text via iter(), so we do a manual stack walk
    # capturing both .text and .tail relative to verse boundaries.
    stack: list[tuple[ET.Element, int]] = [(root, 0)]
    while stack:
        node, state = stack.pop()

        # On first visit, process start of node, then push children, then mark to process tail after children
        if state == 0:
            tag = node.tag.lower()

            if tag in ("book", "b"):
                flush()
                current_book = node.attrib.get("id") or node.attrib.get("code") or node.attrib.get("n") or current_book

            elif tag in ("c", "chapter"):
                flush()
                ch_val = node.attrib.get("id") or node.attrib.get("n") or ""
                m = number_re.search(ch_val)
                current_chapter = int(m.group()) if m else current_chapter

            elif tag in ("v", "verse"):
                # Start a new verse; if one is open, flush it first
                flush()
                v_val = node.attrib.get("id") or node.attrib.get("n") or ""
                m = number_re.search(v_val)
                current_verse = int(m.group()) if m else 0
                # Initial text inside <v>...</v> if any
                if node.text and current_verse is not None:
                    buffer.append(node.text)

            elif tag == "ve":
                # Verse end; include any text in <ve> (rare) then flush
                if node.text and current_verse is not None:
                    buffer.append(node.text)
                flush()

            else:
                # Normal node text inside an open verse
                if node.text and current_verse is not None:
                    buffer.append(node.text)

            # push a marker to process tail after children
            stack.append((node, 1))
            # push children in reverse to simulate pre-order traversal
            children = list(node)
            for child in reversed(children):
                stack.append((child, 0))

        else:
            # state == 1: process tail text after closing this node
            if node.tail and current_verse is not None:
                buffer.append(node.tail)

    # End of document: flush any open verse
    flush()

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed zero verses from USFX; check that <v/> ... text ... <ve/> structure is present.")

    df["text_norm"] = df["text"].apply(normalize_latin)
    return df.sort_values(["book", "chapter", "verse"]).reset_index(drop=True)


class _EventReader:
    """Chunk-fed expat events; no tree, source mutation or entity resolution."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.source: CorpusSource | None = None

    def __iter__(self) -> Iterator[tuple[str, str, dict[str, str]]]:
        events: deque[tuple[str, str, dict[str, str]]] = deque()
        parser = expat.ParserCreate()
        parser.StartElementHandler = lambda name, attrs: events.append(("start", name, attrs))

        def end(name: str) -> None:
            # Expat locates an explicit closing tag at its '<', but an empty
            # element's end after '/>'. Include UTF-16's two byte orders.
            context = parser.GetInputContext() or b""
            explicit = context.startswith((b"</", b"<\x00/\x00", b"\x00<\x00/"))
            events.append(("end" if explicit else "empty_end", name, {}))

        parser.EndElementHandler = end
        parser.CharacterDataHandler = lambda text: events.append(("text", text, {}))

        def unsafe(*args: object) -> None:
            raise ContractError(Code.UNSAFE_XML, "DOCTYPE is not allowed")

        parser.StartDoctypeDeclHandler = unsafe
        digest = hashlib.sha256()
        size = 0
        try:
            with self.path.open("rb") as stream:
                while chunk := stream.read(65536):
                    digest.update(chunk)
                    size += len(chunk)
                    parser.Parse(chunk, False)
                    while events:
                        yield events.popleft()
                parser.Parse(b"", True)
                while events:
                    yield events.popleft()
        except (expat.ExpatError, LookupError, ValueError) as error:
            raise ContractError(Code.INVALID_XML, str(error)) from error
        self.source = CorpusSource(file_name=self.path.name, sha256=f"sha256:{digest.hexdigest()}", bytes=size)


class _StrictBuilder:
    """Structural subset only; legacy projection deliberately keeps its own semantics."""

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

    def invalid(self, message: str) -> None:
        raise ContractError(Code.INVALID_STRUCTURE, message)

    def finish_verse(self) -> None:
        assert self.book is not None and self.chapter is not None and self.verse is not None
        text = canonical_text("".join(self.buffer))
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
            )
        )
        self.verse = None
        self.buffer = []

    def event(self, kind: str, name: str, attrs: dict[str, str]) -> None:
        parent = self.stack[-1] if self.stack else None
        if kind == "text":
            if parent == "h":
                self.title.append(name)
            elif self.verse is not None and parent in {"usfx", "book", "v"}:
                self.buffer.append(name)
            elif name.strip():
                self.invalid("Non-whitespace text outside a verse")
            return
        if kind == "start":
            if name == "usfx":
                if parent is not None:
                    self.invalid("usfx must be the root")
            else:
                if parent not in {"usfx", "book"}:
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
                        self.book, self.chapter = value, None
                        if value not in self.works:
                            self.works[value] = CorpusWork(work=value, order=len(self.works) + 1)
                    elif name == "c":
                        self.chapter = value
                    else:
                        if self.chapter is None:
                            self.invalid("Verse outside a chapter")
                        self.verse, self.buffer = value, []
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
        if name == "v" and kind == "end":
            self.finish_verse()
            self.trailing_ve = True
        elif name == "h":
            assert self.book is not None
            self.works[self.book].title = canonical_text("".join(self.title))
        elif name == "book" and kind == "end":
            if self.verse is not None:
                self.invalid("Unterminated milestone verse")
            self.book, self.chapter = None, None
            self.trailing_ve = False
        elif name == "usfx" and self.verse is not None:
            self.invalid("Unterminated milestone verse")
        self.stack.pop()


def build_usfx(
    source: Path,
    corpus_id: str,
    language: str,
    descriptor: CorpusDescriptor | None,
) -> list[CorpusRecord]:
    """Fully validate before the caller can open an output for replacement."""
    reader = _EventReader(source)
    builder = _StrictBuilder(corpus_id)
    unknown: Counter[str] = Counter()
    failure: ContractError | None = None
    descriptor = descriptor or CorpusDescriptor(format="latinitas-corpus-descriptor", version=1)
    try:
        for kind, name, attrs in reader:
            if kind == "start" and name not in {"usfx", "book", "h", "c", "v", "ve"}:
                unknown[name] += 1
            if failure is None and not unknown:
                try:
                    builder.event(kind, name, attrs)
                except ContractError as error:
                    failure = error
        if unknown:
            raise ContractError(
                Code.UNKNOWN_ELEMENT,
                "Unknown elements: " + ", ".join(f"{name}={count}" for name, count in sorted(unknown.items())),
            )
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
