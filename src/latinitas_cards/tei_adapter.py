"""Perseus/EpiDoc TEI P5 structure, citations and header metadata (`tei/1`, profile `epidoc-cts/1`)."""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    canonical_markup,
    canonical_text,
    encode_ref_component,
    latin_fold,
    passage_hash,
)
from .corpus_validation import validate_records
from .xml_reader import XmlEventReader

TEI_NS = "http://www.tei-c.org/ns/1.0"
WORK_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_XML_NS = "{http://www.w3.org/XML/1998/namespace}"
# An l[@n] inside one of these is flattened text, never a passage.
_NOT_PASSAGE_CONTEXT = frozenset({"note", "rdg", "sic", "abbr", "orig", "del"})
# Elements whose flattened text leaves canonical text as a note of this kind.
_NOTES: dict[str, Literal["note", "variant"]] = {
    "note": "note",
    "bibl": "note",
    "desc": "note",
    "del": "variant",
    "rdg": "variant",
}
_EMPTY = frozenset({"lb", "pb", "cb", "milestone", "gap"})
# Elements that keep their text and record a source_markup span with their attributes.
_SPANS = frozenset(
    {
        "p",
        "ab",
        "add",
        "supplied",
        "unclear",
        "hi",
        "emph",
        "ex",
        "q",
        "quote",
        "foreign",
        "name",
        "persName",
        "placeName",
        "geogName",
        "num",
        "date",
        "w",
        "seg",
        "label",
        "title",
        "cit",
        "reg",
        "corr",
        "expan",
        "sic",
        "abbr",
        "orig",
    }
)
# Inside app or choice: the kept children's text stays, the variant children become variant notes.
_ALTERNATIVES = {
    "app": (frozenset({"lem"}), frozenset({"rdg"})),
    "choice": (frozenset({"corr", "expan", "reg"}), frozenset({"sic", "abbr", "orig"})),
}
# Their meaning needs a position in canonical text, which flattened notes, headings and variants lack.
_POSITIONED = frozenset({"app", "lem", "rdg", "choice", "gap"})
_BETWEEN_LINES = frozenset({"note", "del", "rdg", *_EMPTY})
_ATTACHED = frozenset({"head", *_NOTES, *_EMPTY})
_KNOWN = frozenset({"TEI", "teiHeader", "text", "body", "div", "l", "lg", "lem", *_ATTACHED, *_SPANS, *_ALTERNATIVES})


@dataclass(slots=True)
class _Element:
    """A TEI-namespace element by local name; other names use `{uri}local` (`{}` for no namespace)."""

    name: str
    attrs: dict[str, str]
    children: "list[_Element | str]" = field(default_factory=list)


type _Attachment = CorpusNote | CorpusHeading | CorpusMarkup


@dataclass(slots=True)
class _Loose:
    """A note, heading or empty span outside every passage, attached once passages are known."""

    item: _Attachment
    # The verse textpart whose preceding line takes this item, or None to prefer the next passage.
    part: int | None


@dataclass(slots=True)
class _Passage:
    units: tuple[str, ...]
    values: list[str]
    # The verse textpart holding this line passage; None for a textpart passage.
    part: int | None
    raw: list[str] = field(default_factory=list)
    length: int = 0
    notes: list[CorpusNote] = field(default_factory=list)
    headings: list[CorpusHeading] = field(default_factory=list)
    markup: list[CorpusMarkup] = field(default_factory=list)
    extra: list[CorpusMarkup] = field(default_factory=list)


def _invalid(message: str) -> ContractError:
    return ContractError(Code.INVALID_STRUCTURE, message)


def _name(name: str) -> str:
    if name.startswith("{" + TEI_NS + "}"):
        return name[len(TEI_NS) + 2 :]
    return name if name.startswith("{") else "{}" + name


def _attribute(name: str) -> str:
    return "xml:" + name[len(_XML_NS) :] if name.startswith(_XML_NS) else name


def _read(source: Path) -> tuple[_Element, list[str], Counter[str], XmlEventReader]:
    """Build the document without skipped body divisions, counting unknown elements outside the header."""
    reader = XmlEventReader(source, namespaces=True)
    root: _Element | None = None
    stack: list[_Element] = []
    pending: list[list[str]] = []
    unknown: Counter[str] = Counter()
    skipped: list[str] = []
    skip_depth = header_depth = 0

    def flush() -> None:
        if pending[-1]:
            stack[-1].children.append("".join(pending[-1]))
            pending[-1].clear()

    for kind, value, attrs in reader:
        if skip_depth:
            if kind == "start":
                skip_depth += 1
            elif kind != "text":
                skip_depth -= 1
            continue
        if kind == "text":
            if stack:
                pending[-1].append(value)
            continue
        if kind != "start":
            flush()
            stack.pop()
            pending.pop()
            header_depth = max(header_depth - 1, 0)
            continue
        name = _name(value)
        path = [element.name for element in stack]
        if path == ["TEI", "text", "body"] and name == "div" and attrs.get("type") != "edition":
            skipped.append(attrs.get("type") or "untyped")
            skip_depth = 1
            continue
        if header_depth or (name == "teiHeader" and path == ["TEI"]):
            header_depth += 1
        elif name not in _KNOWN:
            unknown[name] += 1
        element = _Element(name, {_attribute(key): text for key, text in attrs.items()})
        if stack:
            flush()
            stack[-1].children.append(element)
        else:
            root = element
        stack.append(element)
        pending.append([])
    assert root is not None
    return root, skipped, unknown, reader


def _elements(element: _Element | None, name: str) -> list[_Element]:
    if element is None:
        return []
    return [child for child in element.children if isinstance(child, _Element) and child.name == name]


def _child(element: _Element | None, name: str) -> _Element | None:
    return next(iter(_elements(element, name)), None)


def _descendant(element: _Element | None, name: str) -> _Element | None:
    if element is None:
        return None
    for child in element.children:
        if isinstance(child, _Element):
            found = child if child.name == name else _descendant(child, name)
            if found is not None:
                return found
    return None


def _raw_text(element: _Element) -> str:
    return "".join(child if isinstance(child, str) else _raw_text(child) for child in element.children)


def _text(element: _Element | None) -> str | None:
    if element is None:
        return None
    return canonical_text(_raw_text(element)) or None


def _flattened(element: _Element) -> str:
    """The text of a note, variant or heading; positioned or (allowed) unknown elements inside it fail."""
    pending = [element]
    while pending:
        for child in pending.pop().children:
            if isinstance(child, _Element):
                if child.name in _POSITIONED or child.name not in _KNOWN:
                    raise _invalid(f"{child.name} inside {element.name}")
                pending.append(child)
    return _text(element) or ""


def _has_passage_line(element: _Element) -> bool:
    return any(
        isinstance(child, _Element)
        and child.name not in _NOT_PASSAGE_CONTEXT
        and ((child.name == "l" and "n" in child.attrs) or _has_passage_line(child))
        for child in element.children
    )


def _attachment(element: _Element) -> _Attachment:
    """Notes, variants and headings flatten their text out of passage text; empty elements are empty spans."""
    if element.name == "head":
        if _has_passage_line(element):
            raise _invalid("Line passage inside head")
        return CorpusHeading(text=_flattened(element))
    if element.name in _EMPTY:
        if element.children:
            raise _invalid(f"{element.name} must be empty")
        return CorpusMarkup(element=element.name, attributes=element.attrs, span=[0, 0])
    return CorpusNote(kind=_NOTES[element.name], text=_flattened(element))


def _only_elements(element: _Element, allowed: set[str]) -> None:
    for child in element.children:
        if isinstance(child, str):
            if child.strip():
                raise _invalid(f"Text directly inside {element.name}")
        elif child.name not in allowed:
            raise _invalid(f"{child.name} is not allowed inside {element.name}")


class _Extractor:
    """Passages and loose items of the edition division, in document order."""

    def __init__(self) -> None:
        self.items: list[_Passage | _Loose] = []
        self.parts = 0

    def division(self, element: _Element, units: tuple[str, ...], values: list[str], *, edition: bool) -> None:
        textparts = _elements(element, "div")
        if any(child.attrs.get("type") != "textpart" for child in textparts):
            raise _invalid("Only textpart divisions may nest inside the edition")
        if textparts:
            self.container(element, units, values)
        elif _has_passage_line(element):
            self.parts += 1
            self.lines(element, units, values, self.parts)
        elif edition:
            raise _invalid("Edition has no textpart or line passages")
        else:
            passage = _Passage(units, values, None)
            self.fill(passage, element)
            self.items.append(passage)

    def container(self, element: _Element, units: tuple[str, ...], values: list[str]) -> None:
        for child in element.children:
            if isinstance(child, str):
                if child.strip():
                    raise _invalid("Text outside passages")
            elif child.name == "div":
                subtype, n = child.attrs.get("subtype"), child.attrs.get("n")
                if not subtype or not n:
                    raise _invalid("Textpart without @subtype and @n")
                self.division(child, (*units, subtype), [*values, n], edition=False)
            elif child.name in _ATTACHED:
                self.items.append(_Loose(_attachment(child), None))
            else:
                raise _invalid(f"{child.name} outside passages")

    def lines(self, element: _Element, units: tuple[str, ...], values: list[str], part: int) -> None:
        for child in element.children:
            if isinstance(child, str):
                if child.strip():
                    raise _invalid("Text outside lines in a verse textpart")
            elif child.name == "l" and "n" in child.attrs:
                if not child.attrs["n"]:
                    raise _invalid("Line passage with an empty @n")
                passage = _Passage((*units, "line"), [*values, child.attrs["n"]], part)
                self.fill(passage, child)
                self.items.append(passage)
            elif child.name in {"l", "lg"}:
                # An unnumbered l groups like lg: its items attach between lines and an l[@n] in it is a passage.
                self.lines(child, units, values, part)
            elif child.name == "head" or child.name in _BETWEEN_LINES:
                # Headings go to the next line; other items prefer the preceding line.
                self.items.append(_Loose(_attachment(child), None if child.name == "head" else part))
            else:
                raise _invalid(f"{child.name} outside lines in a verse textpart")

    def fill(self, passage: _Passage, element: _Element) -> None:
        for child in element.children:
            if isinstance(child, str):
                passage.raw.append(child)
                passage.length += len(child)
            elif child.name in _ATTACHED:
                item = _attachment(child)
                if isinstance(item, CorpusMarkup):
                    item.span = [passage.length, passage.length]
                    passage.markup.append(item)
                elif isinstance(item, CorpusHeading):
                    passage.headings.append(item)
                else:
                    passage.notes.append(item)
            elif child.name in {"l", "lg"}:
                if child.name == "l" and "n" in child.attrs:
                    raise _invalid("Line passage inside another passage")
                self.fill(passage, child)
            elif child.name in _ALTERNATIVES:
                kept, variants = _ALTERNATIVES[child.name]
                if sum(isinstance(option, _Element) and option.name in kept for option in child.children) > 1:
                    raise _invalid(f"{child.name} holds more than one kept alternative")
                for option in child.children:
                    if isinstance(option, str):
                        if option.strip():
                            raise _invalid(f"Text directly inside {child.name}")
                    elif option.name in kept:
                        self.fill(passage, option)
                    elif option.name in variants:
                        passage.notes.append(CorpusNote(kind="variant", text=_flattened(option)))
                    else:
                        raise _invalid(f"{option.name} is not allowed inside {child.name}")
            elif child.name in _SPANS or child.name not in _KNOWN:
                # Only --allow-unknown lets an unknown element reach here; it is kept in source_extra.
                entry = CorpusMarkup(element=child.name, attributes=child.attrs, span=[passage.length, passage.length])
                (passage.markup if child.name in _SPANS else passage.extra).append(entry)
                self.fill(passage, child)
                entry.span[1] = passage.length
            else:
                raise _invalid(f"{child.name} inside a passage")


def _of_kind[T](entries: list[_Attachment], kind: type[T]) -> list[T]:
    return [entry for entry in entries if isinstance(entry, kind)]


def _resolve(items: list[_Passage | _Loose], texts: dict[int, str]) -> None:
    """Attach loose items: verse items to the preceding kept line of their textpart, others to the next passage.

    Each falls back to the other direction. An omitted (empty) passage takes no items; its own headings, notes
    and empty spans are routed like loose items at its position. Items before a passage precede its own notes,
    headings and spans (at offset 0); items after it follow them (at its text length).
    """
    before: dict[int, list[_Attachment]] = {}
    after: dict[int, list[_Attachment]] = {}
    previous: int | None = None
    following = iter(texts)
    upcoming = next(following, None)
    for index, item in enumerate(items):
        if index in texts:
            previous, upcoming = index, next(following, None)
            continue
        if isinstance(item, _Loose):
            loose = [item]
        else:
            # Known text spans of an omitted passage cover no kept text; only empty spans move.
            empty = [entry for entry in item.markup if entry.element in _EMPTY]
            own: list[_Attachment] = [*item.headings, *item.notes, *empty]
            loose = [_Loose(entry, None if isinstance(entry, CorpusHeading) else item.part) for entry in own]
        for routed in loose:
            same_part = previous is not None and routed.part is not None and items[previous].part == routed.part
            target = previous if same_part or upcoming is None else upcoming
            assert target is not None
            (after if target < index else before).setdefault(target, []).append(routed.item)
    for index, text in texts.items():
        passage = items[index]
        assert isinstance(passage, _Passage)
        leading, trailing = before.get(index, []), after.get(index, [])
        for entry in _of_kind(leading, CorpusMarkup):
            entry.span = [0, 0]
        for entry in _of_kind(trailing, CorpusMarkup):
            entry.span = [len(text), len(text)]
        passage.notes = _of_kind(leading, CorpusNote) + passage.notes + _of_kind(trailing, CorpusNote)
        passage.headings = _of_kind(leading, CorpusHeading) + passage.headings + _of_kind(trailing, CorpusHeading)
        passage.markup = _of_kind(leading, CorpusMarkup) + passage.markup + _of_kind(trailing, CorpusMarkup)


def _header_fields(header: _Element | None) -> tuple[dict[str, str | None], str | None, list[str]]:
    file_desc = _child(header, "fileDesc")
    publication = _child(file_desc, "publicationStmt")
    idnos = {idno.attrs.get("type"): _text(idno) for idno in reversed(_elements(publication, "idno"))}
    licence = _child(_child(publication, "availability"), "licence")
    patterns: list[str] = []
    for refs in _elements(_child(header, "encodingDesc"), "refsDecl"):
        if refs.attrs.get("n") == "CTS":
            patterns = [pattern.attrs["n"] for pattern in _elements(refs, "cRefPattern") if "n" in pattern.attrs]
            break
    fields = {
        "edition": _text(_child(_child(file_desc, "editionStmt"), "edition"))
        or _text(_descendant(_child(file_desc, "sourceDesc"), "title")),
        "publisher": _text(_child(publication, "publisher")),
        "url": idnos.get("URI"),
        "license": (licence.attrs.get("target") or _text(licence)) if licence is not None else None,
        "title": _text(_child(_child(file_desc, "titleStmt"), "title")),
    }
    return fields, idnos.get("filename"), patterns


def _work_id(edition: _Element, filename: str | None, work: str | None, corpus_id: str) -> str:
    if work is not None:
        return work
    urn = re.fullmatch(r"urn:cts:[^:]+:([^.:]+\.[^.:]+)(?:[.:].*)?", edition.attrs.get("n", ""))
    if urn and WORK_ID.fullmatch(urn[1]):
        return urn[1]
    candidate = ".".join((filename or "").split(".")[:2])
    if re.fullmatch(r"[a-z]+[0-9]+\.[a-z]+[0-9]+", candidate):
        return candidate
    return corpus_id


def _edition(root: _Element) -> _Element:
    if root.name != "TEI" or len(_elements(root, "teiHeader")) != 1 or len(_elements(root, "text")) != 1:
        raise _invalid("The root must be TEI holding one teiHeader and one text")
    _only_elements(root, {"teiHeader", "text"})
    (text,) = _elements(root, "text")
    if len(_elements(text, "body")) != 1:
        raise _invalid("text must hold one body")
    _only_elements(text, {"body"})
    (body,) = _elements(text, "body")
    _only_elements(body, {"div"})
    editions = _elements(body, "div")
    if len(editions) != 1:
        raise _invalid("TEI body must hold exactly one edition division")
    if editions[0].attrs.get("xml:lang") not in {"lat", "la"}:
        raise ContractError(Code.UNSUPPORTED_LANGUAGE, "The edition division must have xml:lang lat or la")
    return editions[0]


def _warning(code: Code, message: str, ref: str | None = None) -> Diagnostic:
    return Diagnostic(code=code, severity="warning", message=message, ref=ref)


def build_tei(
    source: Path,
    corpus_id: str,
    language: str,
    descriptor: CorpusDescriptor | None,
    *,
    work: str | None = None,
    allow_unknown: bool = False,
    diagnostics: list[Diagnostic] | None = None,
) -> list[CorpusRecord]:
    """Fully validate before the caller can open an output for replacement."""
    root, skipped, unknown, reader = _read(source)
    warnings: list[Diagnostic] = []
    if unknown:
        message = "Unknown elements: " + ", ".join(f"{name}={count}" for name, count in sorted(unknown.items()))
        if not allow_unknown:
            raise ContractError(Code.UNKNOWN_ELEMENT, message)
        warnings.append(_warning(Code.UNKNOWN_ELEMENT, message))
    edition = _edition(root)
    if skipped:
        warnings.append(_warning(Code.SKIPPED_DIVISION, "Skipped body divisions: " + ", ".join(sorted(set(skipped)))))
    extractor = _Extractor()
    extractor.division(edition, (), [], edition=True)
    items = extractor.items
    found = [(index, item) for index, item in enumerate(items) if isinstance(item, _Passage)]
    units = found[0][1].units
    fields, filename, patterns = _header_fields(_child(root, "teiHeader"))
    if patterns and [label.casefold() for label in reversed(patterns)] != [unit.casefold() for unit in units]:
        warnings.append(
            _warning(
                Code.CITATION_MISMATCH,
                f"CTS cRefPattern labels {', '.join(reversed(patterns))} differ from unit path "
                f"{', '.join(units)}; the unit path is kept",
            )
        )
    refs: dict[int, str] = {}
    seen: set[str] = set()
    texts: dict[int, str] = {}
    for index, passage in found:
        if passage.units != units:
            raise _invalid(f"Citation unit paths differ: {'.'.join(units)} and {'.'.join(passage.units)}")
        passage.values = [unicodedata.normalize("NFC", value) for value in passage.values]
        ref = ".".join(encode_ref_component(value) for value in passage.values)
        if ref in seen:
            raise ContractError(Code.DUPLICATE_SOURCE_REF, f"Duplicate source reference: {ref}")
        seen.add(ref)
        refs[index] = ref
        raw = "".join(passage.raw)
        spans = [*passage.markup, *passage.extra]
        content = canonical_markup(raw, spans) if spans else canonical_text(raw)
        if content:
            texts[index] = content
        else:
            if passage.extra:
                names = ", ".join(sorted({entry.element for entry in passage.extra}))
                raise _invalid(f"Unknown elements {names} in omitted passage {ref} have no canonical text position")
            warnings.append(
                _warning(Code.SKIPPED_EMPTY_PASSAGE, f"Passage {ref} has no canonical text and is omitted", ref)
            )
    if not texts:
        raise _invalid("Edition contains no passages")
    _resolve(items, texts)
    for index, passage in found:
        gaps = sum(entry.element == "gap" for entry in passage.markup)
        if gaps and index in texts:
            message = f"Passage {refs[index]} has {gaps} gap{'s' if gaps > 1 else ''} in its text"
            warnings.append(_warning(Code.GAP_IN_TEXT, message, refs[index]))

    work_id = _work_id(edition, filename, work, corpus_id)
    work_title = fields["title"]
    for name, value in fields.items():
        override = getattr(descriptor, name) if descriptor else None
        if override is not None:
            if value is not None and value != override:
                warnings.append(
                    _warning(
                        Code.DESCRIPTOR_OVERRIDE, f"Descriptor {name} {override!r} replaces header value {value!r}"
                    )
                )
            fields[name] = override
    assert reader.source is not None
    try:
        records: list[CorpusRecord] = [
            CorpusHeader(
                format="latinitas-corpus",
                version=1,
                corpus_id=corpus_id,
                language=language,
                adapter="tei/1",
                source=reader.source,
                edition=fields["edition"],
                publisher=fields["publisher"],
                url=fields["url"],
                license=fields["license"],
            ),
            CorpusMetadata(
                corpus_id=corpus_id,
                title=fields["title"],
                period=descriptor.period if descriptor else None,
                register=descriptor.register_ if descriptor else None,
            ),
            CorpusWork(work=work_id, title=work_title, order=1),
        ]
        for index, passage in found:
            if index not in texts:
                continue
            content = texts[index]
            records.append(
                CorpusPassage(
                    passage_id=f"{corpus_id}:{refs[index]}",
                    work=work_id,
                    native_ref=refs[index],
                    order=len(records) - 2,
                    citation=Citation(
                        scheme="tei-cts",
                        parts=[CitationPart(unit=u, value=v) for u, v in zip(units, passage.values, strict=True)],
                        display=".".join(passage.values),
                    ),
                    text=content,
                    norm=latin_fold(content),
                    passage_hash=passage_hash(content),
                    notes=passage.notes or None,
                    headings=passage.headings or None,
                    source_markup=passage.markup or None,
                    source_extra=passage.extra or None,
                )
            )
    except ValidationError as error:
        raise ContractError(Code.INVALID_CORPUS, str(error)) from error
    _, findings = validate_records(records, path=str(source))
    if findings:
        raise ContractError(Code.INVALID_CORPUS, "; ".join(d.message for d in findings))
    if diagnostics is not None:
        diagnostics.extend(warnings)
    return records
