"""Hardened streaming XML reader shared by the XML corpus adapters."""

import hashlib
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from xml.parsers import expat

from .cli_contract import Code, ContractError
from .corpus_model import CorpusSource


class XmlEventReader:
    """Chunk-fed expat events; no tree, source mutation or entity resolution.

    Yields ``(kind, name, attrs)`` with kind ``start``, ``end`` (explicit closing
    tag), ``empty_end`` (``<x/>``) or ``text`` (name holds the character data).
    With ``namespaces``, element and attribute names take ElementTree's
    ``{uri}local`` form; without it, raw prefixed names are kept. ``source``
    describes the streamed bytes once iteration completes.
    """

    def __init__(self, path: Path, *, namespaces: bool = False) -> None:
        self.path = path
        self.namespaces = namespaces
        self.source: CorpusSource | None = None

    def __iter__(self) -> Iterator[tuple[str, str, dict[str, str]]]:
        limit = 64 * 1024 * 1024
        if self.path.stat().st_size > limit:
            raise ContractError(Code.INPUT_TOO_LARGE, "XML input exceeds 64 MiB")
        events: deque[tuple[str, str, dict[str, str]]] = deque()
        parser = expat.ParserCreate(namespace_separator="}" if self.namespaces else None)
        depth = 0
        previous = ""
        # The latest start tag's stream index and input from its '<' onwards.
        tag_index, tag_context = 0, b""

        def emit(kind: str, name: str, attrs: dict[str, str]) -> None:
            nonlocal previous
            previous = kind
            events.append((kind, name, attrs))

        def expand(name: str) -> str:
            return "{" + name if self.namespaces and "}" in name else name

        def start(name: str, attrs: dict[str, str]) -> None:
            nonlocal depth, tag_index, tag_context
            depth += 1
            if depth > 64:
                raise ContractError(Code.DEPTH_EXCEEDED, "XML element depth exceeds 64")
            tag_index, tag_context = parser.CurrentByteIndex, parser.GetInputContext() or b""
            emit("start", expand(name), {expand(key): value for key, value in attrs.items()})

        parser.StartElementHandler = start

        def end(name: str) -> None:
            nonlocal depth
            depth -= 1
            # Expat reports an empty element's end just after its '/>', always
            # within the context buffered with its start tag. An end tag is
            # either further on, possibly past that context after a comment or
            # PI, or begins just after '>'. Include UTF-16's two byte orders.
            span = parser.CurrentByteIndex - tag_index
            empty = (
                previous == "start"
                and span <= len(tag_context)
                and tag_context.endswith((b"/>", b"/\x00>\x00", b"\x00/\x00>"), 0, span)
            )
            emit("empty_end" if empty else "end", expand(name), {})

        parser.EndElementHandler = end
        parser.CharacterDataHandler = lambda text: emit("text", text, {})

        def unsafe(*args: object) -> None:
            raise ContractError(Code.UNSAFE_XML, "DTD and entities are not allowed")

        def external(*args: object) -> int:
            unsafe()
            return 0

        parser.StartDoctypeDeclHandler = unsafe
        parser.EntityDeclHandler = unsafe
        parser.ExternalEntityRefHandler = external
        parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
        digest = hashlib.sha256()
        size = 0
        try:
            with self.path.open("rb") as stream:
                while chunk := stream.read(65536):
                    size += len(chunk)
                    if size > limit:
                        raise ContractError(Code.INPUT_TOO_LARGE, "XML input exceeds 64 MiB")
                    digest.update(chunk)
                    parser.Parse(chunk, False)
                    while events:
                        yield events.popleft()
                parser.Parse(b"", True)
                while events:
                    yield events.popleft()
        except (expat.ExpatError, LookupError, ValueError) as error:
            raise ContractError(Code.INVALID_XML, str(error)) from error
        self.source = CorpusSource(file_name=self.path.name, sha256=f"sha256:{digest.hexdigest()}", bytes=size)
