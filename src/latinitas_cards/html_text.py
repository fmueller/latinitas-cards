"""Render untrusted source HTML as safe, readable plain text.

Source decks own their field markup, but generated study content must never
trust it.  This module is the normalization boundary for source-owned display
text: markup is parsed structurally so quoted attribute values such as
``title="a > b"`` cannot leak into the text, active elements and comments are
dropped, formatting is flattened, and meaningful block and line boundaries
survive as newline characters so callers can escape the result for display.
Entities are decoded exactly once.  German umlauts, Latin macrons, and other
plain text pass through unchanged.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

_BLOCK_TAGS = frozenset(
    {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_INACTIVE_TAGS = frozenset({"script", "style"})


class _SourceTextExtractor(HTMLParser):
    """Collect readable text from one structural parse of untrusted markup.

    Comments, declarations, processing instructions, and the contents of
    script and style elements are dropped; block and break tags become line
    boundaries; every other tag is discarded while its inner text is kept.
    Character references are decoded exactly once when ``decode_entities`` is
    set, otherwise they pass through untouched.
    """

    def __init__(self, *, decode_entities: bool) -> None:
        super().__init__(convert_charrefs=False)
        self._decode_entities = decode_entities
        self._pieces: list[str] = []
        self._inactive_depth = 0
        self._index = 0

    def updatepos(self, i: int, j: int) -> int:
        self._index = j
        return super().updatepos(i, j)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _INACTIVE_TAGS:
            self._inactive_depth += 1
        elif tag in _BLOCK_TAGS:
            self._pieces.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _INACTIVE_TAGS:
            if self._inactive_depth:
                self._inactive_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if data and not self._inactive_depth:
            self._pieces.append(data)

    def handle_entityref(self, name: str) -> None:
        self._pieces.append(self._reference(f"&{name}"))

    def handle_charref(self, name: str) -> None:
        self._pieces.append(self._reference(f"&#{name}"))

    def _reference(self, body: str) -> str:
        reference = re.compile(f"{re.escape(body)};?").match(self.rawdata, self._index)
        text = reference.group(0) if reference else f"{body};"
        return html.unescape(text) if self._decode_entities else text

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._pieces).split("\n"))
        return "\n".join(line for line in lines if line).strip()


def generated_html_to_text(value: str) -> str:
    """Flatten generated HTML, decoding escaped prose without reparsing it.

    Actual script/style elements and comments are still suppressed. Escaped
    angle brackets are displayed text, not a second layer of source markup.
    """
    decoded = _SourceTextExtractor(decode_entities=True)
    decoded.feed(value)
    decoded.close()
    return decoded.text()


class _SourceBoundaryExtractor(_SourceTextExtractor):
    """Retain source offsets while hiding markup from delimiter recognition."""

    def __init__(self, value: str, *, decode_parentheses: bool = True) -> None:
        super().__init__(decode_entities=True)
        self.boundaries = [" "] * len(value)
        self._decode_parentheses = decode_parentheses
        self.readable: list[tuple[str, int, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        if tag in _BLOCK_TAGS:
            self.readable.append(("\n", self._index, self._index))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_startendtag(tag, attrs)
        if tag in _BLOCK_TAGS:
            self.readable.append(("\n", self._index, self._index))

    def handle_endtag(self, tag: str) -> None:
        super().handle_endtag(tag)
        if tag in _BLOCK_TAGS:
            self.readable.append(("\n", self._index, self._index))

    def handle_data(self, data: str) -> None:
        if not self._inactive_depth:
            self.boundaries[self._index : self._index + len(data)] = data
            self.readable.extend(
                (char, self._index + index, self._index + index + 1) for index, char in enumerate(data)
            )

    def _reference(self, body: str) -> str:
        match = re.compile(f"{re.escape(body)};?").match(self.rawdata, self._index)
        raw = match.group(0) if match else f"{body};"
        if not self._inactive_depth:
            # Only parentheses affect annotation boundaries. Keep other entities
            # literal so decoding cannot introduce a new field separator.
            decoded = html.unescape(raw)
            self.readable.extend((char, self._index, self._index + len(raw)) for char in decoded)
            syntax = decoded.ljust(len(raw)) if self._decode_parentheses and decoded in {"(", ")"} else raw
            self.boundaries[self._index : self._index + len(raw)] = syntax
        return raw


def source_html_boundary_text(value: str) -> str:
    """Return delimiter-readable text at raw offsets, not display text."""
    parser = _SourceBoundaryExtractor(value)
    parser.feed(value)
    parser.close()
    # Match the display normalizer's second parse of entity-exposed markup,
    # but map hidden characters back to the original source offsets.
    decoded_value = "".join(char for char, _, _ in parser.readable)
    resolved = _SourceBoundaryExtractor(decoded_value, decode_parentheses=False)
    resolved.feed(decoded_value)
    resolved.close()
    for character, (_, start, end) in zip(resolved.boundaries, parser.readable, strict=True):
        if character == " ":
            parser.boundaries[start:end] = " " * (end - start)
    return "".join(parser.boundaries)


def source_html_to_text(value: str) -> str:
    """Convert untrusted source HTML to readable plain text without active markup.

    The first parse decodes entities exactly once and handles source markup
    structurally.  Because decoding once can expose markup that the source had
    entity-encoded, the decoded text is parsed a second time without further
    decoding, so encoded comments, scripts, and styles are removed and encoded
    block tags still become line boundaries.  The result is plain text that
    still needs escaping before HTML output.
    """
    if not value:
        return value
    decoded = _SourceTextExtractor(decode_entities=True)
    decoded.feed(value)
    decoded.close()
    resolved = _SourceTextExtractor(decode_entities=False)
    resolved.feed(decoded.text())
    resolved.close()
    return resolved.text()


__all__ = ["generated_html_to_text", "source_html_boundary_text", "source_html_to_text"]
