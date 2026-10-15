"""Fixed, code-point-based segmentation rules over canonical passage text."""

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

from .corpus_model import CorpusPassage

LATIN_TOK = ("latin-tok", "1")
LATIN_SEG = ("latin-seg", "1")

_TOKEN = re.compile(r"[\w\u0300-\u036F]+|[^\w\s]")
_WORD = re.compile(r"[\w\u0300-\u036F]+")
_TERMINATORS = frozenset(".?!")
_CLOSERS = frozenset("\"')»”")


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    kind: Literal["word", "punct"]


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int
    token_start: int
    token_end: int
    continues: bool = False
    continued: bool = False


def tokenize(text: str) -> tuple[Token, ...]:
    """Tokenize without normalizing, folding, or changing the input offsets."""
    return tuple(
        Token(match.start(), match.end(), "word" if _WORD.fullmatch(match.group()) else "punct")
        for match in _TOKEN.finditer(text)
    )


def segment_passage(text: str, tokens: tuple[Token, ...]) -> tuple[Sentence, ...]:
    """Partition tokens from ``tokenize(text)``; whitespace never enters spans."""
    sentences: list[Sentence] = []
    start = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if text[token.start : token.end] not in _TERMINATORS:
            continue
        while (
            index < len(tokens)
            and tokens[index].start == tokens[index - 1].end
            and text[tokens[index].start : tokens[index].end] in _CLOSERS
        ):
            index += 1
        sentences.append(Sentence(tokens[start].start, tokens[index - 1].end, start, index))
        start = index
    if start < len(tokens):
        sentences.append(Sentence(tokens[start].start, tokens[-1].end, start, len(tokens), continues=True))
    return tuple(sentences)


def segment_work(passages: Iterable[CorpusPassage]) -> tuple[tuple[Sentence, ...], ...]:
    """Return one sentence tuple per passage, sorted by corpus passage order.

    Continuation is local to adjacent passages of the same work. No text or
    tokens are joined across passages; a work transition resets continuation.
    """
    result: list[tuple[Sentence, ...]] = []
    previous_work: str | None = None
    continues = False
    for passage in sorted(passages, key=lambda passage: passage.order):
        sentences = segment_passage(passage.text, tokenize(passage.text))
        if sentences and passage.work == previous_work and continues:
            sentences = (replace(sentences[0], continued=True), *sentences[1:])
        result.append(sentences)
        previous_work = passage.work
        continues = bool(sentences and sentences[-1].continues)
    return tuple(result)
