"""Plain UTF-8 lines, source-native templates and normalized corpus records."""

import re
import unicodedata
from pathlib import Path

from pydantic import ValidationError

from .canonical import canonical_json, digest_bytes
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


def parse_template(template: str | None) -> tuple[str, list[str]] | None:
    if template is None:
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*:[A-Za-z][A-Za-z0-9_-]*(\.[A-Za-z][A-Za-z0-9_-]*)*", template):
        raise ContractError(Code.USAGE_ERROR, "Malformed citation template")
    scheme, units_text = template.split(":")
    units = units_text.split(".")
    if len(units) != len(set(units)):
        raise ContractError(Code.USAGE_ERROR, "Citation units must be unique")
    return scheme, units


def read_lines(raw: bytes, template: tuple[str, list[str]] | None) -> tuple[list[str], list[list[str]]]:
    try:
        lines = raw.decode("utf-8-sig").split("\n")
    except UnicodeError as error:
        raise ContractError(Code.UNREADABLE, str(error)) from error
    texts: list[str] = []
    labels: list[list[str]] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, 1):
        if not canonical_text(line):
            continue
        if template:
            label, tab, text = line.partition("\t")
            parts = unicodedata.normalize("NFC", label).split(".")
            if not tab or len(parts) != len(template[1]) or any(not part for part in parts):
                raise ContractError(
                    Code.INVALID_LINE, f"Line {number}: expected LABEL<TAB>text with template components"
                )
            if not canonical_text(text):
                continue
            ref = ".".join(encode_ref_component(part) for part in parts)
            if ref in seen:
                raise ContractError(Code.DUPLICATE_SOURCE_REF, f"Line {number}: duplicate source reference")
            seen.add(ref)
            labels.append(parts)
        else:
            text = line
        texts.append(canonical_text(text))
    return texts, labels


def build_corpus(
    source: Path,
    raw: bytes,
    corpus_id: str,
    language: str,
    descriptor: CorpusDescriptor | None,
    texts: list[str],
    refs: list[str],
    template: tuple[str, list[str]] | None = None,
    labels: list[list[str]] | None = None,
) -> bytes:
    """Build schema-checked records, with works before passages in first-appearance order."""
    descriptor = descriptor or CorpusDescriptor(format="latinitas-corpus-descriptor", version=1)
    try:
        header = CorpusHeader(
            format="latinitas-corpus",
            version=1,
            corpus_id=corpus_id,
            language=language,
            adapter="text/1",
            source=CorpusSource(file_name=source.name, sha256=digest_bytes(raw), bytes=len(raw)),
            edition=descriptor.edition,
            publisher=descriptor.publisher,
            url=descriptor.url,
            license=descriptor.license,
        )
        metadata = CorpusMetadata(
            corpus_id=corpus_id,
            title=descriptor.title,
            period=descriptor.period,
            register=descriptor.register_,
        )
        passages: list[CorpusPassage] = []
        works: dict[str, None] = {}
        for index, text in enumerate(texts):
            work = corpus_id
            ref = refs[index]
            if template:
                assert labels is not None
                scheme, units = template
                values = labels[index]
                if "work" in units:
                    work = values[units.index("work")]
            else:
                scheme, units, values = "ordinal", ["passage"], [ref]
            works[work] = None
            passages.append(
                CorpusPassage(
                    passage_id=f"{corpus_id}:{ref}",
                    work=work,
                    native_ref=ref,
                    order=index + 1,
                    citation=Citation(
                        scheme=scheme,
                        parts=[CitationPart(unit=u, value=v) for u, v in zip(units, values, strict=True)],
                        display=".".join(values),
                    ),
                    text=text,
                    norm=latin_fold(text),
                    passage_hash=passage_hash(text),
                )
            )
        records: list[CorpusRecord] = [header, metadata]
        records.extend(CorpusWork(work=work, order=i + 1) for i, work in enumerate(works or {corpus_id: None}))
        records.extend(passages)
        _, diagnostics = validate_records(records, path=str(source))
        if diagnostics:
            raise ContractError(Code.INVALID_CORPUS, "; ".join(d.message for d in diagnostics))
        return "".join(canonical_json(record.model_dump(mode="json")) + "\n" for record in records).encode("utf-8")
    except ValidationError as error:
        raise ContractError(Code.INVALID_CORPUS, str(error)) from error
