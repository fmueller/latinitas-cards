"""Streaming schema reader and locked canonical writer for corpus JSONL.

The reader gates only JSON/file accessibility and the first-line envelope.
Schema/float findings preserve the raw object for semantic validation by consumers;
cross-record identity, order and content checks belong to corpus validation.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .atomic_io import atomic_replace, reject_aliases, writer_lock
from .canonical import canonical_json
from .cli_contract import Code, ContractError
from .corpus_model import CorpusHeader, CorpusRecord

RECORD_ADAPTER: TypeAdapter[CorpusRecord] = TypeAdapter(CorpusRecord)


@dataclass(frozen=True)
class CorpusFinding:
    line: int
    raw: dict[str, Any]
    code: str
    errors: list[dict[str, Any]]


def read_corpus(path: Path) -> Iterator[CorpusRecord | CorpusFinding]:
    """Yield typed records or recoverable findings, preserving file order.

    ContractError always means exit 3; a finding never aborts iteration. Version
    zero passes the envelope gate but fails the v1 model like any schema violation.
    A later JSON gate error is raised only after earlier items have been yielded.
    """
    try:
        with path.open(encoding="utf-8") as stream:
            seen = False
            floats: list[str] = []

            def parse_float(value: str) -> float:
                floats.append(value)
                return float(value)

            for number, line in enumerate(stream, 1):
                seen = True
                floats.clear()
                try:
                    raw = json.loads(line, parse_float=parse_float, parse_constant=parse_float)
                except (ValueError, RecursionError) as error:
                    raise ContractError(Code.INVALID_JSON, f"Line {number}: {error}", path=str(path)) from error
                if not isinstance(raw, dict):
                    raise ContractError(Code.INVALID_JSON, f"Line {number}: expected an object", path=str(path))
                if number == 1:
                    if raw.get("record") != "header":
                        raise ContractError(Code.MISSING_HEADER, "First line must be a header", path=str(path))
                    if raw.get("format") != "latinitas-corpus":
                        raise ContractError(Code.UNSUPPORTED_FORMAT, "Unknown corpus format", path=str(path))
                    version = raw.get("version")
                    if type(version) is not int or version > 1:
                        raise ContractError(Code.UNSUPPORTED_VERSION, "Unsupported corpus version", path=str(path))
                if floats:
                    yield CorpusFinding(number, raw, "float_value", [{"msg": "JSON floats are forbidden"}])
                    continue
                try:
                    yield RECORD_ADAPTER.validate_python(raw)
                except ValidationError as error:
                    yield CorpusFinding(
                        number, raw, "invalid_record", [dict(e) for e in error.errors(include_url=False)]
                    )
            if not seen:
                raise ContractError(Code.MISSING_HEADER, "Corpus is empty", path=str(path))
    except (OSError, UnicodeError) as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error


def write_corpus(path: Path, records: Iterable[CorpusRecord], protected: Iterable[Path]) -> None:
    """Replace one canonical stream using the shared lock/alias/atomic primitives.

    Callers supply a validated corpus; this layer enforces the header-first wire
    contract and forbids floats, without duplicating semantic corpus validation.
    """
    protected = tuple(protected)
    lock = Path(f"{path.resolve()}.lock")
    reject_aliases(path, [*protected, lock])
    reject_aliases(lock, protected)
    with writer_lock(path):
        reject_aliases(path, [*protected, lock])
        lines: list[str] = []
        for number, record in enumerate(records):
            if number == 0 and not isinstance(record, CorpusHeader):
                raise ValueError("Corpus must start with a header")
            lines.append(canonical_json(record.model_dump(mode="json")) + "\n")
        if not lines:
            raise ValueError("Corpus must start with a header")
        atomic_replace(path, "".join(lines).encode("utf-8"))
