"""Read-only, streaming semantic validation of normalized corpus records."""

import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from .cli_contract import Code, ContractError, Diagnostic
from .corpus_io import RECORD_ADAPTER, CorpusFinding, read_corpus
from .corpus_model import CorpusRecord, canonical_text, latin_fold, passage_hash


def _has_surrogate(value: object) -> bool:
    # Iterative, so nesting that json.loads accepts cannot exhaust the interpreter stack.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if any("\ud800" <= char <= "\udfff" for char in item):
                return True
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def _safe_unicode(value: str) -> str:
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


def validate_records(
    records: Iterable[CorpusRecord | CorpusFinding], *, path: str
) -> tuple[dict[str, int], list[Diagnostic]]:
    """Accumulate findings in stream order, retaining earlier findings on a gate error.

    Raw records remain available after schema errors so independent defects on the
    same line are still checked. Invalid prerequisites do not cascade into findings
    about derived values (for example an invalid corpus ID or noncanonical text).
    Counts include recognized record types, even when their contents are invalid.
    """
    counts = dict.fromkeys(("header", "corpus", "work", "passage"), 0)
    diagnostics: list[Diagnostic] = []
    corpus_id: str | None = None
    works: set[str] = set()
    refs: set[str] = set()
    phase = 0
    missing_corpus = False

    try:
        for number, record in enumerate(records, 1):
            raw = record.raw if isinstance(record, CorpusFinding) else record.model_dump()
            kind = raw.get("record")
            ref = raw.get("native_ref")
            identity = raw.get("passage_id")
            usable_id = (
                isinstance(identity, str)
                and re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}:(?:[A-Za-z0-9._-]|%[0-9A-F]{2})+", identity)
                and len(identity.encode("utf-8")) <= 256
            )
            context = identity if usable_id else ref if isinstance(ref, str) else None
            if context is not None:
                context = _safe_unicode(context)
            line_codes: set[Code] = set()

            def finding(
                code: Code,
                message: str,
                *,
                line_codes: set[Code] = line_codes,
                number: int = number,
                context: str | None = context,
            ) -> None:
                diagnostics.append(
                    Diagnostic(code=code, message=f"Line {number}: {_safe_unicode(message)}", path=path, ref=context)
                )
                line_codes.add(code)

            if _has_surrogate(raw):
                finding(Code.INVALID_RECORD, "Strings must contain Unicode scalar values, not lone surrogates")

            if isinstance(record, CorpusFinding):
                errors = record.errors
                if record.code == "float_value":
                    finding(Code.FLOAT_VALUE, "JSON floats are forbidden")
                    errors = []
                    try:
                        RECORD_ADAPTER.validate_python(raw)
                    except ValidationError as schema_error:
                        errors = [dict(e) for e in schema_error.errors(include_url=False)]
                for error in errors:
                    # The reader already reports the source_extra float invariant.
                    if error["msg"] == "Value error, canonical JSON forbids floats":
                        continue
                    location = error.get("loc", ())
                    if "corpus_id" in location:
                        code = Code.INVALID_CORPUS_ID
                    elif "citation" in location:
                        code = Code.INVALID_CITATION
                    elif kind == "passage" and (location == ("passage",) or "passage_id" in location):
                        code = Code.PASSAGE_ID_MISMATCH
                    else:
                        code = Code.INVALID_RECORD
                    field = ".".join(str(part) for part in location)
                    finding(code, f"{field}: {error['msg']}")

            if not isinstance(kind, str) or kind not in counts:
                continue
            counts[kind] += 1
            next_phase = {"header": 0, "corpus": 1, "work": 2, "passage": 3}[kind]
            if next_phase < phase or (kind in {"header", "corpus"} and counts[kind] > 1):
                finding(Code.INVALID_RECORD, "Expected header, one corpus, works, then passages")
            phase = max(phase, next_phase)
            if kind in {"work", "passage"} and counts["corpus"] == 0 and not missing_corpus:
                finding(Code.INVALID_RECORD, "Missing corpus record")
                missing_corpus = True
            if kind == "header" and number == 1:
                value = raw.get("corpus_id")
                if isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", value):
                    corpus_id = value
            elif kind == "corpus":
                if corpus_id is not None and raw.get("corpus_id") != corpus_id:
                    finding(Code.INVALID_RECORD, "Corpus record ID differs from header")
            elif kind == "work":
                work = raw.get("work")
                if isinstance(work, str):
                    if work in works:
                        finding(Code.INVALID_RECORD, "Duplicate work")
                    works.add(work)
            elif kind == "passage":
                work = raw.get("work")
                if isinstance(work, str) and work not in works:
                    finding(Code.INVALID_RECORD, "Passage refers to an unknown work")
                if isinstance(ref, str):
                    if ref in refs:
                        finding(Code.DUPLICATE_NATIVE_REF, "Native reference is repeated")
                    refs.add(ref)
                    if (
                        corpus_id is not None
                        and identity != f"{corpus_id}:{ref}"
                        and Code.PASSAGE_ID_MISMATCH not in line_codes
                    ):
                        finding(Code.PASSAGE_ID_MISMATCH, "Expected corpus_id:native_ref")
                text = raw.get("text")
                if isinstance(text, str) and not _has_surrogate(text):
                    nfc = unicodedata.normalize("NFC", text)
                    if text != nfc:
                        finding(Code.NON_NFC_TEXT, "Text must be NFC")
                    if nfc != canonical_text(text):
                        finding(Code.NON_CANONICAL_TEXT, "Whitespace must be collapsed and trimmed")
                    if isinstance(raw.get("passage_hash"), str) and raw["passage_hash"] != passage_hash(text):
                        finding(Code.HASH_MISMATCH, "Hash differs from canonical text")
                    if isinstance(raw.get("norm"), str) and raw["norm"] != latin_fold(canonical_text(text)):
                        finding(Code.NORM_MISMATCH, "Norm differs from latin-fold/1")
            if kind in {"work", "passage"} and raw.get("order") != counts[kind]:
                finding(Code.INVALID_RECORD, f"Expected {kind} order {counts[kind]}")
    except ContractError as error:
        diagnostics.append(error.diagnostic.model_copy(update={"path": path}))
        return counts, diagnostics
    if counts["corpus"] == 0 and not missing_corpus:
        diagnostics.append(Diagnostic(code=Code.INVALID_RECORD, message="Missing corpus record", path=path))
    return counts, diagnostics


def validate_corpus(
    path: str, *, annotations: Path | None = None, claims: Path | None = None
) -> tuple[dict[str, int], list[Diagnostic]]:
    """Validate a corpus; sidecar parameters reserve the later inspection seam."""
    return validate_records(read_corpus(Path(path)), path=path)
