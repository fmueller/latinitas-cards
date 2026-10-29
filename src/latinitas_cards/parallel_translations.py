"""Turn an aligned parallel corpus into a translation annotation patch; never a sidecar or claim."""

from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from .annotation_plans import AnnotationPatch, write_document
from .annotation_sidecar import Producer, resolve_created
from .canonical import digest
from .cli_contract import Code, ContractError, Diagnostic
from .corpus_io import read_corpus
from .corpus_model import CorpusHeader, CorpusPassage, LanguageTag
from .corpus_validation import validate_records


def _read(path: Path) -> tuple[list[CorpusHeader | CorpusPassage], list[Diagnostic]]:
    rows = list(read_corpus(path))
    _, findings = validate_records(rows, path=str(path))
    if findings:
        invalid = Diagnostic(code=Code.INVALID_CORPUS, message=f"Invalid corpus: {findings[0].code}", path=str(path))
        return [], [invalid, *findings]
    return [row for row in rows if isinstance(row, CorpusHeader | CorpusPassage)], []


def _primary(language: str) -> str:
    return language.lower().split("-")[0]


def import_translations(
    corpus: Path,
    parallel: Path,
    language: str,
    kind: str,
    name: str,
    version: str,
    output: Path,
    timestamp: str | None,
) -> tuple[JsonValue, list[Diagnostic]]:
    if kind not in ("human", "agent", "llm"):
        raise ContractError(Code.USAGE_ERROR, "--producer-kind must be human, agent or llm")
    try:
        TypeAdapter(LanguageTag).validate_python(language)
    except ValidationError as error:
        raise ContractError(Code.USAGE_ERROR, "Malformed --language tag") from error
    try:
        producer = Producer.model_validate({"kind": kind, "name": name, "version": version})
    except ValidationError as error:
        raise ContractError(Code.USAGE_ERROR, f"Malformed --producer-{error.errors()[0]['loc'][0]}") from error
    created = resolve_created(timestamp, {}) if timestamp is not None else None
    latin_rows, invalid = _read(corpus)
    parallel_rows, invalid_parallel = _read(parallel)
    if invalid or invalid_parallel:
        return None, [*invalid, *invalid_parallel]
    header = next(row for row in parallel_rows if isinstance(row, CorpusHeader))
    latin = [row for row in latin_rows if isinstance(row, CorpusPassage)]
    translated = [row for row in parallel_rows if isinstance(row, CorpusPassage)]
    parallel_primary = _primary(header.language)
    if parallel_primary != _primary(language) or parallel_primary == "la":
        raise ContractError(
            Code.UNSUPPORTED_LANGUAGE,
            f"Parallel corpus language {header.language!r} must be non-Latin and match --language {language!r}",
            path=str(parallel),
        )
    # An empty parallel passage has no translation to offer, so its Latin passage stays unmatched.
    texts = {row.native_ref: row.text for row in translated if row.text}
    refs = {row.native_ref for row in latin}
    diagnostics = [
        Diagnostic(
            code=Code.UNMATCHED_TRANSLATION,
            severity="warning",
            message=f"Latin passage {row.native_ref} has no parallel translation",
            path=str(corpus),
            ref=row.passage_id,
        )
        for row in latin
        if row.native_ref not in texts
    ] + [
        Diagnostic(
            code=Code.UNMATCHED_PARALLEL,
            severity="warning",
            message=f"Parallel passage {row.native_ref} has no Latin passage",
            path=str(parallel),
            ref=row.passage_id,
        )
        for row in translated
        if row.native_ref not in refs
    ]
    operations: list[dict[str, JsonValue]] = []
    for row in latin:
        if row.native_ref in texts:
            operation: dict[str, JsonValue] = {
                "operation": "add",
                "layer": "translation",
                "anchor": {"passage_id": row.passage_id, "passage_hash": row.passage_hash},
                "value": {"language": language, "text": texts[row.native_ref]},
                "producer": producer.model_dump(mode="json"),
            }
            if created is not None:
                operation["created"] = created
            operations.append(operation)
    patch = AnnotationPatch.model_validate(
        {"format": "latinitas-annotation-patch", "version": 1, "operations": operations}
    ).model_dump(mode="json")
    # Like plans: the output may not alias the inputs, their sidecar/claims defaults, locks or manifests.
    protected = [
        path
        for source in (corpus, parallel)
        for path in (source, source.with_suffix(".annotations.jsonl"), Path(f"{source}.claims.json"))
    ]
    written = write_document(output, patch, protected)
    return {"operations": len(operations), "patch": digest(patch), "written": written}, diagnostics
