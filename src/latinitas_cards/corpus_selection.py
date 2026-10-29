"""Sticky corpus exercise selection, report schemas and locked state writing."""

import json
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from .atomic_io import reject_aliases, replace_staged, stage_bytes, writer_locks
from .canonical import canonical_json, digest, digest_bytes
from .cli_contract import Code, ContractError, Diagnostic
from .corpus_exercises import exercise_identity
from .corpus_facts import FactView, accepted_facts
from .corpus_io import read_corpus
from .corpus_model import CorpusHeader, CorpusModel, LanguageTag, Sha256, VersionOne
from .corpus_profile import CorpusProfile, CorpusRecipe, load_corpus_profile, profile_digest, require_profile_corpora
from .known_lemmas import load_known_lemmas

type Reason = Literal[
    "filtered:fragment",
    "filtered:length",
    "filtered:unknown",
    "filtered:reading",
    "deselected",
    "budget:recipe",
    "budget:work",
    "budget:run",
    "near_duplicate",
    "per_lemma",
    "missing_fact",
    "stale_fact",
    "open_fact",
    "withheld_fact",
    "conflicting_facts",
    "unsupported_morphology",
    "wrong_language",
    "no_features",
    "invalid_span",
    "target_missing",
    "context_changed",
]
type Nonnegative = Annotated[int, Field(ge=0)]


class SentenceUnit(CorpusModel):
    kind: Literal["sentence"]
    index: Nonnegative


class PassageUnit(CorpusModel):
    kind: Literal["passage"]


type Unit = Annotated[SentenceUnit | PassageUnit, Field(discriminator="kind")]


class Target(CorpusModel):
    form: str = Field(min_length=1)
    occurrence: Nonnegative


class SelectedEntry(CorpusModel):
    latinitas_id: str
    object_key: str
    recipe: CorpusRecipe
    passage_id: str
    unit: Unit
    target: Target | None
    context_digest: Sha256
    run: Annotated[int, Field(ge=1)]


class DeselectedEntry(SelectedEntry):
    reason: str = Field(min_length=1)


class Selection(VersionOne):
    format: Literal["latinitas-corpus-selection"]
    profile_id: str
    target_language: LanguageTag
    runs: Nonnegative
    selected: list[SelectedEntry]
    deselected: list[DeselectedEntry]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        entries = [*self.selected, *self.deselected]
        ids = [entry.latinitas_id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("selection identities must be unique across both lists")
        for group in (self.selected, self.deselected):
            if [entry.latinitas_id for entry in group] != sorted(entry.latinitas_id for entry in group):
                raise ValueError("selection lists must be sorted by latinitas_id")
        for entry in entries:
            unit = entry.unit.model_dump(mode="json")
            target = entry.target.model_dump(mode="json") if entry.target else None
            key, identity = exercise_identity(entry.passage_id, entry.recipe, self.target_language, unit, target)
            if (key, identity) != (entry.object_key, entry.latinitas_id) or entry.run > self.runs:
                raise ValueError("selection entry identity or run is inconsistent")
            if (entry.recipe == "corpus_sentence_translation") != (entry.target is None):
                raise ValueError("only sentence translation has no target")
        return self


class Factors(CorpusModel):
    reading: int
    target: int
    novelty: int
    length: int
    unknown: int
    coverage: int


class CandidateEntry(CorpusModel):
    object_key: str | None
    latinitas_id: str | None
    recipe: CorpusRecipe
    passage_id: str
    unit: Unit | None
    target: Target | None
    annotation_id: str | None = None
    eligible: bool
    reasons: list[Reason]
    factors: Factors | None
    contributions: Factors | None
    score: int | None
    outcome: Literal["kept", "selected"] | Reason


class CorpusDigests(CorpusModel):
    header_digest: Sha256
    sidecar_digest: Sha256 | None
    claims_digest: Sha256 | None


class KnownDigests(CorpusModel):
    source_digest: Sha256
    profile_digest: Sha256


class CandidatesReport(VersionOne):
    format: Literal["latinitas-corpus-candidates"] = "latinitas-corpus-candidates"
    profile_digest: Sha256
    corpora: dict[str, CorpusDigests]
    known_lemmas: KnownDigests | None
    rule: Literal["latin-rank/1"] = "latin-rank/1"
    runs: Nonnegative
    candidates: list[CandidateEntry]


def load_selection(path: Path, profile: CorpusProfile) -> Selection:
    """An absent file starts empty; a present file must retain its exact profile binding."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return Selection(
            format="latinitas-corpus-selection",
            version=1,
            profile_id=profile.profile_id,
            target_language=profile.target_language,
            runs=0,
            selected=[],
            deselected=[],
        )
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate selection JSON key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
        canonical_json(value)
        selection = Selection.model_validate(value)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ContractError(Code.INVALID_INPUT, str(error), path=str(path)) from error
    if (selection.profile_id, selection.target_language) != (profile.profile_id, profile.target_language):
        raise ContractError(Code.BINDING_DRIFT, "Selection profile_id or target_language differs", path=str(path))
    return selection


def reading_passages(profile: CorpusProfile, views: dict[str, FactView]) -> set[str]:
    """Validate native-reference bounds at use, including both endpoint passages by order."""
    result: set[str] = set()
    for bounds in profile.reading:
        passages = [
            facts.passage for facts in views[bounds.corpus_id].passages.values() if facts.passage.work == bounds.work
        ]
        by_ref = {passage.native_ref: passage for passage in passages}
        first, last = by_ref.get(bounds.from_), by_ref.get(bounds.to)
        if first is None or last is None or first.order > last.order:
            raise ContractError(Code.INVALID_READING, f"Invalid reading range in {bounds.corpus_id}/{bounds.work}")
        result.update(p.passage_id for p in passages if first.order <= p.order <= last.order)
    return result


def select_corpus(
    profile_path: Path,
    corpora: list[Path],
    selection_path: Path,
    report_path: Path,
    *,
    known_profile: Path | None = None,
    known_source: Path | None = None,
    confirm_context: Iterable[str] = (),
) -> tuple[JsonValue, list[Diagnostic]]:
    """Read current state while holding both output locks; stage report before advancing selection."""
    # Deferred to keep the wire models available to the ranking module.
    from .corpus_ranking import rank_selection

    if (known_profile is None) != (known_source is None):
        raise ContractError(Code.USAGE_ERROR, "--known-lemmas and --known-lemmas-source are required together")
    protected = [profile_path]
    for corpus in corpora:
        protected.extend(
            [
                corpus,
                corpus.with_suffix(".annotations.jsonl"),
                Path(f"{corpus}.claims.json"),
                corpus.with_suffix(".manifest.json"),
            ]
        )
    if known_profile is not None and known_source is not None:
        protected.extend([known_profile, known_source, Path(f"{known_source}.latinitas.json")])
    protected += [lock for path in protected for lock in (Path(f"{path}.lock"), Path(f"{path.resolve()}.lock"))]
    outputs = [selection_path, report_path]
    locks = [Path(f"{path.resolve()}.lock") for path in outputs]

    def aliases() -> None:
        for index, output in enumerate(outputs):
            reject_aliases(output, [*protected, *locks, *outputs[:index], *outputs[index + 1 :]])
        for lock in locks:
            reject_aliases(lock, [*protected, *outputs])
        reject_aliases(locks[0], locks[1:])

    aliases()
    with writer_locks(outputs), TemporaryDirectory(prefix="latinitas-selection-") as scratch:
        aliases()
        profile = load_corpus_profile(profile_path)
        existed = selection_path.exists()
        selection = load_selection(selection_path, profile)
        previous_run = selection.runs
        originals: dict[str, str] = {}

        def capture(path: Path, *, optional: bool = False) -> Path:
            # Existing file readers consume private copies so provenance identifies their exact bytes.
            destination = Path(scratch) / str(len(originals)) / path.name
            originals[str(destination)] = str(path)
            try:
                raw = path.read_bytes()
            except FileNotFoundError as error:
                if optional:
                    return destination
                raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error
            except OSError as error:
                raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error
            try:
                destination.parent.mkdir()
                destination.write_bytes(raw)
            except OSError as error:
                raise ContractError(Code.UNREADABLE, "Cannot capture selection input", path=str(path)) from error
            return destination

        def original_diagnostic(diagnostic: Diagnostic) -> Diagnostic:
            message = diagnostic.message
            for private, original in originals.items():
                message = message.replace(private, original)
            return diagnostic.model_copy(
                update={"path": originals.get(diagnostic.path or "", diagnostic.path), "message": message}
            )

        views: dict[str, FactView] = {}
        provenance: dict[str, CorpusDigests] = {}
        diagnostics: list[Diagnostic] = []
        for corpus in corpora:
            captured_corpus = capture(corpus)
            sidecar = capture(corpus.with_suffix(".annotations.jsonl"), optional=True)
            claims = capture(Path(f"{corpus}.claims.json"), optional=True)
            try:
                header = next(read_corpus(captured_corpus))
                view = accepted_facts(captured_corpus, sidecar, claims, target_language=profile.target_language)
            except ContractError as error:
                diagnostic = original_diagnostic(error.diagnostic)
                raise ContractError(Code(diagnostic.code), diagnostic.message, path=diagnostic.path) from error
            except ValueError as error:
                message = original_diagnostic(Diagnostic(code=Code.INVALID_CORPUS, message=str(error))).message
                raise ContractError(Code.INVALID_CORPUS, message, path=str(corpus)) from error
            if not isinstance(header, CorpusHeader):
                raise ContractError(Code.INVALID_CORPUS, "Invalid corpus header", path=str(corpus))
            if header.corpus_id in views:
                raise ContractError(Code.PROFILE_MISMATCH, f"Duplicate corpus: {header.corpus_id}")
            views[header.corpus_id] = view
            diagnostics.extend(original_diagnostic(item) for item in view.diagnostics)
            provenance[header.corpus_id] = CorpusDigests(
                header_digest=digest(header.model_dump(mode="json")),
                sidecar_digest=_file_digest(sidecar),
                claims_digest=_file_digest(claims),
            )
        require_profile_corpora(profile, views)
        reading = reading_passages(profile, views)
        known = None
        if known_profile is not None and known_source is not None:
            captured_profile, captured_source = capture(known_profile), capture(known_source)
            manifest = (
                capture(Path(f"{known_source}.latinitas.json"), optional=True)
                if known_source.suffix.lower() == ".csv"
                else None
            )
            try:
                known = load_known_lemmas(captured_profile, captured_source, manifest_path=manifest)
            except (ValueError, OSError) as error:
                diagnostic = original_diagnostic(Diagnostic(code=Code.INVALID_INPUT, message=str(error)))
                raise ContractError(Code.INVALID_INPUT, diagnostic.message, path=str(known_source)) from error
            diagnostics.extend(
                Diagnostic(code=Code.INVALID_INPUT, severity="warning", message=skip.message, path=skip.location)
                for skip in known.skipped
            )
        selection, entries = rank_selection(
            profile, views, selection, reading, set(known.lemmas) if known else set(), set(confirm_context)
        )
        if not existed and selection.runs == previous_run:
            selection = selection.model_copy(update={"runs": previous_run + 1})
        report = CandidatesReport(
            version=1,
            profile_digest=profile_digest(profile),
            corpora=provenance,
            known_lemmas=KnownDigests(source_digest=known.source_digest, profile_digest=known.profile_digest)
            if known
            else None,
            runs=selection.runs,
            candidates=entries,
        )
        selection_bytes = (canonical_json(selection.model_dump(mode="json")) + "\n").encode("utf-8")
        report_bytes = (canonical_json(report.model_dump(mode="json")) + "\n").encode("utf-8")
        try:
            previous = selection_path.read_bytes() if selection_path.exists() else None
        except OSError as error:
            raise ContractError(Code.UNREADABLE, str(error), path=str(selection_path)) from error
        aliases()
        with ExitStack() as stack:
            staged_report = stack.enter_context(stage_bytes(report_path, report_bytes))
            staged_selection = (
                stack.enter_context(stage_bytes(selection_path, selection_bytes))
                if previous != selection_bytes
                else None
            )
            replace_staged(report_path, staged_report)
            if staged_selection is not None:
                replace_staged(selection_path, staged_selection)
        return {"runs": selection.runs, "selected": len(selection.selected), "candidates": len(entries)}, diagnostics


def _file_digest(path: Path) -> str | None:
    try:
        return digest_bytes(path.read_bytes())
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error
