"""Read-only selected exercise views and initial corpus CSV creation."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from .annotation_sidecar import Annotation, read_sidecar
from .atomic_io import reject_aliases, writer_lock
from .cli_contract import Code, ContractError, Diagnostic
from .corpus_exercises import exercise_candidates
from .corpus_facts import FactView, accepted_facts
from .corpus_io import read_corpus
from .corpus_model import CorpusHeader
from .corpus_notes import CORPUS_SLOTS, corpus_csv, render_corpus_note
from .corpus_profile import CorpusProfile, load_corpus_profile, require_profile_corpora
from .corpus_ranking import rank_selection
from .corpus_selection import load_selection, reading_passages
from .html_text import generated_html_to_text
from .known_lemmas import load_known_lemmas
from .preview_export import commit_csv_artifacts
from .review_inspection import read_export_csv


@dataclass
class CorpusOutput:
    profile: CorpusProfile
    exercises: list[JsonValue]
    notes: list[dict[str, str]]
    diagnostics: list[Diagnostic]


def load_output(
    profile_path: Path,
    corpora: list[Path],
    selection_path: Path,
    *,
    known_profile: Path | None = None,
    known_source: Path | None = None,
) -> CorpusOutput:
    """Re-evaluate selected identities without filling budgets or writing selection state."""
    if (known_profile is None) != (known_source is None):
        raise ContractError(Code.USAGE_ERROR, "--known-lemmas and --known-lemmas-source are required together")
    profile = load_corpus_profile(profile_path)
    selection = load_selection(selection_path, profile)
    views: dict[str, FactView] = {}
    producers: dict[str, JsonValue] = {}
    diagnostics: list[Diagnostic] = []
    for corpus in corpora:
        header = next(read_corpus(corpus))
        if not isinstance(header, CorpusHeader):
            raise ContractError(Code.INVALID_CORPUS, "Invalid corpus header", path=str(corpus))
        if header.corpus_id in views:
            raise ContractError(Code.PROFILE_MISMATCH, f"Duplicate corpus: {header.corpus_id}")
        sidecar = corpus.with_suffix(".annotations.jsonl")
        try:
            view = accepted_facts(
                corpus, sidecar, Path(f"{corpus}.claims.json"), target_language=profile.target_language
            )
        except ValueError as error:
            raise ContractError(Code.INVALID_CORPUS, str(error), path=str(corpus)) from error
        views[header.corpus_id] = view
        diagnostics.extend(view.diagnostics)
        if sidecar.exists():
            producers.update(
                (row.annotation_id, row.producer.model_dump(mode="json"))
                for row in read_sidecar(sidecar)
                if isinstance(row, Annotation)
            )
    require_profile_corpora(profile, views)
    reading = reading_passages(profile, views)
    known = None
    if known_profile is not None and known_source is not None:
        try:
            known = load_known_lemmas(known_profile, known_source)
        except (ValueError, OSError) as error:
            raise ContractError(Code.INVALID_INPUT, str(error), path=str(known_source)) from error
        diagnostics.extend(
            Diagnostic(code=Code.INVALID_INPUT, severity="warning", message=skip.message, path=skip.location)
            for skip in known.skipped
        )
    _, entries = rank_selection(profile, views, selection, reading, known.lemmas if known else {}, set())
    candidates = {
        candidate.latinitas_id: (candidate, view.passages[candidate.passage_id])
        for view in views.values()
        for candidate in exercise_candidates(view, profile)
        if candidate.latinitas_id is not None
    }
    selected = {entry.latinitas_id for entry in selection.selected}
    exercises: list[JsonValue] = []
    notes: list[dict[str, str]] = []
    for entry in entries:
        if entry.latinitas_id not in selected:
            continue
        prompt = answer = citation = latin = passage_text = ""
        facts: list[JsonValue] = []
        current = candidates.get(entry.latinitas_id)
        if current is not None:
            candidate, passage = current
            fields = render_corpus_note(candidate, passage, profile)
            citation = generated_html_to_text(fields["Citation"])
            latin = generated_html_to_text(fields["Latin"])
            passage_text = generated_html_to_text(fields["Passage"])
            slot = next(slot for slot in CORPUS_SLOTS if slot.recipe == entry.recipe)
            if entry.eligible:
                notes.append(fields)
                prompt = generated_html_to_text(fields[slot.prompt_field])
                answer = generated_html_to_text(fields[slot.answer_field])
            facts = [
                {
                    "annotation_id": fact.annotation_id,
                    "layer": fact.layer,
                    "claim_status": fact.review_status,
                    "producer": producers.get(fact.annotation_id),
                    "reason": fact.reason,
                }
                for resolution in candidate.resolutions
                for fact in resolution.facts
            ]
        exercises.append(
            {
                "latinitas_id": entry.latinitas_id,
                "recipe": entry.recipe,
                "prompt": prompt,
                "answer": answer,
                "latin": latin,
                "passage": passage_text,
                "citation": citation,
                "status": "eligible" if entry.eligible else "withheld",
                "reasons": list(entry.reasons),
                "facts": facts,
                "deck_links": [link.model_dump(mode="json") for link in entry.deck_links],
            }
        )
    return CorpusOutput(profile, exercises, notes, diagnostics)


def preview_corpus(
    profile: Path,
    corpora: list[Path],
    selection: Path,
    limit: int,
    *,
    known_profile: Path | None = None,
    known_source: Path | None = None,
) -> tuple[JsonValue, list[Diagnostic]]:
    work = load_output(profile, corpora, selection, known_profile=known_profile, known_source=known_source)
    return {"exercises": work.exercises[:limit], "total": len(work.exercises)}, work.diagnostics


def export_corpus(
    profile: Path,
    corpora: list[Path],
    selection: Path,
    output: Path,
) -> tuple[JsonValue, list[Diagnostic]]:
    protected = [profile, selection]
    for corpus in corpora:
        protected.extend(
            [
                corpus,
                corpus.with_suffix(".annotations.jsonl"),
                Path(f"{corpus}.claims.json"),
                corpus.with_suffix(".manifest.json"),
            ]
        )
    protected += [Path(f"{path}.lock") for path in protected] + [Path(f"{path.resolve()}.lock") for path in protected]
    lock = Path(f"{output.resolve()}.lock")

    def check_paths() -> None:
        if output.exists() or output.is_symlink():
            raise ContractError(Code.EXISTS, "Initial export requires a new output path", path=str(output))
        reject_aliases(output, [*protected, lock])
        reject_aliases(lock, protected)

    check_paths()
    with writer_lock(output):
        check_paths()
        work = load_output(profile, corpora, selection)
        if not work.notes:
            raise ContractError(Code.NOTHING_TO_EXPORT, "No eligible selected exercises")
        payload = corpus_csv(work.profile, work.notes)
        check_paths()
        try:
            commit_csv_artifacts(((output, payload),), create_only=True)
        except FileExistsError as error:
            raise ContractError(Code.EXISTS, "Initial export requires a new output path", path=str(output)) from error
    return {"exported": len(work.notes), "output": str(output)}, work.diagnostics


def inspect_corpus(
    profile: Path,
    corpora: list[Path],
    selection: Path,
    check: Path | None,
) -> tuple[JsonValue, list[Diagnostic]]:
    work = load_output(profile, corpora, selection)
    diagnostics = list(work.diagnostics)
    if check is not None:
        try:
            actual = read_export_csv(
                check, note_type=work.profile.note_type, deck=work.profile.deck, strict_transport=True
            )
            expected = {note["LatinitasID"]: note for note in work.notes}
            seen: set[str] = set()
            for row in actual:
                identity = row.get("LatinitasID", "")
                note = expected.get(identity)
                if (
                    identity in seen
                    or note is None
                    or any(row.get(field) != value for field, value in note.items() if field != "Profile")
                ):
                    raise ValueError(f"Unreviewed, stale or drifted card: {identity}")
                seen.add(identity)
            if seen != expected.keys():
                raise ValueError("CSV identities differ from eligible selected exercises")
        except OSError as error:
            raise ContractError(Code.UNREADABLE, str(error), path=str(check)) from error
        except ValueError as error:
            diagnostics.append(Diagnostic(code=Code.CARD_DRIFT, message=str(error), path=str(check)))
    return {"exercises": work.exercises, "checked": str(check) if check is not None else None}, diagnostics
