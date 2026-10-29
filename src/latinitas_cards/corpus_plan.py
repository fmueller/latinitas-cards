"""Immutable corpus proposal requests for the offline managed planner."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from pydantic import JsonValue

from .atomic_io import write_artifact
from .canonical import canonical_json, digest_bytes
from .cli_contract import Code, ContractError, Diagnostic
from .corpus_output import load_output
from .corpus_selection import load_selection
from .destination_state import BoundDestination, ReconciliationRequired, load_state, read_snapshot
from .managed_plans import compose_plan


def plan_corpus(
    profile: Path,
    corpora: list[Path],
    selection: Path,
    snapshot: Path,
    state: Path,
    output: Path,
) -> tuple[JsonValue, list[Diagnostic]]:
    """Capture consumed bytes and emit proposals, never apply or infer withdrawal."""
    inputs = [profile, selection, snapshot, state]
    for corpus in corpora:
        inputs.extend(
            [
                corpus,
                corpus.with_suffix(".annotations.jsonl"),
                Path(f"{corpus}.claims.json"),
                corpus.with_suffix(".manifest.json"),
            ]
        )
    protected = [*inputs, *(Path(f"{p}.lock") for p in inputs), *(Path(f"{p.resolve()}.lock") for p in inputs)]
    with TemporaryDirectory(prefix="latinitas-plan-") as scratch:
        originals: dict[str, str] = {}
        hashes: dict[str, str | None] = {}

        def capture(path: Path, target: Path | None = None, *, optional: bool = False) -> Path:
            target = target or Path(scratch) / str(len(originals)) / path.name
            originals[str(target)] = str(path)
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                if not optional:
                    raise
                hashes[str(path)] = None
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                hashes[str(path)] = digest_bytes(raw)
            return target

        def original_diagnostic(diagnostic: Diagnostic) -> Diagnostic:
            message = diagnostic.message
            for private, original in originals.items():
                message = message.replace(private, original)
            return diagnostic.model_copy(
                update={
                    "path": originals.get(diagnostic.path or "", diagnostic.path),
                    "message": message,
                }
            )

        try:
            copied_profile, copied_selection = capture(profile), capture(selection)
            copied_snapshot, copied_state = capture(snapshot), capture(state)
            copied_corpora = []
            for corpus in corpora:
                copied = capture(corpus)
                capture(
                    corpus.with_suffix(".annotations.jsonl"), copied.with_suffix(".annotations.jsonl"), optional=True
                )
                capture(Path(f"{corpus}.claims.json"), Path(f"{copied}.claims.json"), optional=True)
                copied_corpora.append(copied)
            work = load_output(copied_profile, copied_corpora, copied_selection)
            stored = load_selection(copied_selection, work.profile)
            baseline = load_state(copied_state)
            if baseline["binding"]["schema"]["version"] != "corpus-v1":
                raise ContractError(Code.PROFILE_MISMATCH, "corpus-v1 manual setup and explicit adoption required")
            observed = read_snapshot(json.loads(copied_snapshot.read_bytes()), BoundDestination(**baseline["binding"]))
            members = {m[0] for m in baseline["binding"]["managed_set"]["members"]}
            anchors = baseline["anchors"]
            eligible = {note["LatinitasID"]: note for note in work.notes}
            proposals: list[dict[str, Any]] = []
            not_imported = []
            for entry in stored.selected:
                identity = entry.latinitas_id
                if identity not in members or (identity not in eligible and identity not in anchors):
                    not_imported.append(identity)
                    continue
                anchor = anchors.get(identity)
                fields = dict(eligible[identity] if identity in eligible else anchor["fields"])
                if anchor is not None:
                    fields["Profile"] = anchor["fields"]["Profile"]
                proposals.append(
                    {
                        "identity": identity,
                        "fields": fields,
                        "status": "approved" if identity in eligible else "withheld",
                        "contributions": {
                            key: anchor.get(key, []) if anchor else []
                            for key in (
                                "source_tags",
                                "configured_tags",
                                "lifecycle_tags",
                            )
                        },
                    }
                )
            for entry in stored.deselected:
                anchor = anchors.get(entry.latinitas_id)
                if anchor is not None and anchor["fields"].get("Source Kind") == "corpus":
                    proposals.append(
                        {
                            "identity": entry.latinitas_id,
                            "fields": dict(anchor["fields"]),
                            "retire": True,
                            "status": "missing",
                            "contributions": {
                                key: anchor.get(key, [])
                                for key in (
                                    "source_tags",
                                    "configured_tags",
                                    "lifecycle_tags",
                                )
                            },
                        }
                    )
            effective = {
                **work.profile.model_dump(mode="json"),
                "generated_note_type": work.profile.note_type,
                "target_deck": work.profile.deck,
                "inputs": hashes,
            }
            # Reject incoherent requests now, as well as when managed plan reads them.
            compose_plan(observed, baseline, proposals, effective)
            request = {
                "binding": baseline["binding"],
                "snapshot": observed.payload,
                "baseline": baseline,
                "proposals": proposals,
                "effective_profile": effective,
            }
            written = write_artifact(output, (canonical_json(request) + "\n").encode("utf-8"), protected)
            return cast(
                JsonValue,
                {
                    "output": str(output),
                    "written": written,
                    "proposals": len(proposals),
                    "not_imported": sorted(not_imported),
                },
            ), [original_diagnostic(d) for d in work.diagnostics]
        except ContractError as error:
            diagnostic = original_diagnostic(error.diagnostic)
            raise ContractError(diagnostic.code, diagnostic.message, path=diagnostic.path) from error
        except ReconciliationRequired as error:
            raise ContractError(Code.BINDING_DRIFT, str(error)) from error
        except (OSError, ValueError) as error:
            message = str(error)
            for private, original in originals.items():
                message = message.replace(private, original)
            raise ContractError(Code.INVALID_INPUT, message) from error
