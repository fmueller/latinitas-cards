"""Offline CSV handoff: emission is pending, never successful application."""

import hashlib
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from .destination_state import (
    MANAGED_FIELDS,
    DestinationSnapshot,
    ReconciliationRequired,
    _text,
    begin_observation,
    load_state,
    observe,
    save_state,
)
from .managed_plans import verify_approval
from .preview_export import serialize_anki_csv


def application_report(state: dict[str, Any], plan_id: str) -> dict[str, Any]:
    plan = state["plans"][plan_id]
    selected = plan["selected_operations"]
    statuses = {
        operation: plan["operations"][
            operation.rsplit("/", 2)[0] if "/field/" in operation else operation.rsplit("/", 1)[0]
        ]["status"]
        for operation in selected
    }
    return {
        "plan_id": plan_id,
        "approved": selected,
        "emitted": selected if plan["emission"]["status"] == "emitted" else [],
        "failed": selected if plan["emission"]["status"] == "failed" else [],
        "pending": [operation for operation, status in statuses.items() if status == "pending"],
        "observed": [operation for operation, status in statuses.items() if status == "confirmed"],
        "unresolved": [operation for operation, status in statuses.items() if status == "unresolved"],
        "emission": plan["emission"],
        "backup": plan["backup"],
        "native_safety": "unverified; T-061 gate required",
        "interval": (
            "Fresh snapshot and no intervening edits through GUI import required; "
            "unknown interval requires reconciliation/replan."
        ),
    }


def emit_updates(
    plan: dict[str, Any],
    approval: dict[str, Any],
    snapshot: DestinationSnapshot,
    state_path: Path,
    backup_path: Path,
    recovery: str,
    output: Path,
    note_type: str,
    deck: str,
) -> dict[str, Any]:
    """Persist pending effects before publishing an atomic CSV artifact.

    The backup is a local operator-attested recoverable collection export, not
    inferred native validity from a file extension. Existing files are refused.
    """
    state = load_state(state_path)
    verify_approval(plan, approval, snapshot, state)
    profile = plan["effective_profile"]
    if profile.get("generated_note_type") != note_type or profile.get("target_deck") != deck:
        raise ReconciliationRequired("import mapping differs from approved effective profile; renew approval")
    _text(recovery)
    if output.resolve() in (state_path.resolve(), backup_path.resolve()):
        raise ReconciliationRequired("output must not overwrite baseline or backup")
    if backup_path.samefile(state_path):
        raise ReconciliationRequired("recoverable destination backup must not alias the baseline journal")
    backup_bytes = backup_path.read_bytes()
    if not backup_bytes:
        raise ReconciliationRequired("empty recoverable backup")
    payload = serialize_anki_csv(
        note_type, deck, tuple(approval["import_columns"]), (tuple(row) for row in approval["import_rows"])
    )
    targets = {}
    notes = {note["identity"]: note for note in plan["notes"]}
    for identity, target in approval["targets"].items():
        anchor = state["anchors"][identity]
        if f"{identity}/tags" in approval["selected_operations"]:
            ownership = notes[identity]["ownership"]
        else:
            managed = set(anchor["source_tags"] + anchor["configured_tags"] + anchor.get("lifecycle_tags", []))
            ownership = {
                **{key: anchor[key] for key in ("source_tags", "configured_tags", "keep_fields")},
                "lifecycle_tags": anchor.get("lifecycle_tags", []),
                "keep_tags": sorted(set(target["tags"]) - managed | (set(anchor["keep_tags"]) & set(target["tags"]))),
                "suppressed_tags": sorted(managed - set(target["tags"])),
            }
        decisions = {
            key: choice
            for key, choice in notes[identity]["decisions"].items()
            if (
                key.startswith("field:")
                and f"{identity}/field/{key.removeprefix('field:')}" in approval["selected_operations"]
            )
            or (key.startswith("tag:") and f"{identity}/tags" in approval["selected_operations"])
        }
        # Only written or already-convergent fields advance the managed baseline;
        # an unapproved divergent destination edit must stay a conflict next time.
        note = notes[identity]
        baseline_fields = {
            name: target["fields"][name]
            if f"{identity}/field/{name}" in approval["selected_operations"]
            or note["destination_fields"][name] == note["proposed_fields"][name]
            else anchor["fields"][name]
            for name in MANAGED_FIELDS
        }
        targets[identity] = {
            **target,
            **ownership,
            "decisions": {**anchor.get("decisions", {}), **decisions},
            "baseline_fields": baseline_fields,
        }
    journal = begin_observation(state, snapshot, plan["plan_id"], targets, approval["review"], reviewed_changes=True)
    pending = journal["plans"][plan["plan_id"]]
    pending.update(
        selected_operations=approval["selected_operations"],
        backup={
            "path": str(backup_path.resolve()),
            "sha256": hashlib.sha256(backup_bytes).hexdigest(),
            "recovery": recovery,
        },
        emission={"status": "pending", "path": str(output.resolve()), "sha256": hashlib.sha256(payload).hexdigest()},
    )
    save_state(state_path, journal)
    name = None
    try:
        try:
            descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # Hard-link publishes without clobbering an existing artifact.
            os.link(name, output)
            pending["emission"]["status"] = "emitted"
        except OSError as exc:
            pending["emission"].update(status="failed", error=str(exc))
        try:
            save_state(state_path, journal)
        except OSError:
            pending["emission"].update(
                status="unknown",
                error=(
                    "Journal persistence uncertain; artifact may exist. Reload journal, "
                    "inspect artifact/hash and destination before retry."
                ),
            )
    finally:
        if name is not None:
            with suppress(FileNotFoundError):
                os.unlink(name)
    return application_report(journal, plan["plan_id"])


def observe_updates(
    state_path: Path,
    plan_id: str,
    snapshot: DestinationSnapshot | None,
    report: str,
    *,
    interval_confirmed: bool,
) -> dict[str, Any]:
    """Reacquire actual state before any retry; persist anchors and receipts together."""
    state = observe(load_state(state_path), plan_id, snapshot, report, interval_confirmed=interval_confirmed)
    save_state(state_path, state)
    return application_report(state, plan_id)
