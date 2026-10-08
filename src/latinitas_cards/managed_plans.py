"""Deterministic offline review and operation-bound approval, never application.

Full-width import targets start from destination values, not regenerated ones.
The selected CSV capability intentionally matches the conservative journal:
answer/prompt/guard changes remain unsupported pending separate native gates.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .card_lifecycle import plan_card_lifecycle
from .cards import TEMPLATE_REGISTRY, RenderedCard
from .destination_reconciliation import ORIGINS, reconcile_note
from .destination_state import (
    MANAGED_FIELDS,
    DestinationSnapshot,
    ReconciliationRequired,
    _binding,
    _digest,
    _encoded,
    _fields,
    _object,
    _state,
    _strings,
    _text,
    require_content_only_card_set,
)
from .notes import (
    CSV_EXPORT_FIELD_NAMES,
    GeneratedNote,
    GeneratedNoteProvenance,
    GenerationMetadata,
    ManagedNoteContent,
)
from .profile import tag_character_violation

CONTENT_FIELDS = frozenset(("Lemma", "Principal Parts", "Meaning", "Generator", "Profile"))


def _generated(fields: dict[str, str], member: list[str]) -> GeneratedNote:
    if len(member) != 4:
        raise ReconciliationRequired("coherent-object membership required")
    return GeneratedNote(
        fields["LatinitasID"],
        member[3],
        ManagedNoteContent(fields["Lemma"], fields["Principal Parts"], fields["Meaning"]),
        GeneratedNoteProvenance(
            fields["Source Kind"], fields["Source Location"], fields["Source Path"] or None, member[2], member[1]
        ),
        GenerationMetadata(fields["Profile"], fields["Note Schema"], fields["Generator"]),
        tuple(
            RenderedCard(
                slot,
                bool(fields[slot.enabled_field].strip()),
                fields[slot.enabled_field],
                fields[slot.prompt_field],
                fields[slot.answer_field],
            )
            for slot in TEMPLATE_REGISTRY
        ),
    )


def compose_plan(
    snapshot: DestinationSnapshot,
    state: object,
    proposals: Sequence[Mapping[str, Any]],
    effective_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose an explicitly selected proposal set; omission is never retirement.

    Proposal records carry complete managed fields, tag contributions, optional
    reconciliation decisions, status, and explicit object-retirement intent.
    Review evidence on a decision is not apply approval. Personal fields are
    rejected rather than exposed as choices. The effective profile includes its
    presentation settings and is independently bound into the plan identity.
    """
    data = snapshot.payload
    baseline = _state(state) if state is not None else None
    if baseline is not None and (
        baseline["binding"] != _binding(data) or baseline["deck_options"] != data["deck_options"]
    ):
        raise ReconciliationRequired("wrong destination or changed deck options")
    notes = {note["identity"]: note for note in data["notes"]}
    members = {member[0]: member for member in data["managed_set"]["members"]}
    tag_sets = [note["tags"] for note in notes.values()]
    if baseline is not None:
        tag_sets.extend(
            anchor.get(key, [])
            for anchor in baseline["anchors"].values()
            for key in ("tags", *ORIGINS, "keep_tags", "suppressed_tags")
        )
    if any(tag_character_violation(tag) for tags in tag_sets for tag in tags):
        raise ReconciliationRequired("CSV tags cannot contain whitespace or control characters")
    selected: set[str] = set()
    entries: list[dict[str, Any]] = []
    for proposal in proposals:
        identity = _text(proposal.get("identity"))
        if identity in selected or identity not in members:
            raise ReconciliationRequired("duplicate or unbound proposal identity")
        selected.add(identity)
        fields = _fields(proposal.get("fields"), identity)
        generated = _generated(fields, members[identity])
        contributions = _object(proposal.get("contributions", {}))
        if set(contributions) - set(ORIGINS):
            raise ReconciliationRequired("unknown tag origin")
        contributions = {key: _strings(contributions.get(key, [])) for key in ORIGINS}
        if any(tag_character_violation(tag) for tags in contributions.values() for tag in tags):
            raise ReconciliationRequired("CSV tags cannot contain whitespace or control characters")
        decisions = _object(proposal.get("decisions", {}))
        retire = proposal.get("retire", False)
        if type(retire) is not bool:
            raise ReconciliationRequired("retirement intent must be explicit boolean")
        destination = notes.get(identity)
        entry: dict[str, Any] = {
            "identity": identity,
            "classification": "unchanged",
            "reasons": [],
            "proposed_fields": fields,
            "proposed_tags": contributions,
            "decisions": decisions,
            "destination_fields": destination["fields"] if destination else None,
            "destination_tags": destination["tags"] if destination else [],
            "baseline_tags": baseline["anchors"].get(identity, {}).get("tags") if baseline else None,
            "card_effects": [],
            "operations": [],
            "blocked": [],
        }
        entries.append(entry)
        if destination is None:
            if baseline is not None and identity in baseline["anchors"]:
                reason = "previously anchored note absent; destination reconciliation required"
                entry.update(classification="conflict", reasons=[reason], blocked=[reason])
                continue
            entry.update(classification="create", reasons=["absent in complete bound destination"])
            entry["operations"] = [
                {
                    "id": f"{identity}/create",
                    "kind": "create",
                    "supported": False,
                    "reason": "structural creation unsupported by selected transport",
                }
            ]
            entry["card_effects"] = [{"semantic_key": key, "effect": "add"} for key in generated.card_keys]
            continue
        if baseline is None or identity not in baseline["anchors"]:
            entry.update(classification="conflict", reasons=["missing applied baseline; explicit adoption required"])
            entry["blocked"] = entry["reasons"][:]
            continue
        try:
            # Tokens here express proposals for inspection only. They are never
            # forwarded as permission or as retirement receipts to an applier.
            lifecycle = plan_card_lifecycle(
                snapshot,
                baseline,
                identity,
                generated,
                proposal_status=proposal.get("status", "approved"),
                approvals={
                    slot.semantic_key: "inspection-only"
                    for slot in TEMPLATE_REGISTRY
                    if slot.semantic_key in generated.card_keys
                    or any(
                        row.get("semantic_key") == slot.semantic_key
                        for row in (destination.get("cards") or {}).get("rows", [])
                    )
                },
                retire_object="inspection-only" if retire else None,
            )
            entry["card_effects"] = lifecycle["effects"]
            for effect in entry["card_effects"]:
                effect.pop("approval", None)
            entry["blocked"] = lifecycle["blocked"]
            contributions["lifecycle_tags"] = sorted(
                set(contributions["lifecycle_tags"]) | set(lifecycle["lifecycle_tags"])
            )
            reconciliation = reconcile_note(snapshot, baseline, identity, fields, contributions, decisions)
        except ReconciliationRequired as exc:
            entry.update(classification="conflict", reasons=[str(exc)], blocked=[str(exc)])
            continue
        entry["resolved_fields"] = reconciliation["fields"]
        entry["resolved_tags"] = reconciliation["tags"]
        entry["tag_diff"] = {
            "additions": sorted(set(reconciliation["tags"]) - set(destination["tags"])),
            "removals": sorted(set(destination["tags"]) - set(reconciliation["tags"])),
            "removal_candidates": reconciliation["removals"],
        }
        entry["ownership"] = reconciliation["ownership"]
        entry["decisions"] = reconciliation["decisions"]
        conflicts = reconciliation["conflicts"]
        retention_unsafe = any("retention" in reason or "uncertain" in reason for reason in entry["blocked"])
        for name in MANAGED_FIELDS:
            d, p, value = destination["fields"][name], fields[name], reconciliation["fields"][name]
            conflict = f"field:{name}" in conflicts
            if d == value and not conflict and f"field:{name}" not in reconciliation["decisions"]:
                continue
            supported = name in CONTENT_FIELDS and not conflict and not retention_unsafe and not retire
            entry["operations"].append(
                {
                    "id": f"{identity}/field/{name}",
                    "kind": "field",
                    "field": name,
                    "baseline": baseline["anchors"][identity]["fields"][name],
                    "destination": d,
                    "proposal": p,
                    "value": value,
                    "supported": supported,
                    "reason": "unresolved conflict"
                    if conflict
                    else "record reviewed field decision without changing destination"
                    if supported and d == value
                    else "compatible content"
                    if supported
                    else "unsupported identity/schema/slot or unsafe retention effect",
                }
            )
        anchor = baseline["anchors"][identity]
        ownership_changed = any(
            reconciliation["ownership"].get(key, []) != anchor.get(key, [])
            for key in ("source_tags", "configured_tags", "lifecycle_tags", "keep_tags", "suppressed_tags")
        )
        if (
            reconciliation["tag_write"]
            or ownership_changed
            or any(key.startswith("tag:") for key in (*conflicts, *reconciliation["decisions"]))
        ):
            lifecycle_tag = ("latinitas::retired" in destination["tags"]) != (
                "latinitas::retired" in reconciliation["tags"]
            )
            supported = not conflicts and not lifecycle_tag and not retention_unsafe and not retire
            entry["operations"].append(
                {
                    "id": f"{identity}/tags",
                    "kind": "tags",
                    "destination": destination["tags"],
                    "value": reconciliation["tags"],
                    "baseline_ownership": {key: anchor.get(key, []) for key in reconciliation["ownership"]},
                    "ownership": reconciliation["ownership"],
                    "supported": supported,
                    "reason": "unresolved or unsupported tag effect"
                    if not supported
                    else "reconciled full tag set"
                    if reconciliation["tag_write"]
                    else "record reviewed tag decision or ownership without changing destination",
                }
            )
        for effect in entry["card_effects"]:
            if effect["effect"] != "retain":
                entry["operations"].append(
                    {
                        "id": f"{identity}/card/{effect['semantic_key']}",
                        "kind": effect["effect"],
                        "supported": False,
                        "reason": "structural/lifecycle application unsupported",
                    }
                )
        entry["classification"] = (
            "retire"
            if retire
            else "conflict"
            if conflicts or entry["blocked"]
            else ("update" if entry["operations"] else "unchanged")
        )
        entry["reasons"] = sorted(set(conflicts + entry["blocked"])) or [
            "reconciled differences" if entry["operations"] else "destination already equals resolved proposal"
        ]
    plan: dict[str, Any] = {
        "format": "latinitas-managed-plan",
        "version": 1,
        "binding": _binding(data),
        "snapshot": snapshot.reference,
        "baseline_version": baseline["version"] if baseline else None,
        "baseline_digest": _digest(baseline),
        "effective_profile": dict(effective_profile),
        "requests": sorted(proposals, key=lambda proposal: proposal["identity"]),
        "transport": {
            "name": "csv",
            "content_only": True,
            "structural": False,
            "lifecycle": False,
            "application_implemented": False,
            "native_preservation_proven": False,
            "slot_payload_changes": False,
        },
        "notes": sorted(entries, key=lambda entry: entry["identity"]),
    }
    plan["plan_id"] = _digest(plan)
    return _object(json.loads(_encoded(plan)))


def _checked_plan(value: object) -> dict[str, Any]:
    plan = _object(json.loads(_encoded(value)))
    identity = plan.pop("plan_id", None)
    if plan.get("format") != "latinitas-managed-plan" or plan.get("version") != 1 or identity != _digest(plan):
        raise ReconciliationRequired("changed or invalid plan identity; renew approval")
    plan["plan_id"] = identity
    return plan


def approve_plan(plan: object, selected_operations: Sequence[str], review: str) -> dict[str, Any]:
    """Bind explicit selected operations and exact full-column write footprint.

    This receipt is not a signature or authorization by another principal. No
    generation/profile/claim confirmation is consulted. No writes are performed.
    """
    data = _checked_plan(plan)
    _text(review)
    selected = _strings(list(selected_operations))
    remaining = set(selected)
    targets: dict[str, Any] = {}
    for note in data["notes"]:
        chosen = [operation for operation in note["operations"] if operation["id"] in remaining]
        if not chosen:
            continue
        target = {"fields": dict(note["destination_fields"] or {}), "tags": note["destination_tags"][:]}
        for operation in chosen:
            remaining.remove(operation["id"])
            if not operation["supported"]:
                raise ReconciliationRequired("unsupported or unresolved selected operation")
            if operation["kind"] == "field":
                name = operation["field"]
                if name not in CONTENT_FIELDS:
                    raise ReconciliationRequired("unsupported or personal field")
                target["fields"][name] = operation["value"]
            elif operation["kind"] == "tags":
                target["tags"] = operation["value"][:]
            else:
                raise ReconciliationRequired("unsupported operation kind")
        _fields(target["fields"], note["identity"])
        targets[note["identity"]] = target
    if remaining:
        raise ReconciliationRequired("unknown selected operation")
    return {
        "format": "latinitas-managed-approval",
        "version": 1,
        "plan_id": data["plan_id"],
        "selected_operations": selected,
        "review": review,
        "targets": targets,
        "write_footprint_digest": _digest(targets),
        "import_columns": list(CSV_EXPORT_FIELD_NAMES),
        "import_rows": [
            [" ".join(target["tags"]) if name == "Tags" else target["fields"][name] for name in CSV_EXPORT_FIELD_NAMES]
            for _, target in sorted(targets.items())
        ],
    }


def verify_approval(plan: object, approval: object, snapshot: DestinationSnapshot, state: object) -> None:
    """Revalidate before any future transport; never advance the observed baseline."""
    data = _checked_plan(plan)
    receipt = _object(approval)
    baseline = _state(state) if state is not None else None
    if (
        data["snapshot"] != snapshot.reference
        or data["binding"] != _binding(snapshot.payload)
        or data["baseline_digest"] != _digest(baseline)
    ):
        raise ReconciliationRequired("stale snapshot/baseline; replan and renew approval")
    if data != compose_plan(snapshot, baseline, data["requests"], data["effective_profile"]):
        raise ReconciliationRequired("plan differs from recomputed derived operations; replan")
    expected = approve_plan(data, receipt.get("selected_operations", []), receipt.get("review", ""))
    if receipt != expected:
        raise ReconciliationRequired("changed approval/payload; renew approval")
    notes = {note["identity"]: note for note in snapshot.payload["notes"]}
    for identity, target in expected["targets"].items():
        require_content_only_card_set(notes[identity], target["fields"])
