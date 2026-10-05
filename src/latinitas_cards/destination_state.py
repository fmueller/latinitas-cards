"""Versioned offline destination evidence and observed application journals.

Inputs are reviewed export assertions, not native proof. This module neither
opens a collection nor authorizes transport. A journal is a single atomic file:
confirmed per-note anchors and their receipts cannot be persisted separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .cards import TEMPLATE_REGISTRY, TEMPLATE_REGISTRY_DIGEST, TEMPLATE_REGISTRY_VERSION, slot_for_key
from .identity import derive_latinitas_id
from .legacy_transition import INCOMPATIBLE_LAYOUT_CHOICES, incompatible_managed_identity
from .notes import AUTHORITATIVE_NOTE_FIELDS, NOTE_SCHEMA_VERSION
from .reference_templates import REFERENCE_CARD_CSS, REFERENCE_CARD_TEMPLATES

MANAGED_FIELDS = tuple(field.name for field in AUTHORITATIVE_NOTE_FIELDS if field.ownership == "managed")


class ReconciliationRequired(ValueError):
    """Evidence is absent, incompatible, ambiguous, or requires explicit review."""


def _encoded(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ReconciliationRequired("evidence must be finite JSON data") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_encoded(value).encode("utf-8")).hexdigest()


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ReconciliationRequired("expected JSON object")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReconciliationRequired("missing text evidence")
    return value


def _strings(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ReconciliationRequired("expected exact non-empty strings")
    if len(set(value)) != len(value):
        raise ReconciliationRequired("duplicate set evidence")
    return sorted(value)


def schema_contract(note_type_id: str) -> dict[str, Any]:
    """Expected reference layout; acquisition must supply actual matching digests."""
    _text(note_type_id)
    return {
        "note_type_id": note_type_id,
        "version": NOTE_SCHEMA_VERSION,
        "fields": [
            [field.name, field.ownership] for field in AUTHORITATIVE_NOTE_FIELDS if field.ownership != "transport"
        ],
        "registry_version": TEMPLATE_REGISTRY_VERSION,
        "registry_digest": TEMPLATE_REGISTRY_DIGEST,
        "templates": [
            {
                "name": template.slot.template_name,
                "ordinal": template.slot.ordinal,
                "semantic_key": template.slot.semantic_key,
                "front_digest": hashlib.sha256(template.front.encode()).hexdigest(),
                "back_digest": hashlib.sha256(template.back.encode()).hexdigest(),
            }
            for template in REFERENCE_CARD_TEMPLATES
        ],
        "css_digest": hashlib.sha256(REFERENCE_CARD_CSS.encode()).hexdigest(),
    }


def _membership(value: object) -> dict[str, Any]:
    data = _object(value)
    scope = _text(data.get("scope"))
    members = data.get("members")
    if not isinstance(members, list):
        raise ReconciliationRequired("missing whole managed-set membership")
    identities: set[str] = set()
    sources: set[tuple[str, ...]] = set()
    keyed_sources: dict[tuple[str, str], bool] = {}
    for member in members:
        if not isinstance(member, list) or len(member) not in (3, 4):
            raise ReconciliationRequired("invalid membership")
        identity, source_scope, source_id, *object_key = (_text(item) for item in member)
        if incompatible_managed_identity(identity):
            raise ReconciliationRequired("incompatible portable identity layout. " + INCOMPATIBLE_LAYOUT_CHOICES)
        source = (source_scope, source_id, *object_key)
        pair = (source_scope, source_id)
        if pair in keyed_sources and keyed_sources[pair] != bool(object_key):
            raise ReconciliationRequired("ambiguous mixed source/object membership")
        keyed_sources[pair] = bool(object_key)
        if object_key and identity != derive_latinitas_id(source_id, object_key[0], source_scope=source_scope):
            raise ReconciliationRequired("inconsistent coherent-object identity")
        if source_scope != scope or identity in identities or source in sources:
            raise ReconciliationRequired("ambiguous managed-set identity")
        identities.add(identity)
        sources.add(source)
    return {"scope": scope, "members": sorted(members)}


@dataclass(frozen=True)
class BoundDestination:
    """Operator-selected portable collection binding, not a filename or deck."""

    destination: str
    profile: str
    schema: dict[str, Any]
    managed_set: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        schema = _object(self.schema)
        if schema != schema_contract(_text(schema.get("note_type_id"))):
            raise ReconciliationRequired(
                "unknown schema/template/style layout; separately reviewed manual setup required; "
                "do not rebind scheduled templates. " + INCOMPATIBLE_LAYOUT_CHOICES
            )
        return _object(
            json.loads(
                _encoded(
                    {
                        "destination": _text(self.destination),
                        "profile": _text(self.profile),
                        "schema": schema,
                        "managed_set": _membership(self.managed_set),
                    }
                )
            )
        )


def _binding(data: dict[str, Any]) -> dict[str, Any]:
    return BoundDestination(
        _text(data.get("destination")),
        _text(data.get("profile")),
        _object(data.get("schema")),
        _object(data.get("managed_set")),
    ).payload()


def _fields(value: object, identity: str) -> dict[str, str]:
    fields = _object(value)
    if set(fields) != set(MANAGED_FIELDS) or any(not isinstance(value, str) for value in fields.values()):
        raise ReconciliationRequired("incomplete managed fields or user-owned field")
    if fields["LatinitasID"] != identity or fields["Note Schema"] != NOTE_SCHEMA_VERSION:
        raise ReconciliationRequired("inconsistent portable identity/schema. " + INCOMPATIBLE_LAYOUT_CHOICES)
    return {name: fields[name] for name in MANAGED_FIELDS}


def _rows(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    evidence = _object(value)
    if type(evidence.get("complete")) is not bool or not isinstance(evidence.get("rows"), list):
        raise ReconciliationRequired("invalid card/history availability")
    rows = [_object(row) for row in evidence["rows"]]
    ids = [_text(row.get("id")) for row in rows]
    if len(set(ids)) != len(ids):
        raise ReconciliationRequired("duplicate card/history identity")
    return {"complete": evidence["complete"], "rows": sorted(rows, key=lambda row: row["id"])}


def _note(value: object, binding: dict[str, Any]) -> dict[str, Any]:
    note = _object(value)
    identity = _text(note.get("identity"))
    membership = {member[0]: member[1:] for member in binding["managed_set"]["members"]}
    if identity not in membership or note.get("source") != membership[identity]:
        raise ReconciliationRequired("unknown or ambiguous source/object identity")
    if note.get("note_type_id") != binding["schema"]["note_type_id"]:
        raise ReconciliationRequired("wrong note type")
    for name in ("guid", "local_id", "personal_digest"):
        if note.get(name) is not None:
            _text(note[name])
    return {
        "identity": identity,
        "source": membership[identity],
        "note_type_id": note["note_type_id"],
        "guid": note.get("guid"),
        "local_id": note.get("local_id"),
        "fields": _fields(note.get("fields"), identity),
        "tags": _strings(note.get("tags")),
        "personal_digest": note.get("personal_digest"),
        "cards": _rows(note.get("cards")),
        "history": _rows(note.get("history")),
    }


def _preservation(note: dict[str, Any]) -> dict[str, Any] | None:
    if any(note.get(name) is None for name in ("guid", "local_id", "personal_digest", "cards", "history")):
        return None
    if not all(note[name]["complete"] for name in ("cards", "history")):
        return None
    return {
        name: note[name]
        for name in ("source", "note_type_id", "guid", "local_id", "personal_digest", "cards", "history")
    }


def bound_card_rows(note: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve actual destination rows to frozen slots; never infer from eligibility."""
    evidence = note.get("cards")
    if not isinstance(evidence, dict) or evidence.get("complete") is not True:
        raise ReconciliationRequired("missing complete card set evidence")
    result: dict[str, dict[str, Any]] = {}
    for row in evidence["rows"]:
        slot = slot_for_key(row.get("semantic_key", ""))
        if slot is None or row.get("ordinal") != slot.ordinal or row.get("template_name") != slot.template_name:
            raise ReconciliationRequired("unknown card set template binding. " + INCOMPATIBLE_LAYOUT_CHOICES)
        if slot.semantic_key in result:
            raise ReconciliationRequired("ambiguous card set binding")
        result[slot.semantic_key] = row
    return result


def require_content_only_card_set(note: Mapping[str, Any], fields: Mapping[str, str]) -> None:
    """CSV cannot add via a guard, clear a front, or substitute eligibility for existence."""
    rows = bound_card_rows(note)
    for slot in TEMPLATE_REGISTRY:
        enabled = bool(fields[slot.enabled_field].strip())
        if enabled != (slot.semantic_key in rows):
            raise ReconciliationRequired("unsupported content-only card set change")
        if enabled and (not fields[slot.prompt_field].strip() or not fields[slot.answer_field].strip()):
            raise ReconciliationRequired("invalid content-only card set content")


@dataclass(frozen=True)
class DestinationSnapshot:
    """Immutable canonical evidence; decoded copies cannot mutate the capture."""

    encoded: str

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ReconciliationRequired("invalid encoded snapshot") from exc
        object.__setattr__(self, "encoded", _encoded(_snapshot_payload(payload, None)))

    @property
    def payload(self) -> dict[str, Any]:
        return _object(json.loads(self.encoded))

    @property
    def fingerprint(self) -> str:
        evidence = self.payload
        for key in ("snapshot_id", "captured_at", "artifact_sha256"):
            del evidence[key]
        return _digest(evidence)

    @property
    def preservation_available(self) -> bool:
        data = self.payload
        return (
            data["level"] == "collection"
            and data["deck_options"] is not None
            and all(_preservation(note) is not None for note in data["notes"])
        )

    def presence(self, identity: str) -> Literal["present", "absent"]:
        data = self.payload
        if identity not in {member[0] for member in data["managed_set"]["members"]}:
            raise ReconciliationRequired("identity outside bound managed set")
        return "present" if any(note["identity"] == identity for note in data["notes"]) else "absent"

    @property
    def reference(self) -> dict[str, str]:
        data = self.payload
        return {
            "snapshot_id": data["snapshot_id"],
            "fingerprint": self.fingerprint,
            "artifact_sha256": data["artifact_sha256"],
        }


def _snapshot_payload(payload: object, expected: BoundDestination | None) -> dict[str, Any]:
    data = _object(json.loads(_encoded(payload)))
    binding = _binding(data)
    if expected is not None and binding != expected.payload():
        raise ReconciliationRequired("wrong destination/schema/managed-set binding")
    if data.get("version") != 1 or data.get("complete") is not True or data.get("fresh") is not True:
        raise ReconciliationRequired("incomplete, stale, or unknown snapshot")
    if data.get("level") not in ("note-only", "collection"):
        raise ReconciliationRequired("unknown evidence level")
    for name in ("snapshot_id", "captured_at", "client", "export_method", "selection"):
        _text(data.get(name))
    try:
        captured = datetime.fromisoformat(data["captured_at"])
        offset = captured.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("not UTC")
    except ValueError as exc:
        raise ReconciliationRequired("capture time must be UTC") from exc
    artifact = _text(data.get("artifact_sha256"))
    if len(artifact) != 64 or any(char not in "0123456789abcdef" for char in artifact):
        raise ReconciliationRequired("invalid artifact digest")
    _object(data.get("export_options"))
    data["exclusions"] = _strings(data.get("exclusions"))
    if data["exclusions"]:
        raise ReconciliationRequired("export exclusions cannot establish whole managed-set absence")
    if not isinstance(data.get("notes"), list):
        raise ReconciliationRequired("missing notes evidence")
    notes = [_note(note, binding) for note in data["notes"]]
    if len({note["identity"] for note in notes}) != len(notes):
        raise ReconciliationRequired("duplicate portable identity")
    for name in ("guid", "local_id"):
        ids = [note[name] for note in notes if note[name] is not None]
        if len(set(ids)) != len(ids):
            raise ReconciliationRequired("duplicate native note identity")
    if type(data.get("expected_count")) is not int or data["expected_count"] != len(notes):
        raise ReconciliationRequired("incomplete expected/observed note count")
    if data.get("deck_options") is not None:
        _object(data["deck_options"])
    data.update(binding)
    data["notes"] = sorted(notes, key=lambda note: note["identity"])
    data.setdefault("deck_options", None)
    return data


def read_snapshot(payload: object, expected: BoundDestination) -> DestinationSnapshot:
    """Reject unsafe input rather than interpreting it as empty destination state."""
    return DestinationSnapshot(_encoded(_snapshot_payload(payload, expected)))


def _ownership(value: object, tags: list[str]) -> dict[str, Any]:
    ownership = _object(value)
    result: dict[str, Any] = {
        name: _strings(ownership.get(name)) for name in ("source_tags", "configured_tags", "keep_tags", "keep_fields")
    }
    for name in ("lifecycle_tags", "suppressed_tags"):
        if name in ownership:
            result[name] = _strings(ownership[name])
    if not set(result["keep_fields"]) <= set(MANAGED_FIELDS) - {"LatinitasID", "Note Schema"}:
        raise ReconciliationRequired("invalid user keep override")
    managed = set(result["source_tags"]) | set(result["configured_tags"]) | set(result.get("lifecycle_tags", []))
    suppressed = set(result.get("suppressed_tags", []))
    if not suppressed <= managed or suppressed & set(tags):
        raise ReconciliationRequired("invalid suppressed tag ownership")
    if (managed - suppressed) | set(result["keep_tags"]) != set(tags):
        raise ReconciliationRequired("unreviewed tag ownership")
    if "decisions" in ownership:
        result["decisions"] = _object(ownership["decisions"])
        for choice in result["decisions"].values():
            _text(_object(choice).get("approval"))
    return result


def adopt(snapshot: DestinationSnapshot, ownership: Mapping[str, object], approval: str) -> dict[str, Any]:
    """Explicit first adoption from observed values and reviewed origins only."""
    if not approval.strip():
        raise ReconciliationRequired("explicit adoption approval required")
    data = snapshot.payload
    if set(ownership) != {note["identity"] for note in data["notes"]}:
        raise ReconciliationRequired("review ownership for every observed note")
    anchors = {
        note["identity"]: {
            **note,
            **_ownership(ownership[note["identity"]], note["tags"]),
            "observed": snapshot.reference,
            "approval": approval,
        }
        for note in data["notes"]
    }
    return {
        "format": "latinitas-observed-baseline",
        "format_version": 1,
        "version": 1,
        "binding": _binding(data),
        "deck_options": data["deck_options"],
        "anchors": anchors,
        "plans": {},
    }


def _state(value: object) -> dict[str, Any]:
    state = _object(json.loads(_encoded(value)))
    if state.get("format") != "latinitas-observed-baseline" or state.get("format_version") != 1:
        raise ReconciliationRequired("not an observed applied baseline (prior exports are not baselines)")
    if type(state.get("version")) is not int or state["version"] < 1:
        raise ReconciliationRequired("invalid baseline version")
    if "deck_options" not in state:
        raise ReconciliationRequired("missing baseline deck options evidence")
    if state["deck_options"] is not None:
        _object(state["deck_options"])
    binding = _binding(_object(state.get("binding")))
    for identity, anchor in _object(state.get("anchors")).items():
        anchor = _object(anchor)
        if _note(anchor, binding)["identity"] != identity:
            raise ReconciliationRequired("anchor identity mismatch")
        _ownership(anchor, anchor["tags"])
        _text(anchor.get("approval"))
        _reference(anchor.get("observed"))
    for plan in _object(state.get("plans")).values():
        plan = _object(plan)
        _text(plan.get("approval"))
        _reference(plan.get("before"))
        if plan.get("status") not in ("pending", "partial", "complete", "abandoned"):
            raise ReconciliationRequired("invalid whole-plan status")
        for identity, operation in _object(plan.get("operations")).items():
            operation = _object(operation)
            if _note(operation.get("before"), binding)["identity"] != identity:
                raise ReconciliationRequired("operation identity mismatch")
            target = _object(operation.get("target"))
            _fields(target.get("fields"), identity)
            _ownership(target, _strings(target.get("tags")))
            if operation.get("status") not in ("pending", "unresolved", "confirmed", "abandoned"):
                raise ReconciliationRequired("invalid operation status")
            if operation.get("status") == "confirmed":
                receipt = _object(operation.get("receipt"))
                _reference(receipt.get("observed"))
                _text(receipt.get("report"))
        statuses = [operation["status"] for operation in plan["operations"].values()]
        expected_status = (
            "complete"
            if all(status == "confirmed" for status in statuses)
            else "pending"
            if all(status == "pending" for status in statuses)
            else "partial"
        )
        if plan["status"] == "abandoned":
            if any(status not in ("confirmed", "abandoned") for status in statuses):
                raise ReconciliationRequired("inconsistent abandoned plan status")
        elif plan["status"] != expected_status:
            raise ReconciliationRequired("whole-plan status inconsistent with operation statuses")
    return state


def _reference(value: object) -> None:
    reference = _object(value)
    for name in ("snapshot_id", "fingerprint", "artifact_sha256"):
        _text(reference.get(name))


def reconcile(snapshot: DestinationSnapshot, state: object) -> tuple[str, ...]:
    """Return inconsistent anchors, including backup restoration; never replay."""
    if state is None:
        raise ReconciliationRequired("absent applied baseline; explicit adoption required")
    baseline = _state(state)
    data = snapshot.payload
    if baseline["binding"] != _binding(data):
        raise ReconciliationRequired("wrong destination baseline")
    if baseline["deck_options"] != data["deck_options"]:
        raise ReconciliationRequired("changed or unknown deck options; explicit reconciliation required")
    notes = {note["identity"]: note for note in data["notes"]}
    return tuple(
        identity
        for identity, anchor in sorted(baseline["anchors"].items())
        if identity not in notes
        or any(
            notes[identity][key] != anchor[key]
            for key in ("fields", "tags", "source", "guid", "local_id", "note_type_id")
        )
        or (_preservation(anchor) is not None and _preservation(notes[identity]) != _preservation(anchor))
    )


def begin_observation(
    state: object, before: DestinationSnapshot, plan_id: str, operations: Mapping[str, object], approval: str
) -> dict[str, Any]:
    """Journal pending approved existing-note effects BEFORE external application.

    This is not an apply API. Persist the returned state before any later import.
    Unresolved plans require observation/reconciliation, not another write attempt.
    """
    result = _state(state)
    _text(plan_id)
    _text(approval)
    if not operations:
        raise ReconciliationRequired("empty plan has no application effects to observe")
    if reconcile(before, result):
        raise ReconciliationRequired("inconsistent anchors; reacquire/reconcile before replanning")
    if any(plan["status"] not in ("complete", "abandoned") for plan in result["plans"].values()):
        raise ReconciliationRequired("pending effects require observation/reconciliation, never blind replay")
    if plan_id in result["plans"]:
        raise ReconciliationRequired("plan already recorded")
    if not before.preservation_available:
        raise ReconciliationRequired("missing preservation evidence")
    data = before.payload
    notes = {note["identity"]: note for note in data["notes"]}
    journal: dict[str, Any] = {}
    for identity, proposed in operations.items():
        if identity not in notes or identity not in result["anchors"]:
            raise ReconciliationRequired("create/adoption is not supported application")
        target = _object(proposed)
        fields = _fields(target.get("fields"), identity)
        tags = _strings(target.get("tags"))
        origins = _ownership(target, tags)
        require_content_only_card_set(notes[identity], fields)
        if ("latinitas::retired" in tags) != ("latinitas::retired" in notes[identity]["tags"]):
            raise ReconciliationRequired("unsupported lifecycle tag effect; tags cannot approximate suspension")
        # Identity/provenance and slot fields cannot be repurposed as structural transport.
        for name in MANAGED_FIELDS:
            if (
                name not in ("Lemma", "Principal Parts", "Meaning", "Generator", "Profile")
                and fields[name] != notes[identity]["fields"][name]
            ):
                raise ReconciliationRequired("unsupported identity/schema/slot effect")
        for name in result["anchors"][identity]["keep_fields"]:
            if fields[name] != notes[identity]["fields"][name]:
                raise ReconciliationRequired("user-owned keep override is not writable")
        journal[identity] = {
            "before": notes[identity],
            "target": {"fields": fields, "tags": tags, **origins},
            "status": "pending",
            "receipt": None,
            "observations": [],
        }
    result["plans"][plan_id] = {
        "approval": approval,
        "baseline_version": result["version"],
        "before": before.reference,
        "deck_options": data["deck_options"],
        "status": "pending",
        "operations": journal,
    }
    return result


def observe(
    state: object,
    plan_id: str,
    after: DestinationSnapshot | None,
    report: str,
    *,
    interval_confirmed: bool,
) -> dict[str, Any]:
    """Reconcile actual results; one note is the indivisible advancement unit."""
    result = _state(state)
    _text(report)
    if plan_id not in result["plans"]:
        raise ReconciliationRequired("unknown pending plan")
    plan = result["plans"][plan_id]
    if plan["status"] == "abandoned":
        raise ReconciliationRequired("abandoned plan requires new approval, not replay")
    notes: dict[str, Any] = {}
    observed_data = after.payload if after is not None else None
    if observed_data is not None:
        if _binding(observed_data) != result["binding"]:
            raise ReconciliationRequired("wrong observed destination")
        notes = {note["identity"]: note for note in observed_data["notes"]}
    advanced = False
    for identity, operation in plan["operations"].items():
        actual = notes.get(identity)
        target = operation["target"]
        confirmed = (
            after is not None
            and observed_data is not None
            and interval_confirmed
            and after.preservation_available
            and observed_data["deck_options"] == plan["deck_options"]
            and actual is not None
            and actual["fields"] == target["fields"]
            and actual["tags"] == target["tags"]
            and _preservation(actual) == _preservation(operation["before"])
        )
        if operation["status"] == "confirmed":
            # Later restoration/edits cannot silently revoke receipts or replay effects.
            if not confirmed:
                raise ReconciliationRequired("recorded success inconsistent; reconcile/invalidate baseline")
            continue
        receipt = {
            "observed": after.reference if after else None,
            "report": report,
            "interval_confirmed": interval_confirmed,
        }
        operation["observations"].append(
            {**receipt, "actual": actual, "status": "confirmed" if confirmed else "unresolved"}
        )
        operation["status"] = "confirmed" if confirmed else "unresolved"
        if confirmed:
            assert actual is not None and after is not None
            operation["receipt"] = receipt
            result["anchors"][identity] = {
                **actual,
                **_ownership(target, actual["tags"]),
                "observed": after.reference,
                "approval": plan["approval"],
            }
            advanced = True
    if advanced:
        result["version"] += 1
    statuses = [operation["status"] for operation in plan["operations"].values()]
    plan["status"] = "complete" if all(status == "confirmed" for status in statuses) else "partial"
    return result


def review_reconciliation(
    state: object, snapshot: DestinationSnapshot, ownership: Mapping[str, object], approval: str
) -> dict[str, Any]:
    """Explicitly accept freshly observed state after mixed outcomes or restoration.

    Historical successful receipts remain evidence of past observations, not a
    claim that their effects survived a restored backup. Pending effects are
    abandoned, never silently promoted. Every observed ownership decision is
    reviewed again; a later plan needs a new approval.
    """
    previous = _state(state)
    if previous["binding"] != _binding(snapshot.payload):
        raise ReconciliationRequired("wrong reconciliation destination")
    result = adopt(snapshot, ownership, approval)
    result["version"] = previous["version"] + 1
    result["plans"] = previous["plans"]
    for plan in result["plans"].values():
        plan["status"] = "abandoned"
        plan["reconciliation"] = {"approval": approval, "observed": snapshot.reference}
        for operation in plan["operations"].values():
            if operation["status"] != "confirmed":
                operation["status"] = "abandoned"
    return result


def save_state(path: Path, state: object) -> None:
    """Atomically replace the complete journal; failure leaves the old anchors intact."""
    checked = _state(state)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_encoded(checked) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(name)


def load_state(path: Path) -> dict[str, Any]:
    """A missing/corrupt file is reconciliation, never implicit first adoption."""
    try:
        return _state(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReconciliationRequired("absent or unreadable applied baseline") from exc
