"""Pure inspectable reconciliation; results are not approval or native proof.

Reconcile once per shared note, never once per card. Callers must resolve all
conflicts before submitting a target to the existing observation journal.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .destination_state import MANAGED_FIELDS, DestinationSnapshot, ReconciliationRequired, _binding, _ownership, _state

ORIGINS = ("source_tags", "configured_tags", "lifecycle_tags")


def reconcile_values(
    baseline: Mapping[str, str] | None,
    destination: Mapping[str, str],
    proposal: Mapping[str, str],
    baseline_tags: Sequence[str],
    destination_tags: Sequence[str],
    ownership: Mapping[str, Any],
    contributions: Mapping[str, Sequence[str]],
    decisions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare B/D/P without changing an anchor. Decisions require review evidence.

    User fields are excluded, even if supplied in all three inputs. Missing
    managed values and unknown ownership require adoption/reconciliation first.
    Suppressed tags record an explicit choice to retain a destination deletion;
    keep_tags records persistent positive user ownership, not inferred intent.
    """
    if baseline is None:
        raise ReconciliationRequired("absent managed baseline")
    normalized: dict[str, Any] = {name: [] for name in (*ORIGINS, "keep_tags", "keep_fields", "suppressed_tags")}
    normalized.update(ownership)
    previous = _ownership(normalized, list(baseline_tags))
    choices = dict(decisions or {})
    recorded: dict[str, Any] = {}
    conflicts: list[str] = []

    def decision(key: str, allowed: set[str]) -> Mapping[str, Any] | None:
        choice = choices.pop(key, None)
        if choice is not None:
            approval = choice.get("approval")
            if choice.get("action") not in allowed or not isinstance(approval, str) or not approval.strip():
                raise ReconciliationRequired("invalid or unreviewed decision")
            recorded[key] = dict(choice)
        return choice

    fields: dict[str, str] = {}
    writes: dict[str, str] = {}
    for name in sorted(set(baseline) | set(destination) | set(proposal)):
        if name not in MANAGED_FIELDS:
            continue
        if name not in baseline or name not in destination or name not in proposal:
            raise ReconciliationRequired("missing managed field baseline/destination/proposal")
        b, d, p = baseline[name], destination[name], proposal[name]
        key = f"field:{name}"
        choice = decision(key, {"keep_destination", "accept_proposal", "replacement"})
        value = d
        if name in previous["keep_fields"]:
            if choice is not None and choice["action"] != "keep_destination":
                raise ReconciliationRequired("user keep field cannot be overwritten")
        elif choice is not None:
            value = d if choice["action"] == "keep_destination" else p
            if choice["action"] == "replacement":
                if not isinstance(choice.get("value"), str):
                    raise ReconciliationRequired("reviewed replacement must be text")
                value = choice["value"]
        elif d == p:
            value = d
        elif d == b:
            value = p
        else:
            conflicts.append(key)
        fields[name] = value
        if value != d:
            writes[name] = value
        if choice is not None:
            recorded[key]["result"] = value

    old = set().union(*(set(previous[name]) for name in ORIGINS))
    proposed = {name: set(contributions.get(name, [])) for name in ORIGINS}
    required = set().union(*proposed.values())
    baseline_tag_set = set(baseline_tags)
    d_tags = set(destination_tags)
    destination_additions = d_tags - baseline_tag_set
    keep = set(previous["keep_tags"]) | destination_additions
    suppressed = set(previous["suppressed_tags"]) & required
    final = set(d_tags)
    removals = sorted((old - required) & d_tags - keep)
    for tag in sorted(old | required | keep):
        key = f"tag:{tag}"
        collision = tag in proposed["lifecycle_tags"] and tag in d_tags and tag not in previous["lifecycle_tags"]
        choice = decision(key, {"keep_destination", "accept_proposal", "replacement", "keep_as_user_owned"})
        # An explicit user ownership override wins on later plans, including
        # reserved lifecycle contributions, unless explicitly reviewed again.
        # It grants no card-state permission.
        if (choice is None or choice["action"] != "accept_proposal") and (
            tag in previous["keep_tags"] or (tag in destination_additions and not collision)
        ):
            for origin in ORIGINS:
                proposed[origin].discard(tag)
            required.discard(tag)
            collision = False
        if choice is not None:
            action = choice["action"]
            present = tag in d_tags if action == "keep_destination" else tag in required
            if action == "replacement":
                if type(choice.get("value")) is not bool:
                    raise ReconciliationRequired("reviewed tag replacement must be boolean")
                present = choice["value"]
            if action == "keep_as_user_owned":
                if tag not in d_tags:
                    raise ReconciliationRequired("cannot keep absent tag as user owned")
                present = True
                keep.add(tag)
                for origin in ORIGINS:
                    proposed[origin].discard(tag)
                required.discard(tag)
            elif collision and action != "accept_proposal" and present:
                proposed["lifecycle_tags"].discard(tag)
                if not any(tag in proposed[origin] for origin in ORIGINS):
                    required.discard(tag)
                keep.add(tag)
            elif action == "accept_proposal":
                keep.discard(tag)
            if present:
                final.add(tag)
                suppressed.discard(tag)
                if tag not in required:
                    keep.add(tag)
            else:
                final.discard(tag)
                keep.discard(tag)
                if tag in required:
                    suppressed.add(tag)
            recorded[key]["result"] = present
        elif collision or (tag in old & required and tag not in d_tags and tag not in suppressed):
            conflicts.append(key)
        elif tag in required and tag not in suppressed:
            final.add(tag)
        elif tag in old and tag not in keep:
            final.discard(tag)
    if choices:
        raise ReconciliationRequired("decision targets unknown or user-owned field/tag")
    result_ownership = {name: sorted(proposed[name]) for name in ORIGINS}
    result_ownership.update(keep_tags=sorted(keep & final), keep_fields=previous["keep_fields"])
    suppressed &= required - final
    if suppressed:
        result_ownership["suppressed_tags"] = sorted(suppressed)
    return {
        "fields": fields,
        "writes": writes,
        "tags": sorted(final),
        "tag_write": final != d_tags,
        "ownership": result_ownership,
        "removals": removals,
        "conflicts": sorted(conflicts),
        "decisions": recorded,
    }


def reconcile_note(
    snapshot: DestinationSnapshot,
    state: object,
    identity: str,
    proposal: Mapping[str, str],
    contributions: Mapping[str, Sequence[str]],
    decisions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reconcile one shared note using the existing bound evidence contract.

    Unresolved results deliberately have no journal target. Changed anchors
    still require the journal's explicit review_reconciliation before recording
    a later pending application; this function never advances the baseline.
    """
    baseline = _state(state)
    data = snapshot.payload
    if baseline["binding"] != _binding(data) or baseline["deck_options"] != data["deck_options"]:
        raise ReconciliationRequired("wrong destination or changed deck options")
    if any(plan["status"] not in ("complete", "abandoned") for plan in baseline["plans"].values()):
        raise ReconciliationRequired("pending effects require observation/reconciliation")
    notes = {note["identity"]: note for note in data["notes"]}
    if identity not in baseline["anchors"] or identity not in notes:
        raise ReconciliationRequired("missing note baseline; explicit adoption required")
    anchor, destination = baseline["anchors"][identity], notes[identity]
    if any(anchor[key] != destination[key] for key in ("source", "guid", "local_id", "note_type_id")):
        raise ReconciliationRequired("changed note identity requires review")
    result = reconcile_values(
        anchor["fields"],
        destination["fields"],
        proposal,
        anchor["tags"],
        destination["tags"],
        anchor,
        contributions,
        decisions,
    )
    if not result["conflicts"]:
        result["target"] = {
            "fields": result["fields"],
            "tags": result["tags"],
            **result["ownership"],
            "decisions": {**anchor.get("decisions", {}), **result["decisions"]},
        }
    return result
