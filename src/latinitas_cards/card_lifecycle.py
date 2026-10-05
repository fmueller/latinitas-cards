"""Offline per-card lifecycle plans, not structural application or native proof.

Actual rows and retirement receipts belong to the bound observed baseline.
GeneratedNote carries the existing coherent-object identity and rendered eligibility;
neither prior-export checkpoints nor visible lemmas establish destination existence.
"""

from collections.abc import Mapping
from typing import Any

from .cards import TEMPLATE_REGISTRY
from .destination_reconciliation import reconcile_note
from .destination_state import (
    MANAGED_FIELDS,
    DestinationSnapshot,
    ReconciliationRequired,
    _binding,
    _reference,
    _state,
    _text,
    bound_card_rows,
    require_content_only_card_set,
)
from .notes import GeneratedNote


def plan_card_lifecycle(
    snapshot: DestinationSnapshot,
    state: object,
    identity: str,
    proposal: GeneratedNote | None,
    *,
    proposal_status: str = "approved",
    approvals: Mapping[str, str] | None = None,
    retire_object: str | None = None,
) -> dict[str, Any]:
    """Plan stable sibling effects; unsupported/blocked plans have no CSV target.

    Approval tokens identify reviewed lifecycle decisions, not profile confirmation.
    A retirement receipt on an observed card row records approval, pre_suspended,
    tool_suspended and the observation reference. This planner never writes one or
    advances the journal. Unknown provenance blocks reversal, even with approval.
    Last-safe retention is a whole-note evidence bundle, explicitly stale/pending;
    it is never smuggled into newly approved shared fields alongside fresh siblings.
    """
    baseline = _state(state)
    data = snapshot.payload
    if baseline["binding"] != _binding(data) or baseline["deck_options"] != data["deck_options"]:
        raise ReconciliationRequired("wrong destination or changed deck options")
    if any(plan["status"] not in ("complete", "abandoned") for plan in baseline["plans"].values()):
        raise ReconciliationRequired("pending effects require observation/reconciliation")
    destination = next((note for note in data["notes"] if note["identity"] == identity), None)
    anchor = baseline["anchors"].get(identity)
    if destination is None or anchor is None:
        raise ReconciliationRequired("missing identity baseline; explicit adoption required")
    if any(
        anchor[key] != destination[key] for key in ("source", "guid", "local_id", "note_type_id", "cards", "history")
    ):
        raise ReconciliationRequired("changed identity/card evidence requires reconciliation")
    rows = bound_card_rows(destination)
    choices = dict(approvals or {})
    for token in choices.values():
        _text(token)
    if retire_object is not None:
        _text(retire_object)
    if proposal_status not in ("approved", "missing", "ambiguous", "withheld", "inapplicable", "parser_failure"):
        raise ReconciliationRequired("unknown proposal status")
    if proposal is not None:
        source = [proposal.provenance.source_scope, proposal.provenance.source_identity]
        if len(destination["source"]) == 3:
            source.append(proposal.object_key)
        if proposal.latinitas_id != identity or source != destination["source"]:
            raise ReconciliationRequired("proposal identity/object requires confirmation")
        if tuple(card.slot for card in proposal.cards) != TEMPLATE_REGISTRY:
            raise ReconciliationRequired("proposal requires complete frozen template bindings")
        for card in proposal.cards:
            if card.eligible != bool(card.guard.strip()) or (
                card.eligible and not (card.prompt.strip() and card.answer.strip())
            ):
                raise ReconciliationRequired("invalid eligible card content")

    uncertain = proposal is None or proposal_status != "approved"
    eligible = set(proposal.card_keys) if proposal is not None and not uncertain and retire_object is None else set()
    blocked: list[str] = []
    if uncertain and retire_object is None:
        blocked.append("uncertain object/data requires confirmation; no implicit retirement")
    effects: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for slot in TEMPLATE_REGISTRY:
        key = slot.semantic_key
        row = rows.get(key)
        approval = choices.pop(key, None)
        if row is None and key not in eligible:
            if approval is not None:
                raise ReconciliationRequired("lifecycle approval has no existing or eligible card")
            continue
        effect: dict[str, Any] = {
            "semantic_key": key,
            "template_name": slot.template_name,
            "ordinal": slot.ordinal,
            "card_id": row["id"] if row else None,
            "effect": "retain",
            "approval": approval,
            "unsuspend": False,
        }
        if row is None:
            effect["effect"] = "add"
        else:
            if type(row.get("suspended")) is not bool:
                raise ReconciliationRequired("unknown pre-retirement suspension state")
            retirement = row.get("retirement")
            if retirement is not None and not isinstance(retirement, dict):
                raise ReconciliationRequired("invalid retirement provenance")
            if retire_object is not None or (not uncertain and key not in eligible and approval is not None):
                effect.update(
                    effect="retire",
                    approval=retire_object or approval,
                    pre_suspended=row["suspended"],
                    suspension_owner="user_or_unknown" if row["suspended"] else "pending_tool",
                )
            elif retirement is not None:
                if key in eligible and approval is not None:
                    effect["effect"] = "reactivate"
                    known = (
                        isinstance(retirement.get("approval"), str)
                        and bool(retirement["approval"].strip())
                        and type(retirement.get("pre_suspended")) is bool
                        and type(retirement.get("tool_suspended")) is bool
                    )
                    if known:
                        _reference(retirement.get("observed"))
                    if not known or (retirement["pre_suspended"] and retirement["tool_suspended"]):
                        blocked.append(f"{key}: unknown/conflicting suspension ownership")
                    elif retirement["tool_suspended"] and not row["suspended"]:
                        blocked.append(f"{key}: changed tool suspension requires review")
                    else:
                        effect["unsuspend"] = not retirement["pre_suspended"] and retirement["tool_suspended"]
                else:
                    blocked.append(f"{key}: retired card requires eligible explicit reactivation")
            elif not uncertain and key not in eligible:
                blocked.append(f"{key}: eligibility loss requires explicit retirement approval")
            if uncertain or key not in eligible or retirement is not None or effect["effect"] == "retire":
                names = (slot.enabled_field, slot.prompt_field, slot.answer_field)
                safe = anchor["fields"]
                if not all(safe[name].strip() for name in names) or any(
                    safe[name] != destination["fields"][name] for name in names
                ):
                    blocked.append(f"{key}: last-safe retention cannot be represented without review")
                effect["retention"] = {
                    "fields": dict(safe),
                    "observed": anchor["observed"],
                    "stale": True,
                    "pending": True,
                    "newly_approved": False,
                }
        if effect["effect"] != "retain":
            unsupported.append(key)
        effects.append(effect)
    if choices:
        raise ReconciliationRequired("approval targets unknown template")
    result: dict[str, Any] = {
        "identity": identity,
        "snapshot": snapshot.reference,
        "baseline_version": baseline["version"],
        "effects": effects,
        "blocked": sorted(blocked),
        "unsupported": unsupported,
        "object_retirement": retire_object,
        "lifecycle_tags": ["latinitas::retired"] if retire_object else [],
    }
    if not blocked and not unsupported and retire_object is None and proposal is not None:
        fields = {name: value for name, value in proposal.to_anki_fields() if name in MANAGED_FIELDS}
        reconciliation = reconcile_note(
            snapshot,
            baseline,
            identity,
            fields,
            {origin: anchor.get(origin, []) for origin in ("source_tags", "configured_tags", "lifecycle_tags")},
        )
        result["reconciliation"] = reconciliation
        if not reconciliation["conflicts"]:
            require_content_only_card_set(destination, reconciliation["fields"])
            result["target"] = reconciliation["target"]
    return result
