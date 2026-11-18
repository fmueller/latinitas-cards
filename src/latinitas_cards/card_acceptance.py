"""Full-assertion gate for opt-in forms, phrases and orchestration consumers.

Consumers enumerate every rendered linguistic assertion, including presuppositions
and explanatory clauses. This module checks their current decisions, not natural
language truth or rendering. Check producers own class-specific corroboration;
provider agreement is not a substitute. Historical exporters are not adapted here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .acceptance import AcceptanceReader, Eligibility
from .acceptance_records import Assertion


@dataclass(frozen=True)
class DecisionSource:
    reader: AcceptanceReader
    path: Path


@dataclass(frozen=True)
class CardClaims:
    note_id: str
    card_key: str
    assertions: tuple[Assertion, ...]
    decisions: tuple[DecisionSource, ...]

    def assess(self) -> Eligibility:
        """Re-read authority on every use, including a previously selected batch."""
        if not self.note_id or not self.card_key or not self.assertions:
            return Eligibility(False, "none", "incomplete_coverage")
        current = {a.id: a for a in self.assertions}
        if len(current) != len(self.assertions):
            return Eligibility(False, "none", "ambiguous_assertion")
        covered: set[str] = set()
        origins: set[str] = set()
        for source in self.decisions:
            # Curated rule/mnemonic/rendered approvals require their own exact
            # fingerprint adapter (T196); a generic human claim cannot replace it.
            if any(c.material == "curated/1" for c in source.reader.cells):
                return Eligibility(False, "none", "human_approval_required")
            try:
                decision, eligibility = source.reader.read(source.path)
            except (ValueError, OSError, TypeError):
                return Eligibility(False, "none", "stale_evidence")
            relevant = tuple(a for a in decision.assertions if a.id in current)
            if not relevant:
                return Eligibility(False, "none", "unrepresentable_claim")
            if any(a != current[a.id] for a in relevant):
                return Eligibility(False, decision.origin, "assertion_mismatch")
            if any(a.id in covered for a in relevant):
                return Eligibility(False, decision.origin, "ambiguous_assertion")
            if not eligibility.eligible:
                return eligibility
            covered.update(a.id for a in relevant)
            origins.add(decision.origin)
        if covered != set(current):
            return Eligibility(False, "none", "incomplete_coverage")
        origin = next(iter(origins)) if len(origins) == 1 else "mixed"
        return Eligibility(True, origin, "accepted")


def preview_cards(cards: Sequence[CardClaims], *, unresolved_only: bool = False) -> tuple[dict[str, object], ...]:
    """Optional inspection only: there is no approve-all or implicit review action."""
    rows: list[dict[str, object]] = []
    for card in cards:
        result = card.assess()
        if unresolved_only and result.eligible:
            continue
        rows.append(
            {
                "note_id": card.note_id,
                "card_key": card.card_key,
                "state": "accepted" if result.eligible else "withheld",
                "acceptance": result.origin,
                "reason": result.reason,
                "assertions": [a.model_dump(mode="json") for a in card.assertions],
            }
        )
    return tuple(rows)


def export_cards(cards: Sequence[CardClaims]) -> tuple[CardClaims, ...]:
    """Return only learning-eligible cards; one withholding does not veto siblings.

    This is an eligibility boundary, not a destination writer. Consumers must call
    it again at output/handoff, rather than cache the result as lasting authority.
    """
    identities = [(c.note_id, c.card_key) for c in cards]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate CardRef")
    return tuple(card for card in cards if card.assess().eligible)
