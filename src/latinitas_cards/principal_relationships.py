"""Offline principal-part comparison; extraction is not morphological approval.

Generation claims use ``<LatinitasID>:<role>`` candidate IDs, including source
scope in the note identity. Each explanation
and split requires its own fresh decision. No suffix heuristic or analyzer is
implicitly trusted here. The display layer consumes this data, not vice versa.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from .claim_review import ClaimAssessment, assess_claim
from .principal_parts import ParsedPrincipalParts
from .profile import DeckProfile

FOURTH_ROLES = ("perfect_passive_participle", "supine")


@dataclass(frozen=True, slots=True)
class RoleComparison:
    role: str
    form: str | None
    status: Literal["present", "absent", "withheld"]
    reason: str
    segmentation: str | None = None
    explanation: str | None = None
    claims: tuple[ClaimAssessment, ...] = ()


@dataclass(frozen=True, slots=True)
class PrincipalPartComparison:
    roles: tuple[RoleComparison, ...]


def compare_principal_parts(
    parsed: ParsedPrincipalParts,
    profile: DeckProfile,
    assessments: Sequence[ClaimAssessment] = (),
    *,
    source_identity: str | None = None,
) -> PrincipalPartComparison:
    """Expose four roles and individually gated linguistic claims for review.

    Selecting a profile role does not adjudicate an uncontextualized fourth
    form. Both fourth-role declarations in one layout require profile repair.
    Conflicting accepted values are withheld, rather than picking a winner.
    """
    identity = source_identity or parsed.source_identity
    fourth = tuple(role for role in profile.principal_parts.roles if role in FOURTH_ROLES)
    if len(fourth) > 1:
        raise ValueError("Explicitly resolve the profile's conflicting PPP and supine roles before generation")
    roles = ("present_1s", "present_infinitive", "perfect_1s", *(fourth or ("fourth_role",)))
    result: list[RoleComparison] = []
    for role in roles:
        part = parsed.by_role.get(role)
        if part is None or part.is_omitted:
            result.append(RoleComparison(role, None, "absent", "Nicht vorhanden; Quelle oder Rollenprofil prüfen."))
            continue
        reviewed: list[ClaimAssessment] = []
        for item in assessments:
            if identity is None or item.claim.candidate_id != f"{identity}:{role}":
                continue
            current = assess_claim(item.claim, item.decision)
            if (
                item.claim.candidate_text != part.display
                or item.claim.profile.model_dump(exclude={"morphology"}) != profile.model_dump(exclude={"morphology"})
                or parsed.evidence is None
                or item.claim.extraction != parsed.evidence
            ):
                current = replace(
                    current,
                    status="withheld",
                    reason="Current candidate, extraction or profile differs; renewed review required",
                )
            reviewed.append(current)
        bound = tuple(reviewed)
        accepted: dict[str, set[str]] = {}
        for item in bound:
            if item.status == "accepted":
                accepted.setdefault(item.claim.kind, set()).add(item.claim.value)
        conflict = len(accepted.get("label", set())) > 1
        label_withheld = any(item.claim.kind == "label" and item.status == "withheld" for item in bound)
        fourth_unreviewed = role in FOURTH_ROLES and accepted.get("label") != {role}
        blocked = part.unresolved or conflict or label_withheld or fourth_unreviewed
        reason = (
            "Zurückgehalten; Alternativen, Rollenprofil und einzelne Belege ausdrücklich prüfen."
            if blocked
            else "Quellform im bestätigten Rollenprofil; Analyse benötigt gesonderte Prüfung."
        )
        segmentation = accepted.get("segmentation", set())
        explanation = accepted.get("explanation", set())
        withheld_kinds = {item.claim.kind for item in bound if item.status == "withheld"}
        result.append(
            RoleComparison(
                role,
                part.display,
                "withheld" if blocked else "present",
                reason,
                next(iter(segmentation))
                if len(segmentation) == 1 and not blocked and "segmentation" not in withheld_kinds
                else None,
                next(iter(explanation))
                if len(explanation) == 1 and not blocked and "explanation" not in withheld_kinds
                else None,
                bound,
            )
        )
    return PrincipalPartComparison(tuple(result))
