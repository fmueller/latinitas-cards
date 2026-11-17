"""Application-owned policy validation. Run files cannot register authority."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from .acceptance_records import Hash, Record, Text, Timestamp, Versions, canonical, digest
from .claim_review import ClaimKind


class ScopeCell(Record):
    claim_class: ClaimKind
    recipe: Text
    language: Literal["de", "en"]
    material: Text


class AutomaticPolicy(Record):
    format: Literal["latinitas-automatic-policy"] = "latinitas-automatic-policy"
    version: Literal[1] = 1
    id: Text
    revision: Text
    checker_sha256: Hash | None
    support_registry_sha256: Hash | None
    comparison_sha256: Hash | None
    versions_sha256: Hash | None
    criteria_sha256: Hash | None
    scope: tuple[ScopeCell, ...]

    @model_validator(mode="after")
    def commitments(self) -> Self:
        keys = [(c.claim_class, c.recipe, c.language, c.material) for c in self.scope]
        if keys != sorted(keys):
            raise ValueError("policy scope must be sorted")
        commits = (
            self.checker_sha256,
            self.support_registry_sha256,
            self.comparison_sha256,
            self.versions_sha256,
            self.criteria_sha256,
        )
        if any(c is None for c in commits) and (
            self.id != "v050-disabled" or self.revision != "1" or self.scope or any(c is not None for c in commits)
        ):
            raise ValueError("only bootstrap may omit commitments")
        return self


class PolicyAuthorization(Record):
    format: Literal["latinitas-policy-authorization"] = "latinitas-policy-authorization"
    version: Literal[1] = 1
    action: Literal["enable", "withdraw"]
    policy_sha256: Hash
    evaluation_report_sha256: Hash
    approved_by: Text
    approved_at: Timestamp
    supersedes: Hash | None


class PolicyRegistryEntry(Record):
    policy: AutomaticPolicy
    state: Literal["disabled", "enabled", "withdrawn"]
    authorization: PolicyAuthorization | None


# Records are intentionally separate from executable code/data commitments.
from .acceptance_registry import AUTHORIZATION_HISTORY, POLICY_REGISTRY  # noqa: E402
from .acceptance_registry import SUPPORT_REGISTRY as SUPPORT_REGISTRY  # noqa: E402


class SupportEntry(Record):
    recipe: Text
    materials: tuple[
        Literal[
            "source-form/1",
            "confirmed-ppp/1",
            "preposition-noun/1",
            "noun-genitive/1",
            "verb-noun/1",
            "adjective-noun/1",
            "target-cloze/1",
            "curated/1",
        ],
        ...,
    ]
    roles: tuple[Literal["present_1s", "present_infinitive", "perfect_1s", "supine", "ppp", "other"], ...]
    target: tuple[
        Literal[
            "meaning",
            "features",
            "person",
            "number",
            "tense",
            "mood",
            "voice",
            "case",
            "gender",
            "context",
            "rule",
            "mnemonic",
        ],
        ...,
    ]


class SupportRegistry(Record):
    version: Literal["v050-support/1"]
    entries: tuple[SupportEntry, ...]
    features: Literal["phrase_inputs.Features/1"]
    source_cell_rule: Literal["confirmed-selected-source-cells/1"]
    evidence_applicability: Literal["source-bound-applicable-rule-or-library/1"]
    ppp_role: Text
    cloze: Text
    unsupported: Text

    @model_validator(mode="after")
    def recipe_keys(self) -> Self:
        if len({e.recipe for e in self.entries}) != len(self.entries):
            raise ValueError("duplicate support recipe")
        if any(not e.materials or not e.roles or not e.target for e in self.entries):
            raise ValueError("support entry requires materials, roles and targets")
        return self


def implementation_digest() -> str:
    """Cover the reader/validator and its local dependencies, excluding authority records."""
    package = Path(__file__).parent
    # All package Python files cover transitive validation/adaptation code; authority
    # data lives separately so a withdrawal does not change evaluated algorithms.
    return digest(
        [
            {"path": p.relative_to(package).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(package.rglob("*.py"))
            if p.name != "acceptance_registry.py"
        ]
    )


def lookup_policy(identity: str, revision: str, sha256: str) -> PolicyRegistryEntry:
    SupportRegistry.read_json(canonical(SUPPORT_REGISTRY))
    history = {}
    for raw in AUTHORIZATION_HISTORY:
        authorization = PolicyAuthorization.read_json(raw)
        history[digest(authorization)] = authorization
    if len(history) != len(AUTHORIZATION_HISTORY):
        raise ValueError("conflicting authorization history")
    seen: set[tuple[str, str]] = set()
    found = None
    for raw in POLICY_REGISTRY:
        entry = PolicyRegistryEntry.read_json(raw)
        key = (entry.policy.id, entry.policy.revision)
        if key in seen:
            raise ValueError("conflicting application policy registry")
        seen.add(key)
        auth = entry.authorization
        if entry.state == "disabled":
            if auth is not None:
                raise ValueError("disabled policy has authorization")
        elif (
            not entry.policy.scope
            or auth is None
            or auth.policy_sha256 != digest(entry.policy)
            or auth.action != ("enable" if entry.state == "enabled" else "withdraw")
            or (entry.state == "withdrawn" and auth.supersedes is None)
        ):
            raise ValueError("invalid application policy authorization")
        if auth is not None and auth.supersedes is not None:
            prior = history.get(auth.supersedes)
            if prior is None or prior.policy_sha256 != auth.policy_sha256 or prior.action == auth.action:
                raise ValueError("invalid authorization supersession")
        if key == (identity, revision):
            found = entry
    if found is None or digest(found.policy) != sha256:
        raise ValueError("unknown_policy: requested policy tuple is not installed")
    return found


def policy_reason(versions: Versions, revision: str, cells: tuple[ScopeCell, ...]) -> str | None:
    entry = lookup_policy(versions.policy, revision, versions.policy_sha256)
    if entry.state != "enabled":
        return "policy_withdrawn" if entry.state == "withdrawn" else "policy_not_enabled"
    policy = entry.policy
    if (
        policy.versions_sha256 != digest(versions.model_dump(exclude={"policy", "policy_sha256"}))
        or policy.checker_sha256 != implementation_digest()
        or policy.comparison_sha256 != implementation_digest()
        or policy.support_registry_sha256 != digest(SUPPORT_REGISTRY)
    ):
        return "stale_evidence"
    if not cells or any(c not in policy.scope or not supported(c) for c in cells):
        return "unsupported_policy_scope"
    if any(c.material == "curated/1" for c in cells):
        return "human_approval_required"
    return None


def supported(cell: ScopeCell) -> bool:
    registry = SupportRegistry.read_json(canonical(SUPPORT_REGISTRY))
    return any(e.recipe == cell.recipe and cell.material in e.materials for e in registry.entries)


BOOTSTRAP = AutomaticPolicy(
    id="v050-disabled",
    revision="1",
    checker_sha256=None,
    support_registry_sha256=None,
    comparison_sha256=None,
    versions_sha256=None,
    criteria_sha256=None,
    scope=(),
)
