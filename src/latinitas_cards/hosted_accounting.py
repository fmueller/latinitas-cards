"""Explicit estimated reservations, never a service-token or financial guarantee.

The owner-only application state directory is authoritative, not the run's path.
It must be retained across runs and restores. Local filesystem administrators are
trusted, just as they are for hosted permission files; copying a run is not a
campaign allocation. No provider communication or acceptance authority lives here.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from .acceptance_records import Hash, Record, Text, Timestamp, canonical
from .pipeline_batches import _atomic

type Phase = Literal["development", "study", "native", "qa"]
PHASE_CALLS: dict[Phase, int] = {"development": 80, "study": 1600, "native": 160, "qa": 160}
type Purpose = Literal["development", "isolated_evaluation", "runtime"]


class Pricing(Record):
    """Operator-checked public rates; integer microdollars per million tokens."""

    endpoint_sha256: Hash
    input_microdollars_per_million: int = Field(gt=0)
    output_microdollars_per_million: int = Field(gt=0)
    source_url: Text
    checked_at: Timestamp
    expires_at: Timestamp

    @model_validator(mode="after")
    def public_source(self) -> Self:
        from urllib.parse import urlsplit

        url = urlsplit(self.source_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("pricing requires a credential-free public HTTPS source")
        if datetime.fromisoformat(self.checked_at) >= datetime.fromisoformat(self.expires_at):
            raise ValueError("invalid pricing observation interval")
        return self

    def applicable(self, endpoint_sha256: str) -> bool:
        return endpoint_sha256 == self.endpoint_sha256 and datetime.fromisoformat(self.checked_at) <= datetime.now(
            UTC
        ) < datetime.fromisoformat(self.expires_at)

    def reservation(self, input_tokens: int, output_tokens: int) -> int:
        # Round upward, without binary floating point or cache/discount assumptions.
        return (
            input_tokens * self.input_microdollars_per_million
            + output_tokens * self.output_microdollars_per_million
            + 999999
        ) // 1000000


class EstimatedAccounting(Record):
    format: Literal["latinitas-estimated-accounting"]
    version: Literal[1]
    kind: Literal["estimated"]
    campaign_id: Hash
    phase: Phase
    purpose: Purpose
    estimator: Literal["wire-padded/1"]
    pricing: tuple[Pricing, ...]
    service_revision: Text | None
    acknowledges_no_guaranteed_cap: Literal[True]

    @model_validator(mode="after")
    def purposes(self) -> Self:
        if self.phase == "study" and self.purpose != "isolated_evaluation":
            raise ValueError("protected study cannot have runtime/export custody")
        if self.phase == "development" and self.purpose != "development":
            raise ValueError("development allocation requires development custody")
        if len({p.endpoint_sha256 for p in self.pricing}) != len(self.pricing):
            raise ValueError("duplicate endpoint pricing")
        return self


class CampaignAllocation(Record):
    format: Literal["latinitas-campaign-allocation"]
    version: Literal[1]
    campaign_id: Hash
    authority: Text
    max_calls: int = Field(gt=0, le=2000)
    max_tokens: int = Field(gt=0, le=40000000)
    max_microdollars: int = Field(gt=0, le=25000000)
    max_execution_seconds: int = Field(gt=0, le=86400)
    phase_calls: dict[Phase, int]

    @model_validator(mode="after")
    def ceilings(self) -> Self:
        if set(self.phase_calls) != set(PHASE_CALLS) or any(
            type(n) is not int or not 0 <= n <= PHASE_CALLS[p] for p, n in self.phase_calls.items()
        ):
            raise ValueError("phase caps cannot be reassigned")
        return self


class Usage(Record):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def total(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("inconsistent usage")
        return self


class Reservation(Record):
    accounting: EstimatedAccounting
    request_sha256: Hash
    scope_sha256: Hash
    endpoint_sha256: Hash
    permission_id: Hash | None
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0, le=2048)
    wire_bytes: int = Field(gt=0, le=16384)
    microdollars: int = Field(gt=0)
    execution_seconds: int = Field(gt=0, le=60)
    pricing_sha256: Hash


class Charge(Record):
    call_id: Text
    run_id: Text
    stage_id: Hash
    reservation: Reservation
    state: Literal["pending", "succeeded", "failed", "unknown", "cancelled", "abandoned"]
    usage: Usage | None
    stop_reason: Text | None
    elapsed_milliseconds: int = Field(ge=0)


class CampaignLedger(Record):
    format: Literal["latinitas-campaign-ledger"] = "latinitas-campaign-ledger"
    version: Literal[1] = 1
    allocation: CampaignAllocation
    charges: tuple[Charge, ...]
    invalidated_endpoints: tuple[Hash, ...] = ()

    @model_validator(mode="after")
    def identities(self) -> Self:
        if len({c.call_id for c in self.charges}) != len(self.charges) or any(
            c.reservation.accounting.campaign_id != self.allocation.campaign_id for c in self.charges
        ):
            raise ValueError("campaign charge identity mismatch")
        return self


def state_root() -> Path:
    return Path.home() / ".local" / "state" / "latinitas-cards" / "hosted-campaigns"


class Campaign:
    def __init__(self, campaign_id: str) -> None:
        # Validate before constructing a path, including for direct API consumers.
        if len(campaign_id) != 64 or any(c not in "0123456789abcdef" for c in campaign_id):
            raise ValueError("invalid campaign identity")
        self.path = state_root() / (campaign_id + ".json")

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        root = state_root()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or root.stat().st_mode & 0o077:
            raise ValueError("campaign state requires an owner-only directory")
        fd = os.open(root / "dispatch.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("campaign_busy") from None
            yield
        finally:
            os.close(fd)

    def initialize(self, allocation: CampaignAllocation) -> None:
        if self.path.name != allocation.campaign_id + ".json":
            raise ValueError("allocation identity mismatch")
        with self.lock():
            if self.path.exists():
                if self.load().allocation != allocation:
                    raise ValueError("campaign allocation is immutable")
            else:
                self.save(CampaignLedger(allocation=allocation, charges=()))

    def load(self) -> CampaignLedger:
        if self.path.is_symlink() or not self.path.exists():
            raise ValueError("missing authoritative campaign; cannot reset spending")
        ledger = CampaignLedger.read_json(self.path.read_bytes())
        if ledger.allocation.campaign_id + ".json" != self.path.name:
            raise ValueError("campaign identity mismatch")
        return ledger

    def save(self, ledger: CampaignLedger) -> None:
        _atomic(self.path, canonical(ledger))

    def abandon(self, call_id: str) -> None:
        """Explicitly close an orphaned/unknown charge, never refund or clear review stops."""
        with self.lock():
            charge = next((c for c in self.load().charges if c.call_id == call_id), None)
            if charge is None or charge.state not in ("pending", "unknown"):
                raise ValueError("not an unknown campaign dispatch")
            self.finish(
                call_id,
                state="abandoned",
                usage=charge.usage,
                stop_reason=charge.stop_reason,
                elapsed_milliseconds=charge.elapsed_milliseconds,
            )

    def reserve(
        self,
        charge: Charge,
        *,
        run_calls: int,
        run_tokens: int,
        permission_calls: int,
        permission_tokens: int,
        permission_microdollars: int,
        permission_seconds: int,
        local_call_ids: set[str],
        duplicate_acknowledged: bool,
    ) -> str | None:
        """Caller holds the global dispatch lock through final persistence."""
        ledger = self.load()
        prior = ledger.charges
        run = [c for c in prior if c.run_id == charge.run_id]
        if {c.call_id for c in run} != local_call_ids:
            return "campaign_run_diverged"
        if any(c.state in ("pending", "unknown") for c in prior):
            return "campaign_unreconciled_call"
        if any(c.stop_reason for c in prior):
            return "campaign_review_required"
        r = charge.reservation
        if (
            any(
                c.state == "abandoned"
                and (c.reservation.request_sha256, c.reservation.endpoint_sha256)
                == (r.request_sha256, r.endpoint_sha256)
                for c in prior
            )
            and not duplicate_acknowledged
        ):
            return "duplicate_processing_acknowledgment_required"
        # Reuse cannot cross a changed material service identity. Null explicitly
        # means unpinnable, not an assertion of immutable deployment identity.
        if r.endpoint_sha256 in ledger.invalidated_endpoints or any(
            c.reservation.endpoint_sha256 == r.endpoint_sha256
            and c.reservation.accounting.service_revision != r.accounting.service_revision
            for c in prior
        ):
            if r.endpoint_sha256 not in ledger.invalidated_endpoints:
                self.save(
                    ledger.model_copy(
                        update={"invalidated_endpoints": (*ledger.invalidated_endpoints, r.endpoint_sha256)}
                    )
                )
            return "service_change_requires_requalification"
        phase = [c for c in prior if c.reservation.accounting.phase == r.accounting.phase]
        permission = [c for c in prior if c.reservation.permission_id == r.permission_id]
        checks = (
            (
                prior,
                ledger.allocation.max_calls,
                ledger.allocation.max_tokens,
                ledger.allocation.max_microdollars,
                ledger.allocation.max_execution_seconds,
            ),
            (run, min(run_calls, 200), min(run_tokens, 200000), 25000000, 86400),
            (phase, ledger.allocation.phase_calls[r.accounting.phase], 40000000, 25000000, 86400),
            (permission, permission_calls, permission_tokens, permission_microdollars, permission_seconds),
        )
        for calls, max_calls, max_tokens, max_cost, max_seconds in checks:
            if (
                len(calls) + 1 > max_calls
                or sum(c.reservation.input_tokens + c.reservation.output_tokens for c in calls)
                + r.input_tokens
                + r.output_tokens
                > max_tokens
                or sum(c.reservation.microdollars for c in calls) + r.microdollars > max_cost
                or sum(c.reservation.execution_seconds for c in calls) + r.execution_seconds > max_seconds
            ):
                return "campaign_budget_exhausted"
        self.save(ledger.model_copy(update={"charges": (*prior, charge)}))
        return None

    def finish(
        self,
        call_id: str,
        *,
        state: Literal["succeeded", "failed", "unknown", "cancelled", "abandoned"],
        usage: Usage | None,
        stop_reason: str | None,
        elapsed_milliseconds: int,
    ) -> None:
        ledger = self.load()
        if not any(c.call_id == call_id for c in ledger.charges):
            raise ValueError("missing campaign charge")
        outcome = {
            "state": state,
            "usage": usage,
            "stop_reason": stop_reason,
            "elapsed_milliseconds": elapsed_milliseconds,
        }
        charges = tuple(c.model_copy(update=outcome) if c.call_id == call_id else c for c in ledger.charges)
        self.save(ledger.model_copy(update={"charges": charges}))


def reported_usage(provider: str, raw: bytes) -> Usage:
    from .acceptance_records import decode_json

    data = decode_json(raw)
    if not isinstance(data, dict):
        raise ValueError("missing usage")
    if provider == "ollama":
        input_tokens = data.get("prompt_eval_count")
        output_tokens = data.get("eval_count")
        total = input_tokens + output_tokens if type(input_tokens) is int and type(output_tokens) is int else None
    else:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("missing usage")
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        total = usage.get("total_tokens")
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total)  # type: ignore[arg-type]
