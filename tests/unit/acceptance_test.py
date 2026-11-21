"""Acceptance persistence is not policy authority or a legacy human review."""

import json
from pathlib import Path
from typing import Literal

import pytest

from latinitas_cards.acceptance import (
    Acceptance,
    AcceptanceReader,
    Assertion,
    Binding,
    Check,
    Human,
    Versions,
    assess,
    check_envelope,
    digest,
    load_acceptance,
    save_acceptance,
    save_stage,
)
from latinitas_cards.acceptance_policy import BOOTSTRAP, ScopeCell
from latinitas_cards.claim_review import Evidence

CELLS = (ScopeCell(claim_class="label", recipe="form_recognition", language="de", material="source-form/1"),)


def candidate(root: Path, *, human: bool = False, versions_override: Versions | None = None) -> Acceptance:
    binding = Binding(
        source_scope="deck",
        source_identity="amo",
        source_sha256=digest("source"),
        extraction_sha256=digest("extraction"),
        mapping_sha256=digest("mapping"),
        task_sha256=digest("task"),
        language="de",
        config_sha256=digest("config"),
    )
    versions = Versions(
        producer="producer/1",
        analyzer="analyzer/1",
        proposal_prompt=None,
        counter_prompt=None,
        checker="checker/1",
        comparison="comparison/1",
        policy="v050-disabled",
        policy_sha256=digest(BOOTSTRAP),
        proposal_model=None,
        counter_model=None,
    )
    versions = versions_override or versions
    assertion = Assertion(
        id="answer",
        kind="label",
        value="present",
        selected_sense=None,
        alternatives=("alternative",),
        evidence=(Evidence("source", "amo", "1", "amo"),),
    )
    method: Literal["human", "rule"] = "human" if human else "rule"
    envelope = check_envelope(binding, versions, method, (assertion,))
    check = Check(
        id="check",
        assertion_ids=("answer",),
        method=method,
        version=versions.checker,
        result="pass",
        evidence=assertion.evidence,
        alternatives=(),
        objections=(),
        input_sha256=digest(envelope),
    )
    save_stage(
        root, run_id="run", stage_id="check", stage="decide", binding=binding, versions=versions, envelope=envelope
    )
    review = (
        Human(
            reviewer="teacher",
            reason="verified",
            reviewed_sha256=digest(
                {
                    "binding": binding,
                    "assertions": (assertion,),
                }
            ),
        )
        if human
        else None
    )
    return Acceptance.create(
        binding=binding,
        versions=versions,
        assertions=(assertion,),
        checks=(check,),
        origin="human" if human else "automatic",
        state="accepted" if human else "withheld",
        human=review,
        stage_ids=("check",),
        reason="reviewed" if human else "policy_not_enabled",
        created_at="2026-11-17T00:00:00Z",
    )


def test_disabled_roundtrip_and_forgery(tmp_path: Path) -> None:
    decision = candidate(tmp_path)
    path = tmp_path / "acceptance.json"
    save_acceptance(path, decision)
    loaded = load_acceptance(path)
    cell = ScopeCell(claim_class="label", recipe="form_recognition", language="de", material="source-form/1")
    assert (
        assess(
            loaded,
            root=tmp_path,
            binding=decision.binding,
            versions=decision.versions,
            assertions=decision.assertions,
            cells=(cell,),
        ).reason
        == "policy_not_enabled"
    )
    forged = Acceptance.create(**{**decision.model_dump(exclude={"decision_id"}), "state": "accepted"})
    assert not assess(
        forged,
        root=tmp_path,
        binding=decision.binding,
        versions=decision.versions,
        assertions=decision.assertions,
        cells=(cell,),
    ).eligible


def test_human_roundtrip_staleness_and_artifact_integrity(tmp_path: Path) -> None:
    decision = candidate(tmp_path, human=True)
    path = tmp_path / "acceptance.json"
    save_acceptance(path, decision)
    loaded = load_acceptance(path)

    def eligibility(binding: Binding = decision.binding, versions: Versions = decision.versions) -> bool:
        return assess(
            loaded, root=tmp_path, binding=binding, versions=versions, assertions=decision.assertions, cells=CELLS
        ).eligible

    result = assess(
        loaded,
        root=tmp_path,
        binding=decision.binding,
        versions=decision.versions,
        assertions=decision.assertions,
        cells=CELLS,
    )
    assert result.eligible and result.origin == "human"
    for field in ("source_sha256", "extraction_sha256", "mapping_sha256", "task_sha256", "config_sha256", "language"):
        changed = decision.binding.model_copy(update={field: "en" if field == "language" else digest("changed")})
        assert not eligibility(binding=changed)
    for field in ("producer", "analyzer", "checker", "comparison", "proposal_prompt"):
        changed_versions = decision.versions.model_copy(update={field: "changed/2"})
        assert not eligibility(versions=changed_versions)
    (tmp_path / "artifacts/check.json").write_text("{}")
    assert not eligibility()


@pytest.mark.parametrize("change", [{"version": True}, {"version": "1"}, {"version": 2}, {"reviewer": "auto"}])
def test_strict_record(tmp_path: Path, change: dict[str, object]) -> None:
    decision = candidate(tmp_path)
    path = tmp_path / "bad.json"
    save_acceptance(path, decision)
    original = path.read_bytes()
    assert load_acceptance(path) == decision
    path.write_text(json.dumps({**decision.model_dump(mode="json"), **change}))
    with pytest.raises(ValueError):
        load_acceptance(path)
    path.write_bytes(original)
    assert load_acceptance(path) == decision


def test_record_normalizes_literal_surrogate_pairs(tmp_path: Path) -> None:
    raw = candidate(tmp_path).binding.model_dump_json().replace('"amo"', '"\ud83d\ude00"')
    for _ in range(2):
        assert Binding.read_json(raw).source_identity == "\U0001f600"


def test_public_reader_never_exports_disabled_or_evaluation_results(tmp_path: Path) -> None:
    decision = candidate(tmp_path)
    path = tmp_path / "decision.json"
    save_acceptance(path, decision)
    cell = ScopeCell(claim_class="label", recipe="form_recognition", language="de", material="source-form/1")
    reader = AcceptanceReader(tmp_path, decision.binding, decision.versions, decision.assertions, (cell,))
    produced = reader.decide(checks=decision.checks, origin="automatic", human=None, stage_ids=decision.stage_ids)
    assert produced.state == "withheld" and produced.origin == "automatic" and produced.reason == "policy_not_enabled"
    assert reader.preview(path)["origin"] == "automatic"
    assert reader.preview(path)["reason"] == "policy_not_enabled"
    with pytest.raises(ValueError, match="policy_not_enabled"):
        reader.export(path)
    handoff = reader.handoff(path, note_id="one", card_key="recognition")
    assert handoff and handoff["note_id"] == "one" and handoff["reason"] == "policy_not_enabled"
    evaluation = tmp_path / "would-accept.json"
    evaluation.write_text(json.dumps({"would_accept": True, "assertions": ["not learning data"]}))
    with pytest.raises(ValueError):
        reader.export(evaluation)


def test_synthetic_installed_policy_and_withdrawal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from latinitas_cards import acceptance_policy as policies
    from latinitas_cards.acceptance_records import canonical

    original = candidate(tmp_path / "original")
    cell = ScopeCell(claim_class="label", recipe="form_recognition", language="de", material="source-form/1")
    policy = policies.AutomaticPolicy(
        id="synthetic-test-only",
        revision="fixture-7",
        checker_sha256=policies.implementation_digest(),
        comparison_sha256=policies.implementation_digest(),
        support_registry_sha256=digest(policies.SUPPORT_REGISTRY),
        versions_sha256=digest(original.versions.model_dump(exclude={"policy", "policy_sha256"})),
        criteria_sha256=digest("synthetic criteria; not empirical approval"),
        scope=(cell,),
    )
    auth = policies.PolicyAuthorization(
        action="enable",
        policy_sha256=digest(policy),
        evaluation_report_sha256=digest("synthetic report"),
        approved_by="fixture",
        approved_at="2026-11-17T00:00:00Z",
        supersedes=None,
    )
    entry = policies.PolicyRegistryEntry(policy=policy, state="enabled", authorization=auth)
    monkeypatch.setattr(policies, "POLICY_REGISTRY", (canonical(entry).decode(),))
    versions = original.versions.model_copy(update={"policy": policy.id, "policy_sha256": digest(policy)})
    decision = candidate(tmp_path / "synthetic", versions_override=versions)
    accepted = Acceptance.create(**{**decision.model_dump(exclude={"decision_id"}), "state": "accepted"})
    path = tmp_path / "accepted.json"
    save_acceptance(path, accepted)
    reader = AcceptanceReader(
        tmp_path / "synthetic", accepted.binding, versions, accepted.assertions, (cell,), "fixture-7"
    )
    produced = reader.decide(checks=decision.checks, origin="automatic", human=None, stage_ids=decision.stage_ids)
    assert produced.state == "accepted" and produced.origin == "automatic"
    produced_path = tmp_path / "produced.json"
    save_acceptance(produced_path, produced)
    assert reader.export(produced_path) == accepted.assertions
    assert reader.handoff(produced_path, note_id="note", card_key="card") is None
    assert reader.export(path) == accepted.assertions
    assert reader.preview(path)["origin"] == "automatic"
    wrong_revision = AcceptanceReader(reader.root, reader.binding, versions, reader.assertions, (cell,), "1")
    with pytest.raises(ValueError):
        wrong_revision.export(path)
    withdrawal = auth.model_copy(update={"action": "withdraw", "supersedes": digest(auth)})
    monkeypatch.setattr(policies, "AUTHORIZATION_HISTORY", (canonical(auth).decode(),))
    monkeypatch.setattr(
        policies,
        "POLICY_REGISTRY",
        (canonical(entry.model_copy(update={"state": "withdrawn", "authorization": withdrawal})).decode(),),
    )
    assert reader.preview(path)["reason"] == "policy_withdrawn"
    with pytest.raises(ValueError, match="policy_withdrawn"):
        reader.export(path)
    handoff = reader.handoff(path, note_id="installed-note", card_key="card-a")
    assert handoff and handoff["previous_decision_ids"] == [accepted.decision_id]


def test_changed_assertion_cannot_reuse_old_check(tmp_path: Path) -> None:
    decision = candidate(tmp_path, human=True)
    changed = decision.assertions[0].model_copy(update={"value": "different", "alternatives": ()})
    assertions = (changed,)
    human = Human(
        reviewer="teacher",
        reason="new content",
        reviewed_sha256=digest({"binding": decision.binding, "assertions": assertions}),
    )
    forged = Acceptance.create(
        **{**decision.model_dump(exclude={"decision_id"}), "assertions": assertions, "human": human}
    )
    assert not assess(
        forged, root=tmp_path, binding=decision.binding, versions=decision.versions, assertions=assertions, cells=CELLS
    ).eligible


def test_human_authority_is_independent_of_automatic_policy(tmp_path: Path) -> None:
    decision = candidate(tmp_path, human=True)
    unrelated_policy = decision.versions.model_copy(
        update={"policy": "withdrawn-automatic-policy", "policy_sha256": digest("withdrawal")}
    )
    assert assess(
        decision,
        root=tmp_path,
        binding=decision.binding,
        versions=unrelated_policy,
        assertions=decision.assertions,
        cells=CELLS,
    ).eligible


def test_record_requires_format_and_version(tmp_path: Path) -> None:
    decision = candidate(tmp_path)
    for missing in ("format", "version"):
        raw = decision.model_dump(mode="json")
        del raw[missing]
        path = tmp_path / f"missing-{missing}.json"
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            load_acceptance(path)


def test_stage_dependency_invalidation_preserves_unrelated_work(tmp_path: Path) -> None:
    from latinitas_cards.acceptance import load_stage, reusable_stage, save_stage_record
    from latinitas_cards.acceptance_records import canonical

    decision = candidate(tmp_path, human=True)
    original = load_stage(tmp_path, "check")
    assert original.output_sha256 is not None
    assert (
        reusable_stage(
            tmp_path,
            "check",
            binding=decision.binding,
            versions=decision.versions,
            input_sha256=decision.checks[0].input_sha256,
        )
        == original
    )
    envelope = check_envelope(
        decision.binding,
        decision.versions,
        "human",
        decision.assertions,
        prerequisite_output_sha256s=(original.output_sha256,),
    )
    child = save_stage(
        tmp_path,
        run_id="run",
        stage_id="child",
        stage="decide",
        binding=decision.binding,
        versions=decision.versions,
        envelope=envelope,
        prerequisite_ids=("check",),
    )
    assert child.reuse_key((original.output_sha256,)) != child.input_sha256
    # An unrelated success remains reusable while a failed/unknown branch cannot.
    for state in ("failed", "unknown", "deferred", "cancelled", "pending", "running"):
        failed = original.model_copy(
            update={
                "stage_id": state,
                "state": state,
                "output_sha256": None,
                "artifact": None,
                "reason": "not successful",
            }
        )
        save_stage_record(tmp_path, failed)
        record = Acceptance.create(**{**decision.model_dump(exclude={"decision_id"}), "stage_ids": (state,)})
        path = tmp_path / f"{state}.json"
        save_acceptance(path, record)
        reader = AcceptanceReader(tmp_path, decision.binding, decision.versions, decision.assertions, CELLS)
        assert reader.preview(path)["state"] == "withheld"
        with pytest.raises(ValueError, match="withheld"):
            reader.export(path)
        assert reader.handoff(path, note_id="installed", card_key="card") is not None
        with pytest.raises(ValueError):
            reusable_stage(
                tmp_path,
                state,
                binding=decision.binding,
                versions=decision.versions,
                input_sha256=original.input_sha256,
            )
    assert (
        reusable_stage(
            tmp_path, "check", binding=decision.binding, versions=decision.versions, input_sha256=original.input_sha256
        )
        == original
    )
    corrupted = original.model_copy(update={"output_sha256": digest("corrupt")})
    (tmp_path / "stages/check.json").write_bytes(canonical(corrupted))
    with pytest.raises(ValueError):
        reusable_stage(
            tmp_path, "child", binding=decision.binding, versions=decision.versions, input_sha256=child.input_sha256
        )


def test_policy_registry_cannot_be_replaced_by_requested_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from latinitas_cards import acceptance_policy as policies
    from latinitas_cards.acceptance_records import canonical

    decision = candidate(tmp_path)
    with pytest.raises(ValueError, match="unknown_policy"):
        policies.lookup_policy("self-approved", "1", digest("self-approved"))
    entry = policies.lookup_policy("v050-disabled", "1", digest(BOOTSTRAP))
    monkeypatch.setattr(policies, "POLICY_REGISTRY", (canonical(entry).decode(), canonical(entry).decode()))
    with pytest.raises(ValueError, match="conflicting"):
        policies.lookup_policy("v050-disabled", "1", decision.versions.policy_sha256)


def test_fresh_human_cannot_override_unsupported_material(tmp_path: Path) -> None:
    decision = candidate(tmp_path, human=True)
    unsupported = ScopeCell(claim_class="label", recipe="ppp_recognition", language="de", material="source-form/1")
    result = assess(
        decision,
        root=tmp_path,
        binding=decision.binding,
        versions=decision.versions,
        assertions=decision.assertions,
        cells=(unsupported,),
    )
    assert not result.eligible and result.reason == "unrepresentable_claim"


@pytest.mark.parametrize("source", ["reference", "", 7])
def test_reference_constraints_are_strict_at_public_boundary(tmp_path: Path, source: str | int) -> None:
    decision = candidate(tmp_path / "seed", human=True)
    root = tmp_path / "store"
    envelope = check_envelope(decision.binding, decision.versions, "human", decision.assertions)
    envelope["reference_constraints"] = [{"source": source, "reference": "r", "version": "1", "text": "t"}]
    save_stage(
        root,
        run_id="run",
        stage_id="check",
        stage="decide",
        binding=decision.binding,
        versions=decision.versions,
        envelope=envelope,
    )
    check = decision.checks[0].model_copy(update={"input_sha256": digest(envelope)})
    record = Acceptance.create(**{**decision.model_dump(exclude={"decision_id"}), "checks": (check,)})
    path = root / "accepted.json"
    save_acceptance(path, record)
    reader = AcceptanceReader(root, decision.binding, decision.versions, decision.assertions, CELLS)
    if source == "reference":
        assert reader.export(path) == decision.assertions
        assert reader.handoff(path, note_id="one", card_key="card") is None
    else:
        assert reader.preview(path)["reason"] == "stale_evidence"
        with pytest.raises(ValueError, match="withheld: stale_evidence"):
            reader.export(path)
        assert reader.handoff(path, note_id="one", card_key="card") is not None


@pytest.mark.parametrize("prefix", [b'"rule"', b"NaN", b"Infinity", b"-Infinity", b"1e999"])
def test_duplicate_check_artifact_keys_withhold_at_public_boundary(tmp_path: Path, prefix: bytes) -> None:
    import hashlib

    from latinitas_cards.acceptance import load_stage
    from latinitas_cards.acceptance_records import canonical

    decision = candidate(tmp_path, human=True)
    path = tmp_path / "accepted.json"
    save_acceptance(path, decision)
    reader = AcceptanceReader(tmp_path, decision.binding, decision.versions, decision.assertions, CELLS)
    assert reader.export(path) == decision.assertions
    stage = load_stage(tmp_path, "check")
    assert stage.artifact is not None
    artifact = tmp_path / stage.artifact
    original = artifact.read_bytes()
    malformed = b'{"method":' + prefix + b"," + original[1:]
    artifact.write_bytes(malformed)
    changed = stage.model_copy(update={"output_sha256": hashlib.sha256(malformed).hexdigest()})
    (tmp_path / "stages/check.json").write_bytes(canonical(changed))
    assert reader.preview(path)["state"] == "withheld"
    with pytest.raises(ValueError, match="withheld"):
        reader.export(path)
    assert reader.handoff(path, note_id="one", card_key="card") is not None
    artifact.write_bytes(original)
    (tmp_path / "stages/check.json").write_bytes(canonical(stage))
    assert reader.export(path) == decision.assertions
    assert reader.handoff(path, note_id="one", card_key="card") is None


@pytest.mark.parametrize(
    ("data", "expected"),
    [(b'{"ratio":0.125}', {"ratio": 0.125}), (b'[1,true,null,"NaN"]', [1, True, None, "NaN"]), (b"0.125", 0.125)],
)
def test_stage_artifact_preserves_generic_json(tmp_path: Path, data: bytes, expected: object) -> None:
    import hashlib

    from latinitas_cards.acceptance import load_stage, stage_artifact

    candidate(tmp_path)
    stage = load_stage(tmp_path, "check")
    assert stage.artifact is not None
    (tmp_path / stage.artifact).write_bytes(data)
    stage = stage.model_copy(update={"output_sha256": hashlib.sha256(data).hexdigest()})
    assert stage_artifact(tmp_path, stage) == expected


@pytest.mark.parametrize(
    "data",
    [b"NaN", b"Infinity", b"-Infinity", b"1e999", b'{"nested":[NaN]}', b'{"nested":{"key":1,"key":2}}'],
)
def test_stage_artifact_rejects_ambiguous_or_nonfinite_json(tmp_path: Path, data: bytes) -> None:
    import hashlib

    from latinitas_cards.acceptance import load_stage, stage_artifact

    candidate(tmp_path)
    stage = load_stage(tmp_path, "check")
    assert stage.artifact is not None
    (tmp_path / stage.artifact).write_bytes(data)
    stage = stage.model_copy(update={"output_sha256": hashlib.sha256(data).hexdigest()})
    with pytest.raises(ValueError):
        stage_artifact(tmp_path, stage)


@pytest.mark.parametrize(
    "field",
    [
        "source_sha256",
        "extraction_sha256",
        "mapping_sha256",
        "task_sha256",
        "language",
        "config_sha256",
        "producer",
        "analyzer",
        "checker",
        "comparison",
        "proposal_prompt",
        "counter_prompt",
        "proposal_model",
        "counter_model",
    ],
)
def test_public_reader_rejects_changed_current_context(tmp_path: Path, field: str) -> None:
    seed = candidate(tmp_path / "seed", human=True)
    versions = seed.versions.model_copy(
        update={
            "proposal_prompt": "p/1",
            "counter_prompt": "c/1",
            "proposal_model": "p-model/1",
            "counter_model": "c-model/1",
        }
    )
    decision = candidate(tmp_path / "store", human=True, versions_override=versions)
    path = tmp_path / "accepted.json"
    save_acceptance(path, decision)
    binding = decision.binding
    if field in type(binding).model_fields:
        binding = binding.model_copy(update={field: "en" if field == "language" else digest("new")})
    else:
        versions = versions.model_copy(update={field: "immutable-revision/2"})
    # Revalidate the changed context: a partial/invalid prompt tuple must not mask freshness bugs.
    versions = Versions.read_json(versions.model_dump_json())
    reader = AcceptanceReader(tmp_path / "store", binding, versions, decision.assertions, CELLS)
    assert reader.preview(path)["state"] == "withheld"
    with pytest.raises(ValueError, match="withheld"):
        reader.export(path)
    handoff = reader.handoff(path, note_id="installed", card_key="recognition")
    assert handoff and handoff["note_id"] == "installed" and handoff["previous_decision_ids"] == [decision.decision_id]
