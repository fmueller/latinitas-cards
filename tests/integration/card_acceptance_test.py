"""Synthetic application checks prove engineering boundaries, not Latin correctness."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from latinitas_cards import acceptance_policy as policies
from latinitas_cards.acceptance import AcceptanceReader, check_envelope, save_acceptance, save_stage
from latinitas_cards.acceptance_records import Acceptance, Assertion, Binding, Check, Human, Versions, canonical, digest
from latinitas_cards.card_acceptance import CardClaims, DecisionSource, export_cards, preview_cards
from latinitas_cards.claim_review import Evidence


@pytest.fixture
def context() -> tuple[Binding, Versions]:
    return (
        Binding(
            source_scope="synthetic",
            source_identity="17",
            source_sha256=digest("source"),
            extraction_sha256=digest("extraction"),
            mapping_sha256=digest("mapping"),
            task_sha256=digest("task"),
            language="de",
            config_sha256=digest("config"),
        ),
        Versions(
            producer="synthetic/1",
            analyzer="synthetic/1",
            proposal_prompt=None,
            counter_prompt=None,
            checker="synthetic-checker/1",
            comparison="synthetic-comparison/1",
            policy=policies.BOOTSTRAP.id,
            policy_sha256=digest(policies.BOOTSTRAP),
            proposal_model=None,
            counter_model=None,
        ),
    )


def install_policy(monkeypatch: pytest.MonkeyPatch, versions: Versions) -> Versions:
    cells = tuple(
        policies.ScopeCell(claim_class=kind, recipe="form_recognition", language="de", material="source-form/1")
        for kind in ("explanation", "label", "segmentation", "translation")
    )
    policy = policies.AutomaticPolicy(
        id="synthetic-test-only",
        revision="1",
        checker_sha256=policies.implementation_digest(),
        comparison_sha256=policies.implementation_digest(),
        support_registry_sha256=digest(policies.SUPPORT_REGISTRY),
        versions_sha256=digest(versions.model_dump(exclude={"policy", "policy_sha256"})),
        criteria_sha256=digest("synthetic criteria, not approval"),
        scope=cells,
    )
    auth = policies.PolicyAuthorization(
        action="enable",
        policy_sha256=digest(policy),
        evaluation_report_sha256=digest("synthetic report, not linguistic evidence"),
        approved_by="fixture",
        approved_at="2026-11-18T00:00:00Z",
        supersedes=None,
    )
    entry = policies.PolicyRegistryEntry(policy=policy, state="enabled", authorization=auth)
    monkeypatch.setattr(policies, "POLICY_REGISTRY", (canonical(entry).decode(),))
    return versions.model_copy(update={"policy": policy.id, "policy_sha256": digest(policy)})


def decision_source(
    root: Path,
    context: tuple[Binding, Versions],
    assertion: Assertion,
    *,
    result: str = "pass",
    human: bool = False,
) -> DecisionSource:
    binding, versions = context
    cell = policies.ScopeCell(
        claim_class=assertion.kind, recipe="form_recognition", language="de", material="source-form/1"
    )
    reader = AcceptanceReader(root, binding, versions, (assertion,), (cell,))
    method: Literal["library", "human"] = "human" if human else "library"
    envelope = check_envelope(binding, versions, method, (assertion,))
    stage = save_stage(
        root,
        run_id="synthetic",
        stage_id=assertion.id,
        stage="analyze",
        binding=binding,
        versions=versions,
        envelope=envelope,
    )
    check = Check.model_validate(
        {
            "id": assertion.id,
            "assertion_ids": (assertion.id,),
            "method": method,
            "version": versions.checker,
            "result": result,
            "evidence": assertion.evidence,
            "alternatives": assertion.alternatives,
            "objections": () if result == "pass" else ("unsupported factual clause",),
            "input_sha256": digest(envelope),
        }
    )
    if human:
        attestation = Human(
            reviewer="teacher",
            reason="reviewed exact synthetic proposition",
            reviewed_sha256=digest({"binding": binding, "assertions": (assertion,)}),
        )
        decision = reader.decide(checks=(check,), origin="human", human=attestation, stage_ids=(stage.stage_id,))
    else:
        decision = reader.decide_automatic(checks=(check,), stage_ids=(stage.stage_id,))
    path = root / "decision.json"
    save_acceptance(path, decision)
    return DecisionSource(reader, path)


def assertions() -> tuple[Assertion, ...]:
    values: tuple[tuple[Literal["label", "translation", "segmentation", "explanation"], str], ...] = (
        ("label", "ablative singular"),
        ("translation", "mit dem Freund"),
        ("segmentation", "amic- + -ō"),
        ("explanation", "The governed form is ablative."),
    )
    return tuple(
        Assertion(
            id=kind,
            kind=kind,
            value=value,
            selected_sense="friend" if kind == "translation" else None,
            alternatives=("separately supported alternative",) if kind == "segmentation" else (),
            evidence=(Evidence("synthetic-library", kind, "1", "isolated checked proposition"),),
        )
        for kind, value in values
    )


def test_full_card_gate_keeps_supported_siblings_and_optional_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: tuple[Binding, Versions]
) -> None:
    binding, versions = context
    versions = install_policy(monkeypatch, versions)
    claims = assertions()
    sources = tuple(decision_source(tmp_path / a.id, (binding, versions), a) for a in claims)
    unsupported = decision_source(tmp_path / "unsupported", (binding, versions), claims[-1], result="unresolved")
    accepted = CardClaims("note", "recognition", claims, sources)
    withheld = CardClaims("note", "explanation", claims, (*sources[:-1], unsupported))
    # Same slot on a different note is a different card, not a deduplication hit.
    sibling = CardClaims("other-note", "recognition", claims[:1], sources[:1])
    cards = (accepted, withheld, sibling)
    previews = preview_cards(cards)
    assert [p["state"] for p in previews] == ["accepted", "withheld", "accepted"]
    assert previews[0]["acceptance"] == "automatic"
    assert previews[0]["assertions"] == [a.model_dump(mode="json") for a in claims]
    assert export_cards(cards) == (accepted, sibling)
    review = preview_cards(cards, unresolved_only=True)
    assert len(review) == 1 and review[0]["card_key"] == "explanation"
    assert export_cards(cards) == (accepted, sibling)  # Inspection grants no authority.


def test_all_checks_pass_under_production_policy_stay_out_of_export(
    tmp_path: Path, context: tuple[Binding, Versions]
) -> None:
    claims = assertions()
    sources = tuple(decision_source(tmp_path / a.id, context, a) for a in claims)
    card = CardClaims("note", "recognition", claims, sources)
    assert preview_cards((card,))[0]["reason"] == "policy_not_enabled"
    assert export_cards((card,)) == ()


def test_card_requires_every_current_assertion_not_just_matching_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: tuple[Binding, Versions]
) -> None:
    binding, versions = context
    versions = install_policy(monkeypatch, versions)
    claims = assertions()
    sources = tuple(decision_source(tmp_path / a.id, (binding, versions), a) for a in claims)
    added = claims[-1].model_copy(update={"value": "A matching label proves every causal explanation."})
    changed = CardClaims("note", "changed", (*claims[:-1], added), sources)
    missing = CardClaims("note", "missing", claims, sources[:-1])
    assert export_cards((changed, missing)) == ()
    assert [p["reason"] for p in preview_cards((changed, missing))] == [
        "assertion_mismatch",
        "incomplete_coverage",
    ]


def test_generic_human_claim_cannot_approve_curated_rule_or_mnemonic(
    tmp_path: Path, context: tuple[Binding, Versions]
) -> None:
    source = decision_source(tmp_path, context, assertions()[-1], human=True)
    for recipe in ("rule_application", "mnemonic_recall"):
        cell = policies.ScopeCell(claim_class="explanation", recipe=recipe, language="de", material="curated/1")
        reader = replace(source.reader, cells=(cell,))
        assert reader.preview(source.path)["reason"] == "human_approval_required"
        with pytest.raises(ValueError, match="human_approval_required"):
            reader.export(source.path)


@pytest.mark.parametrize("damage", ["fail", "unresolved", "not_applicable", "malformed", "stale"])
def test_bad_assertion_withholds_only_affected_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: tuple[Binding, Versions], damage: str
) -> None:
    binding, versions = context
    versions = install_policy(monkeypatch, versions)
    claims = assertions()
    label = decision_source(tmp_path / "label", (binding, versions), claims[0])
    explanation = decision_source(
        tmp_path / "explanation",
        (binding, versions),
        claims[-1],
        result=damage if damage in ("fail", "unresolved", "not_applicable") else "pass",
    )
    if damage == "malformed":
        explanation.path.write_text('{"version":true}')
    elif damage == "stale":
        explanation = replace(
            explanation,
            reader=replace(explanation.reader, binding=binding.model_copy(update={"task_sha256": digest("new")})),
        )
    good = CardClaims("note", "label", claims[:1], (label,))
    bad = CardClaims("note", "explanation", (claims[-1],), (explanation,))
    assert export_cards((bad, good)) == (good,)
    assert preview_cards((bad,))[0]["state"] == "withheld"


def test_old_batch_withdrawal_rechecks_authority_without_invalidating_human_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: tuple[Binding, Versions]
) -> None:
    binding, versions = context
    versions = install_policy(monkeypatch, versions)
    label, translation = assertions()[:2]
    automatic = decision_source(tmp_path / "automatic", (binding, versions), label)
    human = decision_source(tmp_path / "human", context, translation, human=True)
    mixed = CardClaims("note", "mixed", (label, translation), (automatic, human))
    human_card = CardClaims("note", "human", (translation,), (human,))
    old_batch = export_cards((mixed, human_card))
    assert old_batch == (mixed, human_card)
    assert preview_cards((mixed,))[0]["acceptance"] == "mixed"
    entry = policies.lookup_policy(versions.policy, "1", versions.policy_sha256)
    assert entry.authorization is not None
    auth = entry.authorization
    withdrawal = auth.model_copy(update={"action": "withdraw", "supersedes": digest(auth)})
    monkeypatch.setattr(policies, "AUTHORIZATION_HISTORY", (canonical(auth).decode(),))
    monkeypatch.setattr(
        policies,
        "POLICY_REGISTRY",
        (canonical(entry.model_copy(update={"state": "withdrawn", "authorization": withdrawal})).decode(),),
    )
    assert export_cards(old_batch) == (human_card,)
    assert preview_cards(old_batch)[0]["reason"] == "policy_withdrawn"
    handoff = automatic.reader.handoff(automatic.path, note_id=mixed.note_id, card_key=mixed.card_key)
    assert handoff is not None and handoff["reason"] == "policy_withdrawn"


def test_matching_provider_answers_do_not_approve_unsupported_explanation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: tuple[Binding, Versions]
) -> None:
    from latinitas_cards.providers import ModelAnswer, compare_answers

    binding, versions = context
    versions = install_policy(monkeypatch, versions)
    claims = assertions()
    answer = ModelAnswer(
        format="latinitas-model-answer",
        version=1,
        task_sha256=binding.task_sha256,
        assertions=(claims[-1],),
        alternatives=(),
        objections=(),
    )
    assert compare_answers(answer, answer).relation == "equivalent"
    unsupported = decision_source(tmp_path, (binding, versions), claims[-1], result="not_applicable")
    card = CardClaims("note", "matching-but-unsupported", (claims[-1],), (unsupported,))
    assert export_cards((card,)) == ()
    # A serialized accepted flag cannot turn that non-passing check into authority.
    data = unsupported.path.read_text()
    with pytest.raises(ValueError):
        Acceptance.read_json(data.replace('"state":"withheld"', '"state":"accepted"'))


@pytest.mark.parametrize("objection", ["fail", "unresolved", "not_applicable", "cross_assertion", "human"])
def test_automatic_decision_retains_every_relevant_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: tuple[Binding, Versions], objection: str
) -> None:
    binding, versions = context
    versions = install_policy(monkeypatch, versions)
    source = decision_source(tmp_path, (binding, versions), assertions()[0])
    decision, _ = source.reader.read(source.path)
    check = decision.checks[0]
    changes: dict[str, object] = {"id": "additional-check"}
    if objection == "cross_assertion":
        changes["assertion_ids"] = (*check.assertion_ids, "omitted-assertion")
    elif objection == "human":
        changes["method"] = "human"
    else:
        changes.update(result=objection, objections=("open material objection",))
    extra = Check.model_validate_json(check.model_copy(update=changes).model_dump_json())
    produced = source.reader.decide_automatic(checks=(check, extra), stage_ids=decision.stage_ids)
    assert produced.state == "withheld" and produced.origin == "automatic" and produced.human is None


@pytest.mark.parametrize("artifact", ["missing", "ambiguous"])
def test_automatic_decision_requires_unique_bound_check_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context: tuple[Binding, Versions], artifact: str
) -> None:
    binding, versions = context
    versions = install_policy(monkeypatch, versions)
    source = decision_source(tmp_path, (binding, versions), assertions()[0])
    decision, _ = source.reader.read(source.path)
    stages: tuple[str, ...] = ()
    if artifact == "ambiguous":
        duplicate = save_stage(
            tmp_path,
            run_id="synthetic",
            stage_id="duplicate",
            stage="analyze",
            binding=binding,
            versions=versions,
            envelope=check_envelope(binding, versions, "library", decision.assertions),
        )
        stages = (*decision.stage_ids, duplicate.stage_id)
    assert source.reader.decide_automatic(checks=decision.checks, stage_ids=stages).state == "withheld"
