"""Explicit legacy projections preserve authority, not merely reviewer strings."""

from dataclasses import replace
from pathlib import Path

import pytest

from latinitas_cards.acceptance import AcceptanceReader, assess, digest, load_acceptance, save_acceptance
from latinitas_cards.acceptance_adapters import adapt_corpus, adapt_deck, verify_projection
from latinitas_cards.acceptance_policy import BOOTSTRAP
from latinitas_cards.acceptance_records import Binding, Versions
from latinitas_cards.claim_review import Claim, Evidence, review_claim
from latinitas_cards.claim_sidecar import BoundClaim, Sidecar, StoredDecision
from latinitas_cards.profile import DeckProfile


def context() -> tuple[Binding, Versions]:
    return Binding(
        source_scope="deck",
        source_identity="one",
        source_sha256=digest("source"),
        extraction_sha256=digest("extract"),
        mapping_sha256=digest("mapping"),
        task_sha256=digest("task"),
        language="de",
        config_sha256=digest("config"),
    ), Versions(
        producer="1",
        analyzer="1",
        proposal_prompt=None,
        counter_prompt=None,
        checker="1",
        comparison="1",
        policy="v050-disabled",
        policy_sha256=digest(BOOTSTRAP),
        proposal_model=None,
        counter_model=None,
    )


def test_deck_authored_projection_roundtrip_and_added_clause(tmp_path: Path) -> None:
    binding, versions = context()
    claim = Claim(
        candidate_id="one",
        candidate_text="amo",
        kind="label",
        value="present",
        analyzer="1",
        evidence=(Evidence("grammar", "1", "1", "present"),),
        profile=DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms"),
        profile_version="1",
    )
    authored = replace(claim, value="present active", analyzer="reviewer-authored/v1")
    entry = StoredDecision(
        base_fingerprint=claim.fingerprint,
        decision=review_claim(authored, status="accepted", reviewer="Teacher", reason="Checked"),
        value=authored.value,
        evidence=authored.evidence,
    )
    sidecar = tmp_path / "old.json"
    sidecar.write_text(Sidecar(decisions=(entry,)).model_dump_json())
    original = sidecar.read_bytes()
    decision = adapt_deck(
        tmp_path / "store",
        sidecar=sidecar,
        claims=(BoundClaim("one", "label", claim),),
        binding=binding,
        versions=versions,
    )
    save_acceptance(tmp_path / "decision.json", decision)
    loaded = load_acceptance(tmp_path / "decision.json")
    assert loaded.human and loaded.human.reviewer == "Teacher"
    assert loaded.assertions[0].value == "present active"
    assert sidecar.read_bytes() == original
    assert assess(
        loaded, root=tmp_path / "store", binding=binding, versions=versions, assertions=loaded.assertions, cells=()
    ).eligible
    reader = AcceptanceReader(tmp_path / "store", binding, versions, loaded.assertions, ())
    assert reader.preview(tmp_path / "decision.json")["origin"] == "human"
    assert reader.export(tmp_path / "decision.json") == loaded.assertions
    assert reader.handoff(tmp_path / "decision.json", note_id="one", card_key="label") is None
    repeated = adapt_deck(
        tmp_path / "store",
        sidecar=sidecar,
        claims=(BoundClaim("one", "label", claim),),
        binding=binding,
        versions=versions,
    )
    assert repeated.stage_ids == decision.stage_ids
    for field in ("checker", "producer", "policy"):
        changed_versions = versions.model_copy(update={field: "changed/2"})
        fresh = adapt_deck(
            tmp_path / "store",
            sidecar=sidecar,
            claims=(BoundClaim("one", "label", claim),),
            binding=binding,
            versions=changed_versions,
        )
        assert fresh.stage_ids != decision.stage_ids
        assert assess(
            fresh,
            root=tmp_path / "store",
            binding=binding,
            versions=changed_versions,
            assertions=fresh.assertions,
            cells=(),
        ).eligible
    changed = loaded.assertions[0].model_copy(update={"value": "present active; unreviewed clause"})
    with pytest.raises(ValueError, match="unrepresentable_claim"):
        verify_projection(tmp_path / "store", loaded, (changed,))
    from latinitas_cards.acceptance import check_envelope, load_stage, save_stage
    from latinitas_cards.acceptance_records import Acceptance, Human

    root = tmp_path / "store"
    projection = load_stage(root, decision.stage_ids[0])
    assert projection.output_sha256 is not None
    envelope = check_envelope(
        binding, versions, "human", (changed,), prerequisite_output_sha256s=(projection.output_sha256,)
    )
    save_stage(
        root,
        run_id="legacy-adaptation",
        stage_id="changed",
        stage="decide",
        binding=binding,
        versions=versions,
        envelope=envelope,
        prerequisite_ids=(projection.stage_id,),
    )
    check = loaded.checks[0].model_copy(update={"id": "changed", "input_sha256": digest(envelope)})
    assert loaded.human is not None
    human = Human(
        reviewer=loaded.human.reviewer,
        reason=loaded.human.reason,
        reviewed_sha256=digest({"binding": binding, "assertions": (changed,)}),
    )
    forged = Acceptance.create(
        **{
            **loaded.model_dump(exclude={"decision_id"}),
            "assertions": (changed,),
            "checks": (check,),
            "human": human,
            "stage_ids": (projection.stage_id, "changed"),
        }
    )
    forged_path = tmp_path / "forged.json"
    save_acceptance(forged_path, forged)
    forged_reader = AcceptanceReader(root, binding, versions, (changed,), ())
    assert forged_reader.preview(forged_path)["reason"] == "stale_evidence"
    with pytest.raises(ValueError, match="withheld"):
        forged_reader.export(forged_path)
    # Both the legacy proof and human-check artifacts are required on every operation.
    artifacts = [next((root / "artifacts").glob("legacy-*.json"))]
    for identity in loaded.stage_ids:
        stage = load_stage(root, identity)
        assert stage.artifact is not None
        artifacts.append(root / stage.artifact)
    for artifact in artifacts:
        data = artifact.read_bytes()
        for replacement in (None, b"{}"):
            if replacement is None:
                artifact.unlink()
            else:
                artifact.write_bytes(replacement)
            assert reader.preview(tmp_path / "decision.json")["state"] == "withheld"
            with pytest.raises(ValueError, match="withheld"):
                reader.export(tmp_path / "decision.json")
            assert reader.handoff(tmp_path / "decision.json", note_id="one", card_key="label") is not None
            artifact.write_bytes(data)


@pytest.mark.parametrize("authored", [False, True])
def test_phrase_projection_requires_both_reviews(tmp_path: Path, authored: bool) -> None:
    binding, versions = context()
    claim = Claim(
        candidate_id="phrase",
        candidate_text="context",
        kind="explanation",
        value='{"answer":"amo"}',
        analyzer="phrase-reference/v1",
        evidence=(Evidence("source", "1", "1", "amo"),),
        profile=DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms"),
        profile_version="phrase-v1",
    )
    combination = replace(claim, analyzer="phrase-combination/v1", value="explicit combination approval")
    reviewed = replace(claim, analyzer="reviewer-authored/v1", value='{"answer":"amo, ego amo"}') if authored else claim
    answer_entry = StoredDecision(
        base_fingerprint=claim.fingerprint,
        decision=review_claim(reviewed, status="accepted", reviewer="T", reason="Answer checked"),
        value=reviewed.value if authored else None,
        evidence=reviewed.evidence if authored else None,
    )
    sidecar = tmp_path / "old.json"
    sidecar.write_text(Sidecar(decisions=(answer_entry,)).model_dump_json())
    claims = (BoundClaim("phrase", "combination", combination), BoundClaim("answer", "answer", claim))
    with pytest.raises(ValueError):
        adapt_deck(tmp_path / "store", sidecar=sidecar, claims=claims, binding=binding, versions=versions)
    with pytest.raises(ValueError, match="combination"):
        adapt_deck(tmp_path / "alone", sidecar=sidecar, claims=claims[1:], binding=binding, versions=versions)
    mismatched = replace(combination, candidate_text="other context")
    sidecar.write_text(
        Sidecar(
            decisions=(
                answer_entry,
                StoredDecision(
                    base_fingerprint=mismatched.fingerprint,
                    decision=review_claim(mismatched, status="accepted", reviewer="T", reason="Other checked"),
                ),
            )
        ).model_dump_json()
    )
    with pytest.raises(ValueError, match="combination"):
        adapt_deck(
            tmp_path / "mismatched",
            sidecar=sidecar,
            claims=(BoundClaim("other", "combination", mismatched), claims[1]),
            binding=binding,
            versions=versions,
        )
    sidecar.write_text(
        Sidecar(
            decisions=(
                answer_entry,
                StoredDecision(
                    base_fingerprint=combination.fingerprint,
                    decision=review_claim(combination, status="accepted", reviewer="T", reason="Checked"),
                ),
            )
        ).model_dump_json()
    )
    decision = adapt_deck(tmp_path / "store", sidecar=sidecar, claims=claims, binding=binding, versions=versions)
    assert len(decision.assertions) == 2
    save_acceptance(tmp_path / "phrase.json", decision)
    reader = AcceptanceReader(tmp_path / "store", binding, versions, decision.assertions, ())
    assert reader.export(tmp_path / "phrase.json") == decision.assertions
    assert assess(
        decision, root=tmp_path / "store", binding=binding, versions=versions, assertions=decision.assertions, cells=()
    ).eligible


def test_corpus_adapter_uses_current_legacy_bindings(tmp_path: Path) -> None:
    from latinitas_cards.claim_sidecar import sidecar_path
    from latinitas_cards.corpus_claims import corpus_claim_fingerprint, current_annotations, record_corpus_batch

    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    corpus = tmp_path / "corpus.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    corpus.write_bytes((fixtures / "corpus/wsl-three.corpus.jsonl").read_bytes())
    annotations.write_bytes((fixtures / "annotations/wsl.annotations.jsonl").read_bytes())
    row = next(r for r in current_annotations(corpus, annotations) if r.layer == "segmentation")
    assert not record_corpus_batch(
        corpus,
        annotations,
        [
            {
                "fingerprint": corpus_claim_fingerprint(row),
                "status": "accepted",
                "reviewer": "Corpus teacher",
                "reason": "Bound segmentation checked",
            }
        ],
    )
    binding, versions = context()
    sidecar = sidecar_path(corpus)
    decision = adapt_corpus(
        tmp_path / "store",
        corpus=corpus,
        annotations=annotations,
        sidecar=sidecar,
        annotation_ids=(row.annotation_id,),
        binding=binding,
        versions=versions,
    )
    save_acceptance(tmp_path / "decision.json", decision)
    loaded = load_acceptance(tmp_path / "decision.json")
    assert loaded.origin == "human"
    assert loaded.human and loaded.human.reviewer == "Corpus teacher"
    reader = AcceptanceReader(tmp_path / "store", binding, versions, loaded.assertions, ())
    assert reader.preview(tmp_path / "decision.json")["state"] == "accepted"
    assert reader.export(tmp_path / "decision.json") == loaded.assertions
    assert assess(
        loaded, root=tmp_path / "store", binding=binding, versions=versions, assertions=decision.assertions, cells=()
    ).eligible
    # The original reader rejects a copied binding even with the old fingerprint.
    import json

    raw = json.loads(sidecar.read_text())
    raw["decisions"][0]["binding"]["passage_hash"] = "sha256:" + "a" * 64
    sidecar.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        adapt_corpus(
            tmp_path / "stale",
            corpus=corpus,
            annotations=annotations,
            sidecar=sidecar,
            annotation_ids=(row.annotation_id,),
            binding=binding,
            versions=versions,
        )


def test_legacy_artifact_paths_stay_in_store(tmp_path: Path) -> None:
    binding, versions = context()
    claim = Claim(
        candidate_id="one",
        candidate_text="amo",
        kind="label",
        value="present",
        analyzer="1",
        evidence=(Evidence("grammar", "1", "1", "present"),),
        profile_version="1",
        profile=DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms"),
    )
    sidecar = tmp_path / "old.json"
    sidecar.write_text(
        Sidecar(
            decisions=(
                StoredDecision(
                    base_fingerprint=claim.fingerprint,
                    decision=review_claim(claim, status="accepted", reviewer="T", reason="Checked"),
                ),
            )
        ).model_dump_json()
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (escaped / "artifacts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes store"):
        adapt_deck(
            escaped, sidecar=sidecar, claims=(BoundClaim("one", "label", claim),), binding=binding, versions=versions
        )
    assert list(outside.iterdir()) == []
    root = tmp_path / "valid"
    decision = adapt_deck(
        root, sidecar=sidecar, claims=(BoundClaim("one", "label", claim),), binding=binding, versions=versions
    )
    path = root / "decision.json"
    save_acceptance(path, decision)
    legacy = next((root / "artifacts").glob("legacy-*.json"))
    copy = outside / "copy.json"
    copy.write_bytes(legacy.read_bytes())
    legacy.unlink()
    legacy.symlink_to(copy)
    reader = AcceptanceReader(root, binding, versions, decision.assertions, ())
    assert reader.preview(path)["state"] == "withheld"
    with pytest.raises(ValueError, match="withheld"):
        reader.export(path)
