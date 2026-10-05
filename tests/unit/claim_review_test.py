from dataclasses import replace
from typing import Any

import pytest

from latinitas_cards.claim_review import Claim, Evidence, asserted_claims, assess_claim, review_claim
from latinitas_cards.profile import DeckProfile
from latinitas_cards.source_extraction import SourceExtraction


def claim() -> Claim:
    return Claim(
        candidate_id="source-1:slot-3:amāvī",
        candidate_text="amāvī",
        kind="segmentation",
        value="amā- | -v- | -ī",
        analyzer="manual-proposal/v1",
        evidence=(Evidence("grammar", "perfect formation", "v1", "first-conjugation v-perfect"),),
        profile=DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms"),
        profile_version="confirmed/v1",
        alternatives=("amāv- | -ī",),
    )


def test_automatic_acceptance_is_disabled_even_for_supported_claims() -> None:
    result = assess_claim(claim())
    assert result.status == "withheld"
    assert result.reason == "No applicable calibration; explicit claim review required"
    assert asserted_claims((result,)) == ()


def test_review_is_individual_and_never_grants_destination_approval() -> None:
    segmentation = claim()
    label = replace(segmentation, kind="label", value="perfect active indicative, first singular")
    decision = review_claim(label, status="accepted", reviewer="Latin reviewer", reason="Checked named grammar")
    accepted = assess_claim(label, decision)
    withheld = assess_claim(segmentation, decision)
    assert accepted.status == "accepted"
    assert withheld.status == "withheld"
    assert asserted_claims((accepted, withheld)) == (label,)
    assert not hasattr(decision, "destination_approved")


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_id": "source-2:slot-3:amāvī"},
        {"value": "amāv- | -ī"},
        {"alternatives": ()},
        {"analyzer": "manual-proposal/v2"},
        {"profile_version": "confirmed/v2"},
        {
            "profile": DeckProfile.default(
                note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms", separators=(";",)
            )
        },
        {"evidence": (Evidence("grammar", "perfect formation", "v2", "changed text"),)},
        {"extraction": SourceExtraction("amāvī|amavi", ("perfect_1s",), "supported", (("amāvī", "amavi"),))},
    ],
)
def test_changed_claim_or_evidence_invalidates_review(changes: dict[str, Any]) -> None:
    original = claim()
    decision = review_claim(original, status="accepted", reviewer="reviewer", reason="Verified")
    changed = replace(original, **changes)
    result = assess_claim(changed, decision)
    assert result.status == "withheld"
    assert result.reason == "Stale or different claim review; renewed review required"


def test_changed_candidate_text_with_same_identity_invalidates_review() -> None:
    original = claim()
    decision = review_claim(original, status="accepted", reviewer="reviewer", reason="Verified")
    changed = replace(original, candidate_text="amāveram")
    assert assess_claim(changed, decision).status == "withheld"


def test_rejected_ambiguous_and_unsupported_claims_do_not_leak() -> None:
    rejected = review_claim(claim(), status="withheld", reviewer="reviewer", reason="Incorrect detailed split")
    assert assess_claim(claim(), rejected).reason == "Incorrect detailed split"
    ambiguous = replace(claim(), value="PPP", alternatives=("supine",))
    unsupported = replace(
        claim(), kind="explanation", value="tulī derives character-by-character from ferre", supported=False
    )
    with pytest.raises(ValueError, match="unsupported"):
        review_claim(unsupported, status="accepted", reviewer="reviewer", reason="Trust analyzer")
    assert asserted_claims((assess_claim(ambiguous), assess_claim(unsupported), assess_claim(claim(), rejected))) == ()


def test_setup_role_and_extraction_are_not_linguistic_review() -> None:
    extraction = SourceExtraction(
        "amātum|amatum", ("perfect_passive_participle",), "supported", (("amātum", "amatum"),)
    )
    proposed = replace(claim(), kind="label", value="PPP", extraction=extraction)
    result = assess_claim(proposed)
    assert result.status == "withheld"
    assert result.claim.extraction == extraction
    assert result.claim.extraction.candidates == (("amātum", "amatum"),)
    assert result.claim.extraction.raw == "amātum|amatum"


def test_review_requires_evidence_and_accountable_reason() -> None:
    for changes in ({"evidence": ()}, {"profile_version": ""}, {"candidate_id": ""}):
        with pytest.raises(ValueError):
            replace(claim(), **changes)
    with pytest.raises(ValueError):
        review_claim(claim(), status="accepted", reviewer="", reason="")
