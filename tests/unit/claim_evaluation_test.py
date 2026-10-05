import json
from collections import Counter
from pathlib import Path

from latinitas_cards.claim_review import Claim, Evidence, assess_claim, review_claim, summarize_evaluation
from latinitas_cards.profile import DeckProfile


def test_independently_authored_claim_sample_reports_manual_and_automatic_results() -> None:
    path = Path(__file__).parents[1] / "fixtures" / "claim-review" / "evaluation.jsonl"
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    judgments = {case["id"]: case["expected"] for case in cases}
    assert judgments["label-deponent"] == judgments["explanation-deponent"] == "rejected"
    profile = DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")
    automatic = []
    reviewed = []
    for case in cases:
        proposal = Claim(
            candidate_id=case["id"],
            candidate_text=case["form"],
            kind=case["kind"],
            value=case["proposal"],
            analyzer="synthetic-proposals/v1",
            evidence=(Evidence("synthetic evaluation", case["id"], "v1", case["reason"]),),
            profile=profile,
            profile_version="evaluation/v1",
            alternatives=tuple(case["alternatives"]),
            supported=case["expected"] != "unsupported",
        )
        automatic.append((assess_claim(proposal), case["expected"]))
        decision = review_claim(
            proposal,
            status="accepted" if case["expected"] == "correct" else "withheld",
            reviewer="independently reviewed synthetic expectations",
            reason=case["reason"],
        )
        reviewed.append((assess_claim(proposal, decision), case["expected"]))

    auto_report = summarize_evaluation(automatic)
    manual_report = summarize_evaluation(reviewed)
    assert len(auto_report) == len(manual_report) == 3
    for kind in ("label", "segmentation", "explanation"):
        auto = auto_report[("synthetic-proposals/v1", kind)]
        manual = manual_report[("synthetic-proposals/v1", kind)]
        assert auto.total == manual.total == 6
        assert auto.accepted_correct == auto.accepted_errors == 0
        assert auto.precision is None
        assert auto.coverage == 0
        rejected, unsupported = (1, 2) if kind == "segmentation" else (2, 1)
        assert auto.withheld == Counter(correct=2, rejected=rejected, ambiguous=1, unsupported=unsupported)
        assert manual.accepted_correct == 2
        assert manual.accepted_errors == 0
        assert manual.precision == 1
        assert manual.coverage == 2 / 6
        assert manual.withheld == Counter(rejected=rejected, ambiguous=1, unsupported=unsupported)


def test_evaluation_counts_accepted_errors_instead_of_hiding_them() -> None:
    profile = DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms")
    proposal = Claim(
        "bad", "amāvī", "label", "future", "bad-analyzer/v1", (Evidence("test", "bad", "v1", "amāvī"),), profile, "v1"
    )
    decision = review_claim(proposal, status="accepted", reviewer="mistaken reviewer", reason="Mistaken decision")
    report = summarize_evaluation([(assess_claim(proposal, decision), "rejected")])
    result = report[("bad-analyzer/v1", "label")]
    assert result.accepted_correct == 0
    assert result.accepted_errors == 1
    assert result.precision == 0
    assert result.coverage == 1
