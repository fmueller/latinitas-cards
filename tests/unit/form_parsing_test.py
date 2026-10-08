import json
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from latinitas_cards.claim_review import Evidence, review_claim
from latinitas_cards.cli import app
from latinitas_cards.destination_state import BoundDestination, ReconciliationRequired, schema_contract
from latinitas_cards.form_parsing import (
    PARSING_FIELDS,
    ParsingCase,
    ParsingFeature,
    ParsingInput,
    ParsingProposal,
    generate_parsing,
    parsing_csv,
)
from latinitas_cards.profile import DeckProfile


def test_documented_reviewed_example_cli_contract(tmp_path: Path) -> None:
    runner = CliRunner()
    source = tmp_path / "cases.json"
    sample = Path(__file__).parents[2] / "docs/examples/form-parsing-cases.json"
    source.write_bytes(sample.read_bytes())
    preview = runner.invoke(app, ["form-parsing", "preview", str(source)])
    assert preview.exit_code == 0, preview.output
    (result,) = json.loads(preview.stdout)
    assert result["eligible"] is True and result["skips"] == []
    assert result["semantic_key"] == "form_parsing:contextual_analysis"
    assert result["latinitas_id"] == "latinitas-v2-ea8b6ea917204b230894b6b0f5ca2edc91201a958b1734092aea51b1d796e9ef"
    assert result["form"] == "puellae" and result["context"] == "Puellae rosas portant."
    assert [(c["feature"], c["value"], c["status"]) for c in result["claims"]] == [
        ("lemma", "puella", "accepted"),
        ("case", "nominativus", "accepted"),
        ("number", "pluralis", "accepted"),
        ("gender", "femininum", "withheld"),
    ]
    assert result["answer"] == (
        '<section class="morphology-v1 morphology-muted morphology-light">'
        "Lemma: puella<br>Casus: nominativus<br>Numerus: pluralis</section>"
    )
    output = tmp_path / "parsing.csv"
    args = ["form-parsing", "export", str(source), str(output)]
    refused = runner.invoke(app, args)
    assert refused.exit_code == 2 and not output.exists()
    assert "scheduling migration/apply" in refused.output
    exported = runner.invoke(app, [*args, "--approve-fresh-import"])
    assert exported.exit_code == 0, exported.output
    payload = output.read_bytes()
    assert b"Personal Notes" not in payload and b"femininum" not in payload
    assert b"Latinitas Contextual Form Parsing v1" in payload
    overwritten = runner.invoke(app, [*args, "--approve-fresh-import"])
    assert overwritten.exit_code == 2 and output.read_bytes() == payload
    data = json.loads(source.read_text())
    data["cases"][0]["context"] = "Rosam puellae dat."
    source.write_text(json.dumps(data), encoding="utf-8")
    stale = runner.invoke(app, ["form-parsing", "preview", str(source)])
    (changed,) = json.loads(stale.stdout)
    assert changed["latinitas_id"] == result["latinitas_id"]
    assert changed["eligible"] is False and changed["answer"] == ""
    assert all(c["status"] == "withheld" for c in changed["claims"])
    empty = tmp_path / "stale.csv"
    assert (
        runner.invoke(app, ["form-parsing", "export", str(source), str(empty), "--approve-fresh-import"]).exit_code == 0
    )
    assert b"puellae" not in empty.read_bytes()


def example() -> ParsingInput:
    return ParsingInput(
        profile=DeckProfile.default(note_type="Latin", lexical_entry_field="Lemma", principal_parts_field="Forms"),
        cases=(
            ParsingCase(
                source_identity="reviewed-sentence-17",
                source_scope="sanitized-latin/v1",
                object_key="token-2",
                form="puellae",
                context="Puellae rosas portant.",
                applicable_features=("lemma", "case", "number", "gender"),
                required_features=("lemma", "case", "number"),
                proposals=tuple(
                    ParsingProposal(
                        feature=cast(ParsingFeature, feature),
                        value=value,
                        alternatives=alternatives,
                        analyzer="individual-review/v1",
                        evidence=(Evidence("reviewed Latin", "sentence 17", "v1", "Subject of plural portant"),),
                    )
                    for feature, value, alternatives in (
                        ("lemma", "puella", ()),
                        ("case", "nominativus", ("genetivus singularis", "dativus singularis")),
                        ("number", "pluralis", ("singularis",)),
                        ("gender", "femininum", ()),
                    )
                ),
            ),
        ),
    )


def approve(data: ParsingInput, features: tuple[str, ...]) -> ParsingInput:
    case = data.cases[0]
    proposals = tuple(
        proposal.model_copy(
            update={
                "decision": review_claim(
                    proposal.to_claim(case, data.profile),
                    status="accepted",
                    reviewer="fixture reviewer",
                    reason="Reviewed this token in its plural subject context",
                )
            }
        )
        if proposal.feature in features
        else proposal
        for proposal in case.proposals
    )
    return data.model_copy(update={"cases": (case.model_copy(update={"proposals": proposals}),)})


def test_precision_first_withheld_preview_exposes_alternatives_and_no_card() -> None:
    result = generate_parsing(example())
    assert result[0]["eligible"] is False
    assert result[0]["answer"] == ""
    assert result[0]["claims"][1]["alternatives"] == ["genetivus singularis", "dativus singularis"]
    assert "explicit claim review required" in result[0]["claims"][1]["reason"]


def test_only_individually_accepted_applicable_features_are_asserted() -> None:
    data = approve(example(), ("lemma", "case", "number"))
    result = generate_parsing(data)[0]
    assert result["eligible"] is True
    assert "puellae" in result["prompt"] and "Puellae rosas portant." in result["prompt"]
    assert "nominativus" in result["answer"] and "pluralis" in result["answer"]
    assert "femininum" not in result["answer"] and "genetivus" not in result["answer"]
    assert "tense" not in result["answer"]
    assert result["semantic_key"] == "form_parsing:contextual_analysis"


def test_context_changes_invalidate_review_without_rekeying_note() -> None:
    data = approve(example(), ("lemma", "case", "number"))
    before = generate_parsing(data)[0]
    changed = data.model_copy(update={"cases": (data.cases[0].model_copy(update={"context": "Rosam puellae dat."}),)})
    after = generate_parsing(changed)[0]
    assert after["latinitas_id"] == before["latinitas_id"]
    assert after["eligible"] is False
    other = data.model_copy(update={"cases": (data.cases[0].model_copy(update={"object_key": "token-5"}),)})
    assert generate_parsing(other)[0]["latinitas_id"] != before["latinitas_id"]


def test_missing_required_rejected_and_unsupported_features_block_cards() -> None:
    data = approve(example(), ("lemma", "case", "number"))
    case = data.cases[0]
    for proposals in (
        case.proposals[:2],
        tuple(p.model_copy(update={"supported": False}) if p.feature == "case" else p for p in case.proposals),
        tuple(
            p.model_copy(
                update={
                    "decision": review_claim(
                        p.to_claim(case, data.profile),
                        status="withheld",
                        reviewer="reviewer",
                        reason="Ambiguous in this context",
                    )
                }
            )
            if p.feature == "number"
            else p
            for p in case.proposals
        ),
    ):
        changed = data.model_copy(update={"cases": (case.model_copy(update={"proposals": proposals}),)})
        assert generate_parsing(changed)[0]["answer"] == ""


def test_duplicate_contextual_bindings_and_inapplicable_features_are_rejected() -> None:
    data = example()
    with pytest.raises(ValueError, match="duplicate"):
        ParsingInput.model_validate({**data.model_dump(), "cases": [data.cases[0], data.cases[0]]})
    with pytest.raises(ValueError, match="applicable"):
        ParsingCase.model_validate({**data.cases[0].model_dump(), "applicable_features": ["lemma", "gender"]})


def test_cli_preview_export_requires_fresh_setup_and_never_exports_withheld(tmp_path: Path) -> None:
    runner = CliRunner()
    source = tmp_path / "cases.json"
    source.write_text(example().model_dump_json(), encoding="utf-8")
    preview = runner.invoke(app, ["form-parsing", "preview", str(source)])
    assert preview.exit_code == 0 and '"eligible": false' in preview.stdout
    output = tmp_path / "parsing.csv"
    refused = runner.invoke(app, ["form-parsing", "export", str(source), str(output)])
    assert refused.exit_code == 2 and not output.exists()
    data = approve(example(), ("lemma", "case", "number"))
    # Serializable review decisions are separately authored, not an export-time approval switch.
    decision = data.cases[0].proposals[0].decision
    assert decision is not None and decision.reviewer == "fixture reviewer"
    source.write_text(data.model_dump_json(), encoding="utf-8")
    exported = runner.invoke(app, ["form-parsing", "export", str(source), str(output), "--approve-fresh-import"])
    assert exported.exit_code == 0, exported.output
    assert "Latinitas Contextual Form Parsing v1" in output.read_text()
    assert "nominativus" in output.read_text() and "femininum" not in output.read_text()


def test_export_omits_personal_notes_and_managed_binding_refuses_contextual_schema() -> None:
    data = approve(example(), ("lemma", "case", "number"))
    assert b"Personal Notes" not in parsing_csv(data)
    schema = schema_contract("17")
    schema["fields"] = list(PARSING_FIELDS)
    with pytest.raises(ReconciliationRequired, match="unknown schema"):
        BoundDestination("test", "profile", schema, {}).payload()


def test_conflicting_reviewed_alternatives_are_not_rendered_as_facts() -> None:
    data = approve(example(), ("lemma", "case", "number"))
    case = data.cases[0]
    alternative = case.proposals[1].model_copy(update={"value": "dativus", "decision": None})
    alternative = alternative.model_copy(
        update={
            "decision": review_claim(
                alternative.to_claim(case, data.profile),
                status="accepted",
                reviewer="other reviewer",
                reason="Conflicting review",
            )
        }
    )
    changed = data.model_copy(
        update={"cases": (case.model_copy(update={"proposals": (*case.proposals, alternative)}),)}
    )
    result = generate_parsing(changed)[0]
    assert result["eligible"] is False and result["answer"] == ""
    assert result["skips"] == ["Conflicting accepted alternatives: case"]


def test_source_markup_is_escaped_and_theme_does_not_rekey_or_revoke_reviews() -> None:
    data = example()
    data = data.model_copy(
        update={"cases": (data.cases[0].model_copy(update={"context": "<script>alert(1)</script>"}),)}
    )
    data = approve(data, ("lemma", "case", "number"))
    before = generate_parsing(data)[0]
    assert "<script>" not in before["prompt"] and "&lt;script&gt;" in before["prompt"]
    morphology = data.profile.morphology.model_copy(update={"theme": "monochrome", "appearance": "dark"})
    after = generate_parsing(
        data.model_copy(update={"profile": data.profile.model_copy(update={"morphology": morphology})})
    )[0]
    assert after["latinitas_id"] == before["latinitas_id"] and after["eligible"]
    assert "morphology-monochrome morphology-dark" in after["answer"]


@pytest.mark.parametrize("field", ["value", "analyzer"])
def test_cli_nonblank_claim_validation_is_a_normal_input_error(tmp_path: Path, field: str) -> None:
    payload = example().model_dump(mode="json")
    payload["cases"][0]["proposals"][0][field] = " \t"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = CliRunner().invoke(app, ["form-parsing", "preview", str(path)])
    assert result.exit_code == 2
    assert "nonblank" in result.output


def test_cli_preview_bidi_controls_are_visible_and_json_remains_lossless(tmp_path: Path) -> None:
    payload = example().model_dump(mode="json")
    text = "safe\u202e.gpj ā"
    payload["cases"][0]["context"] = text
    payload["cases"][0]["proposals"][0]["evidence"][0]["text"] = text
    path = tmp_path / "bidi.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = CliRunner().invoke(app, ["form-parsing", "preview", str(path)])
    assert result.exit_code == 0
    assert "\u202e" not in result.stdout and "\\u202e" in result.stdout
    parsed = json.loads(result.stdout)
    assert parsed[0]["context"] == text
    assert parsed[0]["claims"][0]["evidence"][0]["text"] == text
