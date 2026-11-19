"""Curated intake is candidate-only and human decisions bind exact content."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from latinitas_cards.cli import app
from latinitas_cards.curated import (
    Approval,
    CuratedInput,
    Permission,
    approval_status,
    load_curated,
    make_approval,
    mnemonic_fingerprint,
    record_reviews,
    rule_fingerprint,
    validate_history,
)


def candidate() -> dict[str, Any]:
    evidence = {"source": "original fixture", "reference": "lesson:1", "version": "1", "text": "Synthetic rule."}
    provenance = {"document": "Original lesson", "section": "1", "reference": "lesson:1", "wording": "quoted"}
    return {
        "format": "latinitas-curated",
        "version": 1,
        "source_scope": "original-course",
        "items": [
            {
                "key": "imperfect-a",
                "rule": {
                    "text": "Use the present stem and imperfect marker.",
                    "language": "en",
                    "applicability": "Regular first-conjugation indicative active.",
                    "caveats": ["Ending adjustments need separate checking."],
                    "evidence": [evidence],
                },
                "mnemonics": [
                    {
                        "key": "ba",
                        "language": "de",
                        "text": "Mit ba war es damals da.",
                        "rule_key": "imperfect-a",
                        "provenance": {**provenance, "wording": "agent_proposed"},
                    }
                ],
                "examples": [
                    {"latin": "amābam", "explanation": "First-person singular imperfect.", "evidence": [evidence]}
                ],
                "tags": ["course::1"],
                "provenance": provenance,
            }
        ],
    }


def permission() -> dict[str, Any]:
    return {
        "format": "latinitas-curated-permission",
        "version": 1,
        "source_scope": "original-course",
        "sources": [
            {
                "document": "Original lesson",
                "source_sha256": "a" * 64,
                "rights": "Original synthetic fixture; permitted for this test.",
                "extraction_permission": "Author permits offline extraction.",
                "processing": "manual",
                "provider": None,
                "provider_permission": None,
            }
        ],
    }


def parse(data: dict[str, Any]) -> CuratedInput:
    return CuratedInput.read_json(json.dumps(data))


def approve(
    data: CuratedInput, *, subject: str = "rule", decision: str = "approved", supersedes: tuple[str, ...] = ()
) -> Approval:
    item = data.items[0]
    language = item.rule.language if subject == "rule" else "de"
    key = item.key if subject == "rule" else "ba"
    fingerprint = rule_fingerprint(item) if subject == "rule" else mnemonic_fingerprint(item, item.mnemonics[0])
    return make_approval(
        source_scope=data.source_scope,
        item_key=item.key,
        subject=subject,
        subject_key=key,
        language=language,
        fingerprint=fingerprint,
        decision=decision,
        reviewer="Human fixture",
        reason="Explicit synthetic test decision, not linguistic approval.",
        reviewed_at="2026-11-19T00:00:00Z",
        supersedes=supersedes,
    )


def test_round_trip_rule_only_and_separate_german_mnemonic() -> None:
    data = parse(candidate())
    rule = approve(data)
    mnemonic = approve(data, subject="mnemonic")
    assert Approval.read_json(rule.model_dump_json()) == rule
    assert approval_status(data, data.items[0], (rule,), subject="rule") == "approved"
    assert (
        approval_status(data, data.items[0], (rule,), subject="mnemonic", language="de", subject_key="ba")
        == "unreviewed"
    )
    assert (
        approval_status(data, data.items[0], (rule, mnemonic), subject="mnemonic", language="de", subject_key="ba")
        == "approved"
    )
    assert (
        approval_status(data, data.items[0], (rule, mnemonic), subject="mnemonic", language="en", subject_key="ba")
        == "missing_language_variant"
    )
    raw = candidate()
    raw["items"][0]["mnemonics"] = []
    only_rule = parse(raw)
    assert approval_status(only_rule, only_rule.items[0], (rule,), subject="rule") == "approved"


@pytest.mark.parametrize(
    "change", ["text", "applicability", "caveats", "evidence", "examples", "provenance", "language"]
)
def test_rule_changes_invalidate_rule_and_associated_mnemonic(change: str) -> None:
    raw = candidate()
    old = parse(raw)
    approvals = (approve(old), approve(old, subject="mnemonic"))
    item = raw["items"][0]
    if change == "examples":
        item["examples"][0]["explanation"] = "Changed grammatical explanation."
    elif change == "provenance":
        item["provenance"]["reference"] = "lesson:2"
    elif change == "evidence":
        item["rule"]["evidence"][0]["version"] = "2"
    elif change == "caveats":
        item["rule"][change] = ["Changed exception."]
    elif change == "language":
        item["rule"][change] = "de"
    else:
        item["rule"][change] += " Changed."
    new = parse(raw)
    assert new.items[0].key == old.items[0].key
    assert approval_status(new, new.items[0], approvals, subject="rule") == "unreviewed"
    assert (
        approval_status(new, new.items[0], approvals, subject="mnemonic", language="de", subject_key="ba")
        == "unreviewed"
    )


def test_mnemonic_edits_preserve_rule_and_other_language_approval() -> None:
    raw = candidate()
    raw["items"][0]["mnemonics"].append(
        {**copy.deepcopy(raw["items"][0]["mnemonics"][0]), "language": "en", "text": "Original English aid."}
    )
    old = parse(raw)
    rule, german = approve(old), approve(old, subject="mnemonic")
    english = make_approval(
        **{
            **german.model_dump(exclude={"approval_id"}),
            "language": "en",
            "fingerprint": mnemonic_fingerprint(old.items[0], old.items[0].mnemonics[1]),
        }
    )
    raw["items"][0]["mnemonics"][0]["text"] += " Neu."
    raw["items"][0]["tags"].append("personal")
    new = parse(raw)
    history = (rule, german, english)
    assert approval_status(new, new.items[0], history, subject="rule") == "approved"
    assert (
        approval_status(new, new.items[0], history, subject="mnemonic", language="de", subject_key="ba") == "unreviewed"
    )
    assert (
        approval_status(new, new.items[0], history, subject="mnemonic", language="en", subject_key="ba") == "approved"
    )


def test_conflicts_need_explicit_supersession_not_timestamp_order() -> None:
    data = parse(candidate())
    yes, no = approve(data), approve(data, decision="rejected")
    no = make_approval(**{**no.model_dump(exclude={"approval_id"}), "reviewed_at": "2026-11-18T00:00:00Z"})
    assert approval_status(data, data.items[0], (yes, no), subject="rule") == "conflict"
    assert approval_status(data, data.items[0], (no, yes), subject="rule") == "conflict"
    partial = approve(data, supersedes=(yes.approval_id,))
    assert approval_status(data, data.items[0], (yes, no, partial), subject="rule") == "conflict"
    replacement = approve(data, supersedes=(yes.approval_id, no.approval_id))
    assert approval_status(data, data.items[0], (yes, no, replacement), subject="rule") == "approved"
    assert approval_status(data, data.items[0], (no,), subject="rule") == "rejected"
    with pytest.raises(ValueError, match="supersed"):
        validate_history((replacement,))
    mnemonic = approve(data, subject="mnemonic", supersedes=(yes.approval_id,))
    with pytest.raises(ValueError, match="subject"):
        validate_history((yes, mnemonic))


@pytest.mark.parametrize("field,value", [("version", True), ("version", 2), ("source_scope", " "), ("approved", True)])
def test_strict_top_level(field: str, value: object) -> None:
    raw = candidate()
    raw[field] = value
    with pytest.raises(ValueError):
        parse(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_mnemonics",
        "empty_evidence",
        "wrong_rule",
        "duplicate_item",
        "duplicate_variant",
        "missing_reference",
        "empty_text",
        "extra_evidence",
        "qa",
    ],
)
def test_malformed_intake_rejected(mutation: str) -> None:
    raw = candidate()
    item = raw["items"][0]
    match mutation:
        case "missing_mnemonics":
            del item["mnemonics"]
        case "empty_evidence":
            item["rule"]["evidence"] = []
        case "wrong_rule":
            item["mnemonics"][0]["rule_key"] = "another"
        case "duplicate_item":
            raw["items"].append(copy.deepcopy(item))
        case "duplicate_variant":
            item["mnemonics"].append(copy.deepcopy(item["mnemonics"][0]))
        case "missing_reference":
            del item["provenance"]["reference"]
        case "empty_text":
            item["mnemonics"][0]["text"] = " "
        case "extra_evidence":
            item["rule"]["evidence"][0]["approved"] = True
        case "qa":
            item["kind"] = "qa"
    with pytest.raises(ValueError):
        parse(raw)


def test_duplicate_json_keys_and_forged_approval_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CuratedInput.read_json(json.dumps(candidate()).replace('"version": 1', '"version": 1, "version": 1'))
    review = approve(parse(candidate())).model_dump()
    review["decision"] = "rejected"
    with pytest.raises(ValueError, match="digest"):
        Approval.read_json(json.dumps(review))


@pytest.mark.parametrize("change", ["scope", "document", "rights", "hosted", "provider", "missing_permission"])
def test_permissions_are_explicit(tmp_path: Path, change: str) -> None:
    data = tmp_path / "candidate.json"
    data.write_text(json.dumps(candidate()))
    raw = permission()
    match change:
        case "scope":
            raw["source_scope"] = "other"
        case "document":
            raw["sources"][0]["document"] = "other"
        case "rights":
            raw["sources"][0]["rights"] = " "
        case "hosted":
            raw["sources"][0]["processing"] = "hosted"
        case "provider":
            raw["sources"][0]["provider"] = "hosted extractor"
        case "missing_permission":
            del raw["sources"][0]["extraction_permission"]
    permissions = tmp_path / "permissions.json"
    permissions.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_curated(data, permissions)


def test_permission_preserves_declared_hosted_processing() -> None:
    raw = permission()
    raw["sources"][0].update(
        processing="hosted",
        provider="disclosed historical extractor",
        provider_permission="Explicit source-owner permission for that prior extraction.",
    )
    assert Permission.read_json(json.dumps(raw)).sources[0].processing == "hosted"


def test_mnemonic_source_needs_its_own_permission(tmp_path: Path) -> None:
    raw, permitted = candidate(), permission()
    raw["items"][0]["mnemonics"][0]["provenance"]["document"] = "Separate aid source"
    data, permissions = tmp_path / "input.json", tmp_path / "permission.json"
    data.write_text(json.dumps(raw))
    permissions.write_text(json.dumps(permitted))
    with pytest.raises(ValueError, match="missing source document permission"):
        load_curated(data, permissions)
    permitted["sources"].append(
        {**permitted["sources"][0], "document": "Separate aid source", "source_sha256": "b" * 64}
    )
    permissions.write_text(json.dumps(permitted))
    assert load_curated(data, permissions)[0].items[0].mnemonics[0].provenance.document == "Separate aid source"


def test_cli_preview_review_no_export_or_automatic_approval(tmp_path: Path) -> None:
    runner = CliRunner()
    data, permissions = tmp_path / "candidate.json", tmp_path / "permissions.json"
    data.write_text(json.dumps(candidate()))
    permissions.write_text(json.dumps(permission()))
    preserved = {path: path.read_bytes() for path in (data, permissions)}
    args = [str(data), "--permissions", str(permissions)]
    result = runner.invoke(app, ["curated", "validate", *args])
    assert result.exit_code == 0, result.output
    assert "not export authorization" in result.output
    result = runner.invoke(app, ["curated", "preview", *args])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["items"][0]["rule"]["status"] == "unreviewed"
    assert report["items"][0]["mnemonics"][0]["text"] == "Mit ba war es damals da."
    assert report["items"][0]["mnemonics"][0]["provenance"]["wording"] == "agent_proposed"
    assert report["export_authorized"] is False
    decisions, history = tmp_path / "decisions.json", tmp_path / "history.json"
    decisions.write_text(json.dumps([approve(parse(candidate())).model_dump(mode="json")]))
    preserved[decisions] = decisions.read_bytes()
    result = runner.invoke(app, ["curated", "review", *args, "--decisions", str(decisions), "--output", str(history)])
    assert result.exit_code == 0, result.output
    assert json.loads(history.read_text())[0]["decision"] == "approved"
    preserved[history] = history.read_bytes()
    result = runner.invoke(app, ["curated", "preview", *args, "--approvals", str(history)])
    assert json.loads(result.output)["items"][0]["rule"]["status"] == "approved"
    assert runner.invoke(app, ["curated", "export", *args]).exit_code != 0
    assert {path: path.read_bytes() for path in preserved} == preserved
    assert (
        runner.invoke(
            app, ["curated", "review", *args, "--decisions", str(decisions), "--output", str(history)]
        ).exit_code
        != 0
    )
    assert {path: path.read_bytes() for path in preserved} == preserved


def test_review_refuses_stale_decisions_without_writing(tmp_path: Path) -> None:
    raw = candidate()
    decision = approve(parse(raw))
    raw["items"][0]["rule"]["text"] += " Changed."
    data, permissions, decisions, output = [
        tmp_path / name for name in ("candidate.json", "permissions.json", "decisions.json", "history.json")
    ]
    data.write_text(json.dumps(raw))
    permissions.write_text(json.dumps(permission()))
    decisions.write_text(json.dumps([decision.model_dump(mode="json")]))
    preserved = {path: path.read_bytes() for path in (data, permissions, decisions)}
    result = CliRunner().invoke(
        app,
        [
            "curated",
            "review",
            str(data),
            "--permissions",
            str(permissions),
            "--decisions",
            str(decisions),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert not output.exists()
    assert {path: path.read_bytes() for path in preserved} == preserved


def test_cli_explicit_supersession_retains_prior_rejection(tmp_path: Path) -> None:
    data = parse(candidate())
    rejected = approve(data, decision="rejected")
    accepted = approve(data, supersedes=(rejected.approval_id,))
    paths = {name: tmp_path / f"{name}.json" for name in ("input", "permission", "history", "decisions", "output")}
    for name, content in (
        ("input", candidate()),
        ("permission", permission()),
        ("history", [rejected.model_dump(mode="json")]),
        ("decisions", [accepted.model_dump(mode="json")]),
    ):
        paths[name].write_text(json.dumps(content))
    preserved = {path: path.read_bytes() for path in paths.values() if path.exists()}
    result = CliRunner().invoke(
        app,
        [
            "curated",
            "review",
            str(paths["input"]),
            "--permissions",
            str(paths["permission"]),
            "--approvals",
            str(paths["history"]),
            "--decisions",
            str(paths["decisions"]),
            "--output",
            str(paths["output"]),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(paths["output"].read_text()) == [
        rejected.model_dump(mode="json"),
        accepted.model_dump(mode="json"),
    ]
    assert {path: path.read_bytes() for path in preserved} == preserved


def test_cli_records_independent_language_specific_mnemonic_decisions(tmp_path: Path) -> None:
    raw = candidate()
    raw["items"][0]["mnemonics"].append(
        {**copy.deepcopy(raw["items"][0]["mnemonics"][0]), "language": "en", "text": "An original English candidate."}
    )
    data = parse(raw)
    german = approve(data, subject="mnemonic")
    english = make_approval(
        **{
            **german.model_dump(exclude={"approval_id"}),
            "language": "en",
            "decision": "rejected",
            "fingerprint": mnemonic_fingerprint(data.items[0], data.items[0].mnemonics[1]),
        }
    )
    paths = {name: tmp_path / f"{name}.json" for name in ("input", "permission", "decisions", "history")}
    paths["input"].write_text(json.dumps(raw))
    paths["permission"].write_text(json.dumps(permission()))
    paths["decisions"].write_text(json.dumps([german.model_dump(mode="json"), english.model_dump(mode="json")]))
    runner = CliRunner()
    args = [str(paths["input"]), "--permissions", str(paths["permission"])]
    result = runner.invoke(
        app, ["curated", "review", *args, "--decisions", str(paths["decisions"]), "--output", str(paths["history"])]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["curated", "preview", *args, "--approvals", str(paths["history"])])
    assert result.exit_code == 0, result.output
    item = json.loads(result.output)["items"][0]
    assert item["rule"]["status"] == "unreviewed"
    assert [(m["language"], m["status"]) for m in item["mnemonics"]] == [("de", "approved"), ("en", "rejected")]


def test_permitted_extraction_fixture() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = root / "tests/fixtures/curated-extraction"
    original = (fixture / "lesson.md").read_bytes()
    source = original.decode("utf-8")
    data, permissions = load_curated(fixture / "candidates.json", fixture / "permissions.json")
    assert permissions.sources[0].source_sha256 == hashlib.sha256(original).hexdigest()
    assert permissions.sources[0].processing == "manual"
    assert "permits copying, offline extraction and redistribution" in source
    assert [item.key for item in data.items] == ["lesson:imperfect-a", "lesson:genitive-plural"]
    assert not data.items[1].mnemonics and not data.items[1].examples
    for item in data.items:
        assert item.provenance.wording == "quoted"
        assert "Rule: " + item.rule.text in source
        assert "Applicability: " + item.rule.applicability in source
        assert all("Caveat: " + caveat in source for caveat in item.rule.caveats)
        assert all(evidence.text in source for evidence in item.rule.evidence)
        assert all(example.latin + " — " + example.explanation in source for example in item.examples)
        assert all(m.text in source and m.provenance.wording == "quoted" for m in item.mnemonics)
    assert data.items[0].mnemonics[0].text == "Mit ba war es damals da."
    assert "What should I revise tomorrow?" not in data.model_dump_json()


def test_fingerprints_commit_exact_rule_content_but_not_mnemonic_or_tags() -> None:
    raw = candidate()["items"][0]
    expected = {name: raw[name] for name in ("key", "rule", "examples", "provenance")}
    encoded = json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert rule_fingerprint(parse(candidate()).items[0]) == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("change", ["scope", "key", "rule_key", "mnemonic_provenance", "mnemonic_language"])
def test_identity_association_and_wording_origin_boundaries(change: str) -> None:
    raw = candidate()
    old = parse(raw)
    rule, mnemonic = approve(old), approve(old, subject="mnemonic")
    item = raw["items"][0]
    if change == "scope":
        raw["source_scope"] = "other-course"
    elif change in {"key", "rule_key"}:
        item["key"] = item["mnemonics"][0]["rule_key"] = "other-rule"
    elif change == "mnemonic_provenance":
        item["mnemonics"][0]["provenance"]["wording"] = "quoted"
    else:
        item["mnemonics"][0]["language"] = "en"
    new = parse(raw)
    assert approval_status(new, new.items[0], (rule, mnemonic), subject="rule") == (
        "approved" if change.startswith("mnemonic_") else "unreviewed"
    )
    assert (
        approval_status(
            new,
            new.items[0],
            (rule, mnemonic),
            subject="mnemonic",
            subject_key="ba",
            language=new.items[0].mnemonics[0].language,
        )
        == "unreviewed"
    )


@pytest.mark.parametrize("change", ["scope", "rendered", "language", "unknown_item", "empty", "automatic", "date"])
def test_explicit_review_rejects_unbound_or_unsupported_decisions(tmp_path: Path, change: str) -> None:
    data = parse(candidate())
    fields = approve(data).model_dump(exclude={"approval_id"})
    match change:
        case "scope":
            fields["source_scope"] = "other"
        case "rendered":
            fields["subject"] = "rendered_card"
        case "language":
            fields["language"] = "de"
        case "unknown_item":
            fields["item_key"] = fields["subject_key"] = "missing"
        case "automatic":
            fields["decision"] = "automatic"
        case "date":
            fields["reviewed_at"] = "2026-02-30T00:00:00Z"
    output = tmp_path / "history.json"
    with pytest.raises(ValueError):
        decision = make_approval(**fields)
        record_reviews(data, (), () if change == "empty" else (decision,), output)
    assert not output.exists()


def test_candidate_preview_preserves_html_as_data_and_escapes_terminal_controls(tmp_path: Path) -> None:
    raw = candidate()
    raw["items"][0]["mnemonics"][0]["text"] = '<script>alert("x")</script>\u001b[31m'
    data, permissions = tmp_path / "input.json", tmp_path / "permission.json"
    data.write_text(json.dumps(raw))
    permissions.write_text(json.dumps(permission()))
    result = CliRunner().invoke(app, ["curated", "preview", str(data), "--permissions", str(permissions)])
    assert result.exit_code == 0, result.output
    assert "\u001b" not in result.output
    assert json.loads(result.output)["items"][0]["mnemonics"][0]["text"] == raw["items"][0]["mnemonics"][0]["text"]
