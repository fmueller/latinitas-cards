import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from latinitas_cards.cli import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/policy"


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("first-match", None),
        ("deny-precedence", None),
        ("threshold", "invalid_policy"),
        ("min-confidence", "invalid_policy"),
        ("calibration", "invalid_policy"),
        ("unknown-field", "invalid_policy"),
        ("auto-supersede", "invalid_policy"),
        ("auto-retract", "invalid_policy"),
        ("auto-any-operation", "invalid_policy"),
        ("version-float", "invalid_policy"),
        ("version-2", "unsupported_version"),
        ("other-format", "unsupported_format"),
        ("not-json", "invalid_json"),
        ("missing", "unreadable"),
    ],
)
def test_validate_fixture_contract(name: str, code: str | None) -> None:
    result = CliRunner().invoke(app, ["policy", "validate", str(FIXTURES / f"{name}.json"), "--json"])
    assert result.exit_code == (3 if code else 0), result.output
    envelope = json.loads(result.stdout)
    assert envelope["command"] == "policy validate"
    assert envelope["ok"] is (code is None)
    assert envelope["exit_code"] == result.exit_code
    assert [d["code"] for d in envelope["diagnostics"]] == ([code] if code else [])
    if code:
        assert envelope["diagnostics"][0]["path"] == str(FIXTURES / f"{name}.json")


@pytest.mark.parametrize("content", ["[]", "null", "{}", '{"version":true}', '{"version":"1"}'])
def test_malformed_document_is_not_internal_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "policy.json"
    path.write_text(content)
    result = CliRunner().invoke(app, ["policy", "validate", str(path), "--json"])
    assert result.exit_code == 3
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "invalid_policy"


@pytest.mark.parametrize(
    "content",
    [
        '{"format":"latinitas-policy","version":1,"rules":[{"operation":"add","layer":"*",'
        '"producer_kind":"*","producer":"*","action":"deny","action":"auto"}]}',
        '{"format":"latinitas-policy","version":1,"rules":[{"operation":"*","layer":"*",'
        '"producer_kind":"*","producer":"*","action":"deny"}],"rules":[]}',
    ],
)
def test_duplicate_keys_cannot_hide_permission_changes(tmp_path: Path, content: str) -> None:
    path = tmp_path / "ambiguous.json"
    path.write_text(content)
    result = CliRunner().invoke(app, ["policy", "validate", str(path), "--json"])
    assert result.exit_code == 3, result.stdout
    envelope = json.loads(result.stdout)
    assert envelope["diagnostics"][0]["code"] == "invalid_policy"
    assert envelope["diagnostics"][0]["path"] == str(path)
