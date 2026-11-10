"""Effective configuration is a read-only, origin-labelled report."""

import json
from importlib.metadata import version
from pathlib import Path

import pytest
from typer.testing import CliRunner

from latinitas_cards.canonical import digest, digest_bytes
from latinitas_cards.cli import app
from latinitas_cards.cli_contract import CliResult
from latinitas_cards.json_schemas import SCHEMAS

CORPUS = Path("tests/fixtures/corpus/valid/ordinal.jsonl")


@pytest.mark.parametrize("actor", [None, "any"])
def test_effective_config_origins_and_digests(tmp_path: Path, actor: str | None) -> None:
    policy = tmp_path / "policy.json"
    data = {"format": "latinitas-policy", "version": 1, "rules": []}
    if actor:
        data["receipt_actor"] = actor
    policy.write_text(json.dumps(data))
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (policy, CORPUS)}
    result = CliRunner().invoke(app, ["corpus", "config", "--policy", str(policy), "--corpus", str(CORPUS), "--json"])
    assert result.exit_code == 0, result.output
    report = CliResult.model_validate_json(result.stdout).result
    assert isinstance(report, dict)
    assert report["tool_version"] == {"value": version("latinitas-cards"), "origin": "default"}
    assert report["supported_formats"] == {
        "value": {
            **dict.fromkeys(SCHEMAS, 1),
            "latinitas-managed-plan": 1,
            "latinitas-managed-approval": 1,
            "latinitas-observed-baseline": 1,
        },
        "origin": "default",
    }
    assert report["rule_versions"] == {
        "value": {
            "text_canon": 1,
            "norm_rules": "latin-fold/1",
            "segmentation": "latin-seg/1",
            "tokenization": "latin-tok/1",
        },
        "origin": "default",
    }
    assert report["policy_path"] == {"value": str(policy.resolve()), "origin": "flag"}
    assert report["corpus_path"] == {"value": str(CORPUS.resolve()), "origin": "flag"}
    assert report["receipt_actor"] == {"value": actor or "human", "origin": "file" if actor else "default"}
    assert report["policy_digest"] == {"value": digest_bytes(policy.read_bytes()), "origin": "file"}
    assert report["corpus_digest"] == {"value": digest_bytes(CORPUS.read_bytes()), "origin": "file"}
    assert report["corpus_header_digest"] == {
        "value": digest(json.loads(CORPUS.read_text().splitlines()[0])),
        "origin": "file",
    }
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["policy.json"]


def test_default_config() -> None:
    result = CliRunner().invoke(app, ["corpus", "config", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)["result"]
    for name in ("policy_path", "corpus_path", "policy_digest", "corpus_digest", "corpus_header_digest"):
        assert report[name] == {"value": None, "origin": "default"}
    assert report["receipt_actor"] == {"value": "human", "origin": "default"}


@pytest.mark.parametrize(
    "option,contents,code",
    [
        ("--policy", None, "unreadable"),
        ("--policy", "{", "invalid_json"),
        ("--policy", '{"format":"other"}', "unsupported_format"),
        ("--policy", '{"version":2}', "unsupported_version"),
        ("--policy", "{}", "invalid_policy"),
        ("--corpus", None, "unreadable"),
        ("--corpus", "{", "invalid_json"),
        ("--corpus", '{"record":"header","format":"other"}', "unsupported_format"),
        ("--corpus", '{"record":"header","format":"latinitas-corpus","version":2}', "unsupported_version"),
        ("--corpus", '{"record":"header","format":"latinitas-corpus","version":1}', "invalid_corpus"),
    ],
)
def test_config_loader_diagnostics(tmp_path: Path, option: str, contents: str | None, code: str) -> None:
    path = tmp_path / "input.json"
    if contents is not None:
        path.write_text(contents)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    result = CliRunner().invoke(app, ["corpus", "config", option, str(path), "--json"])
    assert result.exit_code == 3, result.output
    envelope = CliResult.model_validate_json(result.stdout)
    assert envelope.diagnostics[0].code == code
    if code == "invalid_corpus":
        assert sum(d.code == "invalid_corpus" for d in envelope.diagnostics) == 1
        assert envelope.diagnostics[1].code in envelope.diagnostics[0].message
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
