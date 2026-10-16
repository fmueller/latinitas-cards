"""Read-only effective configuration, using the existing input loaders."""

from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path

from pydantic import JsonValue

from .canonical import digest, digest_bytes
from .cli_contract import Code, ContractError, Diagnostic
from .corpus_io import CorpusFinding, read_corpus
from .corpus_model import CorpusHeader, CorpusRecord
from .corpus_validation import validate_records
from .json_schemas import SCHEMAS
from .policy import load_policy, policy_digest


def build_report(policy_path: Path | None, corpus_path: Path | None) -> tuple[JsonValue, list[Diagnostic]]:
    report: dict[str, JsonValue] = {
        "tool_version": {"value": version("latinitas-cards"), "origin": "default"},
        "supported_formats": {
            "value": {
                **dict.fromkeys(SCHEMAS, 1),
                "latinitas-managed-plan": 1,
                "latinitas-managed-approval": 1,
                "latinitas-observed-baseline": 1,
            },
            "origin": "default",
        },
        "rule_versions": {
            "value": {
                "text_canon": 1,
                "norm_rules": "latin-fold/1",
                "segmentation": "latin-seg/1",
                "tokenization": "latin-tok/1",
            },
            "origin": "default",
        },
        "receipt_actor": {"value": "human", "origin": "default"},
    }
    for name, path in (("policy", policy_path), ("corpus", corpus_path)):
        report[f"{name}_path"] = {
            "value": str(path.resolve()) if path else None,
            "origin": "flag" if path else "default",
        }
        report[f"{name}_digest"] = {"value": None, "origin": "default"}
    report["corpus_header_digest"] = {"value": None, "origin": "default"}
    if policy_path is not None:
        policy = load_policy(policy_path)
        report["receipt_actor"] = {
            "value": policy.receipt_actor,
            "origin": "file" if "receipt_actor" in policy.model_fields_set else "default",
        }
        report["policy_digest"] = {"value": policy_digest(policy_path), "origin": "file"}
    if corpus_path is not None:
        header: CorpusHeader | None = None

        def records() -> Iterator[CorpusRecord | CorpusFinding]:
            nonlocal header
            for row in read_corpus(corpus_path):
                if isinstance(row, CorpusHeader):
                    header = row
                yield row

        _, findings = validate_records(records(), path=str(corpus_path))
        if findings:
            gates = {
                Code.UNREADABLE,
                Code.INVALID_JSON,
                Code.UNSUPPORTED_FORMAT,
                Code.UNSUPPORTED_VERSION,
                Code.MISSING_HEADER,
            }
            if any(f.code in gates for f in findings):
                return None, findings
            return None, [
                Diagnostic(
                    code=Code.INVALID_CORPUS, message=f"Invalid corpus: {findings[0].code}", path=str(corpus_path)
                ),
                *findings,
            ]
        assert header is not None
        try:
            corpus_digest = digest_bytes(corpus_path.read_bytes())
        except OSError as error:
            raise ContractError(Code.UNREADABLE, str(error), path=str(corpus_path)) from error
        report["corpus_digest"] = {"value": corpus_digest, "origin": "file"}
        report["corpus_header_digest"] = {"value": digest(header.model_dump(mode="json")), "origin": "file"}
    return report, []
