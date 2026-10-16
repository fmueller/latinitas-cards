#!/usr/bin/env python3
"""Deliberately generate corpus-only fixtures; tests never invoke regeneration."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from latinitas_cards.atomic_io import atomic_replace
from latinitas_cards.canonical import canonical_json, digest_bytes
from latinitas_cards.cli_contract import ContractError
from latinitas_cards.corpus_io import write_corpus
from latinitas_cards.corpus_model import (
    CorpusDescriptor,
    CorpusHeader,
    CorpusMetadata,
    CorpusPassage,
    CorpusRecord,
    CorpusWork,
    canonical_text,
    encode_ref_component,
    latin_fold,
    passage_hash,
)
from latinitas_cards.json_schemas import SCHEMAS, render_schema
from latinitas_cards.usfx_adapter import build_usfx

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/corpus"


def valid_records(scheme: str) -> list[CorpusRecord]:
    works = ["GEN", "EXO"] if scheme == "usfx-bcv" else ["poem-a", "poem-b"]
    if scheme == "ordinal":
        works = ["test-corpus"]
    texts = ["In principio creavit Deus cælum et terram.", "Māter fīlium amat.", "arma : virum cano"]
    source = ("\n".join(texts) + "\n").encode()
    header = CorpusHeader.model_validate(
        {
            "format": "latinitas-corpus",
            "version": 1,
            "corpus_id": "test-corpus",
            "adapter": "usfx/1" if scheme == "usfx-bcv" else "text/1",
            "source": {"file_name": "source.txt", "sha256": digest_bytes(source), "bytes": len(source)},
        }
    )
    records: list[CorpusRecord] = [header, CorpusMetadata(corpus_id="test-corpus", title="Example corpus")]
    records.extend(CorpusWork(work=work, order=i) for i, work in enumerate(works, 1))
    for index, text in enumerate(texts, 1):
        work = works[0] if index < 3 else works[-1]
        if scheme == "usfx-bcv":
            units, values = ["book", "chapter", "verse"], [work, "1", str(index)]
            display = f"{work} 1:{index}"
        elif scheme == "ordinal":
            units, values = ["passage"], [f"p{index:06}"]
            display = values[0]
        else:
            units, values = ["work", "section", "line"], [work, "é", str(index)]
            display = ".".join(values)
        ref = ".".join(encode_ref_component(value) for value in values)
        records.append(
            CorpusPassage.model_validate(
                {
                    "passage_id": f"test-corpus:{ref}",
                    "work": work,
                    "native_ref": ref,
                    "order": index,
                    "citation": {
                        "scheme": scheme,
                        "parts": [dict(unit=u, value=v) for u, v in zip(units, values, strict=True)],
                        "display": display,
                    },
                    "text": canonical_text(text),
                    "norm": latin_fold(text),
                    "passage_hash": passage_hash(text),
                }
            )
        )
    return records


def invalid_files() -> dict[str, str]:
    baseline = [record.model_dump(mode="json") for record in valid_records("ordinal")]
    cases: dict[str, Any] = {}
    files: dict[str, str] = {}

    def add(name: str, rows: list[dict[str, Any]], code: str) -> None:
        # Float defects deliberately bypass the float-free production serializer.
        files[f"{name}.jsonl"] = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
        )
        # The old free-form source_extra float example also violates its typed item shape.
        cases[f"{name}.jsonl"] = {"exit": 1, "codes": [code, "invalid_record"] if name == "float" else [code]}

    for name, field, value, code in [
        ("non-nfc", "text", "Māter fīlium amat.", "non_nfc_text"),
        ("whitespace", "text", " In principio creavit Deus cælum et terram. ", "non_canonical_text"),
        ("hash", "passage_hash", "sha256:" + "0" * 64, "hash_mismatch"),
        ("norm", "norm", "wrong", "norm_mismatch"),
        ("identity", "passage_id", "test-corpus:wrong", "passage_id_mismatch"),
        ("order", "order", 4, "invalid_record"),
        ("float", "source_extra", {"number": 1.5}, "float_value"),
    ]:
        rows = copy.deepcopy(baseline)
        rows[4 if name == "non-nfc" else 3][field] = value
        add(name, rows, code)
    rows = copy.deepcopy(baseline)
    rows[0]["corpus_id"] = "INVALID"
    add("corpus-id", rows, "invalid_corpus_id")
    rows = copy.deepcopy(baseline)
    rows[4]["native_ref"] = rows[3]["native_ref"]
    rows[4]["passage_id"] = rows[3]["passage_id"]
    add("duplicate", rows, "duplicate_native_ref")
    rows = copy.deepcopy(baseline)
    rows[3]["citation"]["parts"][0]["value"] = 1
    add("citation", rows, "invalid_citation")
    add("missing-corpus", [row for row in baseline if row["record"] != "corpus"], "invalid_record")
    for index, ref in enumerate(["%", "%2", "%GG", "%2e", "é"]):
        rows = copy.deepcopy(baseline)
        rows[3]["native_ref"] = ref
        rows[3]["passage_id"] = f"test-corpus:{ref}"
        add(f"ref-{index}", rows, "invalid_record")
    files["cases.json"] = json.dumps(cases, indent=2, sort_keys=True) + "\n"
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["valid", "invalid", "schemas", "usfx", "markup", "vulgate-digest"], required=True
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.target == "schemas":
        outputs = {
            ROOT / f"docs/schemas/{name}.v1.schema.json": render_schema(model)
            for name, model in SCHEMAS.items()
            if name.startswith("latinitas-corpus")
        }
    elif args.target == "vulgate-digest":
        outputs = {FIXTURES / "usfx/vulgate.digest.json": ""}
    elif args.target == "markup":
        outputs = {}
        cases = {}
        for source in sorted((FIXTURES / "usfx/markup").glob("*.xml")):
            for allow in [False, True] if source.stem == "unknown" else [False]:
                key = source.stem + ("-allowed" if allow else "")
                try:
                    records = build_usfx(source, "test-corpus", "la", None, allow_unknown=allow)
                    outputs[source.with_name(f"{key}.expected.jsonl")] = "".join(
                        canonical_json(record.model_dump(mode="json", by_alias=True)) + "\n" for record in records
                    )
                    cases[key] = {"exit": 0}
                except ContractError as error:
                    cases[key] = {"exit": error.exit_code, "code": error.diagnostic.code}
        outputs[FIXTURES / "usfx/markup/cases.json"] = json.dumps(cases, indent=2, sort_keys=True) + "\n"
    elif args.target == "invalid":
        outputs = {FIXTURES / "invalid" / name: data for name, data in invalid_files().items()}
    elif args.target == "usfx":
        raw = (ROOT / "data/lat-clementine.usfx.xml").read_bytes()
        end = raw.index(b"<ve/>", raw.index(b'<v id="5"/>')) + len(b"<ve/>")
        outputs = {
            FIXTURES / "usfx/gen-1-1-5.usfx.xml": (raw[:end] + b"</book></usfx>\n").decode("utf-8"),
            FIXTURES / "usfx/gen-1-1-5.expected.jsonl": "",
        }
    else:
        outputs = {FIXTURES / f"valid/{name}.jsonl": "" for name in ["usfx-bcv", "ordinal", "wsl"]}
    for path in outputs:
        if any(component.is_symlink() for component in (path, *path.parents)):
            parser.error(f"Refusing symlinked output path {path}")
        if path.exists() and not args.force:
            parser.error(f"Refusing to overwrite {path}; use --force")
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.target == "vulgate-digest":
            source = ROOT / "data/lat-clementine.usfx.xml"
            descriptor = CorpusDescriptor.model_validate_json(
                (ROOT / "data/lat-clementine.descriptor.json").read_bytes()
            )
            records = build_usfx(source, "vulgata-clementina", "la", descriptor)
            raw = "".join(
                canonical_json(record.model_dump(mode="json", by_alias=True)) + "\n" for record in records
            ).encode("utf-8")
            data = (
                json.dumps(
                    {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                        "passages": sum(record.record == "passage" for record in records),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            atomic_replace(path, data.encode("utf-8"))
        elif args.target == "valid":
            write_corpus(path, valid_records(path.stem), [])
        elif args.target == "usfx" and path.suffix == ".jsonl":
            descriptor = CorpusDescriptor.model_validate_json(
                (ROOT / "data/lat-clementine.descriptor.json").read_bytes()
            )
            source = FIXTURES / "usfx/gen-1-1-5.usfx.xml"
            write_corpus(path, build_usfx(source, "vulgata-clementina", "la", descriptor), [source])
        else:
            atomic_replace(path, data.encode("utf-8"))


if __name__ == "__main__":
    main()
