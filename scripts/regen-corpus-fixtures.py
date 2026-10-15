#!/usr/bin/env python3
"""Deliberately generate corpus-only fixtures; tests never invoke regeneration."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from latinitas_cards.atomic_io import atomic_replace
from latinitas_cards.canonical import digest_bytes
from latinitas_cards.corpus_io import write_corpus
from latinitas_cards.corpus_model import (
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
        cases[f"{name}.jsonl"] = {"exit": 1, "codes": [code]}

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
    parser.add_argument("--target", choices=["valid", "invalid", "schemas"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.target == "schemas":
        outputs = {
            ROOT / f"docs/schemas/{name}.v1.schema.json": render_schema(model)
            for name, model in SCHEMAS.items()
            if name.startswith("latinitas-corpus")
        }
    elif args.target == "invalid":
        outputs = {FIXTURES / "invalid" / name: data for name, data in invalid_files().items()}
    else:
        outputs = {FIXTURES / f"valid/{name}.jsonl": "" for name in ["usfx-bcv", "ordinal", "wsl"]}
    for path in outputs:
        if any(component.is_symlink() for component in (path, *path.parents)):
            parser.error(f"Refusing symlinked output path {path}")
        if path.exists() and not args.force:
            parser.error(f"Refusing to overwrite {path}; use --force")
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.target == "valid":
            write_corpus(path, valid_records(path.stem), [])
        else:
            atomic_replace(path, data.encode("utf-8"))


if __name__ == "__main__":
    main()
