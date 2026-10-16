"""Full-edition acceptance, with a raw regex oracle independent of the reader."""

import hashlib
import json
import os
import re
import runpy
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from latinitas_cards.corpus_io import write_corpus
from latinitas_cards.corpus_model import CorpusDescriptor
from latinitas_cards.corpus_validation import validate_corpus
from latinitas_cards.usfx_adapter import build_usfx

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/corpus"
SOURCE_HASH = "05bcb3f669d9b36deab6cf26950b8a24c7609b292e5af1abfa51717677701950"
pytestmark = pytest.mark.skipif("MUTANT_UNDER_TEST" in os.environ, reason="full-corpus acceptance is not mutation work")


def test_digest_generator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real generator on five verses, never a second full edition."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "lat-clementine.usfx.xml").write_bytes((FIXTURES / "usfx/gen-1-1-5.usfx.xml").read_bytes())
    (data / "lat-clementine.descriptor.json").write_bytes((ROOT / "data/lat-clementine.descriptor.json").read_bytes())
    main = runpy.run_path(str(ROOT / "scripts/regen-corpus-fixtures.py"))["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(main.__globals__, "FIXTURES", tmp_path / "fixtures")
    monkeypatch.setattr(sys, "argv", ["regen-corpus-fixtures.py", "--target", "vulgate-digest"])
    main()
    output = tmp_path / "fixtures/usfx/vulgate.digest.json"
    expected_bytes = (
        (FIXTURES / "usfx/gen-1-1-5.expected.jsonl")
        .read_bytes()
        .replace(b"gen-1-1-5.usfx.xml", b"lat-clementine.usfx.xml")
    )
    assert json.loads(output.read_bytes()) == {
        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "bytes": len(expected_bytes),
        "passages": 5,
    }
    original = output.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        main()
    assert output.read_bytes() == original
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--force"])
    main()
    assert output.read_bytes() == original


@pytest.fixture(scope="module")
def normalized_vulgate(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, float]:
    source = ROOT / "data/lat-clementine.usfx.xml"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_HASH
    descriptor_path = ROOT / "data/lat-clementine.descriptor.json"
    descriptor = CorpusDescriptor.model_validate_json(descriptor_path.read_bytes())
    output = tmp_path_factory.mktemp("vulgate") / "corpus.jsonl"
    started = time.perf_counter()
    write_corpus(output, build_usfx(source, "vulgata-clementina", "la", descriptor), [source, descriptor_path])
    seconds = time.perf_counter() - started
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_HASH
    return output, seconds


def test_full_clementine_normalization(
    normalized_vulgate: tuple[Path, float], record_property: Callable[[str, object], None]
) -> None:
    output, seconds = normalized_vulgate
    record_property("normalize_seconds", seconds)
    record_property("normalize_ci_budget_seconds", 15)
    assert seconds < 60, f"normalization took {seconds:.3f}s (CI budget: 15s)"
    raw = output.read_bytes()
    rows: list[dict[str, Any]] = [json.loads(line) for line in raw.splitlines()]
    passages = [row for row in rows if row["record"] == "passage"]
    assert sum(row["record"] == "work" for row in rows) == 73
    assert len(passages) == 35809
    assert len({(p["work"], p["citation"]["parts"][1]["value"]) for p in passages}) == 1334
    ids = [p["passage_id"] for p in passages]
    assert len(set(ids)) == len(ids)
    assert [p["order"] for p in passages] == list(range(1, 35810))

    # This oracle uses only raw source tags, not adapter/model/CLI helpers or constants.
    expected_ids: set[str] = set()
    expected_counts: Counter[str] = Counter()
    book = chapter = ""
    source_text = (ROOT / "data/lat-clementine.usfx.xml").read_text(encoding="utf-8")
    for match in re.finditer(r'<(book|c|v) id="([^"]+)"\s*/?>', source_text):
        tag, value = match.groups()
        if tag == "book":
            book, chapter = value, ""
        elif tag == "c":
            chapter = value
        else:
            assert book and chapter
            expected_ids.add(f"vulgata-clementina:{book}.{chapter}.{value}")
            expected_counts[book] += 1
    assert set(ids) == expected_ids
    assert Counter(p["work"] for p in passages) == expected_counts

    vector = json.loads((FIXTURES / "schema/hash-vectors.json").read_bytes())[0]
    assert passages[0]["text"] == vector["text"] == "In principio creavit Deus cælum et terram."
    assert passages[0]["passage_hash"] == vector["hash"]
    golden = [json.loads(line) for line in (FIXTURES / "usfx/gen-1-1-5.expected.jsonl").read_bytes().splitlines()]
    assert passages[:5] == [row for row in golden if row["record"] == "passage"]
    assert passages[-1]["passage_id"] == "vulgata-clementina:REV.22.21"
    assert passages[-1]["text"] == "Gratia Domini nostri Jesu Christi cum omnibus vobis. Amen."
    assert json.loads((FIXTURES / "usfx/vulgate.digest.json").read_bytes()) == {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "passages": len(passages),
    }
    counts, findings = validate_corpus(str(output))
    assert findings == []
    assert counts == {"header": 1, "corpus": 1, "work": 73, "passage": 35809}
    assert "data/" in tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["mutmut"]["also_copy"]
