import ast
import importlib
import json
import sys
from dataclasses import FrozenInstanceError, astuple
from pathlib import Path

import pytest

from latinitas_cards import corpus_segmentation as segmentation
from latinitas_cards.corpus_model import CorpusPassage

FIXTURES = Path(__file__).parents[1] / "fixtures/corpus"


def test_handwritten_vectors() -> None:
    for case in json.loads((FIXTURES / "segmentation/cases.json").read_text())["passages"]:
        text = case["text"]
        tokens = segmentation.tokenize(text)
        assert [[text[t.start : t.end], t.kind, t.start, t.end] for t in tokens] == case["tokens"], case["name"]
        assert [list(astuple(s)) for s in segmentation.segment_passage(text, tokens)] == case["sentences"], case["name"]


def test_every_fixture_passage_partitions_codepoints_and_tokens() -> None:
    for path in sorted((FIXTURES / "valid").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["record"] != "passage":
                continue
            text = record["text"]
            tokens = segmentation.tokenize(text)
            covered = [i for token in tokens for i in range(token.start, token.end)]
            assert covered == [i for i, char in enumerate(text) if not char.isspace()]
            sentences = segmentation.segment_passage(text, tokens)
            assert [i for s in sentences for i in range(s.token_start, s.token_end)] == list(range(len(tokens)))
            for s in sentences:
                assert (s.start, s.end) == (tokens[s.token_start].start, tokens[s.token_end - 1].end)


def passage(text: str, work: str, order: int) -> CorpusPassage:
    return CorpusPassage.model_validate(
        {
            "passage_id": f"test-corpus:p{order}",
            "native_ref": f"p{order}",
            "work": work,
            "order": order,
            "text": text,
            "norm": "unused",
            "passage_hash": "sha256:" + "0" * 64,
            "citation": {
                "scheme": "ordinal",
                "parts": [{"unit": "passage", "value": str(order)}],
                "display": str(order),
            },
        }
    )


def test_work_order_and_continuation_resets() -> None:
    cases = json.loads((FIXTURES / "segmentation/cases.json").read_text())["work"]
    passages = [passage(c["text"], c["work"], c["order"]) for c in cases]
    for case in cases:
        text = case["text"]
        assert [[text[t.start : t.end], t.kind, t.start, t.end] for t in segmentation.tokenize(text)] == case["tokens"]
    assert [[list(astuple(s)) for s in group] for group in segmentation.segment_work(reversed(passages))] == [
        c["sentences"] for c in cases
    ]


def test_frozen_records_and_rule_identifiers() -> None:
    assert segmentation.LATIN_TOK == ("latin-tok", "1")
    assert segmentation.LATIN_SEG == ("latin-seg", "1")
    token = segmentation.tokenize("A.")[0]
    sentence = segmentation.segment_passage("A.", segmentation.tokenize("A."))[0]
    with pytest.raises(FrozenInstanceError):
        token.start = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        sentence.continued = True  # type: ignore[misc]


def test_dependency_free_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = [astuple(s) for group in segmentation.segment_work([passage("Dixitque: lux!", "GEN", 1)]) for s in group]
    monkeypatch.setitem(sys.modules, "cltk", None)
    reloaded = importlib.reload(segmentation).segment_work([passage("Dixitque: lux!", "GEN", 1)])
    assert [astuple(s) for group in reloaded for s in group] == expected
    tree = ast.parse(Path(segmentation.__file__).read_text())
    imports = [
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    ]
    imports += [
        node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not {"cltk", "stanza", "random", "time", "locale"}.intersection(imports)
