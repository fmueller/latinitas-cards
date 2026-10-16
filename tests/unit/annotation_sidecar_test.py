"""Sidecar wire contracts, literal identity vectors and writer safety."""

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from latinitas_cards.annotation_sidecar import Annotation, annotation_id, read_sidecar
from latinitas_cards.canonical import canonical_json

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLDEN = FIXTURES / "annotations/translation.annotations.jsonl"
LITERAL = (
    '{"anchor":{"passage_hash":"sha256:b26e700a2726398b8b49c503fceb06eeb8d4f79b1c82bebd8a9a58bbe6c55f8c",'
    '"passage_id":"test-corpus:p000002"},'
    '"input_hash":"sha256:b26e700a2726398b8b49c503fceb06eeb8d4f79b1c82bebd8a9a58bbe6c55f8c",'
    '"layer":"translation","producer":{"kind":"human","name":"fixture","version":"1"},'
    '"record":"annotation","value":{"language":"en","text":"The mother loves her son."}}'
)


def test_literal_id_golden_and_canonical_roundtrip() -> None:
    expected = "sha256:" + hashlib.sha256(LITERAL.encode()).hexdigest()
    assert expected == "sha256:2ee246345cc09c45bba19839e626dad38091beae1a34698cc9df7d2bd98a634b"
    rows = list(read_sidecar(GOLDEN))
    record = rows[1]
    assert isinstance(record, Annotation)
    assert record.annotation_id == expected
    raw = record.model_dump(mode="json")
    assert annotation_id(raw) == expected
    raw["created"] = "2027-06-30T12:00:00Z"
    raw["plan"] = "sha256:" + "a" * 64
    assert annotation_id(raw) == expected
    for key, value in [("value", {"language": "en", "text": "Changed"}), ("producer", {"kind": "human"})]:
        assert annotation_id({**raw, key: value}) != expected
    assert annotation_id({**raw, "anchor": {"passage_id": "other"}}) != expected
    assert "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in rows) == GOLDEN.read_text()
    assert json.loads(GOLDEN.read_text().splitlines()[1])["annotation_id"] == expected


def golden_raw() -> dict[str, Any]:
    return dict(json.loads(GOLDEN.read_text().splitlines()[1]))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("confidence",), 500.0),
        (("confidence",), 1001),
        (("value", "extra"), 1.5),
        (("layer",), "syntax"),
        (("producer", "kind"), "tool"),
        (("producer", "version"), 1.0),
        (("producer", "version"), ""),
        (("producer", "name"), "bad/name"),
        (("unknown",), True),
        (("input_hash",), "sha256:" + "a" * 64),
        (("annotation_id",), "sha256:" + "a" * 64),
        (("anchor", "span"), None),
        (("anchor", "segmentation_id"), None),
        (("alternatives",), None),
        (("confidence",), None),
        (("supersedes",), None),
        (("created",), "2026-01-01"),
        (("created",), "2026-01-01T01:00:00+01:00"),
        (("created",), "2026-02-30T00:00:00Z"),
        (("value", "language"), "bad_tag"),
        (("value", "text"), ""),
    ],
)
def test_reject_invalid_records(path: tuple[str, ...], value: Any) -> None:
    raw = golden_raw()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path != ("annotation_id",):
        with contextlib.suppress(ValueError):
            raw["annotation_id"] = annotation_id(raw)
    with pytest.raises(ValidationError):
        Annotation.model_validate(raw)


def test_optional_keys_omitted_plan_required_and_layer_values() -> None:
    raw = golden_raw()
    raw.update(confidence=500, alternatives=[{"language": "la", "text": "Māter."}], supersedes="sha256:" + "b" * 64)
    raw["annotation_id"] = annotation_id(raw)
    assert Annotation.model_validate(raw).model_dump(mode="json") == raw
    for layer, value, span in [
        ("morphology", {"tagset": "UD", "lemma": "māter"}, [0, 5]),
        ("correction", {"text": "Mater"}, [0, 5]),
    ]:
        item = golden_raw()
        item.update(layer=layer, value=value)
        item["anchor"]["span"] = span
        item["annotation_id"] = annotation_id(item)
        assert Annotation.model_validate(item).model_dump(mode="json") == item
    for layer, value in [("morphology", {}), ("morphology", {"tagset": ""}), ("correction", {"text": "a\u0304"})]:
        item = golden_raw()
        item.update(layer=layer, value=value)
        item["annotation_id"] = annotation_id(item)
        with pytest.raises(ValidationError):
            Annotation.model_validate(item)
    raw = golden_raw()
    del raw["plan"]
    with pytest.raises(ValidationError):
        Annotation.model_validate(raw)


@pytest.mark.parametrize("span", [[-1, 3], [5, 2], [1, 1], [0, 3, 5], [0.0, 5]])
def test_invalid_anchor_spans(span: list[int | float]) -> None:
    raw = golden_raw()
    raw["anchor"]["span"] = span
    with contextlib.suppress(ValueError):
        raw["annotation_id"] = annotation_id(raw)
    with pytest.raises(ValidationError):
        Annotation.model_validate(raw)


def test_retractions_and_segmentation_shape() -> None:
    from latinitas_cards.annotation_sidecar import Retraction

    raw = golden_raw()
    raw.update(record="retraction", retracts=raw["annotation_id"])
    del raw["value"]
    raw["annotation_id"] = annotation_id(raw)
    assert Retraction.model_validate(raw).model_dump(mode="json") == raw
    assert raw["annotation_id"] != raw["retracts"]
    raw = golden_raw()
    raw.update(layer="segmentation", value={"tokenizer": "latin-tok/1", "tokens": [], "sentences": []})
    raw["annotation_id"] = annotation_id(raw)
    assert Annotation.model_validate(raw).model_dump(mode="json") == raw
    for value in [
        {"tokenizer": "latin-tok/2", "tokens": [], "sentences": []},
        {"tokenizer": "latin-tok/1", "tokens": [{"kind": "word", "span": [0, 4]}], "sentences": []},
        {"tokenizer": "latin-tok/1", "tokens": [{"kind": "other", "span": [0, 4]}], "sentences": []},
    ]:
        raw["value"] = value
        raw["annotation_id"] = annotation_id(raw)
        with pytest.raises(ValidationError):
            Annotation.model_validate(raw)


def test_contextual_anchor_bounds_and_token_boundaries() -> None:
    from latinitas_cards.annotation_sidecar import validate_anchor
    from latinitas_cards.corpus_io import read_corpus
    from latinitas_cards.corpus_model import CorpusPassage

    passage = next(
        row
        for row in read_corpus(FIXTURES / "corpus/valid/ordinal.jsonl")
        if isinstance(row, CorpusPassage) and row.order == 2
    )
    raw = golden_raw()
    raw.update(
        layer="segmentation",
        value={
            "tokenizer": "latin-tok/1",
            "sentences": [{"continued": False, "continues": False, "span": [0, 18], "tokens": [0, 4]}],
            "tokens": [
                {"kind": "word", "span": [0, 5]},
                {"kind": "word", "span": [6, 12]},
                {"kind": "word", "span": [13, 17]},
                {"kind": "punct", "span": [17, 18]},
            ],
        },
    )
    raw["annotation_id"] = annotation_id(raw)
    segmentation = Annotation.model_validate(raw)
    validate_anchor(segmentation, passage)
    raw = golden_raw()
    raw["anchor"].update(segmentation_id=segmentation.annotation_id, span=[0, 5])
    raw["annotation_id"] = annotation_id(raw)
    validate_anchor(Annotation.model_validate(raw), passage, segmentation)
    for span in [[0, 4], [1, 5], [0, 19]]:
        raw["anchor"]["span"] = span
        raw["annotation_id"] = annotation_id(raw)
        with pytest.raises(ValueError):
            validate_anchor(Annotation.model_validate(raw), passage, segmentation)


def test_retraction_must_copy_target_binding(tmp_path: Path) -> None:
    from latinitas_cards.cli_contract import ContractError

    raw = golden_raw()
    raw.update(record="retraction", retracts=raw["annotation_id"], layer="morphology")
    del raw["value"]
    raw["annotation_id"] = annotation_id(raw)
    sidecar = tmp_path / "bad.jsonl"
    sidecar.write_text(GOLDEN.read_text() + canonical_json(raw) + "\n")
    with pytest.raises(ContractError, match="retraction"):
        list(read_sidecar(sidecar))


def test_morphology_fixture_roundtrip_and_context() -> None:
    from latinitas_cards.annotation_sidecar import validate_anchor
    from latinitas_cards.corpus_io import read_corpus
    from latinitas_cards.corpus_model import CorpusPassage

    path = FIXTURES / "annotations/fixture-morph.annotations.jsonl"
    rows = list(read_sidecar(path))
    assert "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in rows) == path.read_text()
    segmentation, morphology = rows[1:]
    assert isinstance(segmentation, Annotation) and isinstance(morphology, Annotation)
    assert morphology.producer.model_dump() == {"kind": "analyzer", "name": "fixture-morph", "version": "1"}
    passage = next(
        row
        for row in read_corpus(FIXTURES / "corpus/valid/ordinal.jsonl")
        if isinstance(row, CorpusPassage) and row.order == 2
    )
    validate_anchor(morphology, passage, segmentation)
    raw = morphology.model_dump(mode="json")
    del raw["value"]["tagset"]
    raw["annotation_id"] = annotation_id(raw)
    with pytest.raises(ValidationError):
        Annotation.model_validate(raw)


def test_typed_model_construction_and_append_binding_safety(tmp_path: Path) -> None:
    from latinitas_cards.annotation_sidecar import Anchor, Producer, Retraction, append_records
    from latinitas_cards.cli_contract import ContractError

    raw = golden_raw()
    record = Annotation(**{**raw, "anchor": Anchor(**raw["anchor"]), "producer": Producer(**raw["producer"])})
    path = tmp_path / "sidecar.jsonl"
    assert append_records(path, [record], []) == 1
    assert append_records(path, [record], []) == 0
    before = path.read_bytes()
    raw.update(record="retraction", retracts=raw["annotation_id"], layer="morphology")
    del raw["value"]
    raw["annotation_id"] = annotation_id(raw)
    with pytest.raises(ContractError, match="retraction"):
        append_records(path, [Retraction.model_validate(raw)], [])
    assert path.read_bytes() == before


@pytest.mark.parametrize("encoding", ["spaces", "reordered", "no-newline", "duplicate-key", "crlf"])
def test_reader_rejects_noncanonical_lines(tmp_path: Path, encoding: str) -> None:
    from latinitas_cards.cli_contract import Code, ContractError

    header = GOLDEN.read_text().splitlines()[0]
    versions = {
        "spaces": json.dumps(json.loads(header)) + "\n",
        "reordered": '{"version":1,"record":"header","format":"latinitas-annotations"}\n',
        "no-newline": header,
        "duplicate-key": header[:-1] + ',"version":1}\n',
        "crlf": header + "\r\n",
    }
    path = tmp_path / "bad.jsonl"
    path.write_bytes(versions[encoding].encode())
    with pytest.raises(ContractError) as error:
        list(read_sidecar(path))
    assert error.value.diagnostic.code == Code.INVALID_SIDECAR


def test_header_requires_all_wire_fields_in_model_and_schema() -> None:
    from latinitas_cards.annotation_sidecar import AnnotationHeader, AnnotationSchema

    header = {"format": "latinitas-annotations", "record": "header", "version": 1}
    assert AnnotationHeader.model_validate(header).model_dump() == header
    assert set(AnnotationSchema.model_json_schema()["$defs"]["AnnotationHeader"].get("required", [])) == set(header)
    for missing in header:
        with pytest.raises(ValidationError):
            AnnotationHeader.model_validate({key: value for key, value in header.items() if key != missing})
