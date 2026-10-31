import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from latinitas_cards.corpus_model import (
    Citation,
    CorpusDescriptor,
    CorpusHeader,
    CorpusPassage,
    CorpusWork,
    canonical_text,
    encode_ref_component,
    latin_fold,
    passage_hash,
)

FIXTURES = Path(__file__).parents[1] / "fixtures/corpus"


def test_hash_vectors() -> None:
    for vector in json.loads((FIXTURES / "schema/hash-vectors.json").read_text()):
        text = vector["text"]
        assert canonical_text(vector["input"]) == text
        assert vector["hash"] == "sha256:" + hashlib.sha256(text.encode()).hexdigest()
        assert passage_hash(vector["input"]) == passage_hash(text) == vector["hash"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ĀăĔīŏŪ", "aaeiou"),
        ("ĀăĔīŏŪ", "aaeiou"),
        ("Ææ", "aeae"),
        ("Œœ", "oeoe"),
        ("Jj", "ii"),
        ("vV", "vv"),
    ],
)
def test_latin_fold(source: str, expected: str) -> None:
    assert latin_fold(source) == expected


def header_data() -> dict[str, object]:
    return {
        "format": "latinitas-corpus",
        "version": 1,
        "corpus_id": "test-corpus",
        "adapter": "text/1",
        "source": {"file_name": "source.txt", "sha256": "sha256:" + "a" * 64, "bytes": 5},
    }


def test_optional_source_metadata_preserves_legacy_serialization() -> None:
    work = CorpusWork(work="local", order=1)
    assert work.model_dump(mode="json") == {"record": "work", "work": "local", "title": None, "order": 1}
    header = CorpusHeader.model_validate(header_data())
    assert "revision" not in header.model_dump(mode="json")
    assert header.model_dump(mode="json")["edition"] is None
    descriptor = CorpusDescriptor(format="latinitas-corpus-descriptor", version=1)
    assert "revision" not in descriptor.model_dump(mode="json")
    assert descriptor.model_copy(update={"revision": "rev"}).model_dump(mode="json")["revision"] == "rev"
    assert work.model_copy(update={"author": "A"}).model_dump(mode="json")["author"] == "A"


@pytest.mark.parametrize("value", [1.0, True, "1", 2, 0])
def test_strict_versions(value: object) -> None:
    cases: list[tuple[type[BaseModel], dict[str, object]]] = [
        (CorpusHeader, header_data()),
        (CorpusDescriptor, {"format": "latinitas-corpus-descriptor", "version": 1}),
    ]
    for model, data in cases:
        with pytest.raises(ValidationError):
            model.model_validate({**data, "version": value})
    with pytest.raises(ValidationError):
        CorpusHeader.model_validate({**header_data(), "text_canon": value})


@pytest.mark.parametrize(
    "raw,encoded", [(".", "%2E"), ("%", "%25"), ("é", "%C3%A9"), ("é", "%C3%A9"), ("3a", "3a"), ("%2E", "%252E")]
)
def test_encode_components(raw: str, encoded: str) -> None:
    assert encode_ref_component(raw) == encoded


def passage_data(ref: str = "p000001") -> dict[str, object]:
    return {
        "passage_id": f"test-corpus:{ref}",
        "native_ref": ref,
        "work": "test-corpus",
        "order": 1,
        "citation": {"scheme": "ordinal", "parts": [{"unit": "passage", "value": ref}], "display": ref},
        "text": "Arma.",
        "norm": "arma.",
        "passage_hash": passage_hash("Arma."),
    }


@pytest.mark.parametrize("ref", ["GEN.1.1%2E2", "%25", "%C3%A9", "%252E"])
def test_valid_refs(ref: str) -> None:
    assert CorpusPassage.model_validate(passage_data(ref)).native_ref == ref


@pytest.mark.parametrize("ref", ["%", "%2", "%GG", "%2e", "é", ""])
def test_invalid_refs(ref: str) -> None:
    with pytest.raises(ValidationError):
        CorpusPassage.model_validate(passage_data(ref))


@pytest.mark.parametrize(
    "identity", ["b:p000001", "test-corpus:other", "test-corpus:" + "x" * 245, "tést-corpus:p000001"]
)
def test_invalid_passage_identity(identity: str) -> None:
    with pytest.raises(ValidationError):
        CorpusPassage.model_validate({**passage_data(), "passage_id": identity})


def test_identity_byte_boundary() -> None:
    assert len(CorpusPassage.model_validate(passage_data("x" * 244)).passage_id.encode()) == 256
    with pytest.raises(ValidationError):
        CorpusPassage.model_validate(passage_data("x" * 245))


def test_strict_fields() -> None:
    assert CorpusDescriptor.model_validate({"format": "latinitas-corpus-descriptor", "version": 1}).edition is None
    assert CorpusWork(work="GEN", order=1).title is None
    for data in [
        {**header_data(), "corpus_id": "A"},
        {**header_data(), "extra": 1},
        {**header_data(), "source": {"file_name": "../source", "sha256": "bad", "bytes": -1}},
    ]:
        with pytest.raises(ValidationError):
            CorpusHeader.model_validate(data)
    with pytest.raises(ValidationError):
        Citation.model_validate({"scheme": "ordinal", "parts": [{"unit": "passage", "value": 1}], "display": "1"})
    for order in [0, True, 1.0]:
        with pytest.raises(ValidationError):
            CorpusWork.model_validate({"work": "GEN", "order": order})


@pytest.mark.parametrize("missing", ["format", "version"])
def test_envelope_required_in_models_and_schemas(missing: str) -> None:
    cases: list[tuple[type[BaseModel], dict[str, object]]] = [
        (CorpusHeader, header_data()),
        (CorpusDescriptor, {"format": "latinitas-corpus-descriptor", "version": 1}),
    ]
    for model, data in cases:
        del data[missing]
        with pytest.raises(ValidationError):
            model.model_validate(data)
        assert missing in model.model_json_schema()["required"]


@pytest.mark.parametrize(
    "scheme,parts,display",
    [
        ("usfx-bcv", [("book", "GEN"), ("chapter", "1"), ("verse", "3a")], "GEN 1:3a"),
        ("ordinal", [("passage", "p000021")], "p000021"),
        ("wsl", [("work", "poem"), ("section", "é"), ("line", "3")], "poem.é.3"),
    ],
)
def test_citation_display(scheme: str, parts: list[tuple[str, str]], display: str) -> None:
    data = {"scheme": scheme, "parts": [{"unit": u, "value": v} for u, v in parts], "display": display}
    assert Citation.model_validate(data).display == display
    with pytest.raises(ValidationError):
        Citation.model_validate({**data, "display": "wrong"})


@pytest.mark.parametrize(
    "tag",
    [
        "la",
        "la-Latn",
        "en-US",
        "es-419",
        "zh-cmn-Hans-CN",
        "de-CH-1901",
        "sl-rozaj-biske",
        "en-a-foobar-b-baz-x-private",
        "x-private",
        "i-klingon",
        "en-GB-oed",
    ],
)
def test_language_tags(tag: str) -> None:
    assert CorpusHeader.model_validate({**header_data(), "language": tag}).language == tag


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "not a language tag",
        "la_XX",
        "e",
        "latinlanguage",
        "la--Latn",
        "la-x",
        "en-a",
        "en-a-foo-a-bar",
        "sl-rozaj-rozaj",
        "en-\né",
    ],
)
def test_invalid_language_tags(tag: str) -> None:
    with pytest.raises(ValidationError):
        CorpusHeader.model_validate({**header_data(), "language": tag})


@pytest.mark.parametrize(
    "field,item",
    [
        ("notes", {"kind": "other", "text": "note"}),
        ("notes", {"kind": "f"}),
        ("headings", {"text": "heading", "extra": "no"}),
        ("headings", "heading"),
        ("source_markup", {"element": "w", "attributes": {"s": 1}, "span": [0, 1]}),
        ("source_markup", {"element": "w", "attributes": {}, "span": [0, 1, 2]}),
        ("source_extra", {"element": "zz", "attributes": {}, "span": [-1, 1]}),
        ("source_extra", {"element": "zz", "attributes": {}, "span": [2, 1]}),
        ("source_extra", {"element": "zz", "attributes": {}, "span": [0, 1.0]}),
    ],
)
def test_passage_metadata_item_shapes(field: str, item: object) -> None:
    with pytest.raises(ValidationError):
        CorpusPassage.model_validate({**passage_data(), field: [item]})
