import pytest

from latinitas_cards.canonical import canonical_json, digest, digest_bytes


def test_canonical_json_and_hash_vectors() -> None:
    assert canonical_json({"z": [True, None, 2], "a": "ā"}) == '{"a":"ā","z":[true,null,2]}'
    assert digest({}) == "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    assert digest_bytes(b"abc") == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})


@pytest.mark.parametrize("value", [1.0, {"a": [0.5]}, {1.0: "key"}, (float("nan"),), float("inf")])
def test_floats_are_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="float"):
        canonical_json(value)
