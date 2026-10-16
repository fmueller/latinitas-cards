"""Anchor precedence and lifecycle are independent, without rewriting history."""

from pathlib import Path

from latinitas_cards.annotation_sidecar import Annotation, Retraction, annotation_id, read_sidecar
from latinitas_cards.annotation_state import annotation_states, dependents

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/annotations"


def test_states_precedence_lifecycle_and_dependents() -> None:
    segmentation = list(read_sidecar(FIXTURES / "ordinal.annotations.jsonl"))[1]
    assert isinstance(segmentation, Annotation)
    raw = segmentation.model_dump(mode="json")
    raw.update(layer="morphology", value={"tagset": "fixture"})
    raw["anchor"]["segmentation_id"] = segmentation.annotation_id
    raw["anchor"]["span"] = [0, 2]
    raw["annotation_id"] = annotation_id(raw)
    token = Annotation.model_validate(raw)
    replacement = segmentation.model_dump(mode="json")
    replacement["supersedes"] = segmentation.annotation_id
    replacement["annotation_id"] = annotation_id(replacement)
    newer = Annotation.model_validate(replacement)
    records = [segmentation, token, newer]
    hashes = {segmentation.anchor.passage_id: segmentation.input_hash}
    states = annotation_states(hashes, records)
    assert states[segmentation.annotation_id].state == "current"
    assert states[segmentation.annotation_id].superseded is True
    assert states[token.annotation_id].state == "orphaned"
    assert states[newer.annotation_id].state == "current"
    assert dependents(records, segmentation.annotation_id) == [token]
    # Only later records flag a proposal; an earlier forward reference cannot.
    forward = annotation_states(hashes, [newer, segmentation, token])
    assert forward[segmentation.annotation_id].superseded is False
    assert forward[token.annotation_id].state == "current"
    assert annotation_states({}, records)[token.annotation_id].state == "unmatched"
    assert annotation_states(dict.fromkeys(hashes, "sha256:" + "0" * 64), records)[token.annotation_id].state == "stale"
    assert annotation_states(hashes, [token])[token.annotation_id].state == "orphaned"
    assert annotation_states(hashes, [segmentation, token])[token.annotation_id].state == "current"
    retraction = segmentation.model_dump(mode="json", exclude={"value"})
    retraction.update(record="retraction", retracts=segmentation.annotation_id)
    retraction["annotation_id"] = annotation_id(retraction)
    retract = Retraction.model_validate(retraction)
    states = annotation_states(hashes, [segmentation, token, retract])
    assert states[segmentation.annotation_id].retracted is True
    assert states[token.annotation_id].state == "orphaned"
    assert states[retract.annotation_id].state == "current"
