"""Replay v1 decisions and exports independently of newly written sidecars."""

import runpy
from pathlib import Path

import pytest

from latinitas_cards.claim_sidecar import load_sidecar

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("family", ["principal-parts", "form-parsing", "phrases"])
def test_claim_v1_goldens(family: str) -> None:
    harness = runpy.run_path(str(ROOT / "scripts/regen-legacy-goldens.py"))
    harness["claim_case"](family, capture=False)


def test_v1_sidecar_bytes_remain_identical() -> None:
    paths = list((ROOT / "tests/fixtures/goldens/claims-v1").rglob("*.claims.json"))
    assert len(paths) == 4
    for path in paths:
        assert (load_sidecar(path).model_dump_json(indent=2) + "\n").encode() == path.read_bytes()
