"""Replay v1 decisions and exports independently of newly written sidecars."""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("family", ["principal-parts", "form-parsing", "phrases"])
def test_claim_v1_goldens(family: str) -> None:
    harness = runpy.run_path(str(ROOT / "scripts/regen-legacy-goldens.py"))
    harness["claim_case"](family, capture=False)
