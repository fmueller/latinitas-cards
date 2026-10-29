"""Any deliberate golden update must also explain its manifest update."""

import hashlib
import json
from pathlib import Path


def test_capture_manifest_covers_exact_golden_bytes() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures/goldens"
    capture = json.loads((root / "CAPTURE").read_bytes())
    assert len(capture["commit"]) == 40
    actual = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("legacy-vulgate", "claims-v1", "v0-4-0")
        for path in (root / directory).rglob("*")
        if path.is_file()
    }
    assert actual
    assert actual == capture["sha256"]
