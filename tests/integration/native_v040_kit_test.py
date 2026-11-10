"""Offline kit contract; no native-client acceptance is implied."""

import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

from latinitas_cards.managed_layouts import layout_version

ROOT = Path(__file__).resolve().parents[2]


def rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    columns = next(line.removeprefix("#columns:") for line in lines if line.startswith("#columns:"))
    return list(
        csv.DictReader(
            io.StringIO("\n".join(line for line in lines if not line.startswith("#"))), fieldnames=columns.split(",")
        )
    )


def test_offline_kit_retains_update_identity_and_covers_native_matrix(tmp_path: Path) -> None:
    output = tmp_path / "kit"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build-v040-native-kit.py"), str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for language in ("de", "en"):
        initial = rows(output / f"corpus-{language}-light.csv")
        assert len(initial) == 4
        assert {r["Language"] for r in initial} == {language}
        assert all("Personal Notes" not in r for r in initial)
        enabled = {key for r in initial for key, value in r.items() if key.endswith("Enabled") and value == "1"}
        assert enabled == {f"Corpus{name}Enabled" for name in ("Lexical", "Analysis", "Phrase", "Sentence")}
        assert all(sum(r[key] == "1" for key in enabled) == 1 for r in initial)
        for appearance in ("dark", "auto", "update"):
            updated = rows(output / f"corpus-{language}-{appearance}.csv")
            assert [r["LatinitasID"] for r in updated] == [r["LatinitasID"] for r in initial]
            assert [[r[k] for k in sorted(enabled)] for r in updated] == [
                [r[k] for k in sorted(enabled)] for r in initial
            ]
            assert any(a != b for a, b in zip(initial, updated, strict=True))
    family_slots = {
        "principal": {
            f"{recipe}{role}Enabled"
            for recipe in ("Completion", "Recognition")
            for role in ("Present", "Infinitive", "Perfect", "Supine")
        },
        "phrase": {"PhraseTranslationEnabled", "PhraseRecognitionEnabled"},
        "parsing": {"ParsingEnabled"},
    }
    for family, slots in family_slots.items():
        initial = rows(output / f"english-{family}-light.csv")
        for appearance in ("light", "dark", "auto"):
            current = rows(output / f"english-{family}-{appearance}.csv")
            assert len(current) == (2 if family == "phrase" else 1)
            assert [r["LatinitasID"] for r in current] == [r["LatinitasID"] for r in initial]
            assert {k for r in current for k, v in r.items() if k.endswith("Enabled") and v == "1"} == slots
            assert [[r[k] for k in sorted(slots)] for r in current] == [[r[k] for k in sorted(slots)] for r in initial]
            assert all("Personal Notes" not in r for r in current)
            if family == "principal":
                assert "Present, 1st person singular" in current[0]["RecognitionPresentAnswer"]
                assert "Other stems: analysis withheld." in current[0]["RecognitionPresentAnswer"]
            elif family == "phrase":
                answers = " ".join(
                    r[k] for r in current for k in ("PhraseTranslationAnswer", "PhraseRecognitionAnswer")
                )
                assert "with a friend" in answers and "Case: Ablative" in answers
                assert "The preposition cum determines the case here." in answers
            else:
                assert "Case: Nominative" in current[0]["ParsingAnswer"]
                assert "Number: Plural" in current[0]["ParsingAnswer"]
                assert "Gender: Feminine" in current[0]["ParsingAnswer"]
    for version in ("corpus-v1", "3", "phrase-v1", "parsing-v1"):
        layout = layout_version(version)
        directory = output / f"setup-{version}"
        assert (directory / "fields.txt").read_text().splitlines() == list(layout.fields)
        assert (directory / "style.css").read_text() == layout.css
        assert json.loads((directory / "templates.json").read_text()) == [
            {"ordinal": s.ordinal, "name": s.template_name} for s in layout.slots
        ]
        for slot, (front, back) in zip(layout.slots, layout.templates, strict=True):
            assert (directory / f"{slot.ordinal}-front.html").read_text() == front
            assert (directory / f"{slot.ordinal}-back.html").read_text() == back
    deck = output / "reproduction/deck"
    assert {path.name for path in (deck / "principal-parts").iterdir()} == {"de-light-user"}
    assert not (deck / "corpus").exists()
    assert not list(deck.rglob("decisions*.json"))
    archived = list(deck.rglob("*.claims.json.archived"))
    assert len(archived) == 4
    golden = ROOT / "tests/fixtures/goldens/v0-4-0/inputs"
    for path in archived:
        original = golden / path.relative_to(deck).with_suffix("")
        assert path.read_bytes() == original.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text())
    inventory = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert set(manifest) == inventory.keys() - {"manifest.json"}
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    checklist = (output / "CHECKLIST.md").read_text()
    assert "AnkiMobile remains unverified" in checklist
    assert "candidate-bound results are recorded separately" in checklist
    assert "release-v0.4.0.md#candidate-bound-verification-2026-10-31" in checklist
    assert "out of scope for v0.4.0 release verification and readiness" in checklist
    assert "not a compatibility or schedule-preservation claim" in checklist
    second = tmp_path / "second-kit"
    repeated = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build-v040-native-kit.py"), str(second)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert inventory == {str(p.relative_to(second)): p.read_bytes() for p in second.rglob("*") if p.is_file()}
    rejected = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build-v040-native-kit.py"), str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0 and "FileExistsError" in rejected.stderr
    assert inventory == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
