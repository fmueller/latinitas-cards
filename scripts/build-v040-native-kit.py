"""Prepare an offline synthetic kit; never open Anki or certify native acceptance.

Run from the checkout: uv run python scripts/build-v040-native-kit.py NEW_DIRECTORY
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from latinitas_cards.cli import app
from latinitas_cards.managed_layouts import layout_version

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/fixtures/goldens/v0-4-0/inputs"
STAMP = "2026-01-01T00:00:00Z"
RUNNER = CliRunner()


def write(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def cli(*args: str) -> dict[str, Any]:
    result = RUNNER.invoke(app, list(args))
    if result.exit_code:
        raise RuntimeError(f"{' '.join(args)}\n{result.output}")
    report = json.loads(result.stdout) if "--json" in args else {}
    return report["result"] if report.get("format") == "latinitas-cli-result" else report


def retain_current_decisions(path: Path, fingerprints: set[str]) -> set[str]:
    previous = json.loads(path.read_text())
    shutil.copyfile(path, str(path) + ".archived")
    kept = [decision for decision in previous["decisions"] if decision["base_fingerprint"] in fingerprints]
    write(path, previous | {"decisions": kept})
    return {decision["base_fingerprint"] for decision in kept}


def renew(source: str, *options: str) -> None:
    """Archive legacy evidence and remove only obsolete fixture decisions.

    This is a bounded, reviewed fixture migration, not a general prune command.
    New decisions preserve each old claim's accepted/withheld status; translations
    are withheld unless separately supplied in the English recipe fixture.
    """
    report = cli("claims", "list", source, *options, "--json")
    kept = retain_current_decisions(Path(source + ".claims.json"), {c["fingerprint"] for c in report["claims"]})
    # The reviewed fixture's statuses are captured before the language change.
    statuses = json.loads(Path(source + ".statuses.json").read_text())
    items = []
    for claim in report["claims"]:
        if claim["fingerprint"] in kept:
            continue
        key = json.dumps([claim["identity"], claim["feature"], claim["kind"]])
        status = statuses[key]
        if claim["kind"] == "translation":
            status = "withheld"
        items.append(
            {
                "fingerprint": claim["fingerprint"],
                "status": status,
                "reviewer": "native-kit stipulated fixture",
                "reason": "Rechecked unchanged synthetic evidence in English; no calibrated analysis",
            }
        )
    write("decisions.json", items)
    cli("claims", "record", source, *options, "--batch", "decisions.json", "--json")


def remember(source: str, *options: str) -> None:
    report = cli("claims", "list", source, *options, "--json")
    write(
        source + ".statuses.json",
        {json.dumps([c["identity"], c["feature"], c["kind"]]): c["status"] for c in report["claims"]},
    )


def setup(output: Path, version: str) -> None:
    layout = layout_version(version)
    directory = output / f"setup-{version}"
    directory.mkdir()
    (directory / "fields.txt").write_text("\n".join(layout.fields) + "\n")
    (directory / "style.css").write_text(layout.css)
    for slot, (front, back) in zip(layout.slots, layout.templates, strict=True):
        (directory / f"{slot.ordinal}-front.html").write_text(front)
        (directory / f"{slot.ordinal}-back.html").write_text(back)
    write(directory / "templates.json", [{"ordinal": s.ordinal, "name": s.template_name} for s in layout.slots])


def corpus(output: Path, facts: dict[str, Any]) -> None:
    Path("latin.txt").write_text(facts["text"] + "\n")
    cli(
        "corpus",
        "normalize",
        "latin.txt",
        "--adapter",
        "text",
        "--corpus-id",
        "native-kit",
        "--output",
        "latin.jsonl",
        "--json",
    )
    cli("annotations", "segment", "latin.jsonl", "--timestamp", STAMP, "--json")
    passage = next(r for r in map(json.loads, Path("latin.jsonl").read_text().splitlines()) if r["record"] == "passage")
    segment = next(
        r
        for r in map(json.loads, Path("latin.annotations.jsonl").read_text().splitlines())
        if r.get("layer") == "segmentation"
    )
    anchor = {
        "passage_id": passage["passage_id"],
        "passage_hash": passage["passage_hash"],
        "segmentation_id": segment["annotation_id"],
    }
    proposals = [("morphology", [7, 13], facts["morphology"])]
    for language, translations in facts["translations"].items():
        proposals.extend(
            ("translation", span, {"language": language, "text": translations[key]})
            for key, span in (("word", [7, 13]), ("phrase", [7, 19]), ("sentence", [0, 20]))
        )
    write(
        "facts.patch.json",
        {
            "format": "latinitas-annotation-patch",
            "version": 1,
            "operations": [
                {
                    "operation": "add",
                    "layer": layer,
                    "anchor": anchor | {"span": span},
                    "value": value,
                    "producer": {"kind": "human", "name": "native-kit-fixture", "version": "1"},
                }
                for layer, span, value in proposals
            ],
        },
    )
    cli(
        "annotations",
        "plan",
        "facts.patch.json",
        "--corpus",
        "latin.jsonl",
        "--sidecar",
        "latin.annotations.jsonl",
        "--output",
        "facts.plan.json",
        "--json",
    )
    plan = json.loads(Path("facts.plan.json").read_text())
    operations = [arg for op in plan["operations"] for arg in ("--operation", op["operation_id"])]
    cli(
        "annotations",
        "approve",
        "facts.plan.json",
        *operations,
        "--actor",
        "human:synthetic-fixture",
        "--review",
        "Stipulated seven fixture assertions only",
        "--output",
        "facts.receipt.json",
        "--json",
    )
    cli(
        "annotations",
        "apply",
        "facts.plan.json",
        "--corpus",
        "latin.jsonl",
        "--sidecar",
        "latin.annotations.jsonl",
        "--receipt",
        "facts.receipt.json",
        "--timestamp",
        STAMP,
        "--json",
    )
    report = cli("claims", "list", "latin.jsonl", "--annotations", "latin.annotations.jsonl", "--json")
    write(
        "corpus-decisions.json",
        [
            {
                "fingerprint": c["fingerprint"],
                "status": "accepted",
                "reviewer": "native-kit stipulated fixture",
                "reason": "Checked fixed synthetic assertion, not corpus evidence",
            }
            for c in report["claims"]
        ],
    )
    cli(
        "claims",
        "record",
        "latin.jsonl",
        "--annotations",
        "latin.annotations.jsonl",
        "--batch",
        "corpus-decisions.json",
        "--json",
    )
    cli("claims", "inspect", "latin.jsonl", "--annotations", "latin.annotations.jsonl", "--check", "--json")
    for language in ("de", "en"):
        for appearance in ("light", "dark", "auto", "update"):
            profile = {
                "format": "latinitas-corpus-profile",
                "version": 1,
                "profile_id": f"native-{language}",
                "corpora": ["native-kit"],
                "target_language": language,
                "note_type": "Latinitas Native Corpus",
                "deck": f"Native v040::Corpus {language}",
                "recipes": [
                    {"recipe": f"corpus_{r}", "budget": 1}
                    for r in ("lexical_recognition", "guided_analysis", "phrase_translation", "sentence_translation")
                ],
                "morphology": {
                    "appearance": "dark" if appearance == "update" else appearance,
                    "theme": "muted",
                    "terminology": {"version": 1, "language": "user"},
                },
            }
            write("corpus-profile.json", profile)
            cli("corpus-cards", "profile", "validate", "corpus-profile.json", "--json")
            selection = f"selection-{language}.json"
            cli(
                "corpus-cards",
                "select",
                "corpus-profile.json",
                "--corpus",
                "latin.jsonl",
                "--selection",
                selection,
                "--report",
                "candidates.json",
                "--json",
            )
            filename = f"corpus-{language}-{appearance}.csv"
            cli(
                "corpus-cards",
                "export",
                "corpus-profile.json",
                "--corpus",
                "latin.jsonl",
                "--selection",
                selection,
                "--output",
                filename,
                "--json",
            )
            cli(
                "corpus-cards",
                "inspect",
                "corpus-profile.json",
                "--corpus",
                "latin.jsonl",
                "--selection",
                selection,
                "--check",
                filename,
                "--json",
            )
            shutil.copyfile(filename, output / filename)


def english(output: Path, facts: dict[str, Any]) -> None:
    # Copy only exporter inputs and their identity/review sidecars, not unrelated goldens.
    inputs = {
        "principal-parts/de-light-user": (
            "source.csv",
            "source.csv.latinitas.json",
            "profile.json",
            "source.csv.claims.json",
        ),
        "phrases": (
            "inputs.json",
            "inputs.json.claims.json",
            "profile.json",
            "deck.csv",
            "deck.csv.latinitas.json",
            "selected.json",
            "recipes.json",
            "recipes.json.claims.json",
        ),
        "form-parsing": ("parsing.json", "parsing.json.claims.json"),
    }
    for directory, filenames in inputs.items():
        destination = Path("deck") / directory
        destination.mkdir(parents=True)
        for filename in filenames:
            shutil.copyfile(GOLDEN / directory / filename, destination / filename)
    principal = "deck/principal-parts/de-light-user/source.csv"
    profile = "deck/principal-parts/de-light-user/profile.json"
    phrase = "deck/phrases/inputs.json"
    recipes = "deck/phrases/recipes.json"
    selected = "deck/phrases/selected.json"
    parsing = "deck/form-parsing/parsing.json"
    remember(principal, "--profile", profile)
    remember(phrase, "--phrase-input")
    remember(parsing)
    for path in (profile, "deck/phrases/profile.json"):
        data = json.loads(Path(path).read_text())
        data.update(language_tag="en", target_deck="Native v040::English")
        if path != profile:
            data["generated_note_type"] = "Latinitas Native Phrase"
        data["morphology"]["terminology"]["language"] = "user"
        write(path, data)
    data = json.loads(Path(parsing).read_text())
    data["profile"].update(language_tag="en", target_deck="Native v040::English")
    data["profile"]["morphology"]["terminology"]["language"] = "user"
    write(parsing, data)
    data = json.loads(Path(recipes).read_text())
    for exercise in data["exercises"]:
        exercise["reference"]["language"] = "en"
        reading = exercise["reference"]["readings"][0]
        reading.update(
            translations=facts["phrase_reference"]["translations"],
            contributions={"object": facts["phrase_reference"]["contribution"]},
            explanation=facts["phrase_reference"]["explanation"],
        )
    write(recipes, data)
    renew(principal, "--profile", profile)
    renew(phrase, "--phrase-input")
    # Recipe claims use the phrase inputs as the source, but own a separate sidecar.
    report = cli("claims", "list", phrase, "--selections", selected, "--recipes", recipes, "--json")
    kept = retain_current_decisions(Path(recipes + ".claims.json"), {c["fingerprint"] for c in report["claims"]})
    write(
        "phrase-decisions.json",
        [
            {
                "fingerprint": c["fingerprint"],
                "status": "accepted",
                "reviewer": "native-kit stipulated fixture",
                "reason": "Checked cum amico and explicit English reference",
            }
            for c in report["claims"]
            if c["fingerprint"] not in kept
        ],
    )
    cli(
        "claims",
        "record",
        phrase,
        "--selections",
        selected,
        "--recipes",
        recipes,
        "--batch",
        "phrase-decisions.json",
        "--json",
    )
    renew(parsing)
    for appearance in ("light", "dark", "auto"):
        for path in (profile, "deck/phrases/profile.json", parsing):
            data = json.loads(Path(path).read_text())
            target = data["profile"] if path == parsing else data
            target["morphology"]["appearance"] = appearance
            write(path, data)
        commands = {
            "principal": ("generate", "--input", principal, "--profile", profile, "--approve-fresh-import", "--output"),
            "phrase": ("phrases", "export", phrase, selected, recipes),
            "parsing": ("form-parsing", "export", parsing),
        }
        for family, command in commands.items():
            filename = f"english-{family}-{appearance}.csv"
            flags = ("--approve-fresh-import",) if family == "parsing" else ()
            cli(*command, filename, *flags)
            shutil.copyfile(filename, output / filename)


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output / "reproduction"
    work.mkdir()
    facts = json.loads((ROOT / "tests/fixtures/native-v040/facts.json").read_text())
    (output / "fixture.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
    previous = Path.cwd()
    try:
        os.chdir(work)
        corpus(output, facts)
        english(output, facts)
    finally:
        os.chdir(previous)
    # Keep review/annotation provenance, but no incidental lock files or old goldens.
    for lock in work.rglob("*.lock"):
        lock.unlink()
    for version in ("corpus-v1", "3", "phrase-v1", "parsing-v1"):
        setup(output, version)
    shutil.copyfile(ROOT / "docs/native-v040-acceptance.md", output / "CHECKLIST.md")
    write(
        output / "manifest.json",
        {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    print("Offline v0.4.0 kit prepared; native Linux Anki Desktop and AnkiMobile unverified")


if __name__ == "__main__":
    main()
