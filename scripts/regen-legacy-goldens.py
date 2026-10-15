#!/usr/bin/env python3
"""Deliberate baseline capture and read-only comparison harness for T-102.

Run with uv run; tests import this file without invoking main. Only this script
writes golden directories. Application subprocesses always work in disposable /tmp.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tests/fixtures/goldens"
LEGACY = GOLDENS / "legacy-vulgate"
CLAIMS = GOLDENS / "claims-v1"


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def compare(path: Path, actual: bytes, capture: bool) -> None:
    if capture:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(actual)
    else:
        assert actual == path.read_bytes(), f"golden mismatch: {path}"


def invoke(argv: list[str], work: Path, expected: int = 0) -> bytes:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(work),
        "COLUMNS": "120",
        "LINES": "40",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "TZ": "UTC",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    if "MUTANT_UNDER_TEST" in os.environ:
        env["MUTANT_UNDER_TEST"] = os.environ["MUTANT_UNDER_TEST"]
        # mutmut changes the parent sys.path, which a new interpreter cannot
        # inherit. Expose that checkout's package through the disposable cwd;
        # retain the exact -m invocation and fixed environment contract.
        package = work / "latinitas_cards"
        if not package.exists():
            package.symlink_to(ROOT / "src/latinitas_cards", target_is_directory=True)
    result = subprocess.run(
        [sys.executable, "-m", "latinitas_cards", *argv],
        cwd=work,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert result.returncode == expected, (argv, result.returncode, result.stdout.decode())
    stdout = result.stdout
    for path in {str(work), str(work.resolve())}:
        raw = path.encode()
        assert not any(raw in line and len(line.decode()) > 120 for line in stdout.splitlines()), (
            "nonportable Rich wrapping",
            argv,
            stdout,
        )
        stdout = stdout.replace(raw, b"<TMP>")
    return stdout


def legacy_case(case: dict[str, Any], *, capture: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="lg", dir="/tmp") as directory:
        work = Path(directory)
        shutil.copytree(LEGACY / "inputs", work, dirs_exist_ok=True)
        if case.get("apkg"):
            sys.path.insert(0, str(ROOT / "tests/unit"))
            from cli_test import _create_colpkg_with_anki2

            _create_colpkg_with_anki2(work / "input.apkg", [["amo", "I love", ""], ["sum", "I am", ""]])
        if case.get("full_vulgate"):
            shutil.copyfile(ROOT / "data/lat-clementine.usfx.xml", work / "full.xml")
        before = {str(p.relative_to(work)): p.read_bytes() for p in work.rglob("*") if p.is_file()}
        argv = [a.replace("{FIX}", str(work)).replace("{TMP}", str(work)) for a in case["argv"]]
        stdout = invoke(argv, work, case["exit_code"])
        compare(LEGACY / case["stdout"], stdout, capture)
        for output, golden in case["outputs"].items():
            actual = (work / output).read_bytes()
            if case.get("apkg"):
                with zipfile.ZipFile(work / output) as archive:
                    archive.extract("collection.anki2", work)
                with sqlite3.connect(work / "collection.anki2") as connection:
                    actual = encoded(connection.execute("SELECT flds FROM notes ORDER BY id").fetchall())
                (work / "collection.anki2").unlink()
            elif case.get("digest"):
                actual = encoded(
                    {
                        "sha256": hashlib.sha256(actual).hexdigest(),
                        "bytes": len(actual),
                        "rows": len(list(csv.reader(io.StringIO(actual.decode())))) - 1,
                    }
                )
            compare(LEGACY / golden, actual, capture)
        after = {str(p.relative_to(work)): p.read_bytes() for p in work.rglob("*") if p.is_file()}
        changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
        assert changed <= set(case["outputs"]) | set(case["exclusions"]), ("unexpected writes", case["name"], changed)


def claim_case(family: str, *, capture: bool) -> None:
    fixture = CLAIMS / family
    with tempfile.TemporaryDirectory(prefix="lg", dir="/tmp") as directory:
        work = Path(directory)
        shutil.copytree(fixture, work, dirs_exist_ok=True)
        config = json.loads((work / "cases.json").read_bytes())
        # Replay from a pristine input, not from a previously recorded decision.
        for replay in config["replay"]:
            (work / replay["sidecar"]).unlink(missing_ok=True)
        for replay in config["replay"]:
            invoke(["claims", "record", *replay["args"], "--batch", replay["batch"]], work)
            compare(fixture / replay["sidecar"], (work / replay["sidecar"]).read_bytes(), capture)
            compare(fixture / f"{replay['sidecar']}.lock", (work / f"{replay['sidecar']}.lock").read_bytes(), capture)
        # Exports must consume the committed v1 bytes, independently of replay output.
        for replay in config["replay"]:
            shutil.copyfile(fixture / replay["sidecar"], work / replay["sidecar"])
        for export in config["exports"]:
            (work / export["output"]).unlink(missing_ok=True)
        before = {str(p.relative_to(work)): p.read_bytes() for p in work.rglob("*") if p.is_file()}
        for export in config["exports"]:
            invoke(export["argv"], work)
            compare(fixture / export["golden"], (work / export["output"]).read_bytes(), capture)
            for output, golden in export.get("artifacts", {}).items():
                compare(fixture / golden, (work / output).read_bytes(), capture)
        after = {str(p.relative_to(work)): p.read_bytes() for p in work.rglob("*") if p.is_file()}
        changed = {p for p in after if p not in before or after[p] != before[p]}
        compared = {e["output"] for e in config["exports"]}
        compared.update(output for e in config["exports"] for output in e.get("artifacts", {}))
        assert changed <= compared | set(config["exclusions"]), changed


def seed_legacy() -> None:
    inputs = LEGACY / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    texts = {
        "input.csv": "Front,Back,Notes\namo,I love,1\nsum,I am,2\nverbum,a word,3\ntitulus,a title,4\n",
        "existing.csv": "Front,Back,Notes,VulgataCloze\namo,I love,1,old amo\nsum,I am,2,old sum\n",
        "forms.csv": "form,meaning,extra\namo,I love,keep\nsum,I am,other\nverbum,the word,last\n",
        "missing-front.csv": "Word,Back,Notes\namo,I love,1\nsum,I am,2\n",
        "sample.usfx.xml": """<usfx><book id="ZZZ"><c id="2"/>
<v id="1"/>amo amo sum. <f>nota</f> <wj>verbum</wj> est.<ve/>
<v id="1"/>amo duplicatum.<ve/><v id="2"/><ve/>
<v id="3a"/>sum et amas verba.<ve/></book><book id="AAA"/><c n="1"/>
<v n="1">verbum amo.</v><h>titulus</h><ve/>
<v id="2"/>sum verbum.<ve/></usfx>""",
        "empty.usfx.xml": "<usfx><book id='GEN'><c id='1'/></book></usfx>",
        "malformed.usfx.xml": "<usfx><book>",
        "sample.txt": "amo amo sum.\nverbum est bonum.\nsum et amo.\n",
        "parallel.csv": (
            "la,translation_en,translation_de\namo sum,I love and am,ich liebe und bin\n"
            "verbum est,the word is,das Wort ist\n"
        ),
        "stopwords.txt": "sum\n",
        "word-forms.txt": "amo,amas,amat\n",
        "lemmas.txt": "verbum verba\n",
    }
    words = [
        "amo",
        "sum",
        "verbum",
        "deus",
        "terra",
        "caelum",
        "lux",
        "aqua",
        "homo",
        "vita",
        "pater",
        "filius",
        "spiritus",
        "bonus",
        "malum",
        "rex",
        "dominus",
        "populus",
        "cor",
        "manus",
        "dies",
        "nox",
        "annus",
        "via",
        "nomen",
        "vox",
        "pax",
        "lex",
        "amor",
        "mors",
        "panis",
        "vinum",
        "corpus",
        "sanguis",
        "anima",
        "mundus",
        "gloria",
        "gratia",
        "fides",
        "spes",
    ]
    assert len(words) == 40
    texts["vulgate-words.csv"] = "Front,Back,Notes\n" + "".join(f"{w},meaning of {w},fixture\n" for w in words)
    for name, text in texts.items():
        (inputs / name).write_text(text, encoding="utf-8")
    cases: list[dict[str, Any]] = []

    def add(name: str, argv: list[str], *, exit_code: int = 0, output: str | None = None, **extra: Any) -> None:
        cases.append(
            {
                "name": name,
                "argv": argv,
                "exit_code": exit_code,
                "stdout": f"{name}.stdout",
                "outputs": {output: f"{name}.csv"} if output else {},
                "exclusions": {},
                **extra,
            }
        )

    preview = ["preview", "--input", "{FIX}/input.csv", "--usfx", "{FIX}/sample.usfx.xml", "--limit", "5"]
    add("preview", preview)
    add("preview-multi", [*preview, "--multi-cloze-per-verse", "--max-examples", "3"])
    generate = [
        "generate",
        "--input",
        "{FIX}/input.csv",
        "--usfx",
        "{FIX}/sample.usfx.xml",
        "--output",
        "{TMP}/out.csv",
    ]
    add("generate", generate, output="out.csv")
    add(
        "generate-append",
        [a.replace("out.csv", "existing.csv").replace("input.csv", "existing.csv") for a in generate] + ["--append"],
        outputs={"existing.csv": "generate-append.csv"},
    )
    add(
        "generate-filters",
        [
            *generate,
            "--stopwords",
            "{FIX}/stopwords.txt",
            "--word-forms",
            "{FIX}/word-forms.txt",
            "--lemmas",
            "{FIX}/lemmas.txt",
            "--ignore-pattern",
            "^titulus$",
            "--max-examples",
            "5",
        ],
        output="out.csv",
    )
    cloze = ["cloze", "--input", "{FIX}/forms.csv", "--output", "{TMP}/out.csv", "--corpus"]
    for name, corpus, options in [
        ("usfx", "sample.usfx.xml", ["--max-examples", "5"]),
        ("txt", "sample.txt", []),
        ("parallel", "parallel.csv", ["--latin-column", "la", "--parallel-mode", "include"]),
        ("translation", "parallel.csv", ["--translation-lang", "en"]),
        ("prompt", "parallel.csv", ["--parallel-mode", "prompt"]),
        ("easy", "sample.txt", ["--difficulty", "easy"]),
        ("include", "sample.txt", ["--include-column", "form", "--include-column", "CorpusCloze"]),
        ("exclude", "sample.txt", ["--exclude-column", "extra"]),
        ("mixed", "sample.usfx.xml", ["--corpus", "{FIX}/sample.txt"]),
    ]:
        add(f"cloze-{name}", [*cloze, f"{{FIX}}/{corpus}", *options], output="out.csv")
    for name, source, corpus, code in [
        ("validate", "input.csv", "sample.usfx.xml", 0),
        ("validate-empty", "input.csv", "empty.usfx.xml", 1),
        ("validate-malformed", "input.csv", "malformed.usfx.xml", 1),
        ("validate-front", "missing-front.csv", "sample.usfx.xml", 1),
    ]:
        add(name, ["validate", "--input", f"{{FIX}}/{source}", "--usfx", f"{{FIX}}/{corpus}"], exit_code=code)
    add(
        "generate-apkg",
        [a.replace("input.csv", "input.apkg").replace("out.csv", "out.apkg") for a in generate],
        apkg=True,
        outputs={"out.apkg": "generate-apkg.flds.json"},
    )
    full_generate = [
        a.replace("input.csv", "vulgate-words.csv").replace("sample.usfx.xml", "full.xml") for a in generate
    ]
    add(
        "vulgate-generate",
        [*full_generate, "--max-examples", "2"],
        full_vulgate=True,
        digest=True,
        outputs={"out.csv": "vulgate-generate.digest.json"},
    )
    add(
        "vulgate-preview",
        [a.replace("input.csv", "vulgate-words.csv").replace("sample.usfx.xml", "full.xml") for a in preview],
        full_vulgate=True,
    )
    (LEGACY / "cases.json").write_bytes(encoded(cases))


def seed_claims() -> None:
    from latinitas_cards.profile import DeckProfile, SourceIdentityConfig

    sys.path.insert(0, str(ROOT / "tests/unit"))
    from phrase_inputs_test import fixture, saved_example
    from phrase_recipes_test import exercise
    from review_commands_test import parsing_source

    for family in ("principal-parts", "form-parsing", "phrases"):
        destination = CLAIMS / family
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lg", dir="/tmp") as directory:
            work = Path(directory)
            if family == "principal-parts":
                (work / "source.csv").write_text(
                    'ID,Lemma,Forms,Meaning\na,amō,"amō, amāre, amāvī, amātum",love\n'
                    'l,loquor,"loquor, loquī, locūtus sum",speak\ns,sum,"sum, esse, fuī",be\n'
                )
                profile = DeckProfile.default(
                    note_type="Latin",
                    lexical_entry_field="Lemma",
                    principal_parts_field="Forms",
                    meaning_field="Meaning",
                    source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
                    separators=(",",),
                    selected_recipes=("principal_part_completion", "principal_part_recognition"),
                )
                profile.save(work / "profile.json")
                (work / "source.csv.latinitas.json").write_bytes(
                    encoded(
                        {
                            "schema_version": 2,
                            "source_columns": ["ID", "Lemma", "Forms", "Meaning"],
                            "source_scope": "legacy-golden-principal-parts-v1",
                            "entries": [],
                        }
                    )
                )
                args = ["source.csv", "--profile", "profile.json"]
                exports = [
                    {
                        "argv": [
                            "generate",
                            "--input",
                            "source.csv",
                            "--profile",
                            "profile.json",
                            "--recipe",
                            recipe,
                            "--output",
                            f"{recipe}.csv",
                            "--approve-fresh-import",
                        ],
                        "output": f"{recipe}.csv",
                        "golden": f"{recipe}.csv",
                        "artifacts": {
                            "source.csv.latinitas.json": f"{recipe}.manifest.json",
                            "source.csv.latinitas-cards.json": f"{recipe}.checkpoint.json",
                        },
                    }
                    for recipe in ("principal_part_completion", "principal_part_recognition")
                ]
            elif family == "form-parsing":
                parsing_source(work)
                args = ["cases.json"]
                # The replay metadata uses a different filename from the contextual input.
                (work / "cases.json").rename(work / "parsing.json")
                args = ["parsing.json"]
                exports = [
                    {
                        "argv": ["form-parsing", "export", "parsing.json", "out.csv", "--approve-fresh-import"],
                        "output": "out.csv",
                        "golden": "out.csv",
                    }
                ]
            else:
                fixture(work)
                source, selections = saved_example(work)
                candidates = json.loads(invoke(["phrases", "generate", source.name, selections.name, "--json"], work))[
                    "candidates"
                ]
                chosen = exercise().model_copy(update={"candidate_key": candidates[0]["candidate_key"]})
                (work / "recipes.json").write_bytes(
                    encoded({"version": 1, "max_exercises": 1, "exercises": [chosen.model_dump(mode="json")]})
                )
                args = ["inputs.json", "--selections", "selected.json", "--recipes", "recipes.json"]
                exports = [
                    {
                        "argv": ["phrases", "export", "inputs.json", "selected.json", "recipes.json", "out.csv"],
                        "output": "out.csv",
                        "golden": "out.csv",
                    }
                ]
            replays = []
            groups = [["inputs.json", "--phrase-input"], args] if family == "phrases" else [args]
            for index, group in enumerate(groups):
                claims = json.loads(invoke(["claims", "list", *group, "--json"], work))["claims"]
                decisions = [
                    {
                        "fingerprint": c["fingerprint"],
                        "status": "accepted" if c["value"] is not None else "withheld",
                        "reviewer": "Golden fixture reviewer",
                        "reason": "Individually checked synthetic characterization evidence",
                    }
                    for c in claims
                ]
                batch = f"decisions-{index}.json"
                (work / batch).write_bytes(encoded(decisions))
                sidecar = (
                    "recipes.json.claims.json" if family == "phrases" and index == 1 else f"{group[0]}.claims.json"
                )
                (work / sidecar).unlink(missing_ok=True)
                invoke(["claims", "record", *group, "--batch", batch], work)
                replays.append({"args": group, "batch": batch, "sidecar": sidecar})
            for path in work.iterdir():
                if path.is_file() and not path.name.endswith(".lock") and path.name != "decisions.json":
                    shutil.copyfile(path, destination / path.name)
            (destination / "cases.json").write_bytes(encoded({"replay": replays, "exports": exports, "exclusions": {}}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-know-this-rewrites-characterization-goldens", action="store_true", required=True)
    parser.add_argument("--check-only", action="store_true")
    options = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    capture_file = GOLDENS / "CAPTURE"
    if capture_file.exists() and json.loads(capture_file.read_bytes())["commit"] != head:
        parser.error("HEAD differs from the recorded capture commit; use the original baseline checkout")
    if subprocess.check_output(["git", "diff", "HEAD", "--", "src/"], cwd=ROOT):
        parser.error("capture requires unmodified source")
    if options.check_only:
        return
    seed_legacy()
    seed_claims()
    for case in json.loads((LEGACY / "cases.json").read_bytes()):
        legacy_case(case, capture=True)
    for family in ("principal-parts", "form-parsing", "phrases"):
        claim_case(family, capture=True)
    manifest = {
        str(p.relative_to(GOLDENS)): hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in (LEGACY, CLAIMS)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    capture_file.write_bytes(encoded({"commit": head, "sha256": manifest}))


if __name__ == "__main__":
    main()
