import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from itertools import product
from pathlib import Path

from latinitas_cards.identity import derive_latinitas_id
from latinitas_cards.notes import GeneratedNote, GeneratedNoteProvenance, ManagedNoteContent, RecipeMetadata

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "representative-university-latin.apkg"
T014_TASK = ROOT / "planning" / "tasks" / "T-014-publish-v0-1-0.md"
EXPECTED_COLUMNS = (
    "LatinitasID",
    "Prompt",
    "Answer",
    "Tags",
    "Source ID",
    "Source Kind",
    "Source Location",
    "Source Path",
    "Recipe",
    "Exercise Key",
    "Recipe Version",
)
GPU_PACKAGES = (
    "cuda-bindings",
    "cuda-pathfinder",
    "cuda-toolkit",
    "nvidia-cublas",
    "nvidia-cuda-cupti",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-cudnn-cu13",
    "nvidia-cufft",
    "nvidia-cufile",
    "nvidia-curand",
    "nvidia-cusolver",
    "nvidia-cusparse",
    "nvidia-cusparselt-cu13",
    "nvidia-nccl-cu13",
    "nvidia-nvjitlink",
    "nvidia-nvshmem-cu13",
    "nvidia-nvtx",
)


def _invoke(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "latinitas_cards", *arguments],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def _export_rows(path: Path) -> tuple[str, list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    metadata = "".join(lines[:6])
    header = tuple(lines[5].removeprefix("#columns:").rstrip("\n").split(","))
    assert header == EXPECTED_COLUMNS
    raw_rows = list(csv.reader(io.StringIO("".join(lines[6:]))))
    assert all(len(row) == len(header) for row in raw_rows)
    rows = [dict(zip(header, row, strict=True)) for row in raw_rows]
    return metadata, rows


def _audit_rows(audit: str) -> list[tuple[str, str, str]]:
    rows = []
    for line in audit.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 4
        rows.append((cells[0].strip("`"), cells[1].strip("`"), cells[2]))
    return rows


def _scope_tokens(scope: str) -> set[str]:
    normalized = scope.replace("annotate-gpu", "")
    return {
        name
        for name in ("main", "dev", "annotate", "annotate-gpu")
        if (name == "annotate-gpu" and "annotate-gpu" in scope) or (name != "annotate-gpu" and name in normalized)
    }


def _locked_export_inventory(*options: str) -> set[tuple[str, str]]:
    exported = subprocess.run(
        ["uv", "export", "--locked", "--no-emit-project", "--format", "requirements-txt", *options],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    inventory = set()
    for line in exported.stdout.splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)", line)
        if match:
            inventory.add((match.group(1).replace("_", "-"), match.group(2)))
    return inventory


def test_sanitized_fixture_runs_assisted_profile_preview_and_repeatable_cli_export(tmp_path: Path) -> None:
    source = tmp_path / "representative.apkg"
    profile = tmp_path / "profile.json"
    first_output = tmp_path / "first.csv"
    second_output = tmp_path / "second.csv"
    changed_output = tmp_path / "changed.csv"
    shutil.copyfile(FIXTURE, source)
    source_before = source.read_bytes()

    setup = _invoke(
        [
            "setup",
            "--input",
            str(source),
            "--profile",
            str(profile),
            "--recipe",
            "principal_part_completion",
            "--recipe",
            "principal_part_recognition",
            "--non-interactive",
            "--confirm",
            "--json",
        ]
    )

    assert setup.returncode == 0
    setup_payload = json.loads(setup.stdout)
    assert setup_payload["status"] == "saved"
    effective_profile = setup_payload["effective_profile"]
    assert effective_profile["source_identity"] == {"strategy": "note_guid", "field": None}
    assert effective_profile["fields"] == {
        "lexical_entry_field": "Entry",
        "principal_parts_field": "Construction hints",
        "meaning_field": "German gloss",
    }
    assert effective_profile["principal_parts"] == {
        "roles": [
            "present_infinitive",
            "present_1s",
            "perfect_1s",
            "perfect_passive_participle",
        ],
        "separators": [","],
    }
    assert effective_profile["selected_recipes"] == [
        "principal_part_completion",
        "principal_part_recognition",
    ]
    assert effective_profile["language_tag"] == "de"

    preview = _invoke(["preview", "--input", str(source), "--profile", str(profile), "--limit", "2"])

    assert preview.returncode == 0
    assert "Generated: 24" in preview.stdout
    assert "Skipped: 2" in preview.stdout
    assert "Ambiguous: 0" in preview.stdout
    assert "Partizip Perfekt Passiv (PPP)" in preview.stdout
    assert "Output:" not in preview.stdout
    assert source.read_bytes() == source_before

    for output in (first_output, second_output):
        generated = _invoke(
            [
                "generate",
                "--input",
                str(source),
                "--profile",
                str(profile),
                "--output",
                str(output),
            ]
        )
        assert generated.returncode == 0
        assert "Output:" in generated.stdout

    assert first_output.read_bytes() == second_output.read_bytes()
    metadata, rows = _export_rows(first_output)
    assert metadata == (
        "#separator:Comma\n"
        "#html:true\n"
        "#notetype:Latinitas Principal Parts\n"
        "#deck:Latin::Latinitas\n"
        "#tags column:4\n"
        "#columns:LatinitasID,Prompt,Answer,Tags,Source ID,Source Kind,Source Location,Source Path,Recipe,"
        "Exercise Key,Recipe Version\n"
    )
    assert len(rows) == 24
    assert Counter(row["Recipe"] for row in rows) == {
        "principal_part_completion": 12,
        "principal_part_recognition": 12,
    }
    assert {row["Exercise Key"] for row in rows} == {
        "present_infinitive",
        "present_1s",
        "perfect_1s",
        "perfect_passive_participle",
    }
    expected_source_locations = {
        "fixture-guid-001": "note 1001",
        "fixture-guid-002": "note 1002",
        "fixture-guid-005": "note 1005",
    }
    expected_tuples = {
        (source_id, recipe, exercise_key)
        for source_id, recipe, exercise_key in product(
            expected_source_locations,
            ("principal_part_completion", "principal_part_recognition"),
            ("present_infinitive", "present_1s", "perfect_1s", "perfect_passive_participle"),
        )
    }
    assert {(row["Source ID"], row["Recipe"], row["Exercise Key"]) for row in rows} == expected_tuples
    assert len({row["LatinitasID"] for row in rows}) == len(rows)
    assert all(row["Source ID"] in expected_source_locations for row in rows)
    assert all(row["Source Kind"] == "apkg" for row in rows)
    assert all(expected_source_locations[row["Source ID"]] == row["Source Location"] for row in rows)
    assert all(row["Recipe Version"] == "1" for row in rows)
    assert all(row["LatinitasID"].startswith("latinitas-v1-") for row in rows)
    assert all(
        row["LatinitasID"] == derive_latinitas_id(row["Source ID"], row["Recipe"], row["Exercise Key"]) for row in rows
    )
    assert all(row["Tags"] == "latinitas" for row in rows)
    assert all(row["Source Path"] == "" for row in rows)
    assert all("Personal Notes" not in row for row in rows)
    assert any(
        row["Source ID"] == "fixture-guid-001" and row["Exercise Key"] == "perfect_1s" and "dīxī" in row["Answer"]
        for row in rows
    )
    assert any(
        row["Recipe"] == "principal_part_recognition"
        and row["Exercise Key"] == "perfect_1s"
        and row["Prompt"] == "Welche Stammform ist „dīxī“?"
        for row in rows
    )
    ppp_completion = next(
        row
        for row in rows
        if row["Source ID"] == "fixture-guid-001"
        and row["Recipe"] == "principal_part_completion"
        and row["Exercise Key"] == "perfect_passive_participle"
    )
    ppp_recognition = next(
        row
        for row in rows
        if row["Source ID"] == "fixture-guid-001"
        and row["Recipe"] == "principal_part_recognition"
        and row["Exercise Key"] == "perfect_passive_participle"
    )
    assert "<div>" in ppp_completion["Prompt"] and "<strong>" in ppp_completion["Prompt"]
    assert "Partizip Perfekt Passiv (PPP)" in ppp_completion["Prompt"]
    assert "<strong>Fehlende Stammform:</strong> dictum" in ppp_completion["Answer"]
    assert ppp_recognition["Prompt"] == "Welche Stammform ist „dictum“?"
    assert "<strong>Partizip Perfekt Passiv (PPP):</strong> dictum" in ppp_recognition["Answer"]
    assert "<div><strong>Bedeutung:</strong> sagen</div>" in ppp_recognition["Answer"]

    changed_profile = tmp_path / "changed-profile.json"
    changed_profile_payload = json.loads(profile.read_text(encoding="utf-8"))
    changed_profile_payload["fields"]["meaning_field"] = "Reference A"
    changed_profile.write_text(
        json.dumps(changed_profile_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    changed = _invoke(
        [
            "generate",
            "--input",
            str(source),
            "--profile",
            str(changed_profile),
            "--tag",
            "reviewed-v2",
            "--output",
            str(changed_output),
        ]
    )

    assert changed.returncode == 0
    _, changed_rows = _export_rows(changed_output)
    assert changed_output.read_bytes() != first_output.read_bytes()
    assert [row["LatinitasID"] for row in changed_rows] == [row["LatinitasID"] for row in rows]
    assert {row["Tags"] for row in changed_rows} == {"reviewed-v2"}
    assert any("<strong>Bedeutung:</strong> lesson-a" in row["Prompt"] for row in changed_rows)
    assert source.read_bytes() == source_before


def test_release_candidate_managed_updates_preserve_user_owned_notes() -> None:
    note = GeneratedNote.create(
        source_identity="fixture-guid-001",
        provenance=GeneratedNoteProvenance(source_kind="apkg", location="note 1001"),
        recipe=RecipeMetadata(recipe_identity="principal_part_completion", exercise_key="perfect_1s"),
        content=ManagedNoteContent(prompt="original", answer="dīxī", tags=("latinitas",)),
        personal_notes="Review this next week",
    )

    changed = note.with_managed_content(
        ManagedNoteContent(prompt="updated", answer="dīxī (sagen)", tags=("reviewed-v2",))
    )

    assert changed.latinitas_id == note.latinitas_id
    assert changed.content != note.content
    assert changed.personal_notes == "Review this next week"


def test_release_metadata_and_current_lock_audit_are_versioned_for_v010() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    with (ROOT / "uv.lock").open("rb") as lock_file:
        lock = tomllib.load(lock_file)

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    audit = (ROOT / "docs" / "license-compatibility-audit.md").read_text(encoding="utf-8")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert project["project"]["version"] == "0.1.0"
    assert next(package["version"] for package in lock["package"] if package["name"] == "latinitas-cards") == "0.1.0"
    assert "## [0.1.0]" in changelog
    assert "Prepared version: `v0.1.0`" in release
    assert "Prepared tag version: `v0.1.0`" in release
    assert "candidate awaiting user review" in release
    assert "T-014" in release
    t014_front_matter = T014_TASK.read_text(encoding="utf-8").split("---", 2)[1]
    assert re.search(r"^status: todo$", t014_front_matter, flags=re.MULTILINE)
    assert "93 package records" in audit
    assert "2.14.0+cpu" in audit
    assert "LicenseRef-NVIDIA-Proprietary" in audit
    assert "poetry.lock" in release
    assert "GitPython" in release
    assert "GitPython" not in notices
    assert "`cltk` 2.5.1" in notices
    assert "`stanza` 1.14.0" in notices
    assert "PyTorch `2.14.0+cpu`" in notices
    for package in GPU_PACKAGES:
        version = next(item["version"] for item in lock["package"] if item["name"] == package)
        assert f"`{package}` {version}" in notices

    lock_inventory = {
        (package["name"], package["version"]) for package in lock["package"] if package["name"] != "latinitas-cards"
    }
    audit_rows = _audit_rows(audit)
    assert len(audit_rows) == len(lock_inventory) == 92
    assert Counter((name, version) for name, version, _scope in audit_rows) == Counter(lock_inventory)

    expected_scopes: dict[tuple[str, str], set[str]] = {}
    for scope, options in (
        ("main", ("--no-dev",)),
        ("dev", ("--group", "dev")),
        ("annotate", ("--extra", "annotate", "--no-dev")),
        ("annotate-gpu", ("--extra", "annotate-gpu", "--no-dev")),
    ):
        exported = _locked_export_inventory(*options)
        assert exported
        for exported_package in exported:
            expected_scopes.setdefault(exported_package, set()).add(scope)

    actual_scopes = {(name, version): _scope_tokens(scope) for name, version, scope in audit_rows}
    assert actual_scopes == expected_scopes
