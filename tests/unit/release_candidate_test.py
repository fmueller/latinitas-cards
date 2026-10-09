import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

from latinitas_cards.generation import SINGLE_LEXEME_OBJECT_KEY
from latinitas_cards.identity import derive_latinitas_id
from latinitas_cards.notes import GeneratedNote, GeneratedNoteProvenance, GenerationMetadata, ManagedNoteContent

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "representative-university-latin.apkg"
T014_TASK = ROOT / "planning" / "tasks" / "T-014-publish-v0-1-0.md"
EXPECTED_COLUMNS = (
    "LatinitasID",
    "Lemma",
    "Principal Parts",
    "Meaning",
    "Tags",
    "Source ID",
    "Source Scope",
    "Source Kind",
    "Source Location",
    "Source Path",
    "Note Schema",
    "Generator",
    "Profile",
    "CompletionPresentEnabled",
    "CompletionPresentPrompt",
    "CompletionPresentAnswer",
    "CompletionInfinitiveEnabled",
    "CompletionInfinitivePrompt",
    "CompletionInfinitiveAnswer",
    "CompletionPerfectEnabled",
    "CompletionPerfectPrompt",
    "CompletionPerfectAnswer",
    "CompletionPPPEnabled",
    "CompletionPPPPrompt",
    "CompletionPPPAnswer",
    "CompletionSupineEnabled",
    "CompletionSupinePrompt",
    "CompletionSupineAnswer",
    "RecognitionPresentEnabled",
    "RecognitionPresentPrompt",
    "RecognitionPresentAnswer",
    "RecognitionInfinitiveEnabled",
    "RecognitionInfinitivePrompt",
    "RecognitionInfinitiveAnswer",
    "RecognitionPerfectEnabled",
    "RecognitionPerfectPrompt",
    "RecognitionPerfectAnswer",
    "RecognitionPPPEnabled",
    "RecognitionPPPPrompt",
    "RecognitionPPPAnswer",
    "RecognitionSupineEnabled",
    "RecognitionSupinePrompt",
    "RecognitionSupineAnswer",
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
        "pipe_alternatives": False,
        "trailing_poet_hint": False,
    }
    assert effective_profile["selected_recipes"] == [
        "principal_part_completion",
        "principal_part_recognition",
    ]
    assert effective_profile["language_tag"] == "de"

    preview = _invoke(["preview", "--input", str(source), "--profile", str(profile), "--limit", "2"])

    assert preview.returncode == 0
    assert "Objects: 3" in preview.stdout
    assert "Cards: 12" in preview.stdout
    assert "Skipped: 2" in preview.stdout
    assert "Ambiguous: 3" in preview.stdout
    assert "Generated entries with warnings: 3 (overlaps generated)" in preview.stdout
    assert "linguistic_review_required" in preview.stdout
    assert "unresolved_source_evidence" in preview.stdout
    assert "Zero-eligible notes: 1" in preview.stdout
    assert "Partizip Perfekt Passiv (PPP)" in preview.stdout
    assert "Output:" not in preview.stdout
    assert source.read_bytes() == source_before

    gated = _invoke(
        [
            "generate",
            "--input",
            str(source),
            "--profile",
            str(profile),
            "--output",
            str(first_output),
        ]
    )
    assert gated.returncode != 0
    assert "fresh import" in gated.stderr or "fresh import" in gated.stdout
    assert not first_output.exists()

    for output in (first_output, second_output):
        arguments = [
            "generate",
            "--input",
            str(source),
            "--profile",
            str(profile),
            "--output",
            str(output),
        ]
        if output is first_output:
            arguments.append("--approve-fresh-import")
        generated = _invoke(arguments)
        assert generated.returncode == 0
        assert "Output:" in generated.stdout

    assert first_output.read_bytes() == second_output.read_bytes()
    metadata, rows = _export_rows(first_output)
    assert metadata == (
        "#separator:Comma\n"
        "#html:true\n"
        "#notetype:Latinitas Principal Parts\n"
        "#deck:Latin::Latinitas\n"
        f"#tags column:5\n#columns:{','.join(EXPECTED_COLUMNS)}\n"
    )
    assert len(rows) == 2
    expected_source_locations = {
        "fixture-guid-001": "note 1001",
        "fixture-guid-005": "note 1005",
    }
    assert {row["Source ID"] for row in rows} == set(expected_source_locations)
    assert len({row["LatinitasID"] for row in rows}) == len(rows)
    assert all(row["Source ID"] in expected_source_locations for row in rows)
    assert all(row["Source Kind"] == "apkg" for row in rows)
    assert all(row["Source Scope"] == "" for row in rows)
    assert all(expected_source_locations[row["Source ID"]] == row["Source Location"] for row in rows)
    assert all(row["Note Schema"] == rows[0]["Note Schema"] for row in rows)
    assert all(row["Generator"] == rows[0]["Generator"] for row in rows)
    assert all(row["Profile"].startswith("profile-sha256:") for row in rows)
    assert all(row["LatinitasID"].startswith("latinitas-v2-") for row in rows)
    assert all(row["LatinitasID"] == derive_latinitas_id(row["Source ID"], SINGLE_LEXEME_OBJECT_KEY) for row in rows)
    assert all(row["Tags"] == "latinitas" for row in rows)
    assert all(row["Source Path"] == "" for row in rows)
    assert all("Personal Notes" not in row for row in rows)
    dico = next(row for row in rows if row["Source ID"] == "fixture-guid-001")
    assert dico["Lemma"] == "dīcere"
    assert "<strong>Partizip Perfekt Passiv (PPP):</strong> dictum" in dico["Principal Parts"]
    assert dico["Meaning"] == "sagen"
    assert "<div>" not in dico["Principal Parts"] or "<br>" in dico["Principal Parts"]

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
    assert any(row["Meaning"] == "lesson-a" for row in changed_rows)
    assert {row["Profile"] for row in changed_rows} != {row["Profile"] for row in rows}
    assert source.read_bytes() == source_before


def test_release_candidate_managed_updates_preserve_user_owned_notes() -> None:
    note = GeneratedNote.create(
        source_identity="fixture-guid-001",
        provenance=GeneratedNoteProvenance(source_kind="apkg", location="note 1001"),
        object_key=SINGLE_LEXEME_OBJECT_KEY,
        metadata=GenerationMetadata(profile_digest="profile-sha256:original"),
        content=ManagedNoteContent(
            lemma="dīcō",
            principal_parts="original",
            meaning="sagen",
            tags=("latinitas",),
        ),
        personal_notes="Review this next week",
    )

    changed = note.with_managed_content(
        ManagedNoteContent(
            lemma="dīcō",
            principal_parts="updated",
            meaning="sagen; aussprechen",
            tags=("reviewed-v2",),
        )
    )

    assert changed.latinitas_id == note.latinitas_id
    assert changed.content != note.content
    assert changed.personal_notes == "Review this next week"


def test_release_metadata_and_current_lock_audit_are_versioned_for_v020() -> None:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    with (ROOT / "uv.lock").open("rb") as lock_file:
        lock = tomllib.load(lock_file)

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    audit = (ROOT / "docs" / "license-compatibility-audit.md").read_text(encoding="utf-8")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert project["project"]["version"] == "0.2.0"
    assert next(package["version"] for package in lock["package"] if package["name"] == "latinitas-cards") == "0.2.0"
    assert "## [0.2.0] - 2026-10-08" in changelog
    assert "compare/v0.2.0...HEAD" in changelog
    assert "compare/v0.1.1...v0.2.0" in changelog
    assert not changelog.split("## [Unreleased]", 1)[1].split("## [0.2.0]", 1)[0].strip()
    assert "## [0.1.1] - 2026-10-04" in changelog
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "git clone --branch v0.2.0 --depth 1" in readme
    assert "**Release:** [v0.2.0]" in readme
    spec_index = (ROOT / "specs" / "README.md").read_text(encoding="utf-8")
    assert "`specs/v0.2.0.md` defines the completed release spec" in spec_index
    readiness = (ROOT / "docs" / "release-v0.2.0.md").read_text(encoding="utf-8")
    assert "`0.2.0` and `v0.2.0`" in readiness
    assert "2026-10-07" in readiness and "maintainer-reported" in readiness
    assert "urllib3 2.7.0" in readiness
    assert "https://github.com/cltk/cltk/blob/33e1653331fc2499e3f2f5da45237d87db0313cc/" in readiness
    assert "## [0.1.0] - 2026-10-02" in changelog
    assert "## [Unreleased]" in changelog
    assert "Prepared version: `v0.1.0`" in release
    assert "Prepared tag version: `v0.1.0`" in release
    assert "approved for publication" in release
    assert "T-014" in release
    t014_front_matter = T014_TASK.read_text(encoding="utf-8").split("---", 2)[1]
    assert re.search(r"^status: (?:in_progress|completed)$", t014_front_matter, flags=re.MULTILINE)
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
