"""Exercise the aggregate report through its command-line interface."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/summarize-mutations.py"


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    source = tmp_path / "src/latinitas_cards"
    source.mkdir(parents=True)
    for shard, (killed, survived, timeout, no_tests) in enumerate([(1, 1, 1, 1), (8, 1, 0, 0)]):
        name = f"src/latinitas_cards/module_{shard}.py"
        (tmp_path / name).touch()
        artifact = tmp_path / "artifacts" / f"mutation-results-{shard}"
        (artifact / "mutants").mkdir(parents=True)
        (artifact / "mutation-scope.json").write_text(json.dumps({"shard": shard, "shards": 2, "files": [name]}))
        stats = {
            "killed": killed,
            "survived": survived,
            "timeout": timeout,
            "no_tests": no_tests,
            "skipped": 0,
            "suspicious": 0,
            "check_was_interrupted_by_user": 0,
            "segfault": 0,
            "total": killed + survived + timeout + no_tests,
        }
        (artifact / "mutants/mutmut-cicd-stats.json").write_text(json.dumps(stats))
    return tmp_path


def run_summary(root: Path, outcome: str = "success") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / SCRIPT.name),
            str(root / "artifacts"),
            "--shards",
            "2",
            "--shard-result",
            outcome,
            "--output",
            str(root / "report"),
        ],
        capture_output=True,
        text=True,
    )


def test_summary_weights_counts_and_includes_timeouts_and_untested_mutants(campaign: Path) -> None:
    result = run_summary(campaign)
    assert result.returncode == 0, result.stderr
    report = json.loads((campaign / "report/summary.json").read_text())
    assert report["complete"] is True
    assert report["totals"]["total"] == 13
    assert report["totals"]["killed"] == 9
    assert report["totals"]["timeout"] == 1
    assert report["totals"]["survived"] == 2
    assert report["totals"]["no_tests"] == 1
    assert report["score_percent"] == pytest.approx(1000 / 13)
    markdown = (campaign / "report/summary.md").read_text()
    assert "76.92%" in markdown
    assert "50.00%" in markdown
    assert "88.89%" in markdown
    assert "line coverage" in markdown


def test_complete_zero_detection_campaign_has_no_efficacy_floor(campaign: Path) -> None:
    for path in (campaign / "artifacts").glob("*/mutants/mutmut-cicd-stats.json"):
        stats = json.loads(path.read_text())
        stats["survived"] += stats["killed"] + stats["timeout"]
        stats["killed"] = stats["timeout"] = 0
        path.write_text(json.dumps(stats))
    result = run_summary(campaign)
    assert result.returncode == 0, result.stderr
    report = json.loads((campaign / "report/summary.json").read_text())
    assert report["complete"] is True
    assert report["score_percent"] == 0
    assert "Overall mutation score: 0.00%" in (campaign / "report/summary.md").read_text()


@pytest.mark.parametrize(
    "problem", ["missing", "malformed", "total", "negative", "boolean", "unchecked", "interrupted"]
)
def test_incomplete_stats_preserve_report_but_withhold_score(campaign: Path, problem: str) -> None:
    path = campaign / "artifacts/mutation-results-1/mutants/mutmut-cicd-stats.json"
    stats = json.loads(path.read_text())
    if problem == "missing":
        path.unlink()
    elif problem == "malformed":
        path.write_text("not JSON")
    else:
        if problem == "total":
            stats["total"] += 1
        elif problem == "unchecked":
            stats["not_checked"] = 1
            stats["total"] += 1
        elif problem == "negative":
            stats["killed"] = -1
        elif problem == "boolean":
            stats["skipped"] = False
        else:
            stats["killed"] -= 1
            stats["check_was_interrupted_by_user"] = 1
        path.write_text(json.dumps(stats))
    result = run_summary(campaign)
    assert result.returncode != 0
    report = json.loads((campaign / "report/summary.json").read_text())
    assert report["complete"] is False
    assert report["score_percent"] is None
    assert report["errors"]
    assert list(report["shards"]) == ["0"]
    assert report["totals"] == {
        "killed": 1,
        "survived": 1,
        "timeout": 1,
        "no_tests": 1,
        "skipped": 0,
        "suspicious": 0,
        "check_was_interrupted_by_user": 0,
        "segfault": 0,
        "total": 4,
    }
    assert "Incomplete campaign" in (campaign / "report/summary.md").read_text()


@pytest.mark.parametrize("problem", ["missing shard", "wrong id", "overlap", "omitted source", "failed job"])
def test_incomplete_campaign_cannot_publish_overall_score(campaign: Path, problem: str) -> None:
    path = campaign / "artifacts/mutation-results-1/mutation-scope.json"
    scope = json.loads(path.read_text())
    if problem == "missing shard":
        shutil.rmtree(path.parent)
    elif problem == "omitted source":
        (campaign / "src/latinitas_cards/new.py").touch()
    elif problem != "failed job":
        if problem == "wrong id":
            scope["shard"] = 0
        else:
            scope["files"] = ["src/latinitas_cards/module_0.py"]
        path.write_text(json.dumps(scope))
    result = run_summary(campaign, "failure" if problem == "failed job" else "success")
    assert result.returncode != 0
    report = json.loads((campaign / "report/summary.json").read_text())
    assert report["complete"] is False
    assert report["score_percent"] is None
    assert report["errors"]
    if problem == "failed job":
        assert list(report["shards"]) == ["0", "1"]
        assert report["totals"]["total"] == 13
        assert report["totals"]["killed"] == 9
        assert report["totals"]["timeout"] == 1
        assert report["totals"]["survived"] == 2
        assert report["totals"]["no_tests"] == 1
