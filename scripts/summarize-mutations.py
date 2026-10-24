"""Combine disjoint mutation shard artifacts into Markdown and JSON reports."""

import argparse
import json
from pathlib import Path

COUNTS = (
    "killed",
    "survived",
    "timeout",
    "no_tests",
    "skipped",
    "suspicious",
    "check_was_interrupted_by_user",
    "segfault",
    "total",
)


def read_shard(directory: Path, shard: int, shards: int) -> tuple[dict[str, int], set[str]]:
    scope = json.loads((directory / "mutation-scope.json").read_text(encoding="utf-8"))
    if not isinstance(scope, dict) or any(
        type(scope.get(key)) is not int or scope[key] != expected
        for key, expected in (("shard", shard), ("shards", shards))
    ):
        raise ValueError("scope does not identify the expected shard")
    files = scope.get("files")
    if not isinstance(files, list) or not files or not all(isinstance(path, str) for path in files):
        raise ValueError("scope must list source files")
    if len(files) != len(set(files)):
        raise ValueError("scope repeats source files")
    stats = json.loads((directory / "mutants/mutmut-cicd-stats.json").read_text(encoding="utf-8"))
    if not isinstance(stats, dict) or set(stats) != set(COUNTS):
        raise ValueError("stats have missing or unknown outcomes")
    if any(type(value) is not int or value < 0 for value in stats.values()):
        raise ValueError("stats must be nonnegative integer counts")
    if stats["total"] == 0 or sum(stats[key] for key in COUNTS if key != "total") != stats["total"]:
        raise ValueError("stats are empty or contain unchecked mutants")
    if stats["check_was_interrupted_by_user"]:
        raise ValueError("stats contain interrupted mutants")
    return {key: stats[key] for key in COUNTS}, set(files)


def summarize(artifacts: Path, shards: int, shard_result: str, output: Path) -> bool:
    errors: list[str] = []
    totals = dict.fromkeys(COUNTS, 0)
    per_shard: dict[str, dict[str, int]] = {}
    files: set[str] = set()
    if shard_result != "success":
        errors.append(f"Mutation jobs finished with result: {shard_result}.")
    expected = {f"mutation-results-{shard}" for shard in range(shards)}
    if {path.name for path in artifacts.glob("mutation-results-*")} - expected:
        errors.append("Unexpected shard artifacts are present.")
    for shard in range(shards):
        try:
            stats, scope = read_shard(artifacts / f"mutation-results-{shard}", shard, shards)
            if scope & files:
                raise ValueError("scope overlaps an earlier shard")
            files.update(scope)
            per_shard[str(shard)] = stats
            for key in COUNTS:
                totals[key] += stats[key]
        except (OSError, ValueError) as error:
            errors.append(f"Shard {shard}: {error}")
    root = Path(__file__).resolve().parents[1]
    source_files = {path.relative_to(root).as_posix() for path in (root / "src/latinitas_cards").rglob("*.py")}
    if not source_files or files != source_files:
        errors.append("Validated shard scopes do not cover the checkout's source files exactly once.")
    complete = not errors
    score = 100 * (totals["killed"] + totals["timeout"]) / totals["total"] if complete else None
    lines = ["# Mutation testing overview", ""]
    if score is not None:
        lines += [
            f"**Overall mutation score: {score:.2f}%**",
            "",
            f"({totals['killed']:,} killed + {totals['timeout']:,} timed out) / {totals['total']:,} generated mutants.",
        ]
    else:
        lines += ["**Incomplete campaign: no overall mutation score. Counts below are partial.**"]
    lines += [
        "",
        f"Validated shards: {len(per_shard)} / {shards}. Mutation job result: {shard_result}.",
        "",
        "The score sums counts, not shard percentages. Timeouts count as detected; survivors and other outcomes "
        "remain in the denominator. This is not line coverage: only covered lines are mutated.",
        "There is no efficacy floor; surviving mutants do not fail this report-only baseline.",
        "",
        "## Counts across validated shards",
        "",
        "| Outcome | Count |",
        "| --- | ---: |",
        *[f"| {key} | {totals[key]:,} |" for key in COUNTS],
        "",
        "## By shard",
        "",
        "| Shard | Total | Killed | Timeout | Survived | Other | Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for shard in range(shards):
        row = per_shard.get(str(shard))
        if row is None:
            lines.append(f"| {shard} | — | — | — | — | — | unavailable |")
            continue
        detected = row["killed"] + row["timeout"]
        other = row["total"] - detected - row["survived"]
        lines.append(
            f"| {shard} | {row['total']:,} | {row['killed']:,} | {row['timeout']:,} | "
            f"{row['survived']:,} | {other:,} | {100 * detected / row['total']:.2f}% |"
        )
    if errors:
        lines += ["", "## Incomplete campaign details", "", *[f"- {error}" for error in errors]]
    report = {"complete": complete, "score_percent": score, "totals": totals, "shards": per_shard, "errors": errors}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--shard-result", choices=("success", "failure", "cancelled", "skipped"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    raise SystemExit(0 if summarize(args.artifacts, args.shards, args.shard_result, args.output) else 1)
