"""`annotations import-translations` over real normalized Latin and parallel corpora."""

import json
from pathlib import Path
from typing import Any

import pytest
from corpus_exercises_test import profile
from typer.testing import CliRunner

from latinitas_cards.annotation_sidecar import Annotation
from latinitas_cards.claim_sidecar import sidecar_path
from latinitas_cards.cli import app
from latinitas_cards.cli_contract import CliResult
from latinitas_cards.corpus_claims import corpus_claim_fingerprint, record_corpus_batch
from latinitas_cards.corpus_exercises import exercise_candidates
from latinitas_cards.corpus_facts import accepted_facts
from latinitas_cards.corpus_model import latin_fold, passage_hash

RUNNER = CliRunner()
STAMP = "2026-01-01T00:00:00Z"
LATIN = ("poem.1.1\tArma virumque cano.", "poem.1.2\tMater filium amat.", "poem.1.3\tPuer aquam portat.")
GERMAN = ("poem.1.3\tDer Junge trägt Wasser.", "poem.1.1\tWaffen besinge ich und den Mann.", "poem.2.1\tNur hier.")


def invoke(*args: str) -> CliResult:
    result = RUNNER.invoke(app, [*args, "--json"])
    envelope = CliResult.model_validate_json(result.stdout)
    assert envelope.exit_code == result.exit_code, result.output
    return envelope


def normalize(root: Path, name: str, lines: tuple[str, ...], language: str, corpus_id: str) -> Path:
    source, corpus = root / f"{name}.txt", root / f"{name}.jsonl"
    source.write_text("\n".join(lines) + "\n")
    envelope = invoke(
        "corpus",
        "normalize",
        str(source),
        "--adapter",
        "text",
        "--corpus-id",
        corpus_id,
        "--language",
        language,
        "--citation-template",
        "poem:work.section.line",
        "--output",
        str(corpus),
    )
    assert envelope.exit_code == 0, envelope
    return corpus


def passages(corpus: Path) -> list[dict[str, Any]]:
    return [row for row in map(json.loads, corpus.read_text().splitlines()) if row["record"] == "passage"]


def import_translations(latin: Path, parallel: Path, output: Path, *extra: str, language: str = "de") -> CliResult:
    return invoke(
        "annotations",
        "import-translations",
        str(latin),
        "--parallel",
        str(parallel),
        "--language",
        language,
        "--producer-kind",
        "human",
        "--producer-name",
        "wsl-de",
        "--producer-version",
        "2024",
        "--output",
        str(output),
        *extra,
    )


@pytest.fixture
def corpora(tmp_path: Path) -> tuple[Path, Path]:
    return (
        normalize(tmp_path, "latin", LATIN, "la", "reader"),
        normalize(tmp_path, "german", GERMAN, "de", "reader-de"),
    )


def test_patch_plans_approves_and_applies_without_claims(tmp_path: Path, corpora: tuple[Path, Path]) -> None:
    latin, parallel = corpora
    assert invoke("annotations", "segment", str(latin), "--timestamp", STAMP).exit_code == 0
    sidecar = latin.with_suffix(".annotations.jsonl")
    output = tmp_path / "translations.patch.json"
    envelope = import_translations(latin, parallel, output, "--timestamp", "2026-02-03T04:05:06Z")
    assert envelope.exit_code == 0
    assert [(d.code, d.severity, d.ref, d.path) for d in envelope.diagnostics] == [
        ("unmatched_translation", "warning", "reader:poem.1.2", str(latin)),
        ("unmatched_parallel", "warning", "reader-de:poem.2.1", str(parallel)),
    ]
    first, _, third = passages(latin)
    producer = {"kind": "human", "name": "wsl-de", "version": "2024"}
    # Latin corpus order, Latin anchors, parallel canonical text, given tag and producer.
    assert json.loads(output.read_text()) == {
        "format": "latinitas-annotation-patch",
        "version": 1,
        "operations": [
            {
                "operation": "add",
                "layer": "translation",
                "anchor": {"passage_id": row["passage_id"], "passage_hash": row["passage_hash"]},
                "value": {"language": "de", "text": text},
                "producer": producer,
                "created": "2026-02-03T04:05:06Z",
            }
            for row, text in ((first, "Waffen besinge ich und den Mann."), (third, "Der Junge trägt Wasser."))
        ],
    }
    assert not sidecar_path(latin).exists()

    plan, receipt = tmp_path / "plan.json", tmp_path / "receipt.json"
    planned = invoke(
        "annotations", "plan", str(output), "--corpus", str(latin), "--sidecar", str(sidecar), "--output", str(plan)
    )
    assert planned.exit_code == 0
    assert isinstance(planned.result, dict) and isinstance(planned.result["operations"], list)
    ids = [row["operation_id"] for row in planned.result["operations"] if isinstance(row, dict)]
    assert len(ids) == 2
    approve = ["annotations", "approve", str(plan), "--actor", "human:r", "--review", "Checked", "--output"]
    assert invoke(*approve, str(receipt), *(arg for i in ids for arg in ("--operation", str(i)))).exit_code == 0
    apply = ["annotations", "apply", str(plan), "--corpus", str(latin), "--sidecar", str(sidecar)]
    assert invoke(*apply, "--receipt", str(receipt)).exit_code == 0

    applied = [Annotation.model_validate_json(line) for line in sidecar.read_text().splitlines()[1:]]
    translations = [row for row in applied if row.layer == "translation"]
    assert [(row.anchor.passage_id, row.input_hash, row.created) for row in translations] == [
        (first["passage_id"], first["passage_hash"], "2026-02-03T04:05:06Z"),
        (third["passage_id"], third["passage_hash"], "2026-02-03T04:05:06Z"),
    ]
    assert all(row.producer.model_dump() == producer and row.anchor.span is None for row in translations)
    assert not sidecar_path(latin).exists()

    def sentence_reasons(language: str) -> dict[str, tuple[str, ...]]:
        view = accepted_facts(latin, sidecar, sidecar_path(latin), target_language=language)
        return {
            c.passage_id: c.reasons
            for c in exercise_candidates(view, profile(language, corpora=["reader"]))
            if c.recipe == "corpus_sentence_translation"
        }

    # Applied facts alone are open; only an accepted claim makes the unit eligible.
    assert sentence_reasons("de") == {first["passage_id"]: ("open_fact",), third["passage_id"]: ("open_fact",)}
    items = [
        {
            "fingerprint": corpus_claim_fingerprint(translations[0]),
            "status": "accepted",
            "reviewer": "R",
            "reason": "ok",
        }
    ]
    assert record_corpus_batch(latin, sidecar, items) == ()
    assert sentence_reasons("de") == {first["passage_id"]: (), third["passage_id"]: ("open_fact",)}
    assert sentence_reasons("en")[first["passage_id"]] == ("wrong_language",)


def test_untimed_patch_leaves_created_to_apply(
    tmp_path: Path, corpora: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    latin, parallel = corpora
    # Patch bytes never depend on the environment; apply resolves SOURCE_DATE_EPOCH later.
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    output = tmp_path / "patch.json"
    assert import_translations(latin, parallel, output).exit_code == 0
    operations = json.loads(output.read_text())["operations"]
    assert operations and all("created" not in row for row in operations)
    bad = import_translations(latin, parallel, tmp_path / "bad.json", "--timestamp", "2026-02-30T00:00:00Z")
    assert (bad.exit_code, [d.code for d in bad.diagnostics]) == (2, ["timestamp_required"])
    assert not (tmp_path / "bad.json").exists()


def test_rerun_is_byte_identical_noop_and_differing_output_exists(tmp_path: Path, corpora: tuple[Path, Path]) -> None:
    latin, parallel = corpora
    output = tmp_path / "patch.json"
    assert import_translations(latin, parallel, output, "--timestamp", STAMP).exit_code == 0
    before = (output.read_bytes(), output.stat().st_mtime_ns)
    again = import_translations(latin, parallel, output, "--timestamp", STAMP)
    assert again.exit_code == 0 and isinstance(again.result, dict) and again.result["written"] is False
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before
    changed = import_translations(latin, parallel, output, "--timestamp", "2026-01-02T00:00:00Z")
    assert (changed.exit_code, [d.code for d in changed.diagnostics]) == (3, ["exists"])
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before


@pytest.mark.parametrize(
    ("parallel_language", "language"),
    [("la", "la"), ("la-Latn", "la"), ("en", "de"), ("de", "en-GB"), ("la", "de")],
)
def test_latin_or_mismatched_parallel_language_is_rejected(
    tmp_path: Path, parallel_language: str, language: str
) -> None:
    latin = normalize(tmp_path, "latin", LATIN, "la", "reader")
    parallel = normalize(tmp_path, "parallel", GERMAN, parallel_language, "parallel")
    output = tmp_path / "patch.json"
    envelope = import_translations(latin, parallel, output, language=language)
    assert (envelope.exit_code, [d.code for d in envelope.diagnostics]) == (3, ["unsupported_language"])
    assert not output.exists()


def test_primary_subtag_matches_case_insensitively_and_tag_is_kept(tmp_path: Path) -> None:
    latin = normalize(tmp_path, "latin", LATIN, "la", "reader")
    parallel = normalize(tmp_path, "parallel", GERMAN, "de-CH", "parallel")
    output = tmp_path / "patch.json"
    assert import_translations(latin, parallel, output, language="DE-AT").exit_code == 0
    assert {row["value"]["language"] for row in json.loads(output.read_text())["operations"]} == {"DE-AT"}


def rewrite_passage(corpus: Path, target: Path, native_ref: str, text: str, *, rehash: bool) -> Path:
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    for row in rows:
        if row.get("native_ref") == native_ref:
            row["text"] = text
            if rehash:
                row.update(norm=latin_fold(text), passage_hash=passage_hash(text))
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return target


def test_empty_parallel_passage_counts_as_unmatched(tmp_path: Path, corpora: tuple[Path, Path]) -> None:
    latin, parallel = corpora
    # Adapters never write empty passages, but a hand-made corpus with matching hash validates.
    empty = rewrite_passage(parallel, tmp_path / "empty.jsonl", "poem.1.3", "", rehash=True)
    output = tmp_path / "patch.json"
    envelope = import_translations(latin, empty, output)
    assert envelope.exit_code == 0
    assert [(d.code, d.ref) for d in envelope.diagnostics] == [
        ("unmatched_translation", "reader:poem.1.2"),
        ("unmatched_translation", "reader:poem.1.3"),
        ("unmatched_parallel", "reader-de:poem.2.1"),
    ]
    assert [row["value"]["text"] for row in json.loads(output.read_text())["operations"]] == [
        "Waffen besinge ich und den Mann."
    ]


def test_text_mode_warnings_name_the_passage(tmp_path: Path, corpora: tuple[Path, Path]) -> None:
    latin, parallel = corpora
    args = ["annotations", "import-translations", str(latin), "--parallel", str(parallel), "--language", "de"]
    producer = ["--producer-kind", "human", "--producer-name", "p", "--producer-version", "1"]
    result = RUNNER.invoke(app, [*args, *producer, "--output", str(tmp_path / "patch.json")])
    assert result.exit_code == 0
    assert "unmatched_translation: Latin passage poem.1.2 has no parallel translation" in result.stderr
    assert "unmatched_parallel: Parallel passage poem.2.1 has no Latin passage" in result.stderr


def test_invalid_inputs_write_nothing(tmp_path: Path, corpora: tuple[Path, Path]) -> None:
    latin, parallel = corpora
    output = tmp_path / "patch.json"
    for broken_latin, broken_parallel in (
        (rewrite_passage(latin, tmp_path / "broken-la.jsonl", "poem.1.3", "edited", rehash=False), parallel),
        (latin, rewrite_passage(parallel, tmp_path / "broken-de.jsonl", "poem.1.3", "edited", rehash=False)),
    ):
        invalid = import_translations(broken_latin, broken_parallel, output)
        assert (invalid.exit_code, invalid.diagnostics[0].code) == (3, "invalid_corpus")
    language = import_translations(latin, parallel, output, language="not a tag")
    assert (language.exit_code, language.diagnostics[0].message) == (2, "Malformed --language tag")
    for kind, name, version, message in (
        ("rule", "p", "1", "--producer-kind must be human, agent or llm"),
        ("llm", "bad name", "1", "Malformed --producer-name"),
        ("llm", "p", "", "Malformed --producer-version"),
    ):
        envelope = invoke(
            "annotations", "import-translations", str(latin), "--parallel", str(parallel), "--language", "de",
            "--producer-kind", kind, "--producer-name", name, "--producer-version", version, "--output", str(output),
        )  # fmt: skip
        assert (envelope.exit_code, envelope.diagnostics[0].code, envelope.diagnostics[0].message) == (
            2,
            "usage_error",
            message,
        )
    for alias in (
        latin,
        tmp_path / "latin.annotations.jsonl",
        tmp_path / "german.annotations.jsonl",
        tmp_path / "german.jsonl.claims.json",
        tmp_path / "german.jsonl.lock",
    ):
        envelope = import_translations(latin, parallel, alias)
        assert (envelope.exit_code, envelope.diagnostics[0].code) == (3, "alias")
    assert not output.exists() and (tmp_path / "german.jsonl.lock").read_bytes() == b""
    assert not (tmp_path / "german.annotations.jsonl").exists()
