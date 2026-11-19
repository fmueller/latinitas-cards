"""Version-bound development examples; synthetic authority, no model or Anki use.

Run from a development checkout with locked dev dependencies. Fixture builders are
shared with integration tests; expected results below are explicit, not snapshots.
"""

import argparse
import csv
import json
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from latinitas_cards.acceptance import check_envelope, load_acceptance, save_acceptance, save_stage
from latinitas_cards.acceptance_policy import BOOTSTRAP
from latinitas_cards.acceptance_presentation import ACCEPTANCE_FIELDS, Presentation, metadata, note_tags
from latinitas_cards.acceptance_records import Check, Human, canonical, digest
from latinitas_cards.grammar_layout import GRAMMAR_FIELDS
from latinitas_cards.json_schemas import SCHEMAS
from latinitas_cards.pipeline_consumers import Objectives, current_cards
from latinitas_cards.pipeline_profile import PipelineProfile
from latinitas_cards.pipeline_reporting import PipelineReport, Reporting


def write(path: Path, value: object) -> None:
    path.write_bytes(canonical(value))


def display_controls(base: Path, language: str) -> None:
    """Hand-built display controls, not pipeline export or automatic acceptance."""
    human = Presentation(
        "form_translation:standalone",
        "human",
        ("synthetic-human",),
        ("source",),
        ("human: pass",),
        ("SYNTHETIC DISPLAY ONLY; not linguistic approval",),
    )
    automatic = Presentation(
        "form_recognition:standalone",
        "automatic",
        ("synthetic-automatic",),
        ("library",),
        ("library: pass",),
        ("SYNTHETIC DISPLAY ONLY; disabled production policy unchanged",),
    )
    columns = [name for name in (*GRAMMAR_FIELDS, *ACCEPTANCE_FIELDS, "Tags") if name != "Personal Notes"]
    rows = []
    for key, presentations in (("mixed", (human, automatic)), ("human", (human,))):
        row = dict.fromkeys(columns, "")
        row["LatinitasID"] = f"T199-display-{language}-{key}"
        row.update(metadata(presentations, language, "T199-display-only"))
        inherited = ("User::Keep", "SourceCase")
        row["Tags"] = " ".join((*inherited, *note_tags(presentations, inherited)))
        row["CardKey"] = " ".join(p.card_key for p in presentations)
        for ordinal, presentation in zip((12, 13), presentations, strict=False):
            row[f"Grammar{ordinal}Enabled"] = "1"
            row[f"Grammar{ordinal}Prompt"] = "SYNTHETIC: amō" if ordinal == 12 else "SYNTHETIC: amās"
            if ordinal == 12:
                answer = "lieben" if language == "de" else "love"
            else:
                answer = "2 / singular / present / indicative / active"
            row[f"Grammar{ordinal}Answer"] = answer + presentation.answer(language)
        rows.append(row)
    for filename in ("display-initial.csv", "display-update.csv", "display-omission.csv"):
        with (base / filename).open("x", encoding="utf-8", newline="") as stream:
            stream.write(f"#separator:Comma\n#html:true\n#notetype:T199 display {language}\n")
            stream.write(f"#tags column:{len(columns)}\n")
            writer = csv.writer(stream)
            stream.write("#columns:")
            writer.writerow(columns)
            for index, original in enumerate(rows):
                if filename == "display-omission.csv" and index == 0:
                    continue
                row = dict(original)
                if filename == "display-update.csv" and index == 0:
                    row["Grammar12Answer"] = "Controlled correction: " + row["Grammar12Answer"]
                writer.writerow(row[name] for name in columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New disposable directory, never an Anki profile")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    # These helpers create original public engineering material, not reference truth.
    sys.path.insert(0, str(repo / "tests/integration"))
    import curated_cards_test as curated
    import grammar_pipeline_test as grammar

    write(
        root / "source.json", {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()}
    )
    assert BOOTSTRAP.id == "v050-disabled" and BOOTSTRAP.revision == "1" and BOOTSTRAP.scope == ()
    (root / "schemas").mkdir()
    for name, model in {
        **SCHEMAS,
        "latinitas-pipeline-profile": PipelineProfile,
        "latinitas-pipeline-report": PipelineReport,
        "pipeline-reporting-wrapper": Reporting,
    }.items():
        write(root / "schemas" / f"{name}.json", model.model_json_schema())

    # Fail on accidental outbound connections, including loopback. No real server here.
    with patch.object(socket.socket, "connect", side_effect=AssertionError("example must stay offline")):
        for language in ("de", "en"):
            base = root / language
            profile, objectives, run = grammar.fixture(base, language)
            grammar.call("validate", "--profile", str(profile))
            output = base / "workflow"
            first = grammar.call(
                "run",
                "--profile",
                str(profile),
                "--objectives",
                str(objectives),
                "--output",
                str(output),
                "--offline",
            )
            assert first["provider_calls"] == [] and first["file_export"]["exported"] == 0
            assert first["applied_to_anki"] is False
            resumed = grammar.call("resume", "--run", str(output / "run"), "--offline")
            assert resumed["file_export"] == first["file_export"]
            report = grammar.call("report", "--run", str(output / "run"))
            Reporting.read_json(canonical(report["outcome_report"]))
            write(base / "offline-report.json", report)
            grammar.call("stage", "select", "--run", str(output / "run"), "--offline")
            # Missing resources are a processing prerequisite, not a Latin judgment.
            library_profile, _, _ = grammar.fixture(base / "library-inputs", language, preset=None)
            unavailable = grammar.call(
                "run",
                "--profile",
                str(library_profile),
                "--output",
                str(base / "missing-library"),
                "--library-resources",
                str(base / "absent"),
                "--offline",
            )
            problems = unavailable["outcome_report"]["details"]["processing_outcomes"]
            assert any(item["reason"] == "processing_unavailable" for item in problems)
            assert unavailable["provider_calls"] == []
            grammar.call(
                "adapt",
                "--kind",
                "phrase-inputs",
                "--profile",
                str(profile),
                "--objectives",
                str(objectives),
                "--output",
                str(run),
            )
            cards, _ = current_cards(
                profile,
                PipelineProfile.read_json(profile.read_bytes()),
                Objectives.read_json(objectives.read_bytes()),
                run,
            )
            # Explicit synthetic full-card decisions exercise the human path only.
            # They are not a real reviewer, an approval tutorial or automatic policy.
            for index, card in enumerate(cards):
                envelope = check_envelope(card.binding, card.versions, "human", card.claims.assertions)
                stage = save_stage(
                    run,
                    run_id="T199-synthetic",
                    stage_id=f"review-{index}",
                    stage="analyze",
                    binding=card.binding,
                    versions=card.versions,
                    envelope=envelope,
                )
                check = Check(
                    id="human",
                    assertion_ids=tuple(a.id for a in card.claims.assertions),
                    method="human",
                    version=card.versions.checker,
                    result="pass",
                    evidence=(grammar.EVIDENCE,),
                    alternatives=(),
                    objections=(),
                    input_sha256=digest(envelope),
                )
                decision = card.claims.decisions[0].reader.decide(
                    checks=(check,),
                    origin="human",
                    stage_ids=(stage.stage_id,),
                    human=Human(
                        reviewer="T199-synthetic-NOT-human-approval",
                        reason="Disposable engineering control only",
                        reviewed_sha256=digest({"binding": card.binding, "assertions": card.claims.assertions}),
                    ),
                )
                save_acceptance(card.claims.decisions[0].path, decision)
            assert len(cards) == 6
            preview = grammar.call("preview", "--run", str(run))
            write(base / "human-control-preview.json", preview)
            full = grammar.call("export", "--run", str(run), "--output", str(base / "initial.csv"))
            assert full["exported"] == 6
            assert grammar.call("export", "--run", str(run), "--output", str(base / "not-created.csv")) == full
            assert not (base / "not-created.csv").exists()
            assert grammar.call("handoff", "--run", str(run))["ineligible"] == []
            # Fault injection only: replace one synthetic record to test handoff.
            # Production records are immutable; this is NOT a review/update API.
            decision = cards[0].claims.decisions[0].reader
            saved = json.loads(cards[0].claims.decisions[0].path.read_text())
            saved["human"]["reason"] = "Changed synthetic review; installed material needs inspection"
            # Use the public reader to produce a new immutable decision identity.
            renewed = decision.decide(
                checks=load_acceptance(cards[0].claims.decisions[0].path).checks,
                origin="human",
                stage_ids=("review-0",),
                human=Human.model_validate(saved["human"]),
            )
            write(cards[0].claims.decisions[0].path, renewed)
            handoff = grammar.call("handoff", "--run", str(run))
            assert len(handoff["ineligible"]) == 1
            assert handoff["ineligible"][0]["reason"] == "acceptance_or_rendering_changed"
            write(base / "manual-handoff.json", handoff)
            for family in ("grammar", "curated"):
                write(
                    base / f"{family}-migration.json",
                    grammar.call("migration-plan", "--acceptance", "--family", family),
                )
            display_controls(base, language)

            inputs = curated.fixture(base / "curated", language)
            before = curated.call("preview", inputs)
            assert [card["state"] for card in before["cards"]] == ["withheld"]
            history = curated.approve(base / "curated", inputs, [])
            exported = curated.call(
                "export",
                inputs,
                "--approvals",
                str(history),
                "--output",
                str(base / "curated.csv"),
                "--run",
                str(base / "curated-receipts"),
            )
            assert exported["exported"] == 1
            write(base / "curated-result.json", exported)
    print(
        "PASS: de/en offline run/resume/report, zero dispatch, "
        f"six human-control cards each, replay and handoff; {root}"
    )


if __name__ == "__main__":
    main()
