"""Run the README managed sequence on a disposable native collection only.

uv run --with anki==26.9.3 python scripts/check-managed-capture.py --output-dir DIR
Never accepts user collection paths; DIR must not exist. Native provisioning and
import reuse check-managed-anki.py; production capture itself opens only SQLite.
"""

import argparse
import json
import runpy
import shlex
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

from latinitas_cards.destination_state import load_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = runpy.run_path(str(Path(__file__).with_name("check-managed-anki.py")))
    write = reference["write"]
    note = reference["generated"]("fixture-object")
    original = output / "disposable.anki2"
    reference["create"](original, [note], output / "initial.csv")
    # Add personal/history/user suspension evidence only to this owned fixture.
    with closing(sqlite3.connect(original)) as db, db:
        nid, fields = db.execute("SELECT id,flds FROM notes").fetchone()
        values = fields.split("\x1f")
        values[-1] = "synthetic personal ü"
        db.execute("UPDATE notes SET flds=?,tags=' source manual ' WHERE id=?", ("\x1f".join(values), nid))
        cid = db.execute("SELECT id FROM cards ORDER BY id LIMIT 1").fetchone()[0]
        db.execute("UPDATE cards SET queue=-1,due=765,ivl=19,reps=8 WHERE id=?", (cid,))
        db.execute("INSERT INTO revlog VALUES (1000000000000,?,0,3,19,7,2500,1234,1)", (cid,))
        model_id = db.execute("SELECT mid FROM notes").fetchone()[0]
    backup = output / "before.anki2"
    shutil.copyfile(original, backup)
    selection = {
        "destination": "sanitized-original",
        "profile": "sanitized-profile",
        "note_type_id": str(model_id),
        "managed_set": {
            "scope": "native-scope",
            "members": [[note.latinitas_id, "native-scope", "synthetic-entry", note.object_key]],
        },
    }
    write(output / "selection.json", selection)
    ownership = {
        note.latinitas_id: {
            "source_tags": ["source"],
            "configured_tags": [],
            "keep_tags": ["manual"],
            "keep_fields": [],
        }
    }
    write(output / "ownership.json", ownership)
    transcript = []

    def cli(*args: str, artifact: str) -> dict[str, Any]:
        command = ["uv", "run", "--with", "anki==26.9.3", "latinitas-cards", "managed", *args]
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        data: dict[str, Any] = json.loads(result.stdout)
        write(output / artifact, data)
        transcript.append({"command": command, "result": data.get("status", "ok")})
        return data

    snapshot = cli(
        "capture",
        str(backup),
        "--selection",
        str(output / "selection.json"),
        "--closed-backup",
        "--interval-confirmed",
        artifact="snapshot.json",
    )
    before_tables = snapshot["export_options"]["tables"]
    cli(
        "adopt",
        str(output / "snapshot.json"),
        "--ownership",
        str(output / "ownership.json"),
        "--review",
        "reviewed this fixture note: source plus manual keep",
        "--state",
        str(output / "baseline.json"),
        artifact="adoption.json",
    )
    binding = {name: snapshot[name] for name in ("destination", "profile", "schema", "managed_set")}
    write(
        output / "request.json",
        {
            "binding": binding,
            "snapshot": snapshot,
            "baseline": load_state(output / "baseline.json"),
            "proposals": [
                {
                    "identity": note.latinitas_id,
                    "fields": dict(snapshot["notes"][0]["fields"], Meaning="reviewed fixture update"),
                    "contributions": {"source_tags": ["source"]},
                }
            ],
            "effective_profile": reference["PROFILE"],
        },
    )
    plan = cli("plan", str(output / "request.json"), artifact="plan.json")
    operation = next(op["id"] for entry in plan["notes"] for op in entry["operations"] if op.get("field") == "Meaning")
    approval = cli(
        "approve",
        str(output / "plan.json"),
        "--operation",
        operation,
        "--review",
        "approve only fixture Meaning",
        artifact="approval.json",
    )
    write(
        output / "handoff.json",
        {
            "binding": binding,
            "snapshot": snapshot,
            "plan": plan,
            "approval": approval,
            "backup": str(backup),
            "recovery": "restore the untouched before.anki2; reconcile before retry",
            "note_type": reference["MODEL"],
            "deck": reference["DECK"],
        },
    )
    cli(
        "emit",
        str(output / "handoff.json"),
        "--state",
        str(output / "baseline.json"),
        "--output",
        str(output / "updates.csv"),
        artifact="emission.json",
    )
    assert (
        load_state(output / "baseline.json")["anchors"][note.latinitas_id]["fields"]["Meaning"] == "old fixture-object"
    )
    report = reference["native_import"](original, output / "updates.csv")
    write(output / "native-report.json", report)
    after = output / "after.anki2"
    shutil.copyfile(original, after)
    observed = cli(
        "capture",
        str(after),
        "--selection",
        str(output / "selection.json"),
        "--closed-backup",
        "--interval-confirmed",
        artifact="after-snapshot.json",
    )
    after_tables = observed["export_options"]["tables"]
    for table in reference["TABLES"]:
        if table != "notes":
            assert before_tables[table] == after_tables[table], table
    assert before_tables["notes"][0]["flds"].split("\x1f")[-1] == after_tables["notes"][0]["flds"].split("\x1f")[-1]
    assert observed["notes"][0]["tags"] == ["manual", "source"]
    write(
        output / "observation.json",
        {
            "binding": binding,
            "snapshot": observed,
            "plan_id": plan["plan_id"],
            "report": "sanitized native import; full table comparison passed",
        },
    )
    result = cli(
        "observe",
        str(output / "observation.json"),
        "--state",
        str(output / "baseline.json"),
        "--interval-confirmed",
        artifact="observed.json",
    )
    assert result["observed"] == [operation] and result["unresolved"] == []
    assert (
        load_state(output / "baseline.json")["anchors"][note.latinitas_id]["fields"]["Meaning"]
        == "reviewed fixture update"
    )
    write(output / "commands.json", transcript)
    print("PASS: installed CLI capture/adopt/plan/approve/emit/observe; full card/history/model/deck tables unchanged")


if __name__ == "__main__":
    main()
