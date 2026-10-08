"""Optional real Anki Desktop dialog evidence on disposable fixtures only.

Use --prepare once, --gui CASE with native Desktop, then --finish CASE after
graceful GUI exit. Dialog control interaction is performed through its actual
Qt webview (CDP); no update is executed by this script's backend API.
"""

import argparse
import json
import platform
import runpy
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from typing import Any

from latinitas_cards.destination_capture import capture_backup, save_adoption
from latinitas_cards.destination_state import load_state
from latinitas_cards.managed_application import emit_updates, observe_updates
from latinitas_cards.managed_plans import approve_plan, compose_plan

CASES = ("content", "tag-add", "tag-remove", "tag-final-remove", "tag-unmapped", "noop")
REFERENCE = Path(__file__).with_name("check-managed-anki.py")


def prepare(root: Path, ref: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    # Reuse the complete existing asymmetric fixture, including 8 cards/15 logs.
    with (root / "backend-setup.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(REFERENCE), "--output-dir", str(root / "native")],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    notes = [ref["generated"](key) for key in ("sense-A", "sense-B")]
    for case in CASES:
        folder = root / case
        folder.mkdir()
        profile = folder / "profiles" / "Fixture"
        profile.mkdir(parents=True)
        path = profile / "collection.anki2"
        shutil.copyfile(root / "native" / "backup.anki2", path)
        if case == "tag-final-remove":
            with closing(sqlite3.connect(path)) as db, db:
                db.execute(
                    "UPDATE notes SET tags=' source ' WHERE flds LIKE ?",
                    (min(n.latinitas_id for n in notes) + "\x1f%",),
                )
        before = folder / "before.anki2"
        shutil.copyfile(path, before)
        fixture_snapshot = json.loads((root / "native" / "before-snapshot.json").read_text())
        selection = {
            "destination": "desktop-sanitized",
            "profile": "desktop-fixture",
            "note_type_id": fixture_snapshot["schema"]["note_type_id"],
            "managed_set": {
                "scope": "native-scope",
                "members": [[n.latinitas_id, "native-scope", "synthetic-entry", n.object_key] for n in notes],
            },
        }
        ref["write"](folder / "selection.json", selection)
        snap = capture_backup(before, selection, closed_backup=True, interval_confirmed=True)
        ref["write"](folder / "before-snapshot.json", snap.payload)
        if case == "noop":
            shutil.copyfile(root / "native" / "content-tags.csv", folder / "updates.csv")
            continue
        state_path = folder / "state.json"
        save_adoption(state_path, ref["adopt"](snap, ref["ownership"](snap), "review synthetic Desktop ownership"))
        a = ref["propose"](snap, Meaning="approved Desktop A", Lemma="UNAPPROVED lemma")
        a["contributions"] = {
            "source_tags": ["source", "added"] if case in ("content", "tag-add", "tag-unmapped") else []
        }
        if case != "content":
            a["fields"] = dict(snap.payload["notes"][0]["fields"])
        b = ref["propose"](snap, 1, Meaning="UNAPPROVED B")
        plan = compose_plan(snap, load_state(state_path), [a, b], ref["PROFILE"])
        identity = snap.payload["notes"][0]["identity"]
        selected = [
            op["id"]
            for entry in plan["notes"]
            for op in entry["operations"]
            if op["id"].startswith(identity + "/") and (op.get("field") == "Meaning" or op["kind"] == "tags")
        ]
        approval = approve_plan(plan, selected, "review only first note Meaning/tags; exclude lemma and second note")
        emitted = emit_updates(
            plan,
            approval,
            snap,
            state_path,
            before,
            "close GUI; restore before; reconcile before retry",
            folder / "updates.csv",
            ref["MODEL"],
            ref["DECK"],
        )
        for name, value in (("plan", plan), ("approval", approval), ("emission", emitted)):
            ref["write"](folder / f"{name}.json", value)
    ref["write"](
        root / "bindings.json",
        {
            "anki": version("anki"),
            "aqt": version("aqt"),
            "PyQt6": version("PyQt6"),
            "PyQt6-Qt6": version("PyQt6-Qt6"),
            "os": platform.platform(),
            "python": sys.version,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {
                str(p): ref["digest"](p)
                for p in [Path(__file__), REFERENCE, *Path("src/latinitas_cards").rglob("*.py")]
            },
        },
    )
    print("Prepared six isolated profiles; backend setup is NOT Desktop update proof.")


def gui(folder: Path) -> None:
    assert (folder.parent / "bindings.json").exists(), "prepare an owned fixture first"
    (folder / "gui-started").open("x").close()
    if folder.name == "noop":
        content = folder.parent / "content"
        assert (content / "report.json").exists(), "finish actual Desktop content import before no-op"
        for name in ("updates.csv", "before-snapshot.json", "state.json", "plan.json", "approval.json"):
            shutil.copyfile(content / name, folder / name)
        shutil.copyfile(content / "after.anki2", folder / "profiles" / "Fixture" / "collection.anki2")
        shutil.copyfile(content / "after.anki2", folder / "before.anki2")
        selection = json.loads((folder / "selection.json").read_text())
        snap = capture_backup(folder / "before.anki2", selection, closed_backup=True, interval_confirmed=True)
        (folder / "before-snapshot.json").write_text(json.dumps(snap.payload, ensure_ascii=False, indent=2) + "\n")

    import aqt
    from aqt.import_export.import_dialog import ImportDialog
    from aqt.import_export.importing import import_file
    from aqt.profiles import ProfileManager
    from aqt.qt import QApplication, QTimer

    base = folder / "profiles"
    pm = ProfileManager(str(base))
    pm.setupMeta()
    pm.create("Fixture")
    pm.load("Fixture")
    pm.meta["defaultLang"] = "en_US"
    pm.meta["updates"] = False
    pm.profile["autoSync"] = False
    pm.save()
    pm.db.close()

    def poll() -> None:
        request = folder / "capture-request.txt"
        if request.exists():
            name = request.read_text().strip()
            assert name in ("file", "settings", "mapping", "result")
            dialogs = [w for w in QApplication.topLevelWidgets() if isinstance(w, ImportDialog) and w.isVisible()]
            assert len(dialogs) == 1
            assert dialogs[0].grab().save(str(folder / f"{name}.png"))
            request.unlink()
        if (folder / "close-request").exists():
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, ImportDialog) and widget.isVisible():
                    widget.reject()
            timer.stop()
            aqt.mw.unloadProfileAndExit()

    timer = QTimer()
    timer.timeout.connect(poll)

    def ready() -> None:
        timer.start(200)
        QTimer.singleShot(1500, lambda: import_file(aqt.mw, str(folder / "updates.csv")))

    aqt.gui_hooks.profile_did_open.append(ready)
    sys.argv = ["anki", "-b", str(base), "-p", "Fixture", "-l", "en_US"]
    aqt.run()
    (folder / "closed").write_text("native GUI exited gracefully\n")


def drive(folder: Path, ref: dict[str, Any]) -> None:
    def browser(*args: str) -> dict[str, Any]:
        result = subprocess.run(
            ["agent-browser", "--session", "t076", "--cdp", "9223", "--json", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        data = json.loads(result.stdout)
        assert data["success"], data
        return data["data"]

    tabs = browser("tab", "list")["tabs"]
    tab = next(t for t in tabs if t["title"] == "csv import" and t["url"].endswith(str(folder / "updates.csv")))
    browser("tab", tab["tabId"])
    columns = ref["CSV_EXPORT_FIELD_NAMES"]
    expected = ["Comma", ref["MODEL"], ref["DECK"], "Update", "Note Type"]
    expected += [
        f"{columns.index(f) + 1}: {f}" if f in columns else "(Nothing)" for f in ref["REFERENCE_NOTE_TYPE_FIELDS"]
    ]
    expected += ["(Nothing)" if folder.name == "tag-unmapped" else "5: Tags"]
    # Use real pointer clicks through CDP, including the rendered dropdown
    # options. JavaScript below only reads DOM settings, never importer state.
    browser("find", "role", "heading", "click", "--name", "File", "--exact")
    for index, value in enumerate(expected):
        if index in (0, 2):  # Forced delimiter; CSV-selected deck (tree control).
            continue
        browser("eval", f"document.querySelectorAll('[role=combobox]')[{index}].scrollIntoView({{block:'center'}})")
        browser("find", "nth", str(index), "[role=combobox]", "click")
        browser("find", "role", "option", "click", "--name", value, "--exact")
    script = """(() => {
        const expected = EXPECTED;
        const boxes = [...document.querySelectorAll('[role=combobox]')];
        if (boxes.length !== expected.length) throw Error('mapping count');
        const html = document.querySelector('input[type=checkbox]');
        if (!html.checked || !html.disabled || !boxes[0].classList.contains('disabled'))
            throw Error('CSV must force HTML and comma');
        const actual = boxes.map(e => e.innerText.trim());
        if (JSON.stringify(actual) !== JSON.stringify(expected)) throw Error('mapping mismatch');
        return {url: location.href, controls: actual, html: html.checked, htmlForced: html.disabled};
    })()""".replace("EXPECTED", json.dumps(expected))
    ref["write"](folder / "dialog-settings.json", browser("eval", script)["result"])

    def capture(name: str) -> None:
        request = folder / "capture-request.txt"
        request.write_text(name)
        for _ in range(100):
            if not request.exists():
                assert (folder / f"{name}.png").exists()
                return
            time.sleep(0.1)
        raise RuntimeError("native screenshot request not handled")

    browser("eval", "[...document.querySelectorAll('h1')].find(e=>e.innerText==='File').scrollIntoView()")
    capture("file")
    browser("eval", "[...document.querySelectorAll('h1')].find(e=>e.innerText==='Import options').scrollIntoView()")
    capture("settings")
    browser("eval", "document.querySelectorAll('[role=combobox]')[47].scrollIntoView()")
    capture("mapping")
    browser("find", "role", "button", "click", "--name", "Import", "--exact")
    browser("wait", "--text", "Overview")
    result = browser("eval", "document.body.innerText")["result"]
    ref["write"](folder / "dialog-result.json", {"text": result})
    # DOM readiness can precede Qt WebEngine presenting its result texture.
    browser("eval", "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))")
    browser("wait", "500")
    capture("result")
    (folder / "close-request").touch()
    for _ in range(300):
        if (folder / "closed").exists():
            return
        time.sleep(0.1)
    raise RuntimeError("native GUI did not exit gracefully")


def finish(folder: Path, ref: dict[str, Any]) -> None:
    assert (folder / "closed").exists(), "close native GUI before acquiring destination"
    after = folder / "after.anki2"
    shutil.copyfile(folder / "profiles" / "Fixture" / "collection.anki2", after)
    selection = json.loads((folder / "selection.json").read_text())
    snap = capture_backup(after, selection, closed_backup=True, interval_confirmed=True)
    ref["write"](folder / "after-snapshot.json", snap.payload)
    before_tables = ref["tables"](folder / "before.anki2")
    after_tables = ref["tables"](after)
    identity = snap.payload["notes"][0]["identity"]
    desired_tags = {
        "content": ["added", "manual", "source"],
        "tag-add": ["added", "manual", "source"],
        "tag-remove": ["manual"],
        "tag-final-remove": [],
        "tag-unmapped": ["added", "manual", "source"],
    }
    expected = {}
    applied = False
    if folder.name != "noop":
        actual_tags = snap.payload["notes"][0]["tags"]
        applied = actual_tags == desired_tags[folder.name]
        if applied:
            expected = {identity: {"tags": desired_tags[folder.name]}}
            if folder.name == "content":
                expected[identity]["fields"] = {"Meaning": "approved Desktop A"}
        if folder.name == "content":
            assert applied, "required content/tag import did not apply"
        if folder.name == "tag-unmapped":
            assert not applied, "unmapped tag fault unexpectedly applied"
    # A skipped tag effect must leave all fields/tables unchanged. Any partial or
    # unexpected mutation fails the comparison, rather than being called a skip.
    deltas = ref["compare"](before_tables, after_tables, expected)
    report = {"case": folder.name, "deltas": deltas, "counts": {t: len(after_tables[t]) for t in ref["TABLES"]}}
    if (folder / "state.json").exists():
        plan = json.loads((folder / "plan.json").read_text())
        state_path = folder / "state.json"
        prior = load_state(state_path)
        report["observation"] = observe_updates(
            state_path,
            plan["plan_id"],
            snap,
            "actual native Desktop import; edit-free interval",
            interval_confirmed=True,
        )
        observed = report["observation"]
        selected = json.loads((folder / "approval.json").read_text())["selected_operations"]
        if folder.name == "noop":
            assert load_state(state_path) == prior, "no-op observation changed the recorded state"
            assert observed["observed"] == selected and not observed["unresolved"]
        elif applied:
            assert observed["observed"] == selected and not observed["unresolved"]
        else:
            assert not observed["observed"] and observed["unresolved"] == selected
            assert not observed["pending"]  # Actual observation classifies pending effects as unresolved.
            assert load_state(state_path)["anchors"] == prior["anchors"]
        report["capability"] = "observed" if applied or folder.name == "noop" else "skipped; unresolved"
    ref["write"](folder / "report.json", report)
    ref["write"](
        folder / "evidence-sha256.json",
        {
            p.name: ref["digest"](p)
            for p in folder.iterdir()
            if p.is_file() and p.suffix in (".json", ".csv", ".png", ".anki2") and p.name != "evidence-sha256.json"
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--gui", choices=CASES)
    modes.add_argument("--drive", choices=CASES)
    modes.add_argument("--finish", choices=CASES)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    ref = runpy.run_path(str(REFERENCE))
    if args.prepare:
        prepare(root, ref)
    elif args.gui:
        gui(root / args.gui)
    elif args.drive:
        drive(root / args.drive, ref)
    else:
        finish(root / args.finish, ref)


if __name__ == "__main__":
    main()
