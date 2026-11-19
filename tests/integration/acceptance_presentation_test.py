"""Presentation uses isolated engineering records, never linguistic approval."""

import csv
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest
from grammar_pipeline_test import call, fixture, install_synthetic, prove_cards, write

from latinitas_cards.acceptance_presentation import ACCEPTANCE_FIELDS, layout_digest
from latinitas_cards.pipeline_consumers import adapt, export


def test_migration_proposal_has_unique_fields_and_stable_templates() -> None:
    legacy = call("migration-plan")
    assert "confirm-layout" not in legacy["requires"]
    for family in ("grammar", "curated"):
        proposal = call("migration-plan", "--acceptance", "--family", family)
        assert "confirm-layout" in proposal["requires"]
        assert len(proposal["fields"]) == len(set(proposal["fields"]))
        assert proposal["fields"][0] == "LatinitasID"
        from latinitas_cards.curated_cards import CURATED_FIELDS

        old_fields = legacy["fields"] if family == "grammar" else [*CURATED_FIELDS, "Personal Notes"]
        assert proposal["fields"] == [*old_fields, *ACCEPTANCE_FIELDS]
        assert set(ACCEPTANCE_FIELDS) <= set(proposal["fields"])
        if family == "grammar":
            assert proposal["templates"] == legacy["templates"]


def test_explicit_installation_and_answer_only_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, objectives, root = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    raw = json.loads(profile.read_text())
    raw["layout"] = dict(id="acceptance-1", version=1, template_sha256=layout_digest("grammar"))
    write(profile, raw)
    with pytest.raises(ValueError, match="installation"):
        adapt(profile, objectives, root)
    from grammar_pipeline_test import runner

    from latinitas_cards.cli import app

    assert runner.invoke(app, ["pipeline", "confirm-layout", "--profile", str(profile), "--json"]).exit_code != 0
    assert (
        runner.invoke(
            app,
            ["pipeline", "confirm-layout", "--profile", str(profile), "--family", "curated", "--installed", "--json"],
        ).exit_code
        != 0
    )
    call("confirm-layout", "--profile", str(profile), "--installed")
    assert set(json.loads(profile.read_text())["layout"]) == {"id", "version", "template_sha256"}
    confirmation = profile.with_name(profile.name + ".acceptance-1.json")
    installed = json.loads(confirmation.read_text())
    for section in (installed, installed["layout"]):
        section["version"] = True
        write(confirmation, installed)
        with pytest.raises(ValueError, match="installation"):
            adapt(profile, objectives, root)
        section["version"] = 1
    write(confirmation, installed)
    adapt(profile, objectives, root)
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    result = export(root, tmp_path / "accepted.csv")
    from curated_cards_test import csv_note

    row = csv_note(tmp_path / "accepted.csv")
    assert set(ACCEPTANCE_FIELDS) <= row.keys()
    metadata = json.loads(row["Acceptance Data"])
    assert metadata["version"] == 1
    assert all(c["acceptance"] == "automatic" for c in metadata["cards"])
    assert "latinitas::acceptance::automatic" in row["Tags"]
    answers = [v for k, v in row.items() if k.endswith("Answer") and v]
    prompts = [v for k, v in row.items() if k.endswith("Prompt") and v]
    assert answers and all("Warum diese Antwort?" in v and "<details" in v for v in answers)
    assert prompts and all("<details" not in v and "acceptance-status" not in v for v in prompts)
    assert "Personal Notes" not in row
    receipt = json.loads(json.dumps(result["export_receipt"]))
    assert receipt["cards"][0]["owned_tags"]
    assert json.dumps(export(root, tmp_path / "replay.csv"), sort_keys=True) == json.dumps(result, sort_keys=True)
    from latinitas_cards.pipeline_consumers import handoff
    from latinitas_cards.pipeline_reporting import build_report

    assert handoff(root)["ineligible"] == []
    assert build_report(root, {"run_id": "presentation-check"}, None, None)["installed_handoff"] == []


def test_mixed_and_incomplete_summaries_preserve_tags_and_escape() -> None:
    from latinitas_cards.acceptance_presentation import Presentation, metadata, note_tags

    human = Presentation(
        "recognition", "human", ("human-id",), ("source", "curated"), ("human: pass",), ("<script>bad</script>",)
    )
    automatic = Presentation(
        "translation",
        "automatic",
        ("automatic-id",),
        ("library",),
        ("library: pass",),
        ("No empirical accuracy estimate.",),
    )
    tags = note_tags((human, automatic), ("User::Keep", "SourceCase"))
    assert "latinitas::acceptance::mixed" in tags
    assert "latinitas::acceptance::human" not in tags
    assert "latinitas::basis::curated" in tags
    fields = metadata((human, automatic), "de", "run")
    data = json.loads(fields["Acceptance Data"])
    assert [c["acceptance"] for c in data["cards"]] == ["human", "automatic"]
    assert "<script>" not in fields["Evidence Summary"]
    assert "&lt;script&gt;" in fields["Evidence Summary"]
    incomplete = Presentation("withdrawn", "none", (), (), (), ("withdrawn",))
    assert "latinitas::acceptance::incomplete" in note_tags((human, incomplete), ())
    for collision in ("latinitas::acceptance::human", "Latinitas::Acceptance::human", "LATINITAS::BASIS::source"):
        with pytest.raises(ValueError, match="collision"):
            note_tags((human,), (collision,))


@pytest.mark.parametrize("mnemonic", [False, True])
def test_opt_in_curated_theme_reapproval_and_identity(tmp_path: Path, mnemonic: bool) -> None:
    from curated_cards_test import call as curated_call
    from curated_cards_test import decisions
    from curated_cards_test import fixture as curated_fixture

    from latinitas_cards.curated import CuratedInput, load_approvals
    from latinitas_cards.curated_cards import RecipePlan, Settings, render_cards
    from latinitas_cards.pipeline_profile import Layout, PipelineProfile

    profile, _, _ = fixture(tmp_path / "grammar")
    args = curated_fixture(tmp_path / "curated", "de")
    raw_data = json.loads((tmp_path / "curated/input.json").read_text())
    raw_data["items"][0]["tags"] = ["User::Keep", "SourceCase"]
    write(tmp_path / "curated/input.json", raw_data)
    if mnemonic:
        raw_plan = json.loads((tmp_path / "curated/plan.json").read_text())
        raw_plan["targets"][0]["mnemonic_key"] = raw_data["items"][0]["mnemonics"][0]["key"]
        write(tmp_path / "curated/plan.json", raw_plan)
    old = curated_call("preview", args)
    approvals = decisions(tmp_path / "curated", old, ("rule", "mnemonic", "rendered_card"))
    data = CuratedInput.read_json((tmp_path / "curated/input.json").read_bytes())
    plan = RecipePlan.read_json((tmp_path / "curated/plan.json").read_bytes())
    profile_data = PipelineProfile.read_json(profile.read_bytes()).model_copy(
        update={"layout": Layout(id="acceptance-1", version=1, template_sha256=layout_digest("curated"))}
    )
    write(profile, profile_data)
    call("confirm-layout", "--profile", str(profile), "--family", "curated", "--installed")
    light, _ = render_cards(
        data, plan, load_approvals(approvals), Settings("de", profile=profile_data, profile_path=profile)
    )
    assert light[0].note_id == old["cards"][0]["note_id"]
    assert light[0].card_key == old["cards"][0]["card_key"]
    assert [(r.subject, r.status) for r in light[0].requirements] == [
        ("rule", "approved"),
        *(([("mnemonic", "approved")]) if mnemonic else []),
        ("rendered_card", "unreviewed"),
    ]
    assert "Warum diese Antwort?" in light[0].back_html
    assert "acceptance-panel" not in light[0].front_html
    assert "Menschlich geprüft" not in light[0].back_html
    assert light[0].approval_back_html is not None and "Menschlich geprüft" in light[0].approval_back_html
    dark, _ = render_cards(
        data, plan, load_approvals(approvals), Settings("de", theme="dark", profile=profile_data, profile_path=profile)
    )
    assert dark[0].note_id == light[0].note_id
    assert dark[0].fingerprint != light[0].fingerprint
    assert dark[0].requirements[0] == light[0].requirements[0]
    from curated_cards_test import csv_note

    from latinitas_cards.curated_export import export_curated

    settings = Settings("de", profile=profile_data, profile_path=profile)
    approvals = decisions(
        tmp_path / "curated",
        {"source_scope": data.source_scope, "cards": [c.report() for c in light]},
        ("rule", "mnemonic", "rendered_card"),
    )
    history = load_approvals(approvals)
    accepted, _ = render_cards(data, plan, history, settings)
    assert all(c.reason == "accepted" for c in accepted)
    result = export_curated(tmp_path / "export-run", tmp_path / "curated.csv", lambda: (data, plan, history), settings)
    row = csv_note(tmp_path / "curated.csv")
    answer = next(v for k, v in row.items() if k.endswith("Answer") and v)
    assert "Menschlich geprüft" in answer and "Kartendarstellung: freigegeben" in answer
    assert answer == accepted[0].back_html == light[0].approval_back_html
    assert {"User::Keep", "SourceCase"} <= set(row["Tags"].split())
    assert ("Eselsbrücke: freigegeben" in answer) == mnemonic
    assert all(c["acceptance"] == "human" for c in json.loads(row["Acceptance Data"])["cards"])
    assert result["exported"] == len(accepted)
    collision = data.model_copy(
        update={"items": (data.items[0].model_copy(update={"tags": ("LATINITAS::BASIS::curated",)}),)}
    )
    with pytest.raises(ValueError, match="collision"):
        export_curated(
            tmp_path / "export-run", tmp_path / "collision.csv", lambda: (collision, plan, history), settings
        )
    assert not (tmp_path / "collision.csv").exists()
    from latinitas_cards.curated import make_approval

    prior = next(a for a in history if a.subject == "rendered_card")
    rejection = make_approval(
        **{
            k: getattr(prior, k)
            for k in ("source_scope", "item_key", "subject", "subject_key", "language", "fingerprint")
        },
        decision="rejected",
        reviewer="isolated engineering fixture",
        reason="Synthetic withdrawal control",
        reviewed_at="2026-11-19T01:00:00Z",
        supersedes=(prior.approval_id,),
    )
    withdrawn, _ = render_cards(data, plan, (*history, rejection), settings)
    assert withdrawn[0].reason == "rendered_card:rejected"
    assert "Menschlich geprüft" not in withdrawn[0].back_html


def test_theme_and_opt_in_reuse_linguistic_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from grammar_pipeline_test import rebind

    from latinitas_cards.pipeline_consumers import preview

    profile, objectives, root = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    adapt(profile, objectives, root)
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    before = preview(root)
    raw = json.loads(profile.read_text())
    raw["layout"] = dict(id="acceptance-1", version=1, template_sha256=layout_digest("grammar"))
    write(profile, raw)
    call("confirm-layout", "--profile", str(profile), "--installed")
    deck_path = tmp_path / "deck-profile.json"
    deck = json.loads(deck_path.read_text())
    deck["morphology"].update(theme="monochrome", appearance="dark", comparison="disclosure")
    write(deck_path, deck)
    rebind(profile, json.loads((tmp_path / "inputs.json").read_text()))
    after = preview(root, profile_path=profile)
    first = json.loads(json.dumps(before["cards"]))
    second = json.loads(json.dumps(after["cards"]))
    assert [(c["note_id"], c["card_key"], c["state"], c["assertions"]) for c in first] == [
        (c["note_id"], c["card_key"], c["state"], c["assertions"]) for c in second
    ]
    assert all(c["state"] == "accepted" for c in second)
    assert all("morphology-dark" in c["back_html"] and "morphology-monochrome" in c["back_html"] for c in second)
    assert all("acceptance-panel" not in c["front_html"] for c in second)


def test_mixed_sibling_export_receipt_replays_conservative_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latinitas_cards.acceptance import check_envelope, save_acceptance, save_stage
    from latinitas_cards.acceptance_records import Check, Human, canonical, digest
    from latinitas_cards.pipeline_consumers import Objectives, current_cards, decide
    from latinitas_cards.pipeline_profile import PipelineProfile, RecipeOverrides, resolve_recipes

    profile, objectives, root = fixture(tmp_path, preset=None)
    raw = json.loads(profile.read_text())
    choices = RecipeOverrides(include=("ppp_recognition",), exclude=())
    raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
    write(profile, raw)
    install_synthetic(monkeypatch, profile, objectives, root)
    raw = json.loads(profile.read_text())
    raw["layout"] = dict(id="acceptance-1", version=1, template_sha256=layout_digest("grammar"))
    write(profile, raw)
    call("confirm-layout", "--profile", str(profile), "--installed")
    adapt(profile, objectives, root)
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    cards, _ = current_cards(
        profile, PipelineProfile.read_json(profile.read_bytes()), Objectives.read_json(objectives.read_bytes()), root
    )
    card = next(c for c in cards if c.recipe == "ppp_recognition")
    envelope = check_envelope(card.binding, card.versions, "human", card.claims.assertions)
    stage = save_stage(
        root,
        run_id="synthetic",
        stage_id="synthetic-human",
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
        evidence=card.claims.assertions[0].evidence,
        alternatives=(),
        objections=(),
        input_sha256=digest(envelope),
    )
    decision = card.claims.decisions[0].reader.decide(
        checks=(check,),
        origin="human",
        stage_ids=(stage.stage_id,),
        human=Human(
            reviewer="isolated engineering fixture",
            reason="No linguistic approval",
            reviewed_sha256=digest({"binding": card.binding, "assertions": card.claims.assertions}),
        ),
    )
    save_acceptance(card.claims.decisions[0].path, decision)
    decide(root)
    first = export(root, tmp_path / "mixed.csv")
    receipt = json.loads(json.dumps(first["export_receipt"]))
    siblings = [c for c in receipt["cards"] if c["note_id"] == card.claims.note_id]
    assert {c["acceptance"] for c in siblings} == {"human", "automatic"}
    assert all("latinitas::acceptance::mixed" in c["owned_tags"] for c in siblings)
    assert all("latinitas::acceptance::human" not in c["owned_tags"] for c in siblings)
    assert canonical(export(root, tmp_path / "replay.csv")) == canonical(first)


@pytest.mark.parametrize("recipe", [None, "phrase_translation", "phrase_recognition", "phrase_form_cloze"])
def test_package_source_tags_are_preserved_and_bound_to_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipe: str | None
) -> None:
    from curated_cards_test import csv_note
    from grammar_pipeline_test import phrase_fixture

    from latinitas_cards.acceptance_records import canonical
    from latinitas_cards.phrase_inputs import PhraseInputV2
    from latinitas_cards.pipeline_consumers import input_binding, preview
    from latinitas_cards.pipeline_profile import RecipeOverrides, resolve_recipes
    from latinitas_cards.profile import DeckProfile

    profile, objectives, root = phrase_fixture(tmp_path, "de", "verb-noun") if recipe else fixture(tmp_path)
    if recipe is not None and recipe != "phrase_translation":
        raw = json.loads(profile.read_text())
        choices = RecipeOverrides(
            include=(recipe,), exclude=("form_translation", "ppp_translation", "phrase_translation")
        )
        raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
        write(profile, raw)
        targets = json.loads(objectives.read_text())
        targets["phrases"][0].update(recipe=recipe, target="verb", distinction="analysis")
        write(objectives, targets)
    with (tmp_path / "deck.csv").open() as stream:
        rows = list(csv.reader(stream))
    package = tmp_path / "deck.apkg"
    database = tmp_path / "source.anki2"
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["source"] = package.name
    raw = json.loads(profile.read_text())
    next(i for i in raw["inputs"] if i["kind"] == "deck")["path"] = package.name
    write(profile, raw)

    def source_tags(extra: str = "") -> None:
        database.unlink(missing_ok=True)
        with sqlite3.connect(database) as con:
            con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, tags TEXT, flds TEXT)")
            con.execute("CREATE TABLE notetypes (id INTEGER PRIMARY KEY, name TEXT)")
            con.execute("CREATE TABLE fields (ntid INTEGER, ord INTEGER, name TEXT)")
            con.execute("INSERT INTO notetypes VALUES (10, 'Vocabulary')")
            con.executemany("INSERT INTO fields VALUES (10, ?, ?)", enumerate(rows[0]))
            con.executemany(
                "INSERT INTO notes VALUES (?, ?, 10, ?, ?)",
                (
                    (i, r[0], f"User::Keep SourceCase::{r[0]} {extra}", "\x1f".join(r))
                    for i, r in enumerate(rows[1:], 1)
                ),
            )
        with zipfile.ZipFile(package, "w") as archive:
            archive.write(database, "collection.anki2")
        deck = DeckProfile.model_validate_json((tmp_path / "deck-profile.json").read_bytes())
        data["binding"] = input_binding(
            PhraseInputV2.model_validate(data), deck, package.read_bytes(), "synthetic-scope"
        ).model_dump()
        write(tmp_path / "inputs.json", data)
        raw = json.loads(profile.read_text())
        for item in raw["inputs"]:
            item["sha256"] = hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest()
        write(profile, raw)

    source_tags()
    install_synthetic(monkeypatch, profile, objectives, root)
    raw = json.loads(profile.read_text())
    raw["layout"] = dict(id="acceptance-1", version=1, template_sha256=layout_digest("grammar"))
    write(profile, raw)
    call("confirm-layout", "--profile", str(profile), "--installed")
    adapt(profile, objectives, root)
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    before = json.loads(json.dumps(preview(root)))
    first = json.loads(json.dumps(export(root, tmp_path / "first.csv")))
    row = csv_note(tmp_path / "first.csv")
    assert "User::Keep" in row["Tags"].split()
    if recipe:
        assert {"SourceCase::n", "SourceCase::v"} <= set(row["Tags"].split())
    assert all("User::Keep" not in c["owned_tags"] for c in first["export_receipt"]["cards"])
    original = (tmp_path / "first.csv").read_bytes()
    source_tags("AddedCase")
    with pytest.raises(ValueError, match="selection snapshot"):
        export(root, tmp_path / "second.csv", profile_path=profile)
    assert not (tmp_path / "second.csv").exists()
    after = json.loads(json.dumps(preview(root, profile_path=profile)))
    assert [(c["note_id"], c["state"], c["assertions"]) for c in before["cards"]] == [
        (c["note_id"], c["state"], c["assertions"]) for c in after["cards"]
    ]
    source_tags("Latinitas::Acceptance::human")
    with pytest.raises(ValueError, match="collision"):
        export(root, tmp_path / "collision.csv", profile_path=profile)
    assert not (tmp_path / "collision.csv").exists()
    assert (tmp_path / "first.csv").read_bytes() == original
    source_tags()
    assert canonical(export(root, tmp_path / "historical.csv", profile_path=profile)) == canonical(first)


def test_combined_opt_in_profiles_remain_independently_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from curated_cards_test import call as curated_call
    from curated_cards_test import decisions
    from curated_cards_test import fixture as curated_fixture

    args = curated_fixture(tmp_path / "curated", "de")
    profile, objectives, root = fixture(tmp_path / "grammar", preset=None)
    raw = json.loads(profile.read_text())
    input_path = tmp_path / "curated/input.json"
    raw["inputs"].append(
        dict(kind="curated", path=str(input_path), sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
    )
    write(profile, raw)
    install_synthetic(monkeypatch, profile, objectives, root)
    raw = json.loads(profile.read_text())
    raw["layout"] = dict(id="acceptance-1", version=1, template_sha256=layout_digest("grammar"))
    write(profile, raw)
    call("confirm-layout", "--profile", str(profile), "--installed")
    adapt(profile, objectives, root)
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    curated_profile = profile.with_name("curated-profile.json")
    raw["layout"]["template_sha256"] = layout_digest("curated")
    write(curated_profile, raw)
    call("confirm-layout", "--profile", str(curated_profile), "--family", "curated", "--installed")
    extra = ["--profile", str(curated_profile), "--production-profile", str(profile), "--production-run", str(root)]
    report = curated_call("preview", args, *extra)
    assert report["production"]["cards"] and all(c["state"] == "accepted" for c in report["production"]["cards"])
    history = decisions(tmp_path / "curated", report)
    from curated_cards_test import runner

    from latinitas_cards.cli import app

    for production in ([], ["--production-profile", str(curated_profile)]):
        refused = runner.invoke(
            app,
            [
                "curated",
                "export",
                *args,
                "--profile",
                str(curated_profile),
                "--production-run",
                str(root),
                *production,
                "--approvals",
                str(history),
                "--run",
                str(tmp_path / "refused-run"),
                "--output",
                str(tmp_path / "refused.csv"),
                "--production-output",
                str(tmp_path / "refused-production.csv"),
            ],
        )
        assert refused.exit_code != 0
        assert not (tmp_path / "refused.csv").exists()
    result = curated_call(
        "export",
        args,
        *extra,
        "--approvals",
        str(history),
        "--run",
        str(tmp_path / "curated-run"),
        "--output",
        str(tmp_path / "curated.csv"),
        "--production-output",
        str(tmp_path / "production.csv"),
    )
    assert result["exported"] == 1 and result["production"]["exported"] > 0
