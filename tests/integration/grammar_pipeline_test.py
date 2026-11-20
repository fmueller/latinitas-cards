"""Public consumers with isolated synthetic authority, never a linguistic study."""

import csv
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from typer.testing import CliRunner

from latinitas_cards import acceptance_policy as policies
from latinitas_cards.acceptance import check_envelope, save_stage
from latinitas_cards.acceptance_records import Binding, Check, StageName, canonical, digest
from latinitas_cards.claim_review import Evidence
from latinitas_cards.cli import app
from latinitas_cards.grammar_layout import GRAMMAR_LAYOUT, LAYOUT_SHA256
from latinitas_cards.phrase_inputs import PhraseInputV2
from latinitas_cards.pipeline_consumers import Analysis, Rendered, current_cards, input_binding
from latinitas_cards.pipeline_profile import PipelineProfile, RecipeOverrides, resolve_recipes
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig

runner = CliRunner()
EVIDENCE = Evidence("synthetic-reference", "engineering-only", "1", "Isolated fixture, not evaluated authority")


def call(*args: str) -> dict[str, Any]:
    result = runner.invoke(app, ["pipeline", *args, "--json"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return dict(json.loads(result.stdout))


def write(path: Path, value: object) -> None:
    path.write_bytes(canonical(value))


def rebind(profile: Path, data: dict[str, Any]) -> None:
    """Explicitly confirm changed synthetic inputs, without preserving their approvals."""
    base = profile.parent
    deck = DeckProfile.model_validate_json((base / "deck-profile.json").read_bytes())
    parsed = PhraseInputV2.model_validate(data)
    data["binding"] = input_binding(parsed, deck, (base / "deck.csv").read_bytes(), "synthetic-scope").model_dump()
    write(base / "inputs.json", data)
    raw = json.loads(profile.read_text())
    raw["deck_profile"]["sha256"] = hashlib.sha256((base / "deck-profile.json").read_bytes()).hexdigest()
    for item in raw["inputs"]:
        item["sha256"] = hashlib.sha256((base / item["path"]).read_bytes()).hexdigest()
    write(profile, raw)


def fixture(root: Path, language: str = "de", preset: str | None = "grammar-study") -> tuple[Path, Path, Path]:
    root.mkdir(exist_ok=True, parents=True)
    meanings = ("lieben", "Mädchen", "gut", "geliebt") if language == "de" else ("love", "girl", "good", "loved")
    with (root / "deck.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ID", "Lemma", "Parts", "Meaning"))
        writer.writerows(
            (
                ("v", "amō", "amō — amāre — amāvī — amātus", meanings[0]),
                ("n", "puella", "puella — puellae — puellam — puellārum — puellā", meanings[1]),
                ("a", "bonus", "bonus — bona — bonum — bonam — bonārum", meanings[2]),
                ("p", "amō", "amō — amāre — amāvī — amātus", meanings[3]),
            )
        )
    deck = DeckProfile.default(
        note_type="Vocabulary",
        lexical_entry_field="Lemma",
        principal_parts_field="Parts",
        meaning_field="Meaning",
        language_tag=language,
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
        principal_part_roles=("present_1s", "present_infinitive", "perfect_1s", "ppp"),
        generated_note_type="Synthetic grammar-v1",
    )
    raw_deck = deck.model_dump()
    raw_deck["morphology"]["terminology"]["language"] = "user"
    deck = DeckProfile.model_validate(raw_deck)
    write(root / "deck-profile.json", deck)
    write(
        root / "deck.csv.latinitas.json",
        {
            "schema_version": 2,
            "source_columns": ["ID", "Lemma", "Parts", "Meaning"],
            "source_scope": "synthetic-scope",
            "entries": [],
        },
    )

    def form(key: str, text: str, features: dict[str, str], role: str = "other") -> dict[str, Any]:
        return dict(key=key, text=text, features=features, source_role=role, role_evidence=[EVIDENCE])

    nominal = dict(case="nominative", number="singular", gender="feminine")
    entries = [
        dict(
            identity="v",
            pos="verb",
            gender=None,
            evidence=[EVIDENCE],
            senses=[
                dict(
                    key="sense",
                    meaning=meanings[0],
                    evidence=[EVIDENCE],
                    valency={"subject": "nominative", "object": "accusative"},
                )
            ],
            forms=[
                form(
                    "present",
                    "amō",
                    dict(person="1", number="singular", tense="present", mood="indicative", voice="active"),
                    "present_1s",
                ),
                form("infinitive", "amāre", {}, "present_infinitive"),
            ],
        ),
        dict(
            identity="n",
            pos="noun",
            gender="feminine",
            evidence=[EVIDENCE],
            senses=[dict(key="sense", meaning=meanings[1], evidence=[EVIDENCE])],
            forms=[
                form("nom", "puella", nominal),
                form("gen", "puellae", {**nominal, "case": "genitive"}),
                form("acc", "puellam", {**nominal, "case": "accusative"}),
                form("gen-pl", "puellārum", {**nominal, "case": "genitive", "number": "plural"}),
                form("abl", "puellā", {**nominal, "case": "ablative"}),
            ],
        ),
        dict(
            identity="a",
            pos="adjective",
            gender=None,
            evidence=[EVIDENCE],
            senses=[dict(key="sense", meaning=meanings[2], evidence=[EVIDENCE])],
            forms=[
                form("masc", "bonus", {**nominal, "gender": "masculine"}),
                form("fem", "bona", nominal),
                form("neut", "bonum", {**nominal, "gender": "neuter"}),
            ],
        ),
        dict(
            identity="p",
            pos="verb",
            gender=None,
            evidence=[EVIDENCE],
            senses=[dict(key="sense", meaning=meanings[3], evidence=[EVIDENCE])],
            forms=[
                form("ppp", "amātus", {**nominal, "gender": "masculine", "tense": "perfect", "voice": "passive"}, "ppp")
            ],
        ),
    ]
    placeholder = Binding(
        source_scope="synthetic-scope",
        source_identity="phrase-input-v2",
        source_sha256=digest(0),
        extraction_sha256=digest(0),
        mapping_sha256=digest(0),
        task_sha256=digest(0),
        language="de" if language == "de" else "en",
        config_sha256=digest(0),
    )
    data = PhraseInputV2.model_validate(
        dict(
            version=2,
            source="deck.csv",
            profile="deck-profile.json",
            manifest=None,
            entries=entries,
            patterns=[],
            example_selection=None,
            binding=placeholder.model_dump(),
            registry_version="v050-support/1",
        )
    )
    data = data.model_copy(
        update={"binding": input_binding(data, deck, (root / "deck.csv").read_bytes(), "synthetic-scope").model_dump()}
    )
    write(root / "inputs.json", data)
    overrides = RecipeOverrides(include=(), exclude=())
    profile = dict(
        format="latinitas-pipeline-profile",
        version=1,
        deck_profile=dict(
            path="deck-profile.json", sha256=hashlib.sha256((root / "deck-profile.json").read_bytes()).hexdigest()
        ),
        inputs=[
            dict(kind=kind, path=path, sha256=hashlib.sha256((root / path).read_bytes()).hexdigest())
            for kind, path in (("deck", "deck.csv"), ("phrases", "inputs.json"))
        ],
        language=language,
        preset=preset,
        recipe_overrides=overrides,
        recipes=resolve_recipes(preset, overrides),
        selection=dict(patterns=[], entries=[], variants_per_binding=1, output_limit=None),
        processing=dict(max_source_units=None, max_calls=200, max_tokens=200000, timeout_seconds=60, retries=1),
        providers=dict(proposal=None, counter_check=None),
        policy=dict(id="v050-disabled", version="1", sha256=digest(policies.BOOTSTRAP)),
        layout=dict(id="grammar-v1", version=1, template_sha256=LAYOUT_SHA256),
    )
    write(root / "pipeline.json", profile)
    objectives = []
    for entry, recipe, anchors, targets in (
        (
            "v",
            "verb_production",
            ("present", "infinitive"),
            [
                dict(person="2", number="singular", tense="present", mood="indicative", voice="active"),
                dict(person="3", number="plural", tense="present", mood="indicative", voice="active"),
            ],
        ),
        (
            "n",
            "noun_declension",
            ("nom", "gen"),
            [{**nominal, "case": "accusative"}, {**nominal, "case": "genitive", "number": "plural"}],
        ),
        (
            "a",
            "adjective_declension",
            ("masc", "fem", "neut"),
            [{**nominal, "case": "accusative"}, {**nominal, "number": "plural", "gender": "neuter"}],
        ),
    ):
        for target in targets:
            objectives.append(
                dict(
                    object_key="production",
                    entry=entry,
                    sense="sense",
                    form=None,
                    anchors=anchors,
                    recipe=recipe,
                    target=target,
                    distinction="analysis",
                    context="",
                    translations=[],
                    explanation="",
                    caveats=[],
                    evidence=[EVIDENCE],
                )
            )
    write(
        root / "objectives.json", dict(format="latinitas-grammar-objectives", version=1, forms=objectives, phrases=[])
    )
    return root / "pipeline.json", root / "objectives.json", root / "run"


def phrase_fixture(root: Path, language: str, family: str) -> tuple[Path, Path, Path]:
    profile, objectives, run = fixture(root, language, preset=None)
    data = json.loads((root / "inputs.json").read_text())
    third = {"person": "3", "number": "singular", "tense": "present", "mood": "indicative", "voice": "active"}
    data["entries"][0]["forms"].append(
        dict(key="third", text="amat", features=third, source_role="other", role_evidence=[EVIDENCE])
    )
    (root / "deck.csv").write_text((root / "deck.csv").read_text().replace("amātus", "amātus — amat"))
    bindings = {
        "preposition-noun": {"object": ("n", "abl")},
        "noun-genitive": {"head": ("n", "nom"), "dependent": ("n", "gen-pl")},
        "adjective-noun": {"head": ("n", "nom"), "modifier": ("a", "fem")},
        "verb-noun": {"subject": ("n", "nom"), "object": ("n", "acc"), "verb": ("v", "third")},
    }[family]
    entries = {e["identity"]: e for e in data["entries"]}
    features = {
        role: next(f["features"] for f in entries[entry]["forms"] if f["key"] == form)
        for role, (entry, form) in bindings.items()
    }
    data["patterns"] = [
        dict(
            key="pattern",
            family=family,
            function_words=["cum"] if family == "preposition-noun" else [],
            roles={
                role: dict(pos=entries[entry]["pos"], senses=[entry + "/sense"], requires=features[role])
                for role, (entry, _) in bindings.items()
            },
            evidence=[EVIDENCE],
        )
    ]
    rebind(profile, data)
    raw = json.loads(profile.read_text())
    choices = RecipeOverrides(include=(), exclude=("form_translation", "ppp_translation"))
    raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
    write(profile, raw)
    reference = dict(
        language=language,
        evidence=[EVIDENCE],
        readings=[
            dict(
                translations=["synthetic reference", "synthetic alternative"],
                features=features,
                contributions={role: "synthetic " + role for role in bindings},
                explanation="Synthetic engineering reference; no naturalness approval.",
            )
        ],
    )
    write(
        objectives,
        dict(
            format="latinitas-grammar-objectives",
            version=1,
            forms=[],
            phrases=[
                dict(
                    object_key="phrase",
                    selection=dict(
                        pattern="pattern",
                        bindings=[
                            dict(role=role, entry=entry, sense="sense", form=form)
                            for role, (entry, form) in bindings.items()
                        ],
                    ),
                    recipe="phrase_translation",
                    target=None,
                    distinction="meaning",
                    reference=reference,
                )
            ],
        ),
    )
    return profile, objectives, run


@pytest.mark.parametrize("recipe", ["phrase_translation", "phrase_recognition", "phrase_form_cloze"])
@pytest.mark.parametrize("language", ["de", "en"])
@pytest.mark.parametrize(
    "family,text",
    [
        ("preposition-noun", "cum puellā"),
        ("noun-genitive", "puella puellārum"),
        ("adjective-noun", "puella bona"),
        ("verb-noun", "puella puellam amat"),
    ],
)
def test_public_controlled_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str, family: str, text: str, recipe: str
) -> None:
    profile, objectives, root = phrase_fixture(tmp_path, language, family)
    target = {
        "preposition-noun": "object",
        "noun-genitive": "dependent",
        "adjective-noun": "modifier",
        "verb-noun": "verb",
    }[family]
    if recipe != "phrase_translation":
        raw = json.loads(profile.read_text())
        choices = RecipeOverrides(
            include=(recipe,), exclude=("form_translation", "ppp_translation", "phrase_translation")
        )
        raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
        write(profile, raw)
        targets = json.loads(objectives.read_text())
        targets["phrases"][0].update(recipe=recipe, target=target, distinction="analysis")
        write(objectives, targets)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    cards = call("preview", "--run", str(root))["cards"]
    assert len(cards) == 1 and cards[0]["state"] == "accepted"
    selected = json.loads(objectives.read_text())["phrases"][0]["selection"]["bindings"]
    binding = next(b for b in selected if b["role"] == target)
    entries = json.loads((tmp_path / "inputs.json").read_text())["entries"]
    form = next(
        f for e in entries if e["identity"] == binding["entry"] for f in e["forms"] if f["key"] == binding["form"]
    )
    expected_front = text.replace(form["text"], "____") if recipe == "phrase_form_cloze" else text
    assert cards[0]["prompt"].splitlines()[0] == expected_front
    assert "synthetic alternative" in cards[0]["answer"]
    assert "synthetic" not in cards[0]["prompt"]
    if recipe == "phrase_form_cloze":
        assert cards[0]["answer"].splitlines()[0] == form["text"]
        assert form["text"] not in cards[0]["prompt"]
    else:
        assert "____" not in cards[0]["prompt"]
    assert {a["id"] for a in cards[0]["assertions"]} == {
        "construction",
        "translation-0",
        "explanation-0",
        "roles-0",
        "rendered",
    }
    assert call("export", "--run", str(root), "--output", str(tmp_path / "out.csv"))["exported"] == 1


@pytest.mark.parametrize(
    "family,mutation,reason",
    [
        ("preposition-noun", "government", "government"),
        ("adjective-noun", "agreement", "requires"),
        ("verb-noun", "valency", "valency"),
    ],
)
@pytest.mark.parametrize("language", ["de", "en"])
def test_public_wrong_construction_is_not_licensed(
    tmp_path: Path, family: str, mutation: str, reason: str, language: str
) -> None:
    profile, objectives, root = phrase_fixture(tmp_path, language, family)
    data = json.loads((tmp_path / "inputs.json").read_text())
    if mutation == "government":
        data["patterns"][0]["function_words"] = ["ad"]
    elif mutation == "agreement":
        data["entries"][2]["forms"][1]["features"]["gender"] = "masculine"
    else:
        data["entries"][0]["senses"][0]["valency"]["object"] = "dative"
    rebind(profile, data)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    result = call("preview", "--run", str(root))
    assert result["cards"] == []
    assert any(reason in item["reason"] for item in result["skipped"])
    assert call("export", "--run", str(root), "--output", str(tmp_path / "out.csv"))["exported"] == 0


def install_synthetic(monkeypatch: pytest.MonkeyPatch, profile_path: Path, objectives_path: Path, root: Path) -> None:
    from latinitas_cards.pipeline_consumers import Objectives

    profile = PipelineProfile.read_json(profile_path.read_bytes())
    objectives = Objectives.read_json(objectives_path.read_bytes())
    cards, _ = current_cards(profile_path, profile, objectives, root)
    assert cards
    cells = {(a.kind, c.recipe, profile.language, c.material) for c in cards for a in c.claims.assertions}
    policy = policies.AutomaticPolicy(
        id="synthetic-t191-only",
        revision="1",
        checker_sha256=policies.implementation_digest(),
        comparison_sha256=policies.implementation_digest(),
        support_registry_sha256=digest(policies.SUPPORT_REGISTRY),
        versions_sha256=digest(cards[0].versions.model_dump(exclude={"policy", "policy_sha256"})),
        criteria_sha256=digest("not empirical approval"),
        scope=tuple(
            policies.ScopeCell(claim_class=k, recipe=r, language=language, material=m)
            for k, r, language, m in sorted(cells)
        ),
    )
    auth = policies.PolicyAuthorization(
        action="enable",
        policy_sha256=digest(policy),
        evaluation_report_sha256=digest("isolated fixture"),
        approved_by="fixture",
        approved_at="2026-11-18T00:00:00Z",
        supersedes=None,
    )
    monkeypatch.setattr(
        policies,
        "POLICY_REGISTRY",
        (canonical(policies.PolicyRegistryEntry(policy=policy, state="enabled", authorization=auth)).decode(),),
    )
    write(profile_path, {**profile.model_dump(), "policy": dict(id=policy.id, version="1", sha256=digest(policy))})


def prove_cards(profile_path: Path, objectives_path: Path, root: Path, *, unsupported: str = "") -> None:
    from latinitas_cards.pipeline_consumers import Objectives

    profile = PipelineProfile.read_json(profile_path.read_bytes())
    objectives = Objectives.read_json(objectives_path.read_bytes())
    cards, _ = current_cards(profile_path, profile, objectives, root)
    for card in cards:
        checks, stages = [], []
        for assertion in card.claims.assertions:
            envelope = check_envelope(
                card.binding, card.versions, "library", (assertion,), reference_constraints=(EVIDENCE,)
            )
            stage = save_stage(
                root,
                run_id="synthetic-t191",
                stage_id=digest((card.claims.note_id, card.claims.card_key, assertion.id)),
                stage="analyze",
                binding=card.binding,
                versions=card.versions,
                envelope=envelope,
            )
            stages.append(stage.stage_id)
            fail = card.recipe == unsupported and assertion.id == "rendered"
            checks.append(
                Check(
                    id=assertion.id,
                    assertion_ids=(assertion.id,),
                    method="library",
                    version=card.versions.checker,
                    result="unresolved" if fail else "pass",
                    evidence=(EVIDENCE,),
                    alternatives=(),
                    objections=("unsupported rendered clause",) if fail else (),
                    input_sha256=digest(envelope),
                )
            )
        write(
            card.analysis_path,
            Analysis(
                format="latinitas-grammar-analysis",
                version=1,
                binding=card.binding,
                versions=card.versions,
                assertions=card.claims.assertions,
                checks=tuple(checks),
                stage_ids=tuple(stages),
            ),
        )


@pytest.mark.parametrize("language", ["de", "en"])
def test_ppp_default_automatic_and_independent_opt_ins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    profile, objectives, root = fixture(tmp_path, language, preset=None)
    raw = json.loads(profile.read_text())
    choices = RecipeOverrides(
        include=("ppp_recognition", "adjective_declension", "form_cloze"), exclude=("form_translation",)
    )
    raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
    write(profile, raw)
    targets = json.loads(objectives.read_text())
    base = targets["forms"][0]
    targets["forms"] = [
        {
            **base,
            "entry": "p",
            "object_key": "ppp-production",
            "form": "ppp",
            "anchors": [],
            "recipe": "adjective_declension",
            "target": {"case": "accusative", "number": "singular", "gender": "feminine"},
        },
        {
            **base,
            "entry": "p",
            "object_key": "ppp-cloze",
            "form": "ppp",
            "anchors": [],
            "recipe": "form_cloze",
            "target": {},
            "context": "____",
            "translations": [],
        },
    ]
    write(objectives, targets)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    cards = call("preview", "--run", str(root))["cards"]
    assert {c["recipe"] for c in cards} == {"ppp_translation", "ppp_recognition", "adjective_declension", "form_cloze"}
    assert all(c["state"] == "accepted" for c in cards)
    assert next(c for c in cards if c["recipe"] == "adjective_declension")["answer"].splitlines()[0] == "amātam"
    assert next(c for c in cards if c["recipe"] == "ppp_translation")["prompt"] == "amātus\n" + (
        "Übersetze." if language == "de" else "Translate."
    )
    assert call("export", "--run", str(root), "--output", str(tmp_path / "ppp.csv"))["exported"] == 4


@pytest.mark.parametrize("language", ["de", "en"])
def test_public_contrasting_drills_profile_preview_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    profile, objectives, root = fixture(tmp_path, language)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    report = call("preview", "--profile", str(profile), "--run", str(root))
    assert len(report["cards"]) == 6
    assert {c["answer"].splitlines()[0] for c in report["cards"]} == {
        "amās",
        "amant",
        "puellam",
        "puellārum",
        "bonam",
        "bona",
    }
    assert all(c["state"] == "accepted" for c in report["cards"])
    assert len({(c["note_id"], c["card_key"]) for c in report["cards"]}) == 6
    for card in report["cards"]:
        assert card["answer"].splitlines()[0] not in card["prompt"]
        assert "____" not in card["prompt"]
        if card["recipe"] == "verb_production":
            assert all(
                token in card["prompt"]
                for token in (
                    ("Person", "Numerus", "Tempus", "Modus", "Genus verbi")
                    if language == "de"
                    else ("Person", "Number", "Tense", "Mood", "Voice")
                )
            )
        if card["recipe"] == "adjective_declension":
            assert ("Genus" if language == "de" else "Gender") in card["prompt"]
    result = call("export", "--run", str(root), "--output", str(tmp_path / "out.csv"))
    assert result["exported"] == 6
    with (tmp_path / "out.csv").open() as stream:
        rows = list(csv.reader(line for line in stream if not line.startswith("#")))
    assert len({row[0] for row in rows}) == 6
    csv_text = (tmp_path / "out.csv").read_text()
    assert all(form in csv_text for form in ("amās", "amant", "puellam", "puellārum", "bonam", "bona"))
    assert call("preview", "--run", str(root))["cards"] == report["cards"]


def test_default_and_disabled_authority_never_export_production(tmp_path: Path) -> None:
    profile, objectives, root = fixture(tmp_path, preset=None)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    report = call("preview", "--run", str(root))
    assert report["cards"]
    assert {c["recipe"] for c in report["cards"]} <= {"form_translation", "ppp_translation"}
    assert {c["reason"] for c in report["cards"]} == {"policy_not_enabled"}
    assert call("export", "--run", str(root), "--output", str(tmp_path / "out.csv"))["exported"] == 0


def test_cli_preset_saved_profile_and_absent_deck_targets(tmp_path: Path) -> None:
    profile, objectives, root = fixture(tmp_path, preset=None)
    source = tmp_path / "deck.csv"
    source.write_text(
        source.read_text().replace(" — puellam — puellārum — puellā", "").replace(" — bonam — bonārum", "")
    )
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][1]["forms"] = data["entries"][1]["forms"][:2]
    rebind(profile, data)
    original = profile.read_bytes()
    saved = tmp_path / "saved.json"
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
        "--preset",
        "grammar-study",
        "--save-profile",
        str(saved),
    )
    assert profile.read_bytes() == original
    result = call("preview", "--run", str(root))
    assert {c["answer"].splitlines()[0] for c in result["cards"]} == {
        "amās",
        "amant",
        "puellam",
        "puellārum",
        "bonam",
        "bona",
    }
    assert "form_cloze" not in result["recipes"]
    assert PipelineProfile.read_json(saved.read_bytes()).preset == "grammar-study"
    replay = call("preview", "--profile", str(saved), "--run", str(root))
    assert [(c["note_id"], c["card_key"]) for c in result["cards"]] == [
        (c["note_id"], c["card_key"]) for c in replay["cards"]
    ]
    excluded = call("preview", "--run", str(root), "--exclude-recipe", "verb_production")
    assert "verb_production" not in {c["recipe"] for c in excluded["cards"]}


@pytest.mark.parametrize(
    "mutation,reason", [("missing", "incomplete"), ("anchor", "anchor"), ("identity", "source entry")]
)
def test_bad_production_evidence_has_no_card(tmp_path: Path, mutation: str, reason: str) -> None:
    profile, objectives, root = fixture(tmp_path)
    raw = json.loads(objectives.read_text())
    raw["forms"] = raw["forms"][:1]
    if mutation == "missing":
        del raw["forms"][0]["target"]["person"]
    elif mutation == "anchor":
        data = json.loads((tmp_path / "inputs.json").read_text())
        data["entries"][0]["forms"][0]["features"]["person"] = "3"
        rebind(profile, data)
    else:
        raw["forms"][0]["entry"] = "unknown"
    write(objectives, raw)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    report = call("preview", "--run", str(root))
    assert report["cards"] == []
    assert any(reason in s["reason"] for s in report["skipped"])


@pytest.mark.parametrize("mutation", ["supine", "fourth", "contradictory", "no_evidence"])
def test_ppp_status_is_not_an_ending_or_position(tmp_path: Path, mutation: str) -> None:
    profile, objectives, root = fixture(tmp_path, preset=None)
    raw = json.loads(objectives.read_text())
    raw["forms"] = []
    write(objectives, raw)
    data = json.loads((tmp_path / "inputs.json").read_text())
    ppp = data["entries"][3]["forms"][0]
    if mutation in ("supine", "fourth"):
        deck_path = tmp_path / "deck-profile.json"
        deck = json.loads(deck_path.read_text())
        if mutation == "supine":
            deck["principal_parts"]["roles"][-1] = "supine"
        else:
            deck["principal_parts"]["roles"] = deck["principal_parts"]["roles"][:3]
        write(deck_path, deck)
    elif mutation == "contradictory":
        ppp["features"]["voice"] = "active"
    else:
        ppp["role_evidence"] = []
        with pytest.raises(ValueError, match="role evidence"):
            PhraseInputV2.model_validate(data)
        return
    rebind(profile, data)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    assert "ppp_translation" not in {c["recipe"] for c in call("preview", "--run", str(root))["cards"]}


def test_full_card_gate_mixed_siblings_and_authority_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, objectives, root = fixture(tmp_path)
    from latinitas_cards.acceptance_registry import POLICY_REGISTRY

    disabled = POLICY_REGISTRY
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root, unsupported="noun_declension")
    call("stage", "decide", "--run", str(root))
    report = call("preview", "--run", str(root))
    assert {c["recipe"] for c in report["cards"] if c["state"] == "accepted"} == {
        "verb_production",
        "adjective_declension",
    }
    assert call("export", "--run", str(root), "--output", str(tmp_path / "mixed.csv"))["exported"] == 4
    monkeypatch.setattr(policies, "POLICY_REGISTRY", disabled)
    rejected = runner.invoke(
        app, ["pipeline", "export", "--run", str(root), "--output", str(tmp_path / "withdrawn.csv"), "--json"]
    )
    assert rejected.exit_code != 0
    assert not (tmp_path / "withdrawn.csv").exists()


def test_recognition_preserves_syncretism_and_withholds_case_guess(tmp_path: Path) -> None:
    profile, objectives, root = fixture(tmp_path, preset=None)
    data = json.loads((tmp_path / "inputs.json").read_text())
    genitive = data["entries"][1]["forms"][1]
    data["entries"][1]["forms"].append(
        {**genitive, "key": "dative", "features": {**genitive["features"], "case": "dative"}}
    )
    rebind(profile, data)
    raw = json.loads(objectives.read_text())
    base = raw["forms"][0]
    raw["forms"] = [
        {
            **base,
            "object_key": "syncretism",
            "entry": "n",
            "form": "gen",
            "anchors": [],
            "recipe": "form_recognition",
            "target": {},
            "distinction": "analysis",
        }
    ]
    write(objectives, raw)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
        "--include-recipe",
        "form_recognition",
        "--exclude-recipe",
        "form_translation",
        "--exclude-recipe",
        "ppp_translation",
    )
    report = call("preview", "--run", str(root))
    selected = next(c for c in report["cards"] if c["prompt"].startswith("puellae"))
    assert "Genitiv" in selected["answer"] and "Dativ" in selected["answer"]
    raw["forms"][0]["distinction"] = "case"
    write(objectives, raw)
    other = tmp_path / "ambiguous-run"
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(other),
        "--include-recipe",
        "form_recognition",
    )
    report = call("preview", "--run", str(other))
    assert any("ambiguous recognition" in s["reason"] for s in report["skipped"])


@pytest.mark.parametrize("index_damage", [None, "{", "[]", "null"])
def test_bound_human_authority_is_distinct_from_automatic_policy(tmp_path: Path, index_damage: str | None) -> None:
    from latinitas_cards.acceptance import save_acceptance
    from latinitas_cards.acceptance_records import Human
    from latinitas_cards.pipeline_consumers import Objectives

    profile, objectives_path, root = fixture(tmp_path)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives_path),
        "--output",
        str(root),
    )
    cards, _ = current_cards(
        profile,
        PipelineProfile.read_json(profile.read_bytes()),
        Objectives.read_json(objectives_path.read_bytes()),
        root,
    )
    card = cards[0]
    envelope = check_envelope(card.binding, card.versions, "human", card.claims.assertions)
    stage = save_stage(
        root,
        run_id="human-fixture",
        stage_id="review",
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
        evidence=(EVIDENCE,),
        alternatives=(),
        objections=(),
        input_sha256=digest(envelope),
    )
    decision = card.claims.decisions[0].reader.decide(
        checks=(check,),
        origin="human",
        stage_ids=(stage.stage_id,),
        human=Human(
            reviewer="synthetic teacher",
            reason="Exact full-card engineering fixture",
            reviewed_sha256=digest({"binding": card.binding, "assertions": card.claims.assertions}),
        ),
    )
    save_acceptance(card.claims.decisions[0].path, decision)
    if index_damage is None:
        first = call("export", "--run", str(root), "--output", str(tmp_path / "human.csv"))
        assert first["exported"] == 1
        assert first["export_receipt"]["cards"][0]["acceptance"] == "human"
    # A later automatic analysis is not a revocation of this exact human review.
    (root / "analyses").mkdir()
    prove_cards(profile, objectives_path, root)
    call("stage", "decide", "--run", str(root))
    index_path = root / "stage-index.json"
    index_bytes = index_path.read_bytes() if index_path.exists() else None
    if index_damage is not None:
        index_path.write_text(index_damage)
        shown = call("preview", "--run", str(root))
        human = next(c for c in shown["cards"] if c["note_id"] == card.claims.note_id)
        assert human["state"] == "accepted"
        assert shown["skipped"]
    after = call("export", "--run", str(root), "--output", str(tmp_path / "human-after-analysis.csv"))
    assert after["exported"] == 1
    assert after["export_receipt"]["cards"][0]["decision_ids"] == [decision.decision_id]
    assert after["export_receipt"]["cards"][0]["acceptance"] == "human"
    if index_bytes is not None:
        index_path.write_bytes(index_bytes)
    elif index_damage is not None:
        index_path.unlink()
    assert decision.human is not None
    revised_review = card.claims.decisions[0].reader.decide(
        checks=(check,),
        origin="human",
        stage_ids=(stage.stage_id,),
        human=decision.human.model_copy(update={"reason": "Fresh independently bound synthetic review"}),
    )
    write(card.claims.decisions[0].path, revised_review)
    stale_replay = runner.invoke(
        app, ["pipeline", "export", "--run", str(root), "--output", str(tmp_path / "changed-review.csv"), "--json"]
    )
    assert stale_replay.exit_code != 0 and "reconciliation required" in stale_replay.output
    assert not (tmp_path / "changed-review.csv").exists()
    actions = call("handoff", "--run", str(root))["ineligible"]
    assert len(actions) == 1 and actions[0]["reason"] == "acceptance_or_rendering_changed"
    assert actions[0]["previous_decision_ids"] == [decision.decision_id]
    (tmp_path / "deck.csv").write_text((tmp_path / "deck.csv").read_text().replace("bonus", "malus"))
    result = runner.invoke(
        app, ["pipeline", "export", "--run", str(root), "--output", str(tmp_path / "stale.csv"), "--json"]
    )
    assert result.exit_code != 0 and not (tmp_path / "stale.csv").exists()


def test_selection_bounds_and_identity_do_not_follow_wording(tmp_path: Path) -> None:
    profile, objectives, root = fixture(tmp_path)
    raw = json.loads(objectives.read_text())
    first = raw["forms"][0]
    raw["forms"] = [first, {**first, "object_key": "another-variant", "explanation": "Different prose"}]
    write(objectives, raw)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    report = call("preview", "--run", str(root))
    assert len(report["cards"]) == 2
    assert not any(s["reason"] == "variant_limit" for s in report["skipped"])
    before = report["cards"][0]
    raw["forms"] = [{**raw["forms"][1], "explanation": "Replacement prose"}]
    write(objectives, raw)
    other = tmp_path / "rewritten"
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(other),
    )
    after = call("preview", "--run", str(other))["cards"][0]
    assert (before["note_id"], before["card_key"]) == (after["note_id"], after["card_key"])


@pytest.mark.parametrize("language", ["de", "en"])
@pytest.mark.parametrize("addition", [None, "ppp_recognition", "adjective_declension", "form_cloze"])
def test_ppp_recipe_opt_ins_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str, addition: str | None
) -> None:
    profile, objectives, root = fixture(tmp_path, language, preset=None)
    raw = json.loads(profile.read_text())
    choices = RecipeOverrides(include=() if addition is None else (addition,), exclude=("form_translation",))
    raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
    write(profile, raw)
    raw_objectives = json.loads(objectives.read_text())
    base = raw_objectives["forms"][0]
    raw_objectives["forms"] = [
        {
            **base,
            "entry": "p",
            "form": "ppp",
            "anchors": [],
            "recipe": "adjective_declension",
            "target": {"case": "accusative", "number": "singular", "gender": "feminine"},
        },
        {**base, "entry": "p", "form": "ppp", "anchors": [], "recipe": "form_cloze", "target": {}, "context": "____"},
    ]
    write(objectives, raw_objectives)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    first = call("stage", "decide", "--run", str(root))
    assert call("stage", "decide", "--run", str(root)) == first
    cards = call("preview", "--run", str(root))["cards"]
    expected = {"ppp_translation"} | ({addition} if addition else set())
    assert {c["recipe"] for c in cards} == expected
    assert all(c["state"] == "accepted" for c in cards)
    output = tmp_path / "out.csv"
    result = call("export", "--run", str(root), "--output", str(output))
    assert result["exported"] == len(expected)
    with output.open() as stream:
        lines = list(stream)
    columns = next(line.removeprefix("#columns:").strip().split(",") for line in lines if line.startswith("#columns:"))
    rows = list(csv.DictReader((line for line in lines if not line.startswith("#")), fieldnames=columns))
    assert len({r["LatinitasID"] for r in rows}) == len(rows)
    from latinitas_cards.grammar_layout import GRAMMAR_LAYOUT

    assert {t.slot.recipe for t in GRAMMAR_LAYOUT if any(r[t.slot.enabled_field] for r in rows)} == expected


def test_selected_work_limit_and_stale_analysis_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, objectives, root = fixture(tmp_path)
    raw = json.loads(profile.read_text())
    raw["selection"]["entries"] = ["v"]
    raw["processing"]["max_source_units"] = 1
    write(profile, raw)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    assert len(call("preview", "--run", str(root))["cards"]) == 1
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    assert len(call("stage", "decide", "--run", str(root))["decisions"]) == 1
    # A second run exercises both independent decisions; the first malformed file
    # cannot prevent a valid later sibling from being accepted or reused.
    monkeypatch.undo()
    other_profile, other_objectives, other = fixture(tmp_path / "siblings")
    # Reuse the fixture registry scope rather than enabling any installed policy.
    install_synthetic(monkeypatch, other_profile, other_objectives, other)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(other_profile),
        "--objectives",
        str(other_objectives),
        "--output",
        str(other),
    )
    (other / "analyses").mkdir()
    prove_cards(other_profile, other_objectives, other)
    from latinitas_cards.pipeline_consumers import Objectives

    cards, _ = current_cards(
        other_profile,
        PipelineProfile.read_json(other_profile.read_bytes()),
        Objectives.read_json(other_objectives.read_bytes()),
        other,
    )
    cards[0].analysis_path.write_text('{"broken": true}')
    report = call("stage", "decide", "--run", str(other))
    assert len(report["analysis_errors"]) == 1 and len(report["decisions"]) == 5
    assert call("stage", "decide", "--run", str(other)) == report


@pytest.mark.parametrize("language,expected", [("de", "geliebt"), ("en", "loved")])
def test_automatic_ppp_uses_form_meaning_not_lemma_gloss(tmp_path: Path, language: str, expected: str) -> None:
    profile, objectives, root = fixture(tmp_path, language, preset=None)
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][0]["forms"].append(data["entries"][3]["forms"][0])
    rebind(profile, data)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    cards = call("preview", "--run", str(root))["cards"]
    ppp = [c for c in cards if c["recipe"] == "ppp_translation"]
    assert len(ppp) == 2
    assert all(c["answer"].startswith(expected) for c in ppp)
    assert all(c["state"] == "withheld" for c in ppp)


@pytest.mark.parametrize("language", ["de", "en"])
def test_ppp_recognition_preserves_readings_without_guessing_number(tmp_path: Path, language: str) -> None:
    profile, objectives, root = fixture(tmp_path, language, preset=None)
    source = tmp_path / "deck.csv"
    source.write_text(source.read_text().replace("amātus", "amāta"))
    data = json.loads((tmp_path / "inputs.json").read_text())
    participle = data["entries"][3]["forms"][0]
    participle.update(text="amāta", features={**participle["features"], "gender": "feminine"})
    data["entries"][3]["forms"].append(
        {
            **participle,
            "key": "neuter-plural",
            "features": {**participle["features"], "number": "plural", "gender": "neuter"},
        }
    )
    rebind(profile, data)
    raw = json.loads(profile.read_text())
    choices = RecipeOverrides(
        include=("ppp_recognition",), exclude=("form_translation", "ppp_translation", "phrase_translation")
    )
    raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
    write(profile, raw)
    targets = json.loads(objectives.read_text())
    base = targets["forms"][0]
    targets["forms"] = [
        {
            **base,
            "entry": "p",
            "object_key": form,
            "recipe": "ppp_recognition",
            "form": form,
            "anchors": [],
            "target": {},
        }
        for form in ("ppp", "neuter-plural")
    ]
    write(objectives, targets)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    report = call("preview", "--run", str(root))
    assert len(report["cards"]) == 2
    for card in report["cards"]:
        assert "singular" in card["answer"].lower() and "plural" in card["answer"].lower()
        assert "PPP" not in card["prompt"]
    for target in targets["forms"]:
        target["distinction"] = "number"
    write(objectives, targets)
    other = tmp_path / "ambiguous"
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(other),
    )
    report = call("preview", "--run", str(other))
    assert report["cards"] == []
    assert {item["object_key"] for item in report["skipped"] if "ambiguous recognition" in item["reason"]} == {
        "ppp",
        "neuter-plural",
    }


def test_unknown_inflected_translation_needs_explicit_reference(tmp_path: Path) -> None:
    profile, objectives, root = fixture(tmp_path, "en", preset=None)
    source = tmp_path / "deck.csv"
    source.write_text(source.read_text().replace("loved", "carry"))
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][3]["senses"][0]["meaning"] = "carry"
    rebind(profile, data)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    report = call("preview", "--run", str(root))
    assert not any(card["recipe"] == "ppp_translation" for card in report["cards"])
    assert any("missing_form_translation_evidence" in item["reason"] for item in report["skipped"])


def test_export_repeats_authority_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from latinitas_cards import pipeline_consumers as consumers
    from latinitas_cards.acceptance_registry import POLICY_REGISTRY

    disabled = POLICY_REGISTRY
    profile, objectives, root = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    original = consumers.current_cards
    reconstructions = 0

    def revoke(*args: Any, **kwargs: Any) -> Any:
        nonlocal reconstructions
        reconstructions += 1
        if reconstructions == 2:
            monkeypatch.setattr(policies, "POLICY_REGISTRY", disabled)
        return original(*args, **kwargs)

    monkeypatch.setattr(consumers, "current_cards", revoke)
    output = tmp_path / "revoked.csv"
    result = runner.invoke(app, ["pipeline", "export", "--run", str(root), "--output", str(output), "--json"])
    assert result.exit_code != 0 and "eligibility changed before export" in result.output
    assert not output.exists()


def test_aggregate_analysis_decision_replays_its_card_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latinitas_cards.pipeline_consumers import Objectives

    profile, objectives, root = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    cards, _ = current_cards(
        profile, PipelineProfile.read_json(profile.read_bytes()), Objectives.read_json(objectives.read_bytes()), root
    )
    card = cards[0]
    analysis = Analysis.read_json(card.analysis_path.read_bytes())
    unrelated = card.claims.assertions[0].model_copy(update={"id": "unrelated", "value": "another objective"})
    envelope = check_envelope(card.binding, card.versions, "library", (unrelated,), reference_constraints=(EVIDENCE,))
    stage = save_stage(
        root,
        run_id="unrelated",
        stage_id="unrelated",
        stage="analyze",
        binding=card.binding,
        versions=card.versions,
        envelope=envelope,
    )
    extra = analysis.checks[0].model_copy(
        update={"id": "unrelated", "assertion_ids": ("unrelated",), "input_sha256": digest(envelope)}
    )
    write(
        card.analysis_path,
        {
            **analysis.model_dump(),
            "assertions": (*analysis.assertions, unrelated),
            "checks": (*analysis.checks, extra),
            "stage_ids": (*analysis.stage_ids, stage.stage_id),
        },
    )
    first = call("stage", "decide", "--run", str(root))
    assert first["analysis_errors"] == [] and len(first["decisions"]) == 6
    assert call("stage", "decide", "--run", str(root)) == first
    assert call("export", "--run", str(root), "--output", str(tmp_path / "aggregate.csv"))["exported"] == 6


@pytest.mark.parametrize("language,meaning,translation", [("de", "tragen", "getragen"), ("en", "carry", "carried")])
def test_explicit_non_table_ppp_reference_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str, meaning: str, translation: str
) -> None:
    profile, objectives, root = fixture(tmp_path, language, preset=None)
    source = tmp_path / "deck.csv"
    original = "geliebt" if language == "de" else "loved"
    text = source.read_text().replace(original, meaning)
    for before, after in (("amō", "portō"), ("amāre", "portāre"), ("amāvī", "portāvī"), ("amātus", "portātus")):
        text = text.replace(before, after)
    source.write_text(text)
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][3]["forms"][0]["text"] = "portātus"
    data["entries"][3]["senses"][0]["meaning"] = meaning
    rebind(profile, data)
    saved = json.loads(profile.read_text())
    choices = RecipeOverrides(include=(), exclude=("form_translation",))
    saved.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
    write(profile, saved)
    raw = json.loads(objectives.read_text())
    raw["forms"] = [
        {
            **raw["forms"][0],
            "entry": "p",
            "recipe": "ppp_translation",
            "form": "ppp",
            "anchors": [],
            "target": {},
            "translations": [translation],
        }
    ]
    write(objectives, raw)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    cards = call("preview", "--run", str(root))["cards"]
    ppp = next(card for card in cards if card["recipe"] == "ppp_translation")
    assert ppp["prompt"].splitlines()[0] == "portātus" and ppp["answer"].splitlines()[0] == translation
    assert ppp["state"] == "accepted"
    output = tmp_path / "reference.csv"
    assert call("export", "--run", str(root), "--output", str(output))["exported"] == 1
    assert "portātus" in output.read_text() and translation in output.read_text()


def test_exact_conflicting_objectives_withhold_both_representations(tmp_path: Path) -> None:
    profile, objectives, root = fixture(tmp_path)
    raw = json.loads(objectives.read_text())
    original = raw["forms"][0]
    raw["forms"] = [original, {**original, "explanation": "A different asserted explanation."}]
    write(objectives, raw)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    report = call("preview", "--run", str(root))
    assert report["cards"] == []
    assert any(item["reason"] == "duplicate semantic exercise" for item in report["skipped"])


def many_fixture(root: Path) -> tuple[Path, Path, Path]:
    profile, objectives, run = fixture(root)
    data = json.loads((root / "inputs.json").read_text())
    noun = data["entries"][1]
    data["entries"] = [{**noun, "identity": f"n{i:02}"} for i in range(27)]
    with (root / "deck.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ID", "Lemma", "Parts", "Meaning"))
        writer.writerows(
            (f"n{i:02}", "puella", "puella — puellae — puellam — puellārum — puellā", "Mädchen") for i in range(27)
        )
    rebind(profile, data)
    raw = json.loads(objectives.read_text())
    noun_objective = raw["forms"][2]
    raw["forms"] = [{**noun_objective, "entry": f"n{i:02}"} for i in reversed(range(27))]
    write(objectives, raw)
    return profile, objectives, run


def test_full_output_and_receipt_bound_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from latinitas_cards.providers import HTTPTransport

    requests: list[object] = []

    def capture(*args: object, **kwargs: object) -> bytes:
        requests.append((args, kwargs))
        raise AssertionError("offline selection must never dispatch a provider request")

    monkeypatch.setattr(HTTPTransport, "__call__", capture)
    profile, objectives, root = many_fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    full = call("export", "--run", str(root), "--output", str(tmp_path / "full.csv"))
    assert full["exported"] == 27
    from latinitas_cards.pipeline_consumers import Objectives

    current, _ = current_cards(
        profile, PipelineProfile.read_json(profile.read_bytes()), Objectives.read_json(objectives.read_bytes()), root
    )
    identities = {c.claims.note_id: c.binding.source_identity for c in current}
    assert [identities[c["note_id"]] for c in full["cards"]] == [f"n{i:02}" for i in range(27)]
    batches: list[tuple[str, str]] = []
    predecessor: list[str] = []
    for index, count in enumerate((10, 10, 7)):
        args = (
            "export",
            "--run",
            str(root),
            "--output",
            str(tmp_path / f"batch-{index}.csv"),
            "--output-limit",
            "10",
            *predecessor,
        )
        report = call(*args)
        assert report["exported"] == count
        assert len(report["accepted_unselected"]) == 27 - min(10 * (index + 1), 27)
        payload = Path(report["output"]).read_bytes()
        assert call(*args) == report  # Lost command response must not advance.
        assert Path(report["output"]).read_bytes() == payload
        batches.extend((c["note_id"], c["card_key"]) for c in report["cards"])
        predecessor = ["--after-receipt", report["receipt_id"]]
    assert len(set(batches)) == 27
    assert len({key for _, key in batches}) == 1  # Slot keys are note-local.
    assert batches == [(c["note_id"], c["card_key"]) for c in full["cards"]]
    for name, count in (("full", 27), ("batch-0", 10), ("batch-1", 10), ("batch-2", 7)):
        lines = (tmp_path / f"{name}.csv").read_text().splitlines()
        columns = next(line.removeprefix("#columns:") for line in lines if line.startswith("#columns:")).split(",")
        rows = [
            dict(zip(columns, row, strict=True))
            for row in csv.reader(line for line in lines if not line.startswith("#"))
        ]
        assert len(rows) == count
        slot = next(t.slot for t in GRAMMAR_LAYOUT if t.slot.recipe == "noun_declension")
        assert sum(row[slot.enabled_field] == "1" for row in rows) == count
        start = 0 if name == "full" else int(name[-1]) * 10
        assert [(row["LatinitasID"], row["CardKey"]) for row in rows] == [
            (card["note_id"], card["card_key"]) for card in full["cards"][start : start + count]
        ]
    alternate = tmp_path / "alternate.csv"
    assert call("export", "--run", str(root), "--output", str(alternate)) == full
    assert not alternate.exists()
    assert requests == []  # Captured transport count, including every replay, is zero.


def test_batch_invalidation_reconciles_only_changed_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, objectives, root = many_fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    first = call("export", "--run", str(root), "--output", str(tmp_path / "first.csv"), "--output-limit", "10")
    before = call("preview", "--run", str(root))["cards"]
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][0]["evidence"][0]["text"] = "Changed extraction evidence"
    rebind(profile, data)
    after = call("preview", "--run", str(root), "--profile", str(profile))["cards"]
    assert sum(c["state"] == "accepted" for c in after) == 26
    assert [(c["note_id"], c["card_key"]) for c in before] == [(c["note_id"], c["card_key"]) for c in after]
    args = [
        "export",
        "--run",
        str(root),
        "--profile",
        str(profile),
        "--output",
        str(tmp_path / "next.csv"),
        "--after-receipt",
        first["receipt_id"],
        "--json",
    ]
    failed = runner.invoke(app, ["pipeline", *args])
    assert failed.exit_code != 0 and "reconciliation required" in failed.output
    assert not (tmp_path / "next.csv").exists()
    continued = call(
        "export",
        "--run",
        str(root),
        "--profile",
        str(profile),
        "--output",
        str(tmp_path / "next.csv"),
        "--reconcile-receipt",
        first["receipt_id"],
    )
    assert continued["exported"] == 17
    assert len(continued["reconciliation"]["retained"]) == 9
    assert len(continued["reconciliation"]["ineligible"]) == 1
    handoff = continued["export_receipt"]["ineligible"]
    assert len(handoff) == 1
    assert handoff[0]["reason"] == "assertion_mismatch"
    assert handoff[0]["previous_decision_ids"]
    assert handoff[0]["resolution"] == "manual"
    assert "scheduling" in handoff[0]["required_action"]
    assert continued["application"]["state"] == "not_applied"
    assert "destination" in continued["application"]["required_action"]


def test_processing_limit_precedes_output_limit_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, objectives, root = many_fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    raw = json.loads(profile.read_text())
    raw["processing"]["max_source_units"] = 11
    raw["selection"]["output_limit"] = 2
    write(profile, raw)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    first = call("export", "--run", str(root), "--output", str(tmp_path / "first.csv"), "--output-limit", "7")
    assert first["exported"] == 7 and len(first["accepted_unselected"]) == 4
    assert len(first["deferred_source_work"]) == 16
    assert first["output_limit"] == {"source": "cli", "value": 7}
    raw["processing"]["max_source_units"] = None
    write(profile, raw)
    resumed = call(
        "export",
        "--run",
        str(root),
        "--profile",
        str(profile),
        "--output",
        str(tmp_path / "resumed.csv"),
        "--reconcile-receipt",
        first["receipt_id"],
    )
    assert resumed["exported"] == 2 and len(resumed["accepted_unselected"]) == 2
    assert resumed["deferred_source_work"] == []
    assert len(resumed["withheld"]) == 16  # Discovered but not yet checked, not rejected.
    new_refs = {(c["note_id"], c["card_key"]) for c in resumed["withheld"]}
    existing_analyses = {p: p.read_bytes() for p in (root / "analyses").glob("*.json")}
    prove_cards(profile, objectives, root)
    assert all(p.read_bytes() == content for p, content in existing_analyses.items())
    call("stage", "decide", "--run", str(root), "--profile", str(profile))
    completed = call(
        "export",
        "--run",
        str(root),
        "--profile",
        str(profile),
        "--output",
        str(tmp_path / "completed.csv"),
        "--output-limit",
        "100",
        "--reconcile-receipt",
        resumed["receipt_id"],
    )
    completed_refs = {(c["note_id"], c["card_key"]) for c in completed["cards"]}
    assert completed["exported"] == 18 and new_refs <= completed_refs
    assert not completed_refs & {(c["note_id"], c["card_key"]) for c in [*first["cards"], *resumed["cards"]]}


@pytest.mark.parametrize("invalidate_before_recovery", [False, True])
def test_batch_publication_recovery_lock_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalidate_before_recovery: bool
) -> None:
    from latinitas_cards import pipeline_batches as batches

    profile, objectives, root = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    output = tmp_path / "first.csv"
    args = ["pipeline", "export", "--run", str(root), "--output", str(output), "--output-limit", "2", "--json"]
    with batches.run_lock(root):
        busy = runner.invoke(app, args)
        assert busy.exit_code != 0 and "run_busy" in busy.output
        busy = runner.invoke(app, ["pipeline", "stage", "decide", "--run", str(root), "--json"])
        assert busy.exit_code != 0 and "run_busy" in busy.output
        assert not output.exists()
    atomic = batches._atomic

    def interrupted(path: Path, payload: bytes) -> None:
        if path.name == "index.json":
            raise OSError("simulated lost publication response")
        atomic(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(batches, "_atomic", interrupted)
        interrupted_result = runner.invoke(app, args)
        assert interrupted_result.exit_code != 0
    payload = output.read_bytes()
    if invalidate_before_recovery:
        pending = batches.Publication.read_json((root / "selection/pending.json").read_bytes())
        decisions = {p: p.read_bytes() for p in (root / "decisions").rglob("*.json")}
        for path in decisions:
            path.unlink()
        recovered_stale = call("export", "--run", str(root), "--output", str(tmp_path / "stale.csv"))
        assert recovered_stale["exported"] == 0
        assert not output.exists()  # An uncommitted, now-ineligible publication rolls back.
        for path, content in decisions.items():
            path.write_bytes(content)
        output.write_bytes(payload)
        orphan = runner.invoke(app, [*args[:-1], "--after-receipt", digest(pending.receipt), "--json"])
        assert orphan.exit_code != 0 and "foreign predecessor" in orphan.output
        return
    recovered = call(*args[1:-1])
    assert recovered["exported"] == 2 and output.read_bytes() == payload
    assert call(*args[1:-1]) == recovered
    foreign = runner.invoke(app, [*args[:-1], "--after-receipt", "0" * 64, "--json"])
    assert foreign.exit_code != 0 and "foreign predecessor" in foreign.output
    output.write_bytes(payload + b"tampered")
    tampered = runner.invoke(app, args)
    assert tampered.exit_code != 0 and "hash mismatch" in tampered.output


def test_redundancy_and_duplicate_refs_do_not_consume_output_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, objectives, root = fixture(tmp_path, preset=None)
    data = json.loads((tmp_path / "inputs.json").read_text())
    noun = data["entries"][1]
    noun["forms"].append({**noun["forms"][0], "key": "nom-alternative"})
    rebind(profile, data)
    raw = json.loads(objectives.read_text())
    template = raw["forms"][2]
    # Same object and assertion set; only the supplied form key differs. Another
    # objective with identical visible text must survive semantic suppression.
    raw["forms"] = [
        {
            **template,
            "object_key": "same",
            "recipe": "form_translation",
            "form": form,
            "anchors": [],
            "target": noun["forms"][0]["features"],
        }
        for form in ("nom-alternative", "nom")
    ]
    raw["forms"] += [
        {**raw["forms"][0], "context": "unused for translation"},
        {**raw["forms"][0], "object_key": "different-objective"},
    ]
    write(objectives, raw)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    preview = call("preview", "--run", str(root))
    assert {s["reason"] for s in preview["suppressed"]} == {"duplicate_card_ref", "redundant_variant"}
    from latinitas_cards.pipeline_consumers import Objectives

    parsed = Objectives.read_json(objectives.read_bytes())
    expected_winner = min(digest(o) for o in parsed.forms if o.object_key == "same")
    redundant = [s for s in preview["suppressed"] if s["reason"] == "redundant_variant"]
    assert redundant and {s["retained_candidate_key"] for s in redundant} == {expected_winner}
    reconstructed, _ = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), parsed, root)
    object_keys = {c.claims.note_id: c.objective.object_key for c in reconstructed}
    first = call("export", "--run", str(root), "--output", str(tmp_path / "first.csv"), "--output-limit", "1")
    assert first["exported"] == 1
    # Automatically enumerated source cells remain separate objects as well.
    remaining = len(first["accepted_unselected"])
    second = call(
        "export", "--run", str(root), "--output", str(tmp_path / "second.csv"), "--after-receipt", first["receipt_id"]
    )
    assert second["exported"] == remaining
    assert not ({tuple(c.values()) for c in first["cards"]} & {tuple(c.values()) for c in second["cards"]})
    survivors = [object_keys[c["note_id"]] for c in [*first["cards"], *second["cards"]]]
    assert survivors.count("same") == survivors.count("different-objective") == 1


def test_disabled_and_withdrawn_policies_cannot_replay_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, objectives, root = fixture(tmp_path)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    args = ("export", "--run", str(root), "--output", str(tmp_path / "disabled.csv"), "--output-limit", "2")
    disabled = call(*args)
    assert disabled["exported"] == 0
    assert {w["reason"] for w in disabled["withheld"]} == {"policy_not_enabled"}
    assert call(*args) == disabled
    assert (
        call(
            "export",
            "--run",
            str(root),
            "--output",
            str(tmp_path / "disabled-next.csv"),
            "--after-receipt",
            disabled["receipt_id"],
        )["exported"]
        == 0
    )
    profile, objectives, root = fixture(tmp_path / "enabled")
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    args = ("export", "--run", str(root), "--output", str(tmp_path / "enabled.csv"), "--output-limit", "2")
    enabled = call(*args)
    entry = policies.PolicyRegistryEntry.read_json(vars(policies)["POLICY_REGISTRY"][0])
    assert entry.authorization is not None
    withdrawal = entry.authorization.model_copy(
        update={"action": "withdraw", "supersedes": digest(entry.authorization)}
    )
    monkeypatch.setattr(policies, "AUTHORIZATION_HISTORY", (canonical(entry.authorization).decode(),))
    monkeypatch.setattr(
        policies,
        "POLICY_REGISTRY",
        (canonical(entry.model_copy(update={"state": "withdrawn", "authorization": withdrawal})).decode(),),
    )
    replay = runner.invoke(app, ["pipeline", *args, "--json"])
    assert replay.exit_code != 0 and "reconciliation required" in replay.output
    denied = call(
        "export",
        "--run",
        str(root),
        "--output",
        str(tmp_path / "withdrawn.csv"),
        "--reconcile-receipt",
        enabled["receipt_id"],
    )
    assert denied["exported"] == 0
    assert {w["reason"] for w in denied["withheld"]} == {"policy_withdrawn"}
    assert len(denied["reconciliation"]["ineligible"]) == 2
    receipt = enabled["export_receipt"]
    assert receipt["format"] == "latinitas-export-receipt"
    assert len(receipt["cards"]) == 2
    for card in receipt["cards"]:
        assert card["decision_ids"] and card["assertion_ids"]
        assert card["rendered_sha256"]
        assert "Tags" not in card["owned_fields"]
        assert card["owned_tags"] == []
    assert len(denied["export_receipt"]["ineligible"]) == 2
    assert {h["reason"] for h in denied["export_receipt"]["ineligible"]} == {"policy_withdrawn"}
    assert {tuple(h["previous_decision_ids"]) for h in denied["export_receipt"]["ineligible"]} == {
        tuple(c["decision_ids"]) for c in receipt["cards"]
    }
    unresolved = call("handoff", "--run", str(root))
    assert unresolved["ineligible"] == denied["export_receipt"]["ineligible"]
    assert unresolved["state"] == "not_applied"


def test_pattern_ranking_counts_supported_distinct_surfaces_not_aliases(tmp_path: Path) -> None:
    profile, objectives, root = phrase_fixture(tmp_path, "de", "preposition-noun")
    data = json.loads((tmp_path / "inputs.json").read_text())
    noun = data["entries"][1]
    ablative = next(f for f in noun["forms"] if f["key"] == "abl")
    noun["forms"] += [{**ablative, "key": "a-alias"}, {**ablative, "key": "z-plain", "text": "puella"}]
    rebind(profile, data)
    raw = json.loads(objectives.read_text())
    template = raw["phrases"][0]
    raw["phrases"] = [
        {
            **template,
            "object_key": key,
            "selection": {**template["selection"], "bindings": [{**template["selection"]["bindings"][0], "form": key}]},
        }
        for key in ("z-plain", "abl", "a-alias")
    ]
    write(objectives, raw)
    configured = json.loads(profile.read_text())
    configured["selection"]["variants_per_binding"] = 2
    write(profile, configured)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    report = call("preview", "--run", str(root))
    assert len([c for c in report["cards"] if c["recipe"] == "phrase_translation"]) == 3
    assert not any(s["reason"] == "variant_limit" for s in report["suppressed"])
    configured["selection"]["variants_per_binding"] = 1
    configured["selection"]["patterns"] = ["pattern"]
    configured["selection"]["entries"] = ["n"]
    write(profile, configured)
    bounded = call("preview", "--run", str(root), "--profile", str(profile))
    assert {s["object_key"] for s in bounded["suppressed"] if s["reason"] == "variant_limit"} == {"z-plain"}
    for filters in ({"entries": ["v"], "patterns": ["pattern"]}, {"entries": ["n"], "patterns": ["excluded"]}):
        configured["selection"].update(filters)
        write(profile, configured)
        excluded = call("preview", "--run", str(root), "--profile", str(profile))
        assert excluded["cards"] == [] and excluded["deferred_source_work"] == []


@pytest.mark.parametrize("stale", [False, True])
def test_recovery_preserves_foreign_identical_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    from latinitas_cards import pipeline_batches as batches

    profile, objectives, root = fixture(tmp_path)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    output = tmp_path / "foreign.csv"

    def interrupted(self: object, publication: object) -> None:
        raise OSError("crash before output publication")

    with monkeypatch.context() as patch:
        patch.setattr(batches.Batches, "_finish", interrupted)
        failed = runner.invoke(app, ["pipeline", "export", "--run", str(root), "--output", str(output)])
        assert failed.exit_code != 0
    pending = json.loads((root / "selection/pending.json").read_text())
    payload = pending["csv"].encode()
    output.write_bytes(payload)
    inode = output.stat().st_ino
    if stale:
        raw = json.loads(profile.read_text())
        raw["selection"]["entries"] = ["n"]
        write(profile, raw)
    failed = runner.invoke(
        app,
        ["pipeline", "export", "--run", str(root), "--profile", str(profile), "--output", str(tmp_path / "next.csv")],
    )
    assert failed.exit_code != 0 and "conflict" in failed.output
    assert output.read_bytes() == payload and output.stat().st_ino == inode


def test_selection_storage_does_not_follow_symlinks(tmp_path: Path) -> None:
    profile, objectives, root = fixture(tmp_path)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("unrelated bytes")
    store = root / "selection"
    store.mkdir()
    (store / "pending.staged").symlink_to(sentinel)
    call("export", "--run", str(root), "--output", str(tmp_path / "safe.csv"))
    assert sentinel.read_text() == "unrelated bytes"
    external = tmp_path / "external"
    store.rename(external)
    store.symlink_to(external, target_is_directory=True)
    before = {p.name: p.read_bytes() for p in external.iterdir() if p.is_file()}
    failed = runner.invoke(app, ["pipeline", "export", "--run", str(root), "--output", str(tmp_path / "redirect.csv")])
    assert failed.exit_code != 0 and "symlink" in failed.output
    assert before == {p.name: p.read_bytes() for p in external.iterdir() if p.is_file()}


def test_continuation_validates_all_committed_ancestors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, objectives, root = many_fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    first = call("export", "--run", str(root), "--output", str(tmp_path / "first.csv"), "--output-limit", "10")
    second_args = (
        "export",
        "--run",
        str(root),
        "--output",
        str(tmp_path / "second.csv"),
        "--output-limit",
        "10",
        "--after-receipt",
        first["receipt_id"],
    )
    second = call(*second_args)
    original = Path(first["output"]).read_bytes()
    for missing in (False, True):
        if missing:
            Path(first["output"]).unlink()
        else:
            Path(first["output"]).write_text("not the published batch")
        for args in (
            second_args,
            (
                "export",
                "--run",
                str(root),
                "--output",
                str(tmp_path / "third.csv"),
                "--after-receipt",
                second["receipt_id"],
            ),
            (
                "export",
                "--run",
                str(root),
                "--output",
                str(tmp_path / "reconcile.csv"),
                "--reconcile-receipt",
                second["receipt_id"],
            ),
        ):
            failed = runner.invoke(app, ["pipeline", *args, "--json"])
            assert failed.exit_code != 0 and "hash mismatch" in failed.output
    Path(first["output"]).write_bytes(original)
    nonclobber = runner.invoke(
        app,
        [
            "pipeline",
            "export",
            "--run",
            str(root),
            "--output",
            first["output"],
            "--after-receipt",
            second["receipt_id"],
        ],
    )
    assert nonclobber.exit_code != 0
    assert Path(first["output"]).read_bytes() == original


def test_export_reconstructs_analysis_authority_after_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latinitas_cards import pipeline_consumers as consumers
    from latinitas_cards.phrase_recipes import render_phrase_text

    profile, objectives, root = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    cards, _ = current_cards(
        profile, PipelineProfile.model_validate_json(profile.read_bytes()), consumers.load_run(root).objectives, root
    )
    assert cards[0].claims.assess().eligible
    analysis = Analysis.read_json(cards[0].analysis_path.read_bytes())
    changed = analysis.model_copy(
        update={"checks": (analysis.checks[0].model_copy(update={"result": "unresolved"}), *analysis.checks[1:])}
    )
    Analysis.read_json(canonical(changed))

    def replace_analysis(text: str, deck: DeckProfile) -> str:
        cards[0].analysis_path.write_bytes(canonical(changed))
        return render_phrase_text(text, deck)

    monkeypatch.setattr(consumers, "render_phrase_text", replace_analysis)
    output = tmp_path / "stale.csv"
    failed = runner.invoke(app, ["pipeline", "export", "--run", str(root), "--output", str(output)])
    assert failed.exit_code != 0 and "eligibility changed" in failed.output
    assert not output.exists() and not (root / "selection/pending.json").exists()


def test_sibling_continuation_preserves_eligible_note_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, objectives, root = fixture(tmp_path, preset=None)
    raw = json.loads(profile.read_text())
    choices = RecipeOverrides(include=("form_recognition",), exclude=("ppp_translation", "phrase_translation"))
    raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
    raw["selection"]["entries"] = ["n"]
    write(profile, raw)
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][1]["forms"] = [data["entries"][1]["forms"][0]]
    rebind(profile, data)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    first = call("export", "--run", str(root), "--output", str(tmp_path / "first.csv"), "--output-limit", "1")
    second = call(
        "export",
        "--run",
        str(root),
        "--output",
        str(tmp_path / "second.csv"),
        "--output-limit",
        "1",
        "--after-receipt",
        first["receipt_id"],
    )
    assert first["exported"] == second["exported"] == 1
    assert first["cards"][0]["note_id"] == second["cards"][0]["note_id"]
    rows = []
    for report in (first, second):
        lines = Path(report["output"]).read_text().splitlines()
        columns = next(line.removeprefix("#columns:") for line in lines if line.startswith("#columns:")).split(",")
        rows.append(
            dict(zip(columns, next(csv.reader(line for line in lines if not line.startswith("#"))), strict=True))
        )
    slot = next(t.slot for t in GRAMMAR_LAYOUT if t.slot.semantic_key == first["cards"][0]["card_key"])
    for field in (slot.enabled_field, slot.prompt_field, slot.answer_field):
        assert rows[0][field] and rows[1][field] == rows[0][field]


@pytest.mark.parametrize("output_limit", [None, 1])
def test_installed_sibling_invalidation_withholds_note_and_preserves_manual_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_limit: int | None
) -> None:
    from latinitas_cards.pipeline_consumers import load_run
    from latinitas_cards.pipeline_reporting import Reporting, build_report

    profile, objectives, root = fixture(tmp_path, preset=None)
    raw = json.loads(profile.read_text())
    choices = RecipeOverrides(include=("form_recognition",), exclude=("ppp_translation", "phrase_translation"))
    raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
    raw["selection"]["entries"] = ["n", "a"]
    raw["selection"]["output_limit"] = output_limit
    data = json.loads((tmp_path / "inputs.json").read_text())
    for index in (1, 2):
        data["entries"][index]["forms"] = [data["entries"][index]["forms"][0]]
    write(profile, raw)
    rebind(profile, data)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    first = call("export", "--run", str(root), "--output", str(tmp_path / "first.csv"), "--output-limit", "1")
    assert first["exported"] == 1
    cards, _ = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root)
    broken = next(
        c
        for c in cards
        if (c.claims.note_id, c.claims.card_key) == (first["cards"][0]["note_id"], first["cards"][0]["card_key"])
    )
    old_bytes = (tmp_path / "first.csv").read_bytes()
    destination = tmp_path / "destination.json"
    write(
        destination,
        {"Personal Notes": "keep", "Tags": ["Custom", "latinitas::basis::source"], "history": [7], "due": 42},
    )
    destination_bytes = destination.read_bytes()
    analysis_bytes = broken.analysis_path.read_bytes()
    broken.analysis_path.unlink()
    second = call(
        "export",
        "--run",
        str(root),
        "--output",
        str(tmp_path / "second.csv"),
        "--output-limit",
        "4",
        "--reconcile-receipt",
        first["receipt_id"],
    )
    assert second["exported"] == 2  # Only the unrelated note, not the still-eligible sibling of the stale card.
    assert second["notes"] == 1
    assert all(c["note_id"] != broken.claims.note_id for c in second["cards"])
    assert second["unresolved_note_rows"] == [broken.claims.note_id]
    actions = second["export_receipt"]["ineligible"]
    assert [(a["note_id"], a["card_key"]) for a in actions] == [(broken.claims.note_id, broken.claims.card_key)]
    assert {h["note_id"] for h in actions} != {c.claims.note_id for c in cards}
    assert (tmp_path / "first.csv").read_bytes() == old_bytes
    assert destination.read_bytes() == destination_bytes
    report = build_report(
        root, {"run_id": "synthetic-report", "status": "completed", "file_export": second}, None, None
    )
    sibling = next(
        c for c in cards if c.claims.note_id == broken.claims.note_id and c.claims.card_key != broken.claims.card_key
    )
    outcome = next(
        o
        for o in Reporting.read_json(canonical(report["outcome_report"])).record.outcomes
        if o.card_key == sibling.claims.card_key and o.source_identity == sibling.binding.source_identity
    )
    assert (outcome.outcome, outcome.reason) == ("withheld", "installed_content_unresolved")
    assert set(actions[0]["previous_decision_ids"]) <= set(outcome.related_ids)
    pending = call("handoff", "--run", str(root))
    assert pending["ineligible"] == actions
    human = runner.invoke(app, ["pipeline", "handoff", "--run", str(root)])
    assert human.exit_code == 0 and "not_applied" in human.output and "scheduling" in human.output
    # No supported apply/suspend shortcut exists; an inspection file grants no authority.
    denied = runner.invoke(app, ["pipeline", "handoff", "--run", str(root), "--apply", str(destination)])
    assert denied.exit_code != 0 and destination.read_bytes() == destination_bytes
    broken.analysis_path.write_bytes(analysis_bytes)
    assert call("handoff", "--run", str(root))["ineligible"] == actions
    replay = runner.invoke(
        app,
        [
            "pipeline",
            "export",
            "--run",
            str(root),
            "--output",
            str(tmp_path / "restored.csv"),
            "--output-limit",
            "1",
            "--json",
        ],
    )
    assert replay.exit_code != 0 and "reconciliation required" in replay.output
    assert not (tmp_path / "restored.csv").exists()
    assert (tmp_path / "first.csv").read_bytes() == old_bytes
    (tmp_path / "deck.csv").write_text("unconfirmed changed source")
    unavailable = call("handoff", "--run", str(root))
    assert {a["note_id"] for a in unavailable["ineligible"]} == {c.claims.note_id for c in cards}
    assert unavailable["state"] == "not_applied"


def test_processing_ceiling_uses_resolved_recipe_order(tmp_path: Path) -> None:
    profile, objectives, root = fixture(tmp_path, preset=None)
    raw = json.loads(profile.read_text())
    choices = RecipeOverrides(include=("form_recognition",), exclude=())
    raw.update(recipe_overrides=choices.model_dump(), recipes=resolve_recipes(None, choices))
    raw["selection"]["entries"] = ["n"]
    raw["processing"]["max_source_units"] = 1
    write(profile, raw)
    empty = json.loads(objectives.read_text())
    empty.update(forms=[], phrases=[])
    write(objectives, empty)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    report = call("preview", "--run", str(root))
    assert [c["recipe"] for c in report["cards"]] == ["form_translation"]


def test_attested_paradigms_produce_only_target_evidence(tmp_path: Path) -> None:
    """Expected cells transcribed independently from Bennett §§20, 63, 101."""
    from latinitas_cards.latin_library import attested_paradigms
    from latinitas_cards.pipeline_consumers import FormObjective, load_run

    profile, objectives, _ = fixture(tmp_path)
    inputs = json.loads((tmp_path / "inputs.json").read_text())
    noun = inputs["entries"][1]
    noun["forms"] = [{**form, "text": form["text"].replace("puell", "port")} for form in noun["forms"]]
    (tmp_path / "deck.csv").write_text((tmp_path / "deck.csv").read_text().replace("puell", "port"))
    rebind(profile, inputs)
    cases = ("nominative", "genitive", "dative", "accusative", "vocative", "ablative")
    cells: list[tuple[str, dict[str, str], str]] = []
    for person, singular, plural in (("1", "amō", "amāmus"), ("2", "amās", "amātis"), ("3", "amat", "amant")):
        for number, surface in (("singular", singular), ("plural", plural)):
            cells.append(
                ("v", dict(person=person, number=number, tense="present", mood="indicative", voice="active"), surface)
            )
    for number, forms in (
        ("singular", "porta portae portae portam porta portā"),
        ("plural", "portae portārum portīs portās portae portīs"),
    ):
        cells.extend(
            ("n", dict(case=case, number=number, gender="feminine"), form)
            for case, form in zip(cases, forms.split(), strict=True)
        )
    for number, genders in (
        (
            "singular",
            ("bonus bonī bonō bonum bone bonō", "bona bonae bonae bonam bona bonā", "bonum bonī bonō bonum bonum bonō"),
        ),
        (
            "plural",
            (
                "bonī bonōrum bonīs bonōs bonī bonīs",
                "bonae bonārum bonīs bonās bonae bonīs",
                "bona bonōrum bonīs bona bona bonīs",
            ),
        ),
    ):
        for gender, forms in zip(("masculine", "feminine", "neuter"), genders, strict=True):
            cells.extend(
                ("a", dict(case=case, number=number, gender=gender), form)
                for case, form in zip(cases, forms.split(), strict=True)
            )
    raw = json.loads(objectives.read_text())
    templates = {form["entry"]: form for form in raw["forms"]}
    raw["forms"] = [{**templates[entry], "target": features} for entry, features, _ in cells]
    write(objectives, raw)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    saved = load_run(root)
    cards, skipped = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), saved.objectives, root)
    expected = {(entry, canonical(features)): surface for entry, features, surface in cells}
    identities = {"amō": "v", "porta": "n", "bonus": "a"}
    parsed_cells = [
        ((identities[p.lemma], canonical(cell.features.model_dump(exclude_none=True))), cell.surface)
        for p in attested_paradigms()
        for cell in p.cells
    ]
    assert len(parsed_cells) == len(expected) == 54 and dict(parsed_cells) == expected
    excluded = {
        ("n", canonical(dict(case=case, number="singular", gender="feminine"))) for case in ("nominative", "vocative")
    } | {("a", canonical(dict(case="nominative", number="singular", gender="masculine")))}
    rendered = {
        (c.objective.entry, canonical(c.objective.target.model_dump(exclude_none=True)))
        for c in cards
        if isinstance(c.objective, FormObjective)
    }
    assert rendered == expected.keys() - excluded
    work_skipped = [row for row in skipped if "work_id" in row]
    assert {row["work_id"] for row in work_skipped} == {
        digest(o)
        for o in saved.objectives.forms
        if (o.entry, canonical(o.target.model_dump(exclude_none=True))) in excluded
    }
    assert len(work_skipped) == 3 and {row["reason"] for row in work_skipped} == {
        "answer revealed by production prompt"
    }
    for card in cards:
        assert isinstance(card.objective, FormObjective)
        analysis = Analysis.read_json(card.analysis_path.read_bytes())
        checks = {check.id: check for check in analysis.checks}
        target = next(a for a in card.claims.assertions if a.id == "target")
        assert (
            json.loads(target.value)["surface"]
            == expected[card.objective.entry, canonical(card.objective.target.model_dump(exclude_none=True))]
        )
        assert checks["target"].result == "pass"
        assert any("Bennett" in e.source and "sha256" in e.version for e in checks["target"].evidence)
        assert all(c.result != "pass" for c in analysis.checks if c.id != "target")
        assert {a.id for a in analysis.assertions} == {i for c in analysis.checks for i in c.assertion_ids}
    assert not report["provider_calls"] and report["file_export"]["exported"] == 0
    before = {c.analysis_path: c.analysis_path.read_bytes() for c in cards}
    call("resume", "--run", str(root))
    assert all(path.read_bytes() == content for path, content in before.items())
    assert call("export", "--run", str(root), "--output", str(tmp_path / "denied.csv"))["exported"] == 0


@pytest.mark.parametrize(
    "control",
    [
        "unknown-lemma",
        "conflicting-lemma",
        "noun-gender",
        "unknown-noun",
        "missing-resource",
        "tampered-resource",
        "context",
        "wrong-voice",
        "incomplete",
        "contradictory-infinitive",
    ],
)
def test_attested_paradigms_withhold_unsupported_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    from latinitas_cards import latin_library
    from latinitas_cards.pipeline_consumers import load_run

    profile, objectives, _ = fixture(tmp_path)
    data = json.loads((tmp_path / "inputs.json").read_text())
    raw = json.loads(objectives.read_text())
    raw["forms"] = [
        next(f for f in raw["forms"] if f["entry"] == ("n" if control in ("noun-gender", "unknown-noun") else "v"))
    ]
    if control == "unknown-lemma":
        deck = tmp_path / "deck.csv"
        deck.write_text(deck.read_text().replace("amō", "talō").replace("amāre", "talāre"))
        for form in data["entries"][0]["forms"]:
            form["text"] = form["text"].replace("amō", "talō").replace("amāre", "talāre")
    elif control == "conflicting-lemma":
        deck = tmp_path / "deck.csv"
        deck.write_text(deck.read_text().replace("v,amō,", "v,moneō,"))
    elif control == "noun-gender":
        deck = tmp_path / "deck.csv"
        deck.write_text(deck.read_text().replace("puell", "port"))
        noun = data["entries"][1]
        noun["gender"] = "masculine"
        for form in noun["forms"]:
            form["text"] = form["text"].replace("puell", "port")
            form["features"]["gender"] = "masculine"
        raw["forms"][0]["target"]["gender"] = "masculine"
    elif control in ("missing-resource", "tampered-resource"):
        path = tmp_path / "reference.html"
        if control == "tampered-resource":
            path.write_bytes(latin_library.BENNETT_PATH.read_bytes() + b"\n")
        monkeypatch.setattr(latin_library, "BENNETT_PATH", path)
    elif control == "context":
        raw["forms"][0]["context"] = "Unverified contextual applicability"
    elif control == "wrong-voice":
        raw["forms"][0]["target"]["voice"] = "passive"
    elif control == "incomplete":
        raw["forms"][0]["target"].pop("voice")
    elif control == "contradictory-infinitive":
        data["entries"][0]["forms"][1]["features"] = dict(
            tense="perfect", voice="passive", case="ablative", gender="neuter", number="plural"
        )
    rebind(profile, data)
    write(objectives, raw)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    cards, _ = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root)
    assert bool(cards) == (control not in ("wrong-voice", "incomplete"))
    for card in cards:
        analysis = Analysis.read_json(card.analysis_path.read_bytes())
        assert all(c.result != "pass" for c in analysis.checks)
    assert report["provider_calls"] == [] and report["file_export"]["exported"] == 0


def test_attested_paradigm_accepts_compatible_partial_anchor_features(tmp_path: Path) -> None:
    from latinitas_cards.pipeline_consumers import load_run

    profile, objectives, _ = fixture(tmp_path)
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][0]["forms"][1]["features"] = {"tense": "present", "voice": "active"}
    rebind(profile, data)
    raw = json.loads(objectives.read_text())
    raw["forms"] = [raw["forms"][0]]
    write(objectives, raw)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    cards, _ = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root)
    assert len(cards) == 1
    checks = Analysis.read_json(cards[0].analysis_path.read_bytes()).checks
    assert next(c for c in checks if c.id == "target").result == "pass"
    assert all(c.result != "pass" for c in checks if c.id != "target")
    assert report["file_export"]["exported"] == 0


@pytest.mark.parametrize("damage", ["changed", "removed"])
def test_changed_reference_invalidates_saved_derived_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    import shutil

    from latinitas_cards import latin_library
    from latinitas_cards.pipeline_consumers import _analysis_decision, load_run

    package = Path(policies.__file__).parent
    copied = tmp_path / "installed-package"
    shutil.copytree(package, copied, ignore=shutil.ignore_patterns("__pycache__"))
    reference = copied / "references" / "bennett-1918.html"
    original_digest = policies.implementation_digest()
    monkeypatch.setattr(policies, "__file__", str(copied / "acceptance_policy.py"))
    monkeypatch.setattr(latin_library, "BENNETT_PATH", reference)
    assert policies.implementation_digest() == original_digest
    profile, objectives, _ = fixture(tmp_path / "inputs")
    raw = json.loads(objectives.read_text())
    raw["forms"] = [
        {
            **raw["forms"][0],
            "target": dict(person="1", number="plural", tense="present", mood="indicative", voice="active"),
        }
    ]
    write(objectives, raw)
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    parsed = PipelineProfile.read_json(profile.read_bytes())
    cards, _ = current_cards(profile, parsed, load_run(root).objectives, root)
    analysis = Analysis.read_json(cards[0].analysis_path.read_bytes())
    assert next(c for c in analysis.checks if c.id == "target").result == "pass"
    assert _analysis_decision(cards[0]).state == "withheld"
    if damage == "changed":
        reference.write_bytes(reference.read_bytes() + b"\n")
    else:
        reference.unlink()
    assert policies.implementation_digest() != original_digest
    current, _ = current_cards(profile, parsed, load_run(root).objectives, root)
    assert current[0].versions != cards[0].versions
    with pytest.raises(ValueError, match="stale analysis context"):
        _analysis_decision(current[0])
    report = call("resume", "--run", str(root), "--offline")
    updated = Analysis.read_json(current[0].analysis_path.read_bytes())
    assert updated.versions == current[0].versions
    assert all(c.result != "pass" for c in updated.checks)
    assert report["provider_calls"] == [] and report["file_export"]["exported"] == 0


def test_source_bound_rule_producer_keeps_production_policy_disabled(tmp_path: Path) -> None:
    """Disposable source mappings exercise rules, not empirical qualification."""
    from latinitas_cards.acceptance import load_stage, stage_artifact
    from latinitas_cards.pipeline_consumers import _analysis_decision, load_run

    profile, objectives, _ = fixture(tmp_path, preset=None)
    settings = json.loads(profile.read_text())
    overrides = RecipeOverrides(include=("form_recognition",), exclude=())
    settings.update(recipe_overrides=overrides, recipes=resolve_recipes(None, overrides))
    write(profile, settings)
    raw = json.loads(objectives.read_text())
    template = raw["forms"][0]
    raw["forms"] = [
        {**template, "recipe": recipe, "form": "present", "anchors": [], "target": {}}
        for recipe in ("form_recognition", "form_translation")
    ]
    raw["forms"][0]["explanation"] = "Every Latin verb is regular."
    write(objectives, raw)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    cards, _ = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root)
    selected = [c for c in cards if c.objective.object_key == "production"]
    assert len(selected) == 2
    for card in selected:
        analysis = Analysis.read_json(card.analysis_path.read_bytes())
        checks = {c.assertion_ids: c for c in analysis.checks}
        target = checks[("target",)]
        assert target.result == "pass"
        assert {e.source for e in target.evidence} >= {
            "confirmed-deck-mapping",
            "latinitas-principal-part-rules",
        }
        assert any("amō" in e.text and "present_1s" in e.text for e in target.evidence)
        assert checks[("source",)].result != "pass"  # Includes unverified sense/valency claims.
        if ("translation",) in checks:
            assert checks[("translation",)].result == "not_applicable"
        if ("explanation",) in checks:
            assert checks[("explanation",)].result == "not_applicable"
        assert set(a.id for a in analysis.assertions) == {i for c in analysis.checks for i in c.assertion_ids}
        envelopes = [stage_artifact(root, load_stage(root, identity)) for identity in analysis.stage_ids]
        for assertion, check in zip(analysis.assertions, analysis.checks, strict=True):
            envelope = check_envelope(
                analysis.binding, analysis.versions, "rule", (assertion,), reference_constraints=check.evidence
            )
            assert json.loads(canonical(envelope)) in envelopes and digest(envelope) == check.input_sha256
        assert _analysis_decision(card).reason == "policy_not_enabled"
    before = {p: p.read_bytes() for p in [*(root / "stages").glob("*.json"), *(c.analysis_path for c in selected)]}
    call("resume", "--run", str(root))
    assert all(p.read_bytes() == content for p, content in before.items())
    assert report["provider_calls"] == []
    assert report["file_export"]["exported"] == 0
    assert call("export", "--run", str(root), "--output", str(tmp_path / "denied"))["exported"] == 0


@pytest.mark.parametrize("mutation", ["wrong-feature", "wrong-form", "extra-clause", "extra-alternative"])
@pytest.mark.parametrize("derived", [False, True])
def test_source_bound_rule_rejects_candidate_claims(tmp_path: Path, mutation: str, derived: bool) -> None:
    from dataclasses import replace

    from latinitas_cards.acceptance import load_stage, stage_artifact
    from latinitas_cards.pipeline_consumers import load_run, load_source
    from latinitas_cards.pipeline_orchestration import _analyze, load_execution

    profile, objectives, _ = fixture(tmp_path, preset="grammar-study" if derived else None)
    raw = json.loads(objectives.read_text())
    raw["forms"] = (
        [raw["forms"][0]]
        if derived
        else [{**raw["forms"][0], "recipe": "form_translation", "form": "present", "anchors": []}]
    )
    write(objectives, raw)
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    parsed = PipelineProfile.read_json(profile.read_bytes())
    card = next(
        c
        for c in current_cards(profile, parsed, load_run(root).objectives, root)[0]
        if c.objective.object_key == "production"
    )
    target = next(a for a in card.claims.assertions if a.id == "target")
    value = json.loads(target.value)
    if mutation == "wrong-feature":
        value["features"]["person"] = "3"
    elif mutation == "wrong-form":
        value["surface"] = "amant"
    elif mutation == "extra-clause":
        value["explanation"] = "All verbs are regular."
    changed = target.model_copy(
        update={
            "value": canonical(value).decode(),
            "alternatives": ("amant",) if mutation == "extra-alternative" else (),
        }
    )
    claims = replace(card.claims, assertions=tuple(changed if a.id == "target" else a for a in card.claims.assertions))
    candidate = replace(card, claims=claims, analysis_path=card.analysis_path.with_name(mutation + ".json"))
    _analyze(root, load_execution(root), candidate, load_source(profile, parsed))
    analysis = Analysis.read_json(candidate.analysis_path.read_bytes())
    check = next(c for c in analysis.checks if c.assertion_ids == ("target",))
    assert check.result == "fail"
    assert check.objections == ("source_bound_target_contradiction",)
    envelopes = [stage_artifact(root, load_stage(root, identity)) for identity in analysis.stage_ids]
    assert any(digest(envelope) == check.input_sha256 for envelope in envelopes)


def test_source_bound_rule_does_not_infer_exception_applicability(tmp_path: Path) -> None:
    """Even deliberately regular-looking exceptional anchors are not authority."""
    from latinitas_cards.pipeline_consumers import load_run

    profile, objectives, _ = fixture(tmp_path)
    deck = tmp_path / "deck.csv"
    deck.write_text(deck.read_text().replace("amō", "dō").replace("amāre", "dāre"))
    data = json.loads((tmp_path / "inputs.json").read_text())
    for entry in data["entries"]:
        for form in entry["forms"]:
            form["text"] = form["text"].replace("amō", "dō").replace("amāre", "dāre")
    rebind(profile, data)
    raw = json.loads(objectives.read_text())
    raw["forms"] = [raw["forms"][0]]
    raw["forms"][0]["target"].update(person="1", number="plural")
    write(objectives, raw)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    card = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root)[0][
        0
    ]
    assert "dāmus" in card.answer  # The proposal mechanism alone overgeneralizes.
    analysis = Analysis.read_json(card.analysis_path.read_bytes())
    target = next(c for c in analysis.checks if c.assertion_ids == ("target",))
    assert target.result == "not_applicable"
    assert target.objections == ("generated_cell_requires_independent_applicability_evidence",)
    assert all(c.result != "pass" for c in analysis.checks)
    assert report["file_export"]["exported"] == 0


@pytest.mark.parametrize("damage", ["source", "checker", "analyzer", "artifact"])
def test_source_bound_rule_rejects_stale_proofs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    from latinitas_cards import pipeline_consumers as consumers
    from latinitas_cards.acceptance import load_stage

    profile, objectives, _ = fixture(tmp_path, preset=None)
    settings = json.loads(profile.read_text())
    settings["selection"]["entries"] = ["v"]
    write(profile, settings)
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    run = consumers.load_run(root)
    original = next(
        c for c in current_cards(profile, run.profile, run.objectives, root)[0] if c.prompt.startswith("amō\n")
    )
    proof = Analysis.read_json(original.analysis_path.read_bytes())
    assert next(c for c in proof.checks if c.assertion_ids == ("target",)).result == "pass"
    if damage == "artifact":
        stage = load_stage(root, proof.stage_ids[1])
        assert stage.artifact is not None
        (root / stage.artifact).write_text("{}")
        replay = call("stage", "analyze", "--run", str(root))
        assert any(row.get("state") == "failed" for row in replay["stages"]["analyze"])
    else:
        if damage == "source":
            deck = tmp_path / "deck.csv"
            deck.write_text(deck.read_text().replace("lieben", "lieben (changed)"))
            rebind(profile, json.loads((tmp_path / "inputs.json").read_text()))
        elif damage == "checker":
            monkeypatch.setattr(consumers, "implementation_digest", lambda: digest("changed executable bytes"))
        else:
            monkeypatch.setattr(consumers, "RULE_VERSION", "changed-analyzer/2")
        parsed = PipelineProfile.read_json(profile.read_bytes())
        current = next(
            c for c in current_cards(profile, parsed, run.objectives, root)[0] if c.prompt.startswith("amō\n")
        )
        assert current.binding != original.binding or current.versions != original.versions
        write(current.analysis_path, proof)
        with pytest.raises(ValueError, match="stale analysis context"):
            consumers._analysis_decision(current)
    if damage == "source":
        denied = runner.invoke(app, ["pipeline", "export", "--run", str(root), "--output", str(tmp_path / "denied")])
        assert denied.exit_code == 1 and "changed profile/input: deck.csv" in denied.output
        assert not (tmp_path / "denied").exists()
    else:
        assert call("export", "--run", str(root), "--output", str(tmp_path / "denied"))["exported"] == 0


@pytest.mark.parametrize(
    "control", ["other-role", "mapping-mismatch", "different-lemma", "non-active", "wrong-feature"]
)
def test_source_bound_rule_requires_confirmed_citation_role(tmp_path: Path, control: str) -> None:
    from latinitas_cards.pipeline_consumers import load_run

    profile, objectives, _ = fixture(tmp_path, preset=None)
    data = json.loads((tmp_path / "inputs.json").read_text())
    verb = next(e for e in data["entries"] if e["identity"] == "v")
    if control == "other-role":
        verb["forms"][0]["source_role"] = "other"
    elif control == "mapping-mismatch":
        path = tmp_path / "deck-profile.json"
        deck = json.loads(path.read_text())
        deck["principal_parts"]["roles"] = ["perfect_1s", "present_infinitive", "present_1s", "ppp"]
        write(path, deck)
    elif control == "different-lemma":
        path = tmp_path / "deck.csv"
        path.write_text(path.read_text().replace("v,amō,", "v,legō,"))
    elif control == "non-active":
        path = tmp_path / "deck.csv"
        path.write_text(path.read_text().replace("amō", "amor"))
        verb["forms"][0]["text"] = "amor"
    else:
        verb["forms"][0]["features"]["person"] = "3"
    rebind(profile, data)
    settings = json.loads(profile.read_text())
    settings["selection"]["entries"] = ["v"]
    write(profile, settings)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    cards, skipped = current_cards(
        profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root
    )
    assert cards or skipped  # Reject malformed roles rather than silently dropping the work.
    for card in cards:
        analysis = Analysis.read_json(card.analysis_path.read_bytes())
        target = next(c for c in analysis.checks if c.assertion_ids == ("target",))
        assert target.result != "pass"
        if control == "wrong-feature" and card.prompt.startswith("amō\n"):
            assert target.result == "fail" and target.objections == ("source_bound_target_contradiction",)
    assert report["file_export"]["exported"] == 0


@pytest.mark.parametrize("library_state", ["contradiction", "unavailable", "unsupported"])
def test_source_bound_rule_keeps_library_objections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library_state: str
) -> None:
    from types import SimpleNamespace

    from latinitas_cards import pipeline_orchestration as workflow
    from latinitas_cards.pipeline_consumers import load_run

    profile, objectives, _ = fixture(tmp_path, preset=None)
    settings = json.loads(profile.read_text())
    settings["selection"]["entries"] = ["v"]
    write(profile, settings)
    # Only the optional analyzer boundary is synthetic; no passing Check or policy is injected.
    monkeypatch.setattr(workflow, "LocalStanza", lambda path: SimpleNamespace(version="synthetic-engineering/1"))
    monkeypatch.setattr(workflow, "analyze_deck", lambda *a, **kw: SimpleNamespace(state=library_state, checks=()))
    output = tmp_path / "workflow"
    report = call(
        "run",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(output),
        "--library-resources",
        "engineering-only",
        "--offline",
    )
    root = output / "run"
    card = next(
        c
        for c in current_cards(
            profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root
        )[0]
        if c.prompt.startswith("amō\n")
    )
    target = next(
        c for c in Analysis.read_json(card.analysis_path.read_bytes()).checks if c.assertion_ids == ("target",)
    )
    expected = {"contradiction": "fail", "unavailable": "unresolved", "unsupported": "pass"}
    assert target.result == expected[library_state]
    assert (
        target.objections
        == {
            "contradiction": ("library_contradiction",),
            "unavailable": ("processing_unavailable",),
            "unsupported": (),
        }[library_state]
    )
    assert report["file_export"]["exported"] == 0


def test_source_bound_rule_missing_lexical_field_preserves_unsupported_siblings(tmp_path: Path) -> None:
    from latinitas_cards.pipeline_consumers import load_run

    profile, objectives, _ = fixture(tmp_path, preset=None)
    deck = tmp_path / "deck.csv"
    with deck.open() as stream:
        rows = list(csv.DictReader(stream))
    with deck.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ID", "Parts", "Meaning"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / "deck.csv.latinitas.json"
    raw = json.loads(manifest.read_text())
    raw["source_columns"] = ["ID", "Parts", "Meaning"]
    write(manifest, raw)
    rebind(profile, json.loads((tmp_path / "inputs.json").read_text()))
    settings = json.loads(profile.read_text())
    settings["selection"]["entries"] = ["n", "a"]
    write(profile, settings)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    cards, _ = current_cards(profile, PipelineProfile.read_json(profile.read_bytes()), load_run(root).objectives, root)
    assert {c.binding.source_identity for c in cards} == {"n", "a"}
    assert len(report["stages"]["analyze"]) == len(cards)
    for card in cards:
        analysis = Analysis.read_json(card.analysis_path.read_bytes())
        assert all(c.result == "not_applicable" for c in analysis.checks)
    assert report["file_export"]["exported"] == 0


def test_orchestrated_offline_stages_and_replay(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    assert report["file_export"]["exported"] == 0
    assert report["applied_to_anki"] is False
    assert report["provider_calls"] == []
    assert report["coverage_limit"]
    assert set(report["stages"]) == {
        "ingest",
        "analyze",
        "propose",
        "counter_check",
        "decide",
        "select",
        "render",
        "export",
    }
    root = output / "run"
    first = {p.name: p.read_bytes() for p in (root / "stages").glob("*.json")}
    replay = call("resume", "--run", str(root))
    assert replay["file_export"] == report["file_export"]
    assert first == {p.name: p.read_bytes() for p in (root / "stages").glob("*.json")}
    assert call("report", "--run", str(root))["run_id"] == report["run_id"]
    assert call("stage", "select", "--run", str(root))["cards"]
    call("cancel", "--run", str(root))
    assert call("resume", "--run", str(root))["status"] == "cancelled"


def report_call(*args: str) -> dict[str, Any]:
    result = call(*args)
    view = result["outcome_report"]
    return {**result, **view["record"], **view["details"]}


def test_pipeline_report_export_only_is_incomplete(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only")
    result = report_call("stage", "export", "--run", str(output / "run"))
    assert not result["complete"]


def test_pipeline_report_missing_pattern_retains_source_work(tmp_path: Path) -> None:
    profile, objectives, _ = phrase_fixture(tmp_path, "en", "preposition-noun")
    raw = json.loads(objectives.read_text())
    first = raw["phrases"][0]
    raw["phrases"].append({**first, "selection": {**first["selection"], "pattern": "missing-pattern"}})
    write(objectives, raw)
    result = report_call(
        "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(tmp_path / "out")
    )
    missing = [o for o in result["outcomes"] if o["card_key"] is None and o["related_ids"]]
    assert len(missing) == 1 and missing[0]["outcome"] == "not_generated" and missing[0]["candidate_id"] is None
    assert missing[0]["source_links"] and not any("#/patterns/" in link for link in missing[0]["source_links"])
    assert any(o["card_key"] and o["outcome"] == "withheld" for o in result["outcomes"])


def test_pipeline_report_exact_approved_record_and_outcome_shape() -> None:
    from latinitas_cards.pipeline_reporting import Outcome, PipelineReport

    assert set(PipelineReport.model_fields) == {
        "format",
        "version",
        "run_id",
        "profile_sha256",
        "effective_profile",
        "source_units",
        "stage_counts",
        "outcomes",
        "calls",
        "decision_ids",
        "receipt_ids",
        "permission_sha256",
        "complete",
    }
    assert set(Outcome.model_fields) == {
        "source_identity",
        "candidate_id",
        "card_key",
        "outcome",
        "reason",
        "related_ids",
    }
    assert PipelineReport.model_json_schema()["additionalProperties"] is False


def test_pipeline_report_conflicting_objectives_account_for_both_work_units(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    raw = json.loads(objectives.read_text())
    first = raw["forms"][0]
    raw["forms"] = [first, {**first, "explanation": "a distinct synthetic explanation"}]
    write(objectives, raw)
    result = report_call(
        "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(tmp_path / "out")
    )
    assert result["source_units"] == {"selected": 2, "attempted": 2, "deferred": 0}
    assert result["counts"]["cards"] == 0
    assert len({i for o in result["outcomes"] for i in o["related_ids"]}) == 2
    assert len([o for o in result["outcomes"] if o["related_ids"] and o["source_links"]]) == 2


def test_pipeline_report_read_boundary_and_terminal_controls(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "out"
    result = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    result["historical_diagnostics"] = {"error": "Bearer SYNTHETIC_OLD_TOKEN"}
    view = result["outcome_report"]
    view["record"]["run_id"] += "\x1b[2J\x1b[H"
    for rows in (view["record"]["outcomes"], view["details"]["outcomes"]):
        rows[0]["reason"] = "\x1b[2J\x1b[H"
    view["details"]["outcomes"][2]["objections"] = ["distinct material objection"]
    view["details"]["outcomes"][2]["limitations"] = ["distinct evidence limitation"]
    write(output / "report.json", result)
    stored = call("report", "--run", str(output / "run"))
    assert "SYNTHETIC_OLD_TOKEN" not in canonical(stored).decode()
    human = runner.invoke(app, ["pipeline", "report", "--run", str(output / "run")], color=True)
    assert human.exit_code == 0 and "\x1b" not in human.output
    assert "distinct material objection" in human.output and "distinct evidence limitation" in human.output


@pytest.mark.parametrize("collision", ["1", "SYNTHETIC_KEY_TOKEN", "card_identity"])
def test_pipeline_report_secret_collision_fails_without_corrupt_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision: str
) -> None:
    profile, objectives, _ = fixture(tmp_path)
    configure_pipeline_http(profile, "http://127.0.0.1:1")
    raw = json.loads(profile.read_text())
    for endpoint in raw["providers"].values():
        endpoint["credential_ref"] = "SYNTHETIC_REPORT_TOKEN"
    write(profile, raw)
    output = tmp_path / "out"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--offline")
    original = (output / "report.json").read_bytes()
    if collision == "card_identity":
        collision = next(
            o["note_id"] for o in json.loads(original)["outcome_report"]["details"]["outcomes"] if o["note_id"]
        )
    elif collision != "1":
        index = json.loads((output / "run" / "stage-index.json").read_text())
        index[collision] = "missing-stage"
        write(output / "run" / "stage-index.json", index)
    monkeypatch.setenv("SYNTHETIC_REPORT_TOKEN", collision)
    failed = runner.invoke(app, ["pipeline", "stage", "render", "--run", str(output / "run"), "--offline", "--json"])
    assert failed.exit_code == 1
    assert "pipeline_report_redaction_or_validation_failed" in failed.output
    assert "SYNTHETIC_KEY_TOKEN" not in failed.output
    assert (output / "report.json").read_bytes() == original


def test_pipeline_report_explains_actual_analysis_reuse(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "out"
    first = report_call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    assert first["reuse"].get("processed") == 6 and not first["reuse"].get("reused")
    resumed = report_call("resume", "--run", str(output / "run"))
    assert resumed["reuse"].get("reused") == 6 and not resumed["reuse"].get("processed")
    human = runner.invoke(app, ["pipeline", "report", "--run", str(output / "run")])
    assert human.exit_code == 0 and '"reused": 6' in human.output


@pytest.mark.parametrize("damage", ["format", "binding", "assertions", "checks"])
def test_pipeline_report_decision_failure_is_card_linked_and_redacted(tmp_path: Path, damage: str) -> None:
    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--offline")
    path = next((output / "run" / "analyses").glob("*.json"))
    analysis = json.loads(path.read_text())
    if damage == "format":
        analysis["format"] = "synthetic-private-error-marker"
    elif damage == "binding":
        analysis["binding"]["config_sha256"] = "0" * 64
    elif damage == "assertions":
        analysis["assertions"][0]["value"] = "changed assertion"
    else:
        analysis["checks"][0]["input_sha256"] = "0" * 64
    write(path, analysis)
    result = report_call("stage", "decide", "--run", str(output / "run"))
    if damage == "checks":
        # The disabled policy withholds this check; it is not a processing exception.
        assert not result["stages"]["decide"][0]["analysis_errors"]
        assert {o["outcome"] for o in result["outcomes"] if o["card_key"]} == {"withheld"}
        again = report_call("stage", "export", "--run", str(output / "run"))
        assert {o["outcome"] for o in again["outcomes"] if o["card_key"]} == {"withheld"}
        return
    damaged = result["stages"]["decide"][0]["analysis_errors"][0]
    assert "synthetic-private-error-marker" not in canonical(result).decode()
    assert any(
        o["note_id"] == damaged["note_id"]
        and o["card_key"] == damaged["card_key"]
        and o["outcome"] == "processing_failed"
        for o in result["outcomes"]
    )
    again = report_call("stage", "export", "--run", str(output / "run"))
    assert not again["complete"]
    assert any(
        o["outcome"] == "processing_failed"
        for o in again["outcomes"]
        if (o["note_id"], o["card_key"]) == (damaged["note_id"], damaged["card_key"])
    )


def test_pipeline_report_stage_only_preserves_export_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, objectives, _ = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, tmp_path / "unused")
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only")
    root = output / "run"
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    exported = report_call("resume", "--run", str(root))
    for stage in ("select", "render"):
        result = report_call("stage", stage, "--run", str(root))
        assert result["receipt_ids"] == exported["receipt_ids"]
        assert result["counts"]["selected_cards"] == 6
        assert {o["outcome"] for o in result["outcomes"] if o["card_key"]} == {"exported"}
    raw = json.loads(profile.read_text())
    raw["selection"]["entries"] = ["n"]
    write(profile, raw)
    changed = report_call("stage", "select", "--run", str(root), "--profile", str(profile))
    assert changed["receipt_ids"] == [] and changed["counts"]["selected_cards"] == 0
    assert len(changed["installed_handoff"]) == 4
    assert {h["reason"] for h in changed["installed_handoff"]} == {"current_eligibility_unavailable"}
    readable = runner.invoke(app, ["pipeline", "report", "--run", str(root)])
    assert readable.exit_code == 0 and "Unresolved installed content" in readable.output
    assert "not approved for direct reimport" in readable.output


def test_pipeline_report_reads_historical_v1_without_invented_handoff(tmp_path: Path) -> None:
    from latinitas_cards.export_receipts import Handoff
    from latinitas_cards.pipeline_reporting import safe_snapshot

    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only")
    root = output / "run"
    stored = json.loads((output / "report.json").read_text())
    stored["outcome_report"]["details"].pop("installed_handoff", None)
    stored.pop("installed_handoff", None)
    write(output / "report.json", stored)
    result = call("report", "--run", str(root))
    assert result["outcome_report"] == stored["outcome_report"]
    assert "installed_handoff" not in result
    readable = runner.invoke(app, ["pipeline", "report", "--run", str(root)])
    assert readable.exit_code == 0 and "not assessed" in readable.output
    action = Handoff(
        note_id="synthetic-handoff-identity",
        card_key="slot",
        previous_decision_ids=(),
        reason="current_eligibility_unavailable",
        resolution="manual",
        required_action="Inspect destination first",
    ).model_dump(mode="json")
    stored["installed_handoff"] = [action]
    write(output / "report.json", stored)
    assert call("report", "--run", str(root))["installed_handoff"] == [action]
    with pytest.raises(ValueError, match="report_credential_collision"):
        safe_snapshot(stored, ("synthetic-handoff-identity",))
    action["unexpected"] = "private-marker"
    write(output / "report.json", stored)
    bad = runner.invoke(app, ["pipeline", "report", "--run", str(root), "--json"])
    assert bad.exit_code == 1 and "pipeline_report_unavailable_or_invalid" in bad.output
    assert "private-marker" not in bad.output


def test_versioned_pipeline_outcomes_and_readable_yield(tmp_path: Path) -> None:
    from latinitas_cards.pipeline_reporting import Reporting

    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "workflow"
    result = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    view = Reporting.read_json(canonical(result["outcome_report"]))
    report = view.record
    assert report.format == "latinitas-pipeline-report" and report.version == 1
    assert report.profile_sha256 == digest(report.effective_profile)
    assert view.details.counts["cards"] == 6
    assert view.details.counts["candidates"] == 6
    assert report.source_units.model_dump() == {"selected": 6, "attempted": 6, "deferred": 0}
    cards = [o for o in view.details.outcomes if o.card_key is not None]
    assert len(cards) == 6 and {o.outcome for o in cards} == {"withheld"}
    assert {o.reason for o in cards} == {"policy_not_enabled"}
    assert all(o.source_links and o.policy_version == "1" for o in cards)
    assert any(o.outcome == "not_generated" and o.card_key is None for o in report.outcomes)
    for item in cards:
        for link in item.source_links:
            assert link.startswith("file:") and "#/entries/" in link
    assert report.calls == () and report.permission_sha256 is None
    stored = call("report", "--run", str(output / "run"))
    assert stored == result
    human = runner.invoke(app, ["pipeline", "report", "--run", str(output / "run")])
    assert human.exit_code == 0, human.output
    assert "Yield:" in human.output and "withheld" in human.output
    assert "Example:" in human.output and "private reasoning" not in human.output
    assert "not applied to Anki" in human.output


def test_pipeline_report_partial_resume_keeps_accepted_and_deferred_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, objectives, _ = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, tmp_path / "unused")
    raw = json.loads(profile.read_text())
    raw["processing"]["max_source_units"] = 2
    raw["selection"]["output_limit"] = 1
    write(profile, raw)
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only")
    root = output / "run"
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    first = report_call("resume", "--run", str(root))
    assert first["source_units"] == dict(selected=6, attempted=2, deferred=4)
    assert first["counts"]["cards"] == first["counts"]["candidates"] == 2
    assert {o["outcome"] for o in first["outcomes"] if o["card_key"]} == {"exported", "accepted_unselected"}
    deferred = [o for o in first["outcomes"] if o["reason"] == "processing_limit"]
    assert len(deferred) == 4 and all(o["card_key"] is None and o["candidate_id"] is None for o in deferred)
    assert all(o["source_links"] for o in deferred)
    assert not first["complete"]
    accepted_ids = {i for o in first["outcomes"] if o["card_key"] for i in o["related_ids"]}
    raw["processing"]["max_source_units"] = None
    write(profile, raw)
    prove_cards(profile, objectives, root)
    resumed = report_call(
        "resume", "--run", str(root), "--profile", str(profile), "--reconcile-receipt", first["receipt_ids"][0]
    )
    assert resumed["source_units"] == dict(selected=6, attempted=6, deferred=0)
    assert resumed["counts"]["cards"] == 6
    assert resumed["counts"]["selected_cards"] == 2
    assert accepted_ids <= {i for o in resumed["outcomes"] for i in o["related_ids"]}
    assert {o["origin"] for o in resumed["outcomes"] if o["card_key"]} == {"automatic"}
    assert resumed["receipt_ids"] == [resumed["file_export"]["receipt_id"]]
    assert resumed["complete"]


def test_pipeline_report_one_failed_claim_withholds_multiple_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, objectives, _ = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, tmp_path / "unused")
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only")
    root = output / "run"
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    for path in (root / "analyses").glob("*.json"):
        analysis = Analysis.read_json(path.read_bytes())
        if analysis.binding.source_identity != "n":
            continue
        checks = tuple(
            c.model_copy(
                update={"result": "fail", "objections": ("synthetic contradiction",), "alternatives": ("other",)}
            )
            if c.assertion_ids == ("source",)
            else c
            for c in analysis.checks
        )
        write(path, analysis.model_copy(update={"checks": checks}))
    report = report_call("resume", "--run", str(root))
    assert report["counts"]["failed_claims"] == 1
    assert report["counts"]["cards_affected_by_failed_claims"] == 2
    failed = [o for o in report["outcomes"] if o["failed_claim_ids"]]
    assert len(failed) == 2 and {o["outcome"] for o in failed} == {"withheld"}
    assert all("synthetic contradiction" in o["objections"] and "other" in o["alternatives"] for o in failed)
    assert report["counts"]["selected_cards"] == 4


def test_pipeline_report_provider_prerequisite_is_not_linguistic_rejection(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    configure_pipeline_http(profile, "http://127.0.0.1:1")
    output = tmp_path / "workflow"
    report = report_call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    cards = [o for o in report["outcomes"] if o["card_key"]]
    assert len(cards) == 1 and cards[0]["outcome"] == "deferred"
    assert cards[0]["reason"] == "missing_token_bound"
    assert report["calls"] == [] and not report["complete"]
    assert any(p["reason"] == "missing_token_bound" for p in report["processing_outcomes"])


def test_pipeline_report_processing_failures_and_secret_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, objectives, _ = fixture(tmp_path)
    secret = "synthetic-runtime-secret"
    monkeypatch.setenv("SYNTHETIC_REPORT_TOKEN", secret)
    with synthetic_pipeline_http() as (url, requests, status):
        configure_pipeline_http(profile, url)
        raw = json.loads(profile.read_text())
        for endpoint in raw["providers"].values():
            endpoint["credential_ref"] = "SYNTHETIC_REPORT_TOKEN"
        write(profile, raw)
        status[0] = 503
        output = tmp_path / "workflow"
        result = report_call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        assert requests and {o["outcome"] for o in result["outcomes"] if o["card_key"]} == {"processing_failed"}
        assert not result["complete"]
        root = output / "run"
        identity = result["stage_ids"]["ingest"]
        stage_path = root / "stages" / (identity + ".json")
        stage = json.loads(stage_path.read_text())
        stage["state"] = "arbitrary-secret-from-error"
        write(stage_path, stage)
        dispatched = len(requests)
        failed = report_call("resume", "--run", str(root))
        assert len(requests) == dispatched and not failed["reuse"].get("processed")
        assert failed["invalid_stages"] and failed["status"] in ("failed", "unknown")
        assert any(p["reason"] == "invalid_stage_artifact" for p in failed["processing_outcomes"])
        stored = (output / "report.json").read_text()
        assert "arbitrary-secret-from-error" not in stored and secret not in stored
        assert "diagnostic_details_redacted" in stored
        assert report_call("report", "--run", str(root)) == failed
        human = runner.invoke(app, ["pipeline", "report", "--run", str(root)])
        assert human.exit_code == 0 and "processing_failed" in human.output
        assert secret not in human.output and "arbitrary-secret-from-error" not in human.output


def test_pipeline_report_validates_historical_profile_shape_without_authorizing_it(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "out"
    result = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    record = result["outcome_report"]["record"]
    historical = json.loads(profile.read_text())
    historical["policy"]["id"] = "historical-uninstalled-policy"
    record["effective_profile"] = historical
    record["profile_sha256"] = digest(historical)
    write(output / "report.json", result)
    assert call("report", "--run", str(output / "run"))["outcome_report"]["record"]["effective_profile"] == historical
    for malformed in ({}, {**historical, "selection": {}}, {**historical, "unexpected": "private-profile-marker"}):
        record["effective_profile"] = malformed
        record["profile_sha256"] = digest(malformed)
        write(output / "report.json", result)
        bad = runner.invoke(app, ["pipeline", "report", "--run", str(output / "run"), "--json"])
        assert bad.exit_code == 1 and "pipeline_report_unavailable_or_invalid" in bad.output
        assert "private-profile-marker" not in bad.output


def test_pipeline_report_schema_rejects_invalid_version_and_counts(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from latinitas_cards.pipeline_reporting import PipelineReport, Reporting

    profile, objectives, _ = fixture(tmp_path)
    result = call(
        "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(tmp_path / "workflow")
    )
    assert PipelineReport.model_json_schema()["properties"]["format"]["const"] == "latinitas-pipeline-report"
    view = result["outcome_report"]
    for mutation in (
        {"version": 2},
        {"version": True},
        {"source_units": {"selected": -1, "attempted": 0, "deferred": 0}},
        {"source_units": {"selected": True, "attempted": 0, "deferred": 0}},
        {"stage_counts": {"invented_state": 1}},
        {"extra": "not_an_approved_key"},
    ):
        with pytest.raises(ValueError):
            PipelineReport.read_json(canonical({**view["record"], **mutation}))
    with pytest.raises(ValidationError):
        Reporting.read_json(canonical({**view, "details": {**view["details"], "counts": {"cards": -1}}}))
    view["record"]["version"] = "private-error-string"
    write(tmp_path / "workflow" / "report.json", result)
    bad = runner.invoke(app, ["pipeline", "report", "--run", str(tmp_path / "workflow" / "run"), "--json"])
    assert bad.exit_code == 1 and "private-error-string" not in bad.output
    result.pop("outcome_report")
    result["historical_diagnostics"] = {"error": "Bearer SYNTHETIC_OLD_TOKEN"}
    write(tmp_path / "workflow" / "report.json", result)
    legacy = call("report", "--run", str(tmp_path / "workflow" / "run"))
    assert "outcome_report" not in legacy and "SYNTHETIC_OLD_TOKEN" not in canonical(legacy).decode()


def test_pipeline_report_links_confirmed_phrase_passages(tmp_path: Path) -> None:
    profile, objectives, _ = phrase_fixture(tmp_path, "en", "preposition-noun")
    report = report_call(
        "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(tmp_path / "workflow")
    )
    cards = [o for o in report["outcomes"] if o["card_key"]]
    assert cards and all(any("#/patterns/" in link for link in o["source_links"]) for o in cards)
    for item in cards:
        for link in item["source_links"]:
            pointer = link.split("#/", 1)[1].split("/")
            assert json.loads((tmp_path / "inputs.json").read_text())[pointer[0]][int(pointer[1])]


def provider_answer(body: dict[str, Any]) -> bytes:
    task = json.loads(body["messages"][1]["content"])
    answer = dict(
        format="latinitas-model-answer",
        version=1,
        task_sha256=task["task_sha256"],
        assertions=[
            dict(
                id="independent",
                kind="label",
                value="synthetic-only",
                selected_sense=None,
                alternatives=[],
                evidence=[EVIDENCE],
            )
        ],
        alternatives=[],
        objections=[],
    )
    return canonical({"choices": [{"finish_reason": "stop", "message": {"content": canonical(answer).decode()}}]})


@contextmanager
def synthetic_pipeline_http(*, usage: bool = False) -> Iterator[tuple[str, list[dict[str, Any]], list[int]]]:
    requests: list[dict[str, Any]] = []
    status = [200]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            payload = provider_answer(body)
            if usage:
                payload = canonical(
                    {
                        **json.loads(payload),
                        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
                    }
                )
            self.send_response(status[0])
            self.send_header("x-request-id", f"synthetic-{len(requests)}")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # This controlled server's accounting is part of the fixture, not a
        # qualification of a model label or a production-compatible endpoint.
        from latinitas_cards import pipeline_orchestration as workflow

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(workflow, "input_token_bound", lambda endpoint: len)
            yield f"http://127.0.0.1:{server.server_port}", requests, status
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def configure_pipeline_http(profile: Path, url: str, *, calls: int = 200) -> None:
    raw = json.loads(profile.read_text())
    endpoint = dict(url=url, provider="openai", model="gpt-4o-mini-2024-07-18", locality="local", credential_ref=None)
    raw["providers"] = dict(proposal=endpoint, counter_check=endpoint)
    raw["selection"]["entries"] = ["n"]
    raw["processing"].update(max_source_units=1, max_calls=calls)
    write(profile, raw)


@pytest.mark.parametrize("phase,purpose", [("development", "development"), ("study", "isolated_evaluation")])
def test_estimated_installed_pipeline_separates_usage_custody_and_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, purpose: str
) -> None:
    from latinitas_cards.hosted_accounting import PHASE_CALLS, Campaign, CampaignAllocation

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    profile, objectives, _ = fixture(tmp_path)
    allocation = CampaignAllocation(
        format="latinitas-campaign-allocation",
        version=1,
        campaign_id=digest("pipeline-test"),
        authority="synthetic offline engineering only",
        max_calls=2,
        max_tokens=40000000,
        max_microdollars=25000000,
        max_execution_seconds=86400,
        phase_calls=PHASE_CALLS,
    )
    Campaign(allocation.campaign_id).initialize(allocation)
    with synthetic_pipeline_http(usage=True) as (url, requests, _):
        configure_pipeline_http(profile, url)
        raw = json.loads(profile.read_text())
        raw["version"] = 2
        raw["processing"].update(
            retries=0,
            accounting={
                "format": "latinitas-estimated-accounting",
                "version": 1,
                "kind": "estimated",
                "campaign_id": allocation.campaign_id,
                "phase": phase,
                "purpose": purpose,
                "estimator": "wire-padded/1",
                "service_revision": None,
                "acknowledges_no_guaranteed_cap": True,
                "pricing": [
                    {
                        "endpoint_sha256": digest(raw["providers"]["proposal"]),
                        "input_microdollars_per_million": 1000000,
                        "output_microdollars_per_million": 3000000,
                        "source_url": "https://example.invalid/pricing",
                        "checked_at": "2020-01-01T00:00:00Z",
                        "expires_at": "2099-01-01T00:00:00Z",
                    }
                ],
            },
        )
        write(profile, raw)
        output = tmp_path / "estimated"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        assert len(requests) == 2
        assert [c["used_tokens"] for c in report["provider_calls"]] == [130, 130]
        assert report["outcome_report"]["record"]["version"] == 2
        accounting = raw["processing"]["accounting"]
        charges = {c["call_id"]: c for c in report["accounting_provenance"]["charges"]}
        assert set(charges) == {c["call_id"] for c in report["provider_calls"]}
        captured = {digest(body): body for body in requests}
        assert len(captured) == 2
        for provider_call in report["provider_calls"]:
            charge = charges[provider_call["call_id"]]
            reservation = charge["reservation"]
            assert charge["run_id"] == report["run_id"]
            assert charge["stage_id"] == provider_call["stage_id"]
            assert charge["state"] == provider_call["state"] == "succeeded"
            assert charge["stop_reason"] is None
            assert reservation["request_sha256"] == provider_call["request_sha256"]
            assert reservation["request_sha256"] in captured
            disclosure = next(d for d in report["disclosure"] if reservation["request_sha256"] in d["request_sha256s"])
            assert reservation["scope_sha256"] == digest(disclosure["data_scope"])
            assert reservation["endpoint_sha256"] == provider_call["endpoint_sha256"]
            assert reservation["endpoint_sha256"] == digest(raw["providers"]["proposal"])
            assert reservation["accounting"] == disclosure["accounting"] == accounting
            assert reservation["pricing_sha256"] == digest(accounting["pricing"][0])
            assert reservation["permission_id"] is None
            assert charge["usage"] == {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130}
            assert reservation["input_tokens"] > charge["usage"]["input_tokens"]
            assert reservation["output_tokens"] > charge["usage"]["output_tokens"]
            assert provider_call["reserved_tokens"] == reservation["input_tokens"] + reservation["output_tokens"]
            assert provider_call["reserved_tokens"] > provider_call["used_tokens"]
        limits = report["outcome_report"]["details"]["limits"]
        assert limits["accounting_kind"] == "estimated" and limits["estimator"] == "wire-padded/1"
        assert limits["campaign_id"] == allocation.campaign_id and limits["purpose"] == purpose
        assert limits["no_guaranteed_token_or_financial_cap"] is True
        assert limits["actual_plan_deduction"] == "unknown"
        assert limits["tokens_reserved"] == sum(c["reserved_tokens"] for c in report["provider_calls"])
        assert report["stages"]["export"][0]["reason"] == "isolated_artifact_not_learning_export"
        assert not list(output.glob("*.csv")) and "file_export" not in report
        root = output / "run"
        assert json.loads((output / "report.json").read_text()) == call("report", "--run", str(root)) == report
        resumed = call("resume", "--run", str(root), "--offline")
        assert resumed["accounting_provenance"] == report["accounting_provenance"]
        assert resumed["provider_calls"] == report["provider_calls"]
        assert call("report", "--run", str(root)) == resumed
        assert len(requests) == 2  # Completed capture needs neither pricing renewal nor another dispatch.
        rejected = runner.invoke(app, ["pipeline", "export", "--run", str(root), "--output", str(tmp_path / "bad.csv")])
        assert rejected.exit_code != 0 and "isolated" in rejected.output
        assert not (tmp_path / "bad.csv").exists()
        second = call(
            "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(tmp_path / "another")
        )
        assert len(requests) == 2
        assert second["stages"]["propose"][0]["reason"] == "campaign_budget_exhausted"

        adapted = tmp_path / "adapted"
        call(
            "adapt",
            "--kind",
            "phrase-inputs",
            "--profile",
            str(profile),
            "--objectives",
            str(objectives),
            "--output",
            str(adapted),
        )
        replacement = tmp_path / "replacement.json"
        raw["version"] = 1
        raw["processing"].pop("accounting")
        write(replacement, raw)
        call("validate", "--profile", str(replacement))
        changed = runner.invoke(
            app, ["pipeline", "resume", "--run", str(root), "--profile", str(replacement), "--offline", "--json"]
        )
        assert changed.exit_code != 0 and "immutable" in changed.output
        for isolated in (root, adapted):
            custody = (isolated / "accounting-custody.json").read_bytes()
            assert json.loads(custody) == accounting
            destination = tmp_path / (isolated.name + "-replacement.csv")
            rejected = runner.invoke(
                app,
                [
                    "pipeline",
                    "export",
                    "--run",
                    str(isolated),
                    "--profile",
                    str(replacement),
                    "--output",
                    str(destination),
                    "--json",
                ],
            )
            assert rejected.exit_code != 0 and "isolated" in rejected.output
            assert not destination.exists()
            assert (isolated / "accounting-custody.json").read_bytes() == custody
        assert len(requests) == 2


def test_orchestrated_http_success_lost_response_resume_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latinitas_cards import pipeline_orchestration as workflow

    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "workflow"
    with synthetic_pipeline_http() as (url, requests, _):
        configure_pipeline_http(profile, url)
        original = workflow._stage

        def interrupt(
            root: Path,
            state: workflow.Execution,
            card: Rendered,
            name: StageName,
            envelope: dict[str, object],
        ) -> str:
            if name == "propose":
                raise KeyboardInterrupt
            return original(root, state, card, name, envelope)

        monkeypatch.setattr(workflow, "_stage", interrupt)
        result = runner.invoke(
            app,
            [
                "pipeline",
                "run",
                "--profile",
                str(profile),
                "--objectives",
                str(objectives),
                "--output",
                str(output),
                "--json",
            ],
        )
        assert result.exit_code != 0 and len(requests) == 1
        monkeypatch.setattr(workflow, "_stage", original)
        root = output / "run"
        report = call("resume", "--run", str(root))
        assert len(requests) == 2 and report["file_export"]["exported"] == 0
        assert [r["state"] for r in report["provider_calls"]] == ["succeeded", "succeeded"]
        assert json.loads(requests[0]["messages"][1]["content"]) == json.loads(requests[1]["messages"][1]["content"])
        assert "proposal/1" in requests[0]["messages"][0]["content"]
        assert "blind/1" in requests[1]["messages"][0]["content"]
        assert "synthetic-only" not in requests[1]["messages"][1]["content"]
        call("resume", "--run", str(root))
        call("resume", "--run", str(root), "--offline")
        call("report", "--run", str(root))
        call("preview", "--run", str(root))
        call("validate", "--profile", str(profile))
        assert len(requests) == 2


def test_orchestrated_unknown_retrieval_and_budget(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    output = tmp_path / "workflow"
    with synthetic_pipeline_http() as (url, requests, status):
        configure_pipeline_http(profile, url)
        status[0] = 503
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        assert len(requests) == 1
        unknown = report["provider_calls"][0]
        assert unknown["state"] == "unknown" and unknown["reserved_tokens"] > 2048
        call("resume", "--run", str(root))
        assert len(requests) == 1
        response = tmp_path / "retrieved.json"
        response.write_bytes(provider_answer(requests[0]))
        reconciled = call(
            "reconcile-call",
            "--run",
            str(root),
            "--call-id",
            unknown["call_id"],
            "--response",
            str(response),
            "--request-sha256",
            unknown["request_sha256"],
            "--provider-request-id",
            "synthetic-1",
        )
        assert reconciled["state"] == "succeeded"
        status[0] = 200
        finished = call("resume", "--run", str(root))
        assert len(requests) == 2 and finished["file_export"]["exported"] == 0
        assert sum(c["reserved_tokens"] for c in finished["provider_calls"]) > unknown["reserved_tokens"]
        configure_pipeline_http(profile, url, calls=2)
        changed = json.loads(profile.read_text())
        changed["providers"]["counter_check"]["url"] += "/changed-checker"
        write(profile, changed)
        # A changed endpoint/ceiling preserves the proposal, not a fresh budget.
        exhausted = call("resume", "--run", str(root), "--profile", str(profile))
        assert len(requests) == 2 and exhausted["status"] == "deferred"
        assert exhausted["stages"]["propose"][0]["state"] == "succeeded"
        assert exhausted["stages"]["counter_check"][0]["reason"] == "budget_exhausted"


def test_orchestrated_changed_source_reuses_unrelated_calls(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    with synthetic_pipeline_http() as (url, requests, _):
        configure_pipeline_http(profile, url)
        raw = json.loads(profile.read_text())
        raw["selection"]["entries"] = ["n", "v"]
        raw["processing"]["max_source_units"] = 3
        write(profile, raw)
        output = tmp_path / "workflow"
        first = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        assert len(requests) == 6
        data = json.loads((tmp_path / "inputs.json").read_text())
        data["entries"][0]["senses"][0]["valency"]["subject"] = "accusative"
        rebind(profile, data)
        resumed = call(
            "resume",
            "--run",
            str(output / "run"),
            "--profile",
            str(profile),
            "--reconcile-receipt",
            first["file_export"]["receipt_id"],
        )
        assert len(requests) == 8
        assert len(resumed["provider_calls"]) == 8
        assert resumed["file_export"]["exported"] == 0
        assert {c["call_id"] for c in first["provider_calls"]} <= {c["call_id"] for c in resumed["provider_calls"]}


def test_orchestrated_permission_exact_requests_and_renewal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime, timedelta

    from latinitas_cards import pipeline_orchestration as workflow
    from latinitas_cards.providers import Endpoint

    profile, objectives, _ = fixture(tmp_path)
    configure_pipeline_http(profile, "https://synthetic.invalid")
    raw = json.loads(profile.read_text())
    for endpoint in raw["providers"].values():
        endpoint["locality"] = "hosted"
    write(profile, raw)
    output = tmp_path / "workflow"
    prepared = call(
        "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only"
    )
    disclosed = prepared["disclosure"][0]
    now = datetime.now(UTC)
    permission: dict[str, Any] = dict(
        format="latinitas-hosted-permission",
        version=1,
        run_id=prepared["run_id"],
        endpoint_sha256s=[digest(disclosed["endpoints"]["proposal"])],
        config_sha256=disclosed["sent_task"]["binding"]["config_sha256"],
        input_sha256s=[*disclosed["input_sha256s"], *disclosed["request_sha256s"]],
        data_scope=disclosed["data_scope"],
        max_calls=2,
        max_tokens=200000,
        timeout_seconds=60,
        expires_at=(now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        confirmed_by="synthetic operator",
        confirmed_at=now.isoformat().replace("+00:00", "Z"),
    )
    permission["permission_id"] = digest(permission)
    grant = tmp_path / "permission.json"
    write(grant, permission)
    sent: list[dict[str, Any]] = []

    class ControlledTransport:
        def __init__(self, credential: str | None = None) -> None:
            pass

        def __call__(self, endpoint: Endpoint, body: bytes, timeout: float, cancel: workflow.Cancellation) -> bytes:
            decoded = json.loads(body)
            sent.append(decoded)
            return provider_answer(decoded)

    monkeypatch.setattr(workflow, "HTTPTransport", ControlledTransport)
    monkeypatch.setattr(workflow, "input_token_bound", lambda endpoint: len)
    root = output / "run"
    denied = call("resume", "--run", str(root))
    assert sent == [] and denied["stages"]["propose"][0]["reason"] == "permission_or_scope"
    permitted = call("resume", "--run", str(root), "--allow-hosted", str(grant))
    assert len(sent) == 2 and len(permitted["provider_calls"]) == 2
    assert permitted["outcome_report"]["record"]["permission_sha256"] == hashlib.sha256(grant.read_bytes()).hexdigest()
    assert permitted["outcome_report"]["details"]["limits"]["hosted"]["max_calls"] == 2
    assert permitted["outcome_report"]["details"]["limits"]["local"]["max_calls"] == 200
    assert (root / "permissions" / (permission["permission_id"] + ".json")).exists()
    raw["providers"]["counter_check"]["url"] += "/new-scope"
    write(profile, raw)
    changed = call("resume", "--run", str(root), "--profile", str(profile), "--allow-hosted", str(grant))
    assert len(sent) == 2 and changed["stages"]["counter_check"][0]["reason"] == "permission_or_scope"
    permission.pop("permission_id")
    permission["max_calls"] = 10
    permission["permission_id"] = digest(permission)
    write(grant, permission)
    # A renewed ceiling still does not grant the changed endpoint/body.
    call("resume", "--run", str(root), "--allow-hosted", str(grant))
    assert len(sent) == 2
    local = runner.invoke(app, ["pipeline", "resume", "--run", str(root), "--local-only", "--json"])
    assert local.exit_code != 0 and "local_only" in local.stdout
    assert call("resume", "--run", str(root), "--offline")["provider_calls"] == changed["provider_calls"]
    assert len(sent) == 2


def test_orchestrated_abandon_cancel_and_tampered_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from latinitas_cards import pipeline_orchestration as workflow
    from latinitas_cards.providers import Endpoint, TransportFailure

    profile, objectives, _ = fixture(tmp_path)
    configure_pipeline_http(profile, "http://127.0.0.1:1")
    output = tmp_path / "workflow"
    call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only")
    root = output / "run"
    dispatches = []

    class CancelTransport:
        def __init__(self, credential: str | None = None) -> None:
            pass

        def __call__(self, endpoint: Endpoint, body: bytes, timeout: float, cancel: workflow.Cancellation) -> bytes:
            dispatches.append(body)
            workflow.cancel(root)
            assert cancel.is_set()
            raise TransportFailure("unknown", "interrupted-1")

    monkeypatch.setattr(workflow, "HTTPTransport", CancelTransport)
    monkeypatch.setattr(workflow, "input_token_bound", lambda endpoint: len)
    cancelled = call("resume", "--run", str(root))
    assert cancelled["status"] == "cancelled" and len(dispatches) == 1
    unknown = cancelled["provider_calls"][0]
    assert unknown["state"] == "unknown"
    undispatched = call("resume", "--run", str(root), "--continue-cancelled")
    assert len(dispatches) == 1
    assert not undispatched["outcome_report"]["details"]["reuse"].get("processed")
    call("reconcile-call", "--run", str(root), "--call-id", unknown["call_id"], "--abandon")
    call("resume", "--run", str(root), "--retry-unknown")
    assert len(dispatches) == 1
    repeated = call("resume", "--run", str(root), "--retry-unknown", "--confirm-duplicate-billing")
    assert len(dispatches) == 2 and len(repeated["provider_calls"]) == 2
    assert all(c["reserved_tokens"] > 2048 for c in repeated["provider_calls"])
    stage_id = repeated["stage_ids"]["ingest"]
    stage = json.loads((root / "stages" / (stage_id + ".json")).read_text())
    (root / stage["artifact"]).write_text("{}")
    failed = call("resume", "--run", str(root), "--continue-cancelled")
    assert failed["status"] == "unknown"
    assert any(r["stage_id"] == stage_id and "digest mismatch" in r["detail"] for r in failed["invalid_stages"])
    assert failed["stage_ids"]["ingest"] == stage_id
    assert len(dispatches) == 2


def test_orchestrated_library_unavailable_and_independent_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latinitas_cards.acceptance import save_acceptance
    from latinitas_cards.acceptance_records import Human
    from latinitas_cards.pipeline_consumers import Objectives

    profile, objectives, _ = fixture(tmp_path / "human", preset=None)
    output = profile.parent / "workflow"
    call("run", "--profile", str(profile), "--output", str(output), "--prepare-only")
    root = output / "run"
    parsed = PipelineProfile.read_json(profile.read_bytes())
    empty = Objectives(format="latinitas-grammar-objectives", version=1, forms=(), phrases=())
    card = current_cards(profile, parsed, empty, root)[0][0]
    envelope = check_envelope(card.binding, card.versions, "human", card.claims.assertions)
    stage = save_stage(
        root,
        run_id="synthetic-review",
        stage_id="review",
        stage="analyze",
        binding=card.binding,
        versions=card.versions,
        envelope=envelope,
    )
    check = Check(
        id="review",
        assertion_ids=tuple(a.id for a in card.claims.assertions),
        method="human",
        version=card.versions.checker,
        result="pass",
        evidence=(EVIDENCE,),
        alternatives=(),
        objections=(),
        input_sha256=digest(envelope),
    )
    decision = card.claims.decisions[0].reader.decide(
        checks=(check,),
        origin="human",
        stage_ids=(stage.stage_id,),
        human=Human(
            reviewer="synthetic operator",
            reason="independent exact card review",
            reviewed_sha256=digest({"binding": card.binding, "assertions": card.claims.assertions}),
        ),
    )
    save_acceptance(card.claims.decisions[0].path, decision)
    human_report = report_call("resume", "--run", str(root), "--offline")
    assert human_report["file_export"]["exported"] == 1
    accepted = [o for o in human_report["outcomes"] if o["outcome"] == "exported"]
    assert len(accepted) == 1 and accepted[0]["origin"] == "human"
    assert any(o["origin"] != "human" for o in human_report["outcomes"] if o["card_key"])
    human_ref = (accepted[0]["note_id"], accepted[0]["card_key"])
    for stage_name in ("select", "render", "export"):
        restored = report_call("stage", stage_name, "--run", str(root))
        assert not [o for o in restored["processing_outcomes"] if (o.get("note_id"), o.get("card_key")) == human_ref]
        assert [o for o in restored["outcomes"] if (o["note_id"], o["card_key"]) == human_ref] == accepted
        assert restored["complete"] == human_report["complete"]

    library_output = profile.parent / "missing-library"
    missing = report_call(
        "run",
        "--profile",
        str(profile),
        "--output",
        str(library_output),
        "--library-resources",
        str(tmp_path / "absent-models"),
    )
    assert missing["status"] == "failed" and missing["provider_calls"] == []
    assert any(r.get("reason") == "processing_unavailable" for r in missing["stages"]["analyze"])
    assert any(
        o["outcome"] == "processing_failed" and o["reason"] == "processing_unavailable" for o in missing["outcomes"]
    )

    profile2, objectives2, _ = fixture(tmp_path / "isolated")
    install_synthetic(monkeypatch, profile2, objectives2, tmp_path / "unused")
    output2 = profile2.parent / "workflow"
    call(
        "run", "--profile", str(profile2), "--objectives", str(objectives2), "--output", str(output2), "--prepare-only"
    )
    root2 = output2 / "run"
    (root2 / "analyses").mkdir()
    prove_cards(profile2, objectives2, root2)
    positive = call("resume", "--run", str(root2))
    assert positive["file_export"]["exported"] == 6
    assert positive["provider_calls"] == [] and positive["applied_to_anki"] is False


def test_orchestrated_completed_calls_need_no_runtime_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, objectives, _ = fixture(tmp_path)
    with synthetic_pipeline_http() as (url, requests, _):
        configure_pipeline_http(profile, url)
        raw = json.loads(profile.read_text())
        for endpoint in raw["providers"].values():
            endpoint["credential_ref"] = "SYNTHETIC_T193_KEY"
        write(profile, raw)
        monkeypatch.setenv("SYNTHETIC_T193_KEY", "controlled-fixture-only")
        output = tmp_path / "workflow"
        first = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        monkeypatch.delenv("SYNTHETIC_T193_KEY")
        replay = call("resume", "--run", str(output / "run"))
        assert len(requests) == 2 and replay["provider_calls"] == first["provider_calls"]
        assert all(replay["stages"][s][0]["state"] == "succeeded" for s in ("propose", "counter_check"))


def test_orchestrated_allocation_is_published_only_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latinitas_cards.providers import ProviderRun

    profile, _, _ = fixture(tmp_path)
    output = tmp_path / "workflow"
    initialize = ProviderRun.initialize

    def interrupted(self: ProviderRun) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(ProviderRun, "initialize", interrupted)
    result = runner.invoke(app, ["pipeline", "run", "--profile", str(profile), "--output", str(output)])
    assert result.exit_code != 0 and not output.exists()
    monkeypatch.setattr(ProviderRun, "initialize", initialize)
    call("run", "--profile", str(profile), "--output", str(output), "--prepare-only")
    (output / "run/providers/provider-ledger.json").unlink()
    damaged = runner.invoke(app, ["pipeline", "resume", "--run", str(output / "run"), "--json"])
    assert damaged.exit_code != 0 and "missing durable provider ledger" in damaged.stdout


def test_orchestrated_transient_library_failure_is_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from latinitas_cards import pipeline_orchestration as workflow

    profile, objectives, _ = fixture(tmp_path, preset=None)
    raw = json.loads(profile.read_text())
    raw["selection"]["entries"] = ["n"]
    raw["processing"]["max_source_units"] = 1
    overrides = RecipeOverrides(include=("form_recognition",), exclude=())
    raw.update(recipe_overrides=overrides.model_dump(), recipes=resolve_recipes(None, overrides))
    write(profile, raw)
    monkeypatch.setattr(workflow, "LocalStanza", lambda path: SimpleNamespace(version="fixed-synthetic-version"))
    attempted: list[int] = []

    def analyze(*args: object, **kwargs: object) -> SimpleNamespace:
        attempted.append(1)
        return SimpleNamespace(state="unavailable" if len(attempted) == 1 else "unsupported", checks=())

    monkeypatch.setattr(workflow, "analyze_deck", analyze)
    output = tmp_path / "workflow"
    first = call(
        "run",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(output),
        "--library-resources",
        "unused",
        "--offline",
    )
    assert first["status"] == "failed"
    replay = call("resume", "--run", str(output / "run"))
    assert len(attempted) == 2 and replay["stages"]["analyze"][0].get("state") != "failed"
    call("resume", "--run", str(output / "run"))
    assert len(attempted) == 2


def test_orchestrated_partial_processing_and_decision_errors_are_explicit(tmp_path: Path) -> None:
    from latinitas_cards.acceptance import load_stage

    profile, objectives, _ = fixture(tmp_path)
    raw = json.loads(profile.read_text())
    raw["processing"]["max_source_units"] = 1
    write(profile, raw)
    output = tmp_path / "workflow"
    report = call(
        "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--offline"
    )
    assert any(s["reason"] == "processing_limit" for s in report["skipped"])
    assert report["status"] == "deferred"
    root = output / "run"
    next((root / "analyses").glob("*.json")).write_text("{}")
    failed = call("stage", "decide", "--run", str(root))
    assert failed["stages"]["decide"][0]["analysis_errors"]
    assert failed["status"] == "failed"
    assert load_stage(root, failed["stage_ids"]["decide"]).state == "failed"


def test_orchestrated_standalone_provider_requires_each_cards_analysis(tmp_path: Path) -> None:
    from latinitas_cards.acceptance import load_stage, stage_artifact

    profile, objectives, _ = fixture(tmp_path)
    with synthetic_pipeline_http() as (url, requests, _):
        configure_pipeline_http(profile, url)
        output = tmp_path / "workflow"
        call(
            "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only"
        )
        root = output / "run"
        call("stage", "analyze", "--run", str(root))
        raw = json.loads(profile.read_text())
        raw["selection"]["entries"] = ["n", "v"]
        raw["processing"]["max_source_units"] = 3
        write(profile, raw)
        proposed = call("stage", "propose", "--run", str(root), "--profile", str(profile))
        assert len(requests) == 1
        ready = [r for r in proposed["stages"]["propose"] if r["state"] == "succeeded"]
        assert len(ready) == 1
        assert sum(r.get("reason") == "prerequisite_analyze_incomplete" for r in proposed["stages"]["propose"]) == 2
        stage = load_stage(root, ready[0]["execution_stage_id"])
        assert stage.prerequisite_ids
        stage_artifact(root, stage)


def test_orchestrated_network_restrictions_survive_resume(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    with synthetic_pipeline_http() as (url, requests, _):
        configure_pipeline_http(profile, url)
        output = tmp_path / "workflow"
        call(
            "run",
            "--profile",
            str(profile),
            "--objectives",
            str(objectives),
            "--output",
            str(output),
            "--offline",
            "--local-only",
        )
        replay = call("resume", "--run", str(output / "run"))
        assert requests == [] and replay["network_policy"] == {"offline": True, "local_only": True}
        raw = json.loads(profile.read_text())
        for endpoint in raw["providers"].values():
            endpoint.update(url="https://synthetic.invalid", locality="hosted")
        write(profile, raw)
        denied = runner.invoke(
            app, ["pipeline", "resume", "--run", str(output / "run"), "--profile", str(profile), "--json"]
        )
        assert denied.exit_code != 0 and "local_only" in denied.stdout


def test_orchestrated_unqualified_provider_labels_do_not_supply_token_bounds(tmp_path: Path) -> None:
    profile, objectives, _ = fixture(tmp_path)
    configure_pipeline_http(profile, "http://127.0.0.1:1")
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    assert report["provider_calls"] == []
    assert report["stages"]["propose"][0]["reason"] == "missing_token_bound"


@pytest.mark.parametrize("damage", ["analysis", "proof", "ancestor"])
def test_orchestrated_provider_validates_underlying_prerequisites(tmp_path: Path, damage: str) -> None:
    from latinitas_cards.acceptance import load_stage
    from latinitas_cards.pipeline_consumers import Objectives

    profile, objectives, _ = fixture(tmp_path)
    with synthetic_pipeline_http() as (url, requests, _):
        configure_pipeline_http(profile, url)
        raw = json.loads(profile.read_text())
        raw["selection"]["entries"] = ["n", "v"]
        raw["processing"]["max_source_units"] = 3
        write(profile, raw)
        output = tmp_path / "workflow"
        call(
            "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only"
        )
        root = output / "run"
        call("stage", "analyze", "--run", str(root))
        cards, _ = current_cards(
            profile,
            PipelineProfile.read_json(profile.read_bytes()),
            Objectives.read_json(objectives.read_bytes()),
            root,
        )
        damaged = cards[0]
        if damage == "ancestor":
            call("stage", "propose", "--run", str(root))
            assert len(requests) == 3
        if damage == "proof":
            analysis = Analysis.read_json(damaged.analysis_path.read_bytes())
            proof = load_stage(root, analysis.stage_ids[0])
            assert proof.artifact is not None
            (root / proof.artifact).write_text("{}")
        else:
            damaged.analysis_path.write_text("{}")
        before = len(requests)
        stage = "counter_check" if damage == "ancestor" else "propose"
        report = call("stage", stage, "--run", str(root))
        assert len(requests) == before + 2
        failed = [r for r in report["stages"][stage] if r["note_id"] == damaged.claims.note_id]
        assert len(failed) == 1 and failed[0]["state"] == "failed"
        assert failed[0]["reason"] == "prerequisite_artifact_invalid"
        assert all(
            json.loads(r["messages"][1]["content"])["task_sha256"] != damaged.binding.task_sha256
            for r in requests[before:]
        )


@pytest.mark.parametrize("reselect", [False, True], ids=["healthy-siblings", "unrelated-selection"])
def test_orchestrated_corrupt_wrapper_is_branch_local(tmp_path: Path, reselect: bool) -> None:
    from latinitas_cards.acceptance import load_stage
    from latinitas_cards.pipeline_consumers import Objectives

    profile, objectives, _ = fixture(tmp_path)
    with synthetic_pipeline_http() as (url, requests, _):
        configure_pipeline_http(profile, url)
        raw = json.loads(profile.read_text())
        raw["selection"]["entries"] = ["n", "a"]
        raw["processing"]["max_source_units"] = None
        write(profile, raw)
        output = tmp_path / "workflow"
        prepared = call(
            "run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output), "--prepare-only"
        )
        root = output / "run"
        cards, _ = current_cards(
            profile,
            PipelineProfile.read_json(profile.read_bytes()),
            Objectives.read_json(objectives.read_bytes()),
            root,
        )
        damaged = next(c for c in cards if c.objective.model_dump().get("entry") == "n")
        wrapper = next(
            r
            for r in prepared["stages"]["ingest"]
            if (r["note_id"], r["card_key"]) == (damaged.claims.note_id, damaged.claims.card_key)
        )
        identity = wrapper["execution_stage_id"]
        stage = load_stage(root, identity)
        assert stage.artifact is not None
        artifact = root / stage.artifact
        artifact.write_text("{}")
        record = (root / "stages" / (identity + ".json")).read_bytes()
        if reselect:
            raw["selection"]["entries"] = ["v"]
            write(profile, raw)
            ingested = call("stage", "ingest", "--run", str(root), "--profile", str(profile))
            assert ingested["stages"]["ingest"]
            assert all(
                load_stage(root, r["execution_stage_id"]).state == "succeeded" for r in ingested["stages"]["ingest"]
            )
        analyzed = call("stage", "analyze", "--run", str(root))
        assert analyzed["invalid_stages"]
        healthy = [r for r in analyzed["stages"]["analyze"] if r.get("state") != "failed"]
        assert healthy
        if not reselect:
            failed = [r for r in analyzed["stages"]["analyze"] if r.get("state") == "failed"]
            assert len(failed) == 1 and failed[0]["note_id"] == damaged.claims.note_id
            assert failed[0]["reason"] == "prerequisite_artifact_invalid"
        proposed = call("stage", "propose", "--run", str(root))
        assert len(requests) == len(healthy)
        if not reselect:
            blocked = [r for r in proposed["stages"]["propose"] if r["note_id"] == damaged.claims.note_id]
            assert len(blocked) == 1 and blocked[0]["state"] == "deferred"
            assert blocked[0]["reason"] == "prerequisite_analyze_incomplete"
        assert all(
            json.loads(r["messages"][1]["content"])["task_sha256"] != damaged.binding.task_sha256 for r in requests
        )
        assert sum(r.get("state") == "succeeded" for r in proposed["stages"]["propose"]) == len(healthy)
        resumed = call("resume", "--run", str(root))
        count = len(requests)
        assert count == 2 * len(healthy)
        replay = call("resume", "--run", str(root))
        assert len(requests) == count
        assert replay["file_export"] == resumed["file_export"]
        assert replay["file_export"]["exported"] == 0 and replay["applied_to_anki"] is False
        assert artifact.read_text() == "{}"
        assert (root / "stages" / (identity + ".json")).read_bytes() == record
        assert identity in json.loads((root / "stage-index.json").read_text()).values()


def test_installed_checker_change_covers_all_batches_and_corrupt_history_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latinitas_cards import pipeline_consumers as consumers

    profile, objectives, root = fixture(tmp_path)
    install_synthetic(monkeypatch, profile, objectives, root)
    call(
        "adapt",
        "--kind",
        "phrase-inputs",
        "--profile",
        str(profile),
        "--objectives",
        str(objectives),
        "--output",
        str(root),
    )
    (root / "analyses").mkdir()
    prove_cards(profile, objectives, root)
    call("stage", "decide", "--run", str(root))
    first = call("export", "--run", str(root), "--output", str(tmp_path / "first.csv"), "--output-limit", "1")
    second = call(
        "export",
        "--run",
        str(root),
        "--output",
        str(tmp_path / "second.csv"),
        "--output-limit",
        "1",
        "--after-receipt",
        first["receipt_id"],
    )
    monkeypatch.setattr(consumers, "implementation_digest", lambda: digest("new checker revision"))
    actions = call("handoff", "--run", str(root))["ineligible"]
    assert {(a["note_id"], a["card_key"]) for a in actions} == {
        (r["note_id"], r["card_key"]) for report in (first, second) for r in report["cards"]
    }
    assert {a["reason"] for a in actions} == {"binding_mismatch"}
    denied = call(
        "export",
        "--run",
        str(root),
        "--output",
        str(tmp_path / "denied.csv"),
        "--reconcile-receipt",
        second["receipt_id"],
    )
    assert denied["exported"] == 0 and denied["export_receipt"]["ineligible"] == actions
    (tmp_path / "first.csv").write_text("corrupted original artifact")
    invalid = runner.invoke(app, ["pipeline", "handoff", "--run", str(root), "--json"])
    assert invalid.exit_code != 0 and "hash mismatch" in invalid.output
