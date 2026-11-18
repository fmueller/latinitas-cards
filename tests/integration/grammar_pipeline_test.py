"""Public consumers with isolated synthetic authority, never a linguistic study."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from latinitas_cards import acceptance_policy as policies
from latinitas_cards.acceptance import check_envelope, save_stage
from latinitas_cards.acceptance_records import Binding, Check, canonical, digest
from latinitas_cards.claim_review import Evidence
from latinitas_cards.cli import app
from latinitas_cards.grammar_layout import GRAMMAR_LAYOUT, LAYOUT_SHA256
from latinitas_cards.phrase_inputs import PhraseInputV2
from latinitas_cards.pipeline_consumers import Analysis, current_cards, input_binding
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


def test_bound_human_authority_is_distinct_from_automatic_policy(tmp_path: Path) -> None:
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
    assert call("export", "--run", str(root), "--output", str(tmp_path / "human.csv"))["exported"] == 1
    # A later automatic analysis is not a revocation of this exact human review.
    (root / "analyses").mkdir()
    prove_cards(profile, objectives_path, root)
    call("stage", "decide", "--run", str(root))
    assert call("export", "--run", str(root), "--output", str(tmp_path / "human-after-analysis.csv"))["exported"] == 1
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
