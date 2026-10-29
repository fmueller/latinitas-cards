import html
import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from latinitas_cards.cli import app

runner = CliRunner()


@pytest.mark.parametrize("terminology", ["latin", "user"])
@pytest.mark.parametrize(
    ("lemma", "forms", "meaning", "fourth", "translations", "sense"),
    [
        (
            "amō",
            "amō, amāre, amāvī, amātum",
            "lieben",
            "supine",
            ["ich liebe", "lieben", "ich habe geliebt", "um zu lieben"],
            "love",
        ),
        (
            "loquor",
            "loquor, loquī, locūtus sum, locūtus",
            "sprechen",
            "perfect_passive_participle",
            ["ich spreche", "sprechen", "ich habe gesprochen", "gesprochen"],
            "speak",
        ),
        (
            "petō",
            "petō, petere, petīvī, petītum",
            "suchen; verlangen",
            "supine",
            ["ich verlange", "verlangen", "ich habe verlangt", "um zu verlangen"],
            "request",
        ),
        (
            "amō",
            "amō, amāre, amāvī, amātus",
            "lieben",
            "perfect_passive_participle",
            ["ich liebe", "lieben", "ich habe geliebt", "geliebt"],
            "love",
        ),
        ("sum", "sum, esse, fuī, ", "sein", "supine", ["ich bin", "sein", "ich bin gewesen"], "be"),
    ],
)
def test_reviewed_form_translations(
    tmp_path: Path,
    terminology: str,
    lemma: str,
    forms: str,
    meaning: str,
    fourth: str,
    translations: list[str],
    sense: str,
) -> None:
    from latinitas_cards.preview_export import prepare_principal_part_export, write_principal_part_csv
    from latinitas_cards.profile import DeckProfile, MorphologySettings, SourceIdentityConfig

    source = tmp_path / "source.csv"
    source.write_text(f'ID,Lemma,Forms,Meaning\nentry-1,{lemma},"{forms}",{meaning}\n')
    profile = DeckProfile.default(
        note_type="Latin",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="Meaning",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
        principal_part_roles=("present_1s", "present_infinitive", "perfect_1s", fourth),
        separators=(",",),
        selected_recipes=("principal_part_completion", "principal_part_recognition"),
    ).model_copy(
        update={
            "morphology": MorphologySettings.model_validate({"terminology": {"version": 1, "language": terminology}})
        }
    )
    profile_path = tmp_path / "profile.json"
    profile.save(profile_path)
    write_principal_part_csv(
        prepare_principal_part_export(source, profile, approve_new_scope=True), tmp_path / "base.csv"
    )
    initial = invoke("inspect", str(source), "--profile", str(profile_path))
    claims = invoke("list", str(source), "--profile", str(profile_path))["claims"]
    candidates = [c for c in claims if c["kind"] == "translation"]
    assert len(candidates) == len(translations)
    assert all(c["status"] == "withheld" for c in candidates)
    batch = tmp_path / "decisions.json"
    evidence = [
        {
            "source": "Latin reviewer",
            "reference": sense,
            "version": "1",
            "text": "Checked form, role and active meaning",
        }
    ]
    # Accepting an unsupported placeholder or omitting the selected sense must fail closed.
    for item in [decision(candidates[0]), decision(candidates[0]) | {"value": translations[0], "evidence": evidence}]:
        batch.write_text(json.dumps([item]))
        assert (
            runner.invoke(
                app, ["claims", "record", str(source), "--profile", str(profile_path), "--batch", str(batch)]
            ).exit_code
            != 0
        )
    batch.write_text(
        json.dumps(
            [
                decision(c) | {"value": value, "evidence": evidence, "selected_sense": sense}
                for c, value in zip(candidates, translations, strict=True)
            ]
        )
    )
    invoke("record", str(source), "--profile", str(profile_path), "--batch", str(batch))
    # Even an accepted translation cannot adjudicate the fourth role.
    unconfirmed = invoke("inspect", str(source), "--profile", str(profile_path))
    assert [c["prompt"] for c in unconfirmed["cards"]] == [c["prompt"] for c in initial["cards"]]
    if len(translations) == 4:
        assert all(not c["eligible"] for c in unconfirmed["cards"] if c["semantic_key"].endswith(f":{fourth}"))
        batch.write_text(
            json.dumps([decision(next(c for c in claims if c["feature"] == fourth and c["kind"] == "label"))])
        )
        invoke("record", str(source), "--profile", str(profile_path), "--batch", str(batch))
    reviewed = invoke("inspect", str(source), "--profile", str(profile_path), "--check")
    assert [(c["identity"], c["slot"]) for c in initial["cards"]] == [
        (c["identity"], c["slot"]) for c in reviewed["cards"]
    ]
    for card in reviewed["cards"]:
        assert all(value not in card["prompt"] for value in translations if value not in meaning)
        if card["eligible"]:
            assert all(value in card["answer"] for value in translations)
    reported = invoke("list", str(source), "--profile", str(profile_path))["claims"]
    assert all(c["selected_sense"] == sense and c["language"] == "de" for c in reported if c["kind"] == "translation")
    output = tmp_path / "reviewed.csv"
    write_principal_part_csv(prepare_principal_part_export(source, profile), output)
    assert (
        invoke("inspect", str(source), "--profile", str(profile_path), "--artifact", str(output), "--check")[
            "diagnostics"
        ]
        == []
    )
    # Editing either the selected sense or translation without renewed review invalidates it.
    sidecar = Path(f"{source}.claims.json")
    saved = sidecar.read_text()
    for field, value in [("selected_sense", "different-sense"), ("value", "Ungeprüfte Übersetzung")]:
        data = json.loads(saved)
        entry = next(e for e in data["decisions"] if e.get("selected_sense") == sense)
        entry[field] = value
        sidecar.write_text(json.dumps(data))
        tampered = invoke("inspect", str(source), "--profile", str(profile_path))
        assert tampered["diagnostics"]
        assert all(
            line != translations[0] and not line.endswith(f" · {translations[0]}")
            for c in tampered["cards"]
            for line in c["answer"].splitlines()
        )
        assert all(value not in c["answer"] for c in tampered["cards"])
    sidecar.write_text(saved)
    current = next(c for c in reported if c["kind"] == "translation")
    batch.write_text(
        json.dumps(
            [
                decision(current)
                | {
                    "status": "withheld",
                    "value": "Nicht behaupten",
                    "evidence": evidence,
                    "selected_sense": sense,
                }
            ]
        )
    )
    invoke("record", str(source), "--profile", str(profile_path), "--batch", str(batch))
    withheld = invoke("inspect", str(source), "--profile", str(profile_path), "--check")
    assert all("Nicht behaupten" not in c["answer"] for c in withheld["cards"])
    assert all(
        line != translations[0] and not line.endswith(f" · {translations[0]}")
        for c in withheld["cards"]
        for line in c["answer"].splitlines()
    )
    drift = runner.invoke(
        app,
        [
            "claims",
            "inspect",
            str(source),
            "--profile",
            str(profile_path),
            "--artifact",
            str(output),
            "--check",
            "--json",
        ],
    )
    assert drift.exit_code != 0 and "drift" in drift.stdout
    sidecar.write_text(saved)
    # Meaning edits must not reuse a selected-sense decision, or alter note identities.
    source.write_text(source.read_text().replace(meaning, "anderer Sinn"))
    stale = invoke("inspect", str(source), "--profile", str(profile_path))
    assert stale["diagnostics"]
    assert all(translations[0] not in c["answer"] for c in stale["cards"])


def test_language_switch_stales_reviews_and_english_needs_its_own(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/goldens/v0-4-0/inputs/principal-parts/de-light-user")
    shutil.copytree(fixture, tmp_path, dirs_exist_ok=True)
    source, profile_path = tmp_path / "source.csv", tmp_path / "profile.json"
    sidecar = Path(f"{source}.claims.json")
    german_claims = invoke("list", str(source), "--profile", str(profile_path))["claims"]
    german_translations = [
        c["value"] for c in german_claims if c["kind"] == "translation" and c["status"] == "accepted"
    ]
    assert german_translations == ["ich liebe", "lieben", "ich habe geliebt", "um zu lieben"]
    german = invoke("inspect", str(source), "--profile", str(profile_path), "--check")
    # "lieben" is also the source gloss, which every language shows verbatim.
    reviewed_only = [text for text in german_translations if text != "lieben"]
    eligible = [card["semantic_key"] for card in german["cards"] if card["eligible"]]
    assert len(eligible) == 8

    def switch(tag: str, terminology: str = "user") -> None:
        profile = json.loads(profile_path.read_text())
        profile["language_tag"] = tag
        profile["morphology"]["terminology"]["language"] = terminology
        profile_path.write_text(json.dumps(profile))

    # The unchanged fingerprint contract binds language_tag, so every reviewed claim goes stale.
    switch("en")
    english = invoke("inspect", str(source), "--profile", str(profile_path))
    decisions = json.loads(sidecar.read_text())["decisions"]
    assert [d["code"] for d in english["diagnostics"]] == ["stale"] * len(decisions) == ["stale"] * 16
    assert [(c["identity"], c["semantic_key"], c["slot"]) for c in english["cards"]] == [
        (c["identity"], c["semantic_key"], c["slot"]) for c in german["cards"]
    ]
    assert not any(card["eligible"] for card in english["cards"] if card["semantic_key"].endswith(":supine"))
    claims = invoke("list", str(source), "--profile", str(profile_path))["claims"]
    assert all(c["status"] != "accepted" and c["review_status"] == "open" for c in claims)
    answered = [card for card in english["cards"] if card["answer"]]
    assert answered
    for card in answered:
        assert all(text not in card["answer"] for text in reviewed_only)
        assert "Stammformen" not in card["answer"] and "Compare principal parts" in card["answer"]

    # English translations appear only after their own accepted decisions.
    sidecar.rename(tmp_path / "german.claims.json")
    evidence = [{"source": "Latin reviewer", "reference": "love", "version": "1", "text": "Checked form"}]
    english_translations = ["I love", "to love", "I have loved", "to love (supine)"]
    values = iter(english_translations)
    batch = tmp_path / "english.json"
    batch.write_text(
        json.dumps(
            [
                decision(c)
                | (
                    {"value": next(values), "evidence": evidence, "selected_sense": "love"}
                    if c["kind"] == "translation"
                    else {}
                )
                for c in claims
                if c["kind"] in {"label", "translation"}
            ]
        )
    )
    invoke("record", str(source), "--profile", str(profile_path), "--batch", str(batch))
    assert all(
        c["language"] == "en"
        for c in invoke("list", str(source), "--profile", str(profile_path))["claims"]
        if c["kind"] == "translation"
    )
    reviewed = invoke("inspect", str(source), "--profile", str(profile_path), "--check")
    assert reviewed["diagnostics"] == []
    assert [card["semantic_key"] for card in reviewed["cards"] if card["eligible"]] == eligible
    assert all(
        all(text in card["answer"] for text in english_translations)
        and all(text not in card["answer"] for text in reviewed_only)
        for card in reviewed["cards"]
        if card["eligible"]
    )

    # Other tags render German but cannot assert a form translation.
    sidecar.unlink()
    switch("fr", "latin")
    fresh = invoke("list", str(source), "--profile", str(profile_path))["claims"]
    target = next(c for c in fresh if c["kind"] == "translation")
    batch.write_text(
        json.dumps([decision(target) | {"value": "j'aime", "evidence": evidence, "selected_sense": "love"}])
    )
    rejected = runner.invoke(
        app, ["claims", "record", str(source), "--profile", str(profile_path), "--batch", str(batch), "--json"]
    )
    assert rejected.exit_code != 0
    assert "German (de) or English (en)" in rejected.stdout
    assert not sidecar.exists()


def parsing_source(tmp_path: Path) -> Path:
    data = json.loads(Path("docs/examples/form-parsing-cases.json").read_text())
    for proposal in data["cases"][0]["proposals"]:
        proposal.pop("decision", None)
    source = tmp_path / "cases.json"
    source.write_text(json.dumps(data))
    return source


def invoke(*args: str) -> dict[str, Any]:
    result = runner.invoke(app, ["claims", *args, "--json"])
    assert result.exit_code == 0, result.output
    return cast(dict[str, Any], json.loads(result.stdout))


def decision(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": claim["fingerprint"],
        "status": "accepted",
        "reviewer": "Latin reviewer",
        "reason": "Checked context",
    }


@pytest.mark.parametrize("failure", ["invalid", "stale", "conflict"])
def test_atomic_batch_rejection_and_parsing_export(tmp_path: Path, failure: str) -> None:
    source = parsing_source(tmp_path)
    claims = invoke("list", str(source))["claims"]
    assert {c["feature"] for c in claims} == {"lemma", "case", "number", "gender"}
    assert all(c["status"] == "withheld" and c["evidence"] and c["identity"] for c in claims)
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([decision(claims[0])]))
    invoke("record", str(source), "--batch", str(batch))
    sidecar = Path(f"{source}.claims.json")
    before = sidecar.read_bytes()
    bad = decision(claims[2])
    if failure == "invalid":
        bad["reviewer"] = " "
    elif failure == "stale":
        bad["fingerprint"] = "0" * 64
    else:
        bad = decision(claims[0]) | {"status": "withheld"}
    batch.write_text(json.dumps([decision(claims[1]), bad]))
    rejected = runner.invoke(app, ["claims", "record", str(source), "--batch", str(batch), "--json"])
    assert rejected.exit_code != 0
    diagnostics = json.loads(rejected.stdout)["diagnostics"]
    assert any(d["item"] == 1 and d["code"] == failure for d in diagnostics)
    assert sidecar.read_bytes() == before
    batch.write_text(json.dumps([decision(c) for c in claims[1:3]]))
    invoke("record", str(source), "--batch", str(batch))
    output = tmp_path / "out.csv"
    exported = runner.invoke(app, ["form-parsing", "export", str(source), str(output), "--approve-fresh-import"])
    assert exported.exit_code == 0, exported.output
    assert b"nominativus" in output.read_bytes() and b"femininum" not in output.read_bytes()
    checked = invoke("inspect", str(source), "--artifact", str(output), "--check")
    assert checked["diagnostics"] == [] and checked["cards"][0]["eligible"]
    assert "Casus: nominativus" in checked["cards"][0]["answer"]
    output.write_bytes(output.read_bytes().replace(b"nominativus", b"genetivus"))
    tampered = runner.invoke(app, ["claims", "inspect", str(source), "--artifact", str(output), "--check", "--json"])
    assert tampered.exit_code != 0 and "drift" in tampered.stdout


def test_authored_claim_and_stale_context(tmp_path: Path) -> None:
    source = parsing_source(tmp_path)
    original = invoke("list", str(source))["claims"][1]
    authored = decision(original) | {
        "value": "nominativus pluralis",
        "evidence": [
            {"source": "Reviewer", "reference": "sentence", "version": "1", "text": "Subject agrees with portant"}
        ],
    }
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([authored]))
    invoke("record", str(source), "--batch", str(batch))
    current = invoke("list", str(source))["claims"][1]
    assert current["fingerprint"] != original["fingerprint"]
    assert current["value"] == "nominativus pluralis" and current["status"] == "accepted"
    data = json.loads(source.read_text())
    data["cases"][0]["context"] = "Rosam puellae dat."
    source.write_text(json.dumps(data))
    stale = invoke("inspect", str(source))
    assert stale["diagnostics"] and stale["cards"][0]["identity"] == original["identity"]
    assert "nominativus pluralis" not in stale["cards"][0]["answer"]


@pytest.mark.parametrize("explanation_text", ["BEFORE <script> AFTER", "BEFORE <datum> AFTER", "BEFORE AFTER"])
def test_principal_recipes_authored_values_and_profile_binding(tmp_path: Path, explanation_text: str) -> None:
    from latinitas_cards.preview_export import prepare_principal_part_export, write_principal_part_csv
    from latinitas_cards.profile import DeckProfile, SourceIdentityConfig

    source = tmp_path / "source.csv"
    source.write_text('ID,Lemma,Forms\nentry-1,dīcō,"dīcō, dīcere, dīxī, dictum"\n')
    profile = DeckProfile.default(
        note_type="Latin",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
        principal_part_roles=("present_1s", "present_infinitive", "perfect_1s", "supine"),
        separators=(",",),
        selected_recipes=("principal_part_completion", "principal_part_recognition"),
    )
    profile_path = tmp_path / "profile.json"
    profile.save(profile_path)
    write_principal_part_csv(
        prepare_principal_part_export(source, profile, approve_new_scope=True), tmp_path / "bootstrap.csv"
    )
    initial = invoke("inspect", str(source), "--profile", str(profile_path))
    claims = invoke("list", str(source), "--profile", str(profile_path))["claims"]
    label = next(c for c in claims if c["feature"] == "supine" and c["kind"] == "label")
    segmentation = next(c for c in claims if c["feature"] == "supine" and c["kind"] == "segmentation")
    explanation = next(c for c in claims if c["feature"] == "supine" and c["kind"] == "explanation")
    evidence = [{"source": "Latin reference", "reference": "dico", "version": "1", "text": "Reviewed supine formation"}]
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            [
                decision(label),
                decision(segmentation) | {"value": "dict-um", "evidence": evidence},
                decision(explanation) | {"value": explanation_text, "evidence": evidence},
            ]
        )
    )
    invoke("record", str(source), "--profile", str(profile_path), "--batch", str(batch))
    reviewed = invoke("inspect", str(source), "--profile", str(profile_path))
    cards = [c for c in reviewed["cards"] if c["semantic_key"].endswith(":supine")]
    assert len(cards) == 2 and all(c["eligible"] for c in cards)
    assert all("dict-um" in c["answer"] and explanation_text in c["answer"] for c in cards)
    assert all("dīcere" in c["answer"] for c in cards)
    assert [(c["identity"], c["slot"]) for c in initial["cards"]] == [
        (c["identity"], c["slot"]) for c in reviewed["cards"]
    ]
    output = tmp_path / "reviewed.csv"
    rerun = tmp_path / "rerun.csv"
    write_principal_part_csv(prepare_principal_part_export(source, profile), output)
    write_principal_part_csv(prepare_principal_part_export(source, profile), rerun)
    assert output.read_bytes() == rerun.read_bytes()
    assert html.escape(explanation_text) in output.read_text()
    checked = invoke("inspect", str(source), "--profile", str(profile_path), "--artifact", str(output), "--check")
    assert checked["diagnostics"] == []
    assert all(explanation_text in c["answer"] for c in checked["cards"] if c["semantic_key"].endswith(":supine"))
    standalone = invoke("inspect", str(output))
    assert all(explanation_text in c["answer"] for c in standalone["cards"] if c["semantic_key"].endswith(":supine"))
    text = runner.invoke(app, ["claims", "inspect", str(source), "--profile", str(profile_path)])
    assert text.exit_code == 0 and explanation_text in text.stdout
    changed = profile.model_copy(update={"morphology": profile.morphology.model_copy(update={"appearance": "dark"})})
    changed.save(profile_path)
    assert invoke("list", str(source), "--profile", str(profile_path))["diagnostics"] == []
    changed = profile.model_copy(update={"note_type": "Different linguistic source"})
    changed.save(profile_path)
    stale = runner.invoke(app, ["claims", "inspect", str(source), "--profile", str(profile_path), "--check", "--json"])
    assert stale.exit_code != 0 and "stale" in stale.stdout


@pytest.mark.parametrize("layout", ["supported", "javascript", "wrong-slot"])
def test_offline_package_contract(tmp_path: Path, layout: str) -> None:
    import csv
    import io
    import sqlite3
    import zipfile

    from latinitas_cards.form_parsing import PARSING_BACK, PARSING_FIELDS, PARSING_FRONT

    source = parsing_source(tmp_path)
    claims = invoke("list", str(source))["claims"]
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([decision(c) for c in claims[:3]]))
    invoke("record", str(source), "--batch", str(batch))
    output = tmp_path / "out.csv"
    assert (
        runner.invoke(app, ["form-parsing", "export", str(source), str(output), "--approve-fresh-import"]).exit_code
        == 0
    )
    rows = list(
        csv.reader(io.StringIO("\n".join(line for line in output.read_text().splitlines() if not line.startswith("#"))))
    )
    values = [
        *rows[0][:-1],
        "Personal annotation &lt;script&gt; AFTER &lt;datum&gt; END"
        "<script>ACTIVE_SCRIPT</script><style>ACTIVE_STYLE</style><!-- ACTIVE_COMMENT -->",
    ]
    model: dict[str, Any] = {
        "name": "Latinitas Contextual Form Parsing v1",
        "flds": [{"name": f} for f in PARSING_FIELDS],
        "tmpls": [{"ord": 0, "name": "Contextual Analysis", "qfmt": PARSING_FRONT, "afmt": PARSING_BACK}],
    }
    if layout == "javascript":
        model["tmpls"][0]["qfmt"] += "<script>runArbitraryTemplate()</script>"
    database = tmp_path / "collection.anki2"
    with sqlite3.connect(database) as con:
        con.executescript(
            "CREATE TABLE col(id INTEGER, models TEXT); "
            "CREATE TABLE notes(id INTEGER,mid INTEGER,flds TEXT,guid TEXT,tags TEXT); "
            "CREATE TABLE cards(nid INTEGER,ord INTEGER);"
        )
        con.execute("INSERT INTO col VALUES(1,?)", (json.dumps({"1": model}),))
        con.execute("INSERT INTO notes VALUES(1,1,?,'guid','')", ("\x1f".join(values),))
        con.execute("INSERT INTO cards VALUES(1,?)", (7 if layout == "wrong-slot" else 0,))
    package = tmp_path / "deck.colpkg"
    with zipfile.ZipFile(package, "w") as archive:
        archive.write(database, "collection.anki2")
    before = package.read_bytes()
    result = runner.invoke(app, ["claims", "inspect", str(source), "--artifact", str(package), "--check", "--json"])
    report = json.loads(result.stdout)
    assert package.read_bytes() == before
    if layout == "supported":
        assert result.exit_code == 0, result.stdout
        assert "Puellae rosas portant." in report["cards"][0]["prompt"]
        assert "Personal annotation <script> AFTER <datum> END" in report["cards"][0]["answer"]
        assert "ACTIVE_" not in report["cards"][0]["answer"]
    else:
        assert result.exit_code != 0
        assert report["diagnostics"]


def test_correct_previously_accepted_claim_and_inspect_export_alone(tmp_path: Path) -> None:
    source = parsing_source(tmp_path)
    original = invoke("list", str(source))["claims"][1]
    invoke(
        "record",
        str(source),
        "--fingerprint",
        original["fingerprint"],
        "--status",
        "accepted",
        "--reviewer",
        "Reviewer",
        "--reason",
        "Reviewed sentence",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            [{"source": "Reviewer", "reference": "sentence", "version": "2", "text": "Corrected the canonical value"}]
        )
    )
    invoke(
        "record",
        str(source),
        "--fingerprint",
        original["fingerprint"],
        "--status",
        "accepted",
        "--reviewer",
        "Reviewer",
        "--reason",
        "Reviewed correction",
        "--value",
        "nominativus pluralis",
        "--evidence",
        str(evidence),
    )
    claims = invoke("list", str(source))["claims"]
    assert claims[1]["value"] == "nominativus pluralis"
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([decision(claims[0]), decision(claims[2])]))
    invoke("record", str(source), "--batch", str(batch))
    output = tmp_path / "out.csv"
    assert (
        runner.invoke(app, ["form-parsing", "export", str(source), str(output), "--approve-fresh-import"]).exit_code
        == 0
    )
    inspected = invoke("inspect", str(output))
    assert "nominativus pluralis" in inspected["cards"][0]["answer"]
    assert inspected["cards"][0]["claims"] and inspected["diagnostics"]
    assert runner.invoke(app, ["claims", "inspect", str(output), "--check", "--json"]).exit_code != 0


def test_conflicting_alternative_acceptances_are_atomic(tmp_path: Path) -> None:
    source = parsing_source(tmp_path)
    data = json.loads(source.read_text())
    alternative = data["cases"][0]["proposals"][1] | {"value": "genetivus"}
    data["cases"][0]["proposals"].append(alternative)
    source.write_text(json.dumps(data))
    claims = invoke("list", str(source))["claims"]
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([decision(claims[0])]))
    invoke("record", str(source), "--batch", str(batch))
    before = Path(f"{source}.claims.json").read_bytes()
    batch.write_text(json.dumps([decision(claims[2]), decision(claims[1]), decision(claims[-1])]))
    result = runner.invoke(app, ["claims", "record", str(source), "--batch", str(batch), "--json"])
    assert result.exit_code != 0 and "conflict" in result.stdout
    assert Path(f"{source}.claims.json").read_bytes() == before


def test_author_missing_features_without_supplying_missing_forms(tmp_path: Path) -> None:
    source = parsing_source(tmp_path)
    data = json.loads(source.read_text())
    data["cases"][0]["proposals"] = []
    source.write_text(json.dumps(data))
    claims = invoke("list", str(source))["claims"]
    assert {c["feature"] for c in claims} == {"lemma", "case", "number", "gender"}
    assert all(c["value"] is None for c in claims)
    evidence = [{"source": "Reviewed sentence", "reference": "puellae", "version": "1", "text": "Subject of portant"}]
    values = {"lemma": "puella", "case": "nominativus", "number": "pluralis"}
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            [
                decision(c) | {"value": values[c["feature"]], "evidence": evidence}
                for c in claims
                if c["feature"] in values
            ]
        )
    )
    invoke("record", str(source), "--batch", str(batch))
    inspected = invoke("inspect", str(source))
    assert inspected["cards"][0]["eligible"] and "Casus: nominativus" in inspected["cards"][0]["answer"]


def test_duplicate_sidecar_bindings_fail_closed(tmp_path: Path) -> None:
    source = parsing_source(tmp_path)
    claim = invoke("list", str(source))["claims"][0]
    invoke(
        "record",
        str(source),
        "--fingerprint",
        claim["fingerprint"],
        "--status",
        "accepted",
        "--reviewer",
        "Reviewer",
        "--reason",
        "Checked",
    )
    path = Path(f"{source}.claims.json")
    sidecar = json.loads(path.read_text())
    sidecar["decisions"] *= 3
    path.write_text(json.dumps(sidecar))
    report = invoke("list", str(source))
    assert report["diagnostics"] and report["claims"][0]["status"] == "withheld"


def test_open_review_status_is_distinct_from_explicit_withholding(tmp_path: Path) -> None:
    source = parsing_source(tmp_path)
    claims = invoke("list", str(source))["claims"]
    assert all(c["review_status"] == "open" and c["status"] == "withheld" for c in claims)
    invoke(
        "record",
        str(source),
        "--fingerprint",
        claims[0]["fingerprint"],
        "--status",
        "withheld",
        "--reviewer",
        "Reviewer",
        "--reason",
        "Insufficient evidence",
    )
    updated = invoke("list", str(source))["claims"]
    assert updated[0]["review_status"] == "withheld" and updated[0]["reason"] == "Insufficient evidence"
    assert all(c["review_status"] == "open" for c in updated[1:])


def test_malformed_csv_returns_item_diagnostic_not_exception(tmp_path: Path) -> None:
    from latinitas_cards.form_parsing import PARSING_FIELDS, PARSING_NOTE_TYPE
    from latinitas_cards.preview_export import serialize_anki_csv

    artifact = tmp_path / "broken.csv"
    artifact.write_bytes(
        serialize_anki_csv(PARSING_NOTE_TYPE, "Latin", (*PARSING_FIELDS[:-1], "Tags"), ()) + b'"unterminated'
    )
    result = runner.invoke(app, ["claims", "inspect", str(artifact), "--check", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["diagnostics"][0]["code"] == "invalid"
    assert "CSV" in json.loads(result.stdout)["diagnostics"][0]["message"]


@pytest.mark.parametrize("mutation", ["fresh", "note-type", "deck"])
def test_check_uses_export_escaping_and_current_import_metadata(tmp_path: Path, mutation: str) -> None:
    source = parsing_source(tmp_path)
    data = json.loads(source.read_text())
    if mutation == "fresh":
        data["cases"][0]["context"] = 'Puellae & pueri dicunt: "salvē".'
    source.write_text(json.dumps(data))
    claims = invoke("list", str(source))["claims"]
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([decision(c) for c in claims[:3]]))
    invoke("record", str(source), "--batch", str(batch))
    output = tmp_path / "out.csv"
    assert (
        runner.invoke(app, ["form-parsing", "export", str(source), str(output), "--approve-fresh-import"]).exit_code
        == 0
    )
    if mutation == "note-type":
        output.write_text(
            output.read_text().replace("#notetype:Latinitas Contextual Form Parsing v1", "#notetype:Wrong model")
        )
    elif mutation == "deck":
        lines = output.read_text().splitlines(keepends=True)
        output.write_text("".join("#deck:Wrong deck\n" if line.startswith("#deck:") else line for line in lines))
    result = runner.invoke(app, ["claims", "inspect", str(source), "--artifact", str(output), "--check", "--json"])
    if mutation == "fresh":
        assert result.exit_code == 0, result.stdout
        assert (
            json.loads(result.stdout)["cards"][0]["prompt"]
            == 'puellae\nPuellae & pueri dicunt: "salvē".\nFormam resolve.'
        )
    else:
        assert result.exit_code == 1 and json.loads(result.stdout)["diagnostics"]
