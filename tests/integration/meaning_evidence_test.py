"""Public finite meaning evidence; disabled policy still denies all exports."""

import json
from pathlib import Path
from typing import Any

import pytest
from grammar_pipeline_test import bennett_phrase_fixture, call, fixture, rebind, write

from latinitas_cards.acceptance_records import canonical
from latinitas_cards.pipeline_consumers import Analysis, current_cards, load_run

CONTEXTS = {
    "preposition-noun": {
        "en": (
            "Suppose someone has set out (profectus est …). "
            "Translate only the fragment, using indefinite noun reference."
        ),
        "de": (
            "Nimm an, jemand ist aufgebrochen (profectus est …). "
            "Übersetze nur das Fragment und verwende eine unbestimmte Nominalgruppe."
        ),
    },
    "noun-genitive": {
        "en": "Use definite reference for both nouns.",
        "de": "Verwende für beide Substantive eine bestimmte Referenz.",
    },
    "adjective-noun": {
        "en": "Use indefinite reference for the noun.",
        "de": "Verwende für das Substantiv eine unbestimmte Referenz.",
    },
    "verb-noun": {
        "en": "Use definite reference for both nouns.",
        "de": "Verwende für beide Substantive eine bestimmte Referenz.",
    },
}


def meaning_fixture(root: Path, language: str, family: str = "form") -> tuple[Path, Path]:
    if family == "form":
        profile, objectives, _ = fixture(root, language)
    else:
        profile, objectives, _ = bennett_phrase_fixture(root, family)
    data = json.loads((root / "inputs.json").read_text())
    raw = json.loads(objectives.read_text())
    if family == "verb-noun":
        # Five source cells are not a four-position principal-part list.
        # Keep them all, with no claim of confirmed parser-assigned roles.
        for form in data["entries"][0]["forms"]:
            form["source_role"] = "other"
    if family == "form":
        data["entries"][0]["senses"][0]["key"] = "love"
        original = raw["forms"][0]
        raw["forms"] = [
            {
                **original,
                "recipe": "form_translation",
                "form": "present",
                "anchors": [],
                "sense": "love",
                "translations": ["I love" if language == "en" else "ich liebe"],
                "explanation": "The verb is present indicative active."
                if language == "en"
                else "Das Verb steht im Präsens Indikativ Aktiv.",
                "caveats": [
                    "No claim of exhaustive contextual interpretation is made."
                    if language == "en"
                    else "Eine vollständige kontextuelle Deutung wird nicht behauptet."
                ],
            }
        ]
    else:
        texts = {
            "preposition-noun": (
                "with a gate",
                "mit einem Tor",
                "Cum governs the ablative and expresses accompaniment here.",
                "Cum regiert den Ablativ und drückt hier Begleitung aus.",
                {"object": ("accompaniment: gate", "Begleitung: Tor")},
            ),
            "noun-genitive": (
                "the gate of the garden",
                "das Tor des Gartens",
                "The genitive relates the gate to the garden without specifying the kind of relation.",
                "Der Genitiv setzt das Tor zum Garten in Beziehung, ohne die Art der Beziehung festzulegen.",
                {
                    "head": ("head: gate", "Bezugswort: Tor"),
                    "dependent": ("genitive relation: garden", "Genitivbeziehung: Garten"),
                },
            ),
            "adjective-noun": (
                "a good garden",
                "ein guter Garten",
                "Bonus agrees with hortus in case, number and gender.",
                "Bonus stimmt mit hortus in Kasus, Numerus und Genus überein.",
                {"head": ("head: garden", "Bezugswort: Garten"), "modifier": ("quality: good", "Eigenschaft: gut")},
            ),
            "verb-noun": (
                "the boy loves the gate",
                "der Junge liebt das Tor",
                "Puer is the subject and portam is the object of amat.",
                "Puer ist das Subjekt und portam das Objekt von amat.",
                {
                    "subject": ("lover: boy", "Liebender: Junge"),
                    "object": ("loved entity: gate", "geliebtes Objekt: Tor"),
                    "verb": ("present active love", "lieben im Präsens Aktiv"),
                },
            ),
        }[family]
        index = 0 if language == "en" else 1
        phrase = raw["phrases"][0]
        phrase["reference"]["evidence"].append(
            {
                "source": "constructed-exercise-context",
                "reference": "development/" + family,
                "version": "1",
                "text": CONTEXTS[family][language],
            }
        )
        phrase["reference"]["language"] = language
        reading = phrase["reference"]["readings"][0]
        phrase["reference"]["readings"] = [
            {
                **reading,
                "translations": [texts[index]],
                "explanation": texts[index + 2],
                "contributions": {k: v[index] for k, v in texts[4].items()},
            }
        ]
        if language == "de":
            replacements = {"gate": "Tor", "garden": "Garten", "boy": "Junge", "good": "gut", "love": "lieben"}
            deck = root / "deck.csv"
            text = deck.read_text()
            for entry in data["entries"]:
                for sense in entry["senses"]:
                    old = sense["meaning"]
                    if old in replacements:
                        sense["meaning"] = replacements[old]
                        text = text.replace("," + old + "\n", "," + replacements[old] + "\n")
            deck.write_text(text)
            deck_profile = json.loads((root / "deck-profile.json").read_text())
            deck_profile["language_tag"] = "de"
            write(root / "deck-profile.json", deck_profile)
    settings = json.loads(profile.read_text())
    settings["language"] = language
    if family == "form":
        settings["preset"] = None
        settings["recipe_overrides"] = {"include": [], "exclude": ["ppp_translation", "phrase_translation"]}
        settings["recipes"] = ["form_translation"]
    settings["selection"]["entries"] = ["v"] if family == "form" else []
    write(profile, settings)
    rebind(profile, data)
    write(objectives, raw)
    return profile, objectives


@pytest.mark.parametrize("language", ["en", "de"])
@pytest.mark.parametrize("family", ["form", "preposition-noun", "noun-genitive", "adjective-noun", "verb-noun"])
def test_complete_meaning_analysis_stays_disabled(tmp_path: Path, language: str, family: str) -> None:
    profile, objectives = meaning_fixture(tmp_path, language, family)
    output = tmp_path / "workflow"
    result = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    run = load_run(root)
    cards, _ = current_cards(profile, run.profile, run.objectives, root)
    card = (
        next(c for c in cards if c.objective.object_key == "production")
        if family == "form"
        else next(c for c in cards if c.recipe == "phrase_translation")
    )
    analysis = Analysis.read_json(card.analysis_path.read_bytes())
    assert all(c.result == "pass" for c in analysis.checks), [(c.id, c.result, c.objections) for c in analysis.checks]
    assert result["file_export"]["exported"] == 0


def captured_answer(body: dict[str, Any], paraphrase: bool = False) -> bytes:
    """Controlled interpreter of captured requests, not renderer/reference output."""
    from grammar_pipeline_test import EVIDENCE

    task = json.loads(body["messages"][1]["content"])
    objective = json.loads(task["objective"])
    language = task["binding"]["language"]
    supplied = {
        e["reference"].rsplit("/", 1)[-1]: json.loads(e["text"])
        for e in task["source_evidence"]
        if e["source"] in ("unverified-form-entry", "unverified-phrase-entry")
    }
    values: dict[str, str] = {}
    senses = {}
    if objective["recipe"] == "form_translation":
        entry = next(iter(supplied.values()))["entry"]
        form = next(f for f in entry["forms"] if f["key"] == objective["selected_form"])
        sense = next(s for s in entry["senses"] if s["key"] == objective["selected_sense"])
        # This loopback fixture qualifies only the captured amō first-person cell.
        assert form["text"] == "amō" and sense["key"] == "love"
        translation = "I love" if language == "en" else "ich liebe"
        explanation = (
            ("This is an active present indicative form." if paraphrase else "The verb is present indicative active.")
            if language == "en"
            else (
                "Dies ist eine aktive Präsensform im Indikativ."
                if paraphrase
                else "Das Verb steht im Präsens Indikativ Aktiv."
            )
        )
        caveat = (
            "No claim of exhaustive contextual interpretation is made."
            if language == "en"
            else "Eine vollständige kontextuelle Deutung wird nicht behauptet."
        )
        values["source"] = canonical(dict(entry=entry, sense=sense)).decode()
        values["target"] = canonical(
            dict(
                surface=form["text"], features=form["features"], source_role=form["source_role"], anchors=[], rule=None
            )
        ).decode()
        values.update(translation=translation, explanation=explanation)
        for a in objective["assertions"]:
            if a["id"].startswith("caveat-"):
                values[a["id"]] = caveat
        prompt = form["text"] + ("\nTranslate." if language == "en" else "\nÜbersetze.")
        answer = translation + (
            "\nReference guidance; self-assessment."
            if language == "en"
            else "\nReferenz als Orientierung; Selbstbeurteilung."
        )
        answer += "\n" + explanation + "\n" + caveat
        selected_sense = sense["key"]
    else:
        pattern = next(
            json.loads(e["text"]) for e in task["source_evidence"] if e["source"] == "unverified-phrase-pattern"
        )
        family = objective["family"]
        components = {}
        for b in objective["bindings"]:
            item = supplied[b["entry"]]
            entry = item["entry"]
            sense = next(s for s in entry["senses"] if s["key"] == b["sense"])
            components[b["role"]] = dict(
                identity=item["identity"],
                entry=entry,
                sense=sense,
                form=next(f for f in entry["forms"] if f["key"] == b["form"]),
            )
            senses[b["role"]] = sense["key"]
        # Known literal composition from the request's Latin roles and senses.
        assert {r: c["sense"]["key"] for r, c in components.items()} == {
            "preposition-noun": {"object": "gate"},
            "noun-genitive": {"head": "gate", "dependent": "garden"},
            "adjective-noun": {"head": "garden", "modifier": "good"},
            "verb-noun": {"subject": "boy", "object": "gate", "verb": "love"},
        }[family]
        translations = {
            "preposition-noun": {
                "en": ("with a gate", "together with a gate"),
                "de": ("mit einem Tor", "zusammen mit einem Tor"),
            },
            "noun-genitive": {
                "en": ("the gate of the garden", "the garden's gate"),
                "de": ("das Tor des Gartens", "des Gartens Tor"),
            },
            "adjective-noun": {
                "en": ("a good garden", "a garden that is good"),
                "de": ("ein guter Garten", "ein Garten, der gut ist"),
            },
            "verb-noun": {
                "en": ("the boy loves the gate", "the gate is loved by the boy"),
                "de": ("der Junge liebt das Tor", "das Tor wird vom Jungen geliebt"),
            },
        }
        translation = translations[family][language][int(paraphrase)]
        explanations = {
            "preposition-noun": (
                "Cum governs the ablative and expresses accompaniment here.",
                "Cum regiert den Ablativ und drückt hier Begleitung aus.",
            ),
            "noun-genitive": (
                "The genitive relates the gate to the garden without specifying the kind of relation.",
                "Der Genitiv setzt das Tor zum Garten in Beziehung, ohne die Art der Beziehung festzulegen.",
            ),
            "adjective-noun": (
                "Bonus agrees with hortus in case, number and gender.",
                "Bonus stimmt mit hortus in Kasus, Numerus und Genus überein.",
            ),
            "verb-noun": (
                "Puer is the subject and portam is the object of amat.",
                "Puer ist das Subjekt und portam das Objekt von amat.",
            ),
        }
        contributions = {
            "preposition-noun": {"object": ("accompaniment: gate", "Begleitung: Tor")},
            "noun-genitive": {
                "head": ("head: gate", "Bezugswort: Tor"),
                "dependent": ("genitive relation: garden", "Genitivbeziehung: Garten"),
            },
            "adjective-noun": {
                "head": ("head: garden", "Bezugswort: Garten"),
                "modifier": ("quality: good", "Eigenschaft: gut"),
            },
            "verb-noun": {
                "subject": ("lover: boy", "Liebender: Junge"),
                "object": ("loved entity: gate", "geliebtes Objekt: Tor"),
                "verb": ("present active love", "lieben im Präsens Aktiv"),
            },
        }
        index = int(language == "de")
        explanation = explanations[family][index]
        roles = {r: v[index] for r, v in contributions[family].items()}
        values.update(
            {
                "construction": canonical(dict(pattern=pattern, components=components)).decode(),
                "translation-0": translation,
                "explanation-0": explanation,
                "roles-0": canonical(roles).decode(),
            }
        )
        order = {
            "preposition-noun": ("object",),
            "noun-genitive": ("head", "dependent"),
            "adjective-noun": ("head", "modifier"),
            "verb-noun": ("subject", "object", "verb"),
        }[family]
        text = " ".join([*pattern["function_words"], *(components[r]["form"]["text"] for r in order)])
        prompt = text + (
            "\nTranslate, then compare with the reference."
            if language == "en"
            else "\nÜbersetze und vergleiche anschließend mit der Referenz."
        )
        for context in objective["contexts"]:
            prompt += "\n" + context
        count = sum(a["kind"] == "translation" for a in objective["assertions"])
        answer = "\n".join(
            ("Reading " if language == "en" else "Lesart ") + str(i + 1) + ": " + translation + "\n" + explanation
            for i in range(count)
        )
        for i in range(1, count):
            for name in ("translation", "explanation", "roles"):
                values[f"{name}-{i}"] = values[f"{name}-0"]
        answer += (
            "\nSelf-assessment: compare the meaning, not the exact wording."
            if language == "en"
            else "\nSelbsteinschätzung: Vergleiche den Sinn, nicht den genauen Wortlaut."
        )
        selected_sense = canonical(senses).decode()
    values["rendered"] = canonical(dict(prompt=prompt, answer=answer)).decode()
    assertions = [
        dict(
            id=a["id"],
            kind=a["kind"],
            value=values[a["id"]],
            selected_sense=selected_sense if a["kind"] == "translation" else None,
            alternatives=[],
            evidence=[EVIDENCE],
        )
        for a in objective["assertions"]
    ]
    response = dict(
        format="latinitas-model-answer",
        version=1,
        task_sha256=task["task_sha256"],
        assertions=assertions,
        alternatives=[],
        objections=[],
    )
    return canonical({"choices": [{"finish_reason": "stop", "message": {"content": canonical(response).decode()}}]})


@pytest.mark.parametrize("language", ["en", "de"])
@pytest.mark.parametrize("family", ["form", "preposition-noun", "noun-genitive", "adjective-noun", "verb-noun"])
def test_complete_public_comparison_and_saved_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str, family: str
) -> None:
    import grammar_pipeline_test as fixtures
    from grammar_pipeline_test import configure_pipeline_http, synthetic_pipeline_http

    from latinitas_cards.pipeline_orchestration import read_source_comparison

    invocations: list[dict[str, Any]] = []

    def respond(body: dict[str, Any]) -> bytes:
        result = captured_answer(body, paraphrase=bool(invocations))
        invocations.append(body)
        return result

    monkeypatch.setattr(fixtures, "provider_answer", respond)
    profile, objectives = meaning_fixture(tmp_path, language, family)
    # Avoid unrelated automatic infinitive objectives in this loopback journey.
    if family == "form":
        data = json.loads((tmp_path / "inputs.json").read_text())
        data["entries"][0]["forms"] = data["entries"][0]["forms"][:1]
        rebind(profile, data)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"] if family == "form" else []
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        run = load_run(root)
        card = current_cards(profile, run.profile, run.objectives, root)[0][0]
        proof = read_source_comparison(root, card)
        assert proof is not None and proof["relation"] == "equivalent", proof
        checks, _ = card.claims.decisions[0].reader.model_checks()
        assert all(c.result == "pass" for c in checks), checks
        assert {c.method for c in checks} == {"semantic_comparison", "blind_model"}
        assert report["file_export"]["exported"] == 0
        assert len(sent) == 2 and sent[0]["messages"][1]["content"] == sent[1]["messages"][1]["content"]
        task = json.loads(sent[1]["messages"][1]["content"])
        assert task["reference_constraints"] == []
        if family != "form":
            assert json.loads(task["objective"])["contexts"] == [CONTEXTS[family][language]]
            assert [e["text"] for e in task["source_evidence"] if e["source"] == "unverified-phrase-context"] == [
                CONTEXTS[family][language]
            ]
        assert "independently_grounded" not in sent[1]["messages"][1]["content"]
    before = {p: p.read_bytes() for p in (root / "stages").glob("*.json")}
    assert call("resume", "--run", str(root))["file_export"]["exported"] == 0
    assert len(sent) == 2 and all(p.read_bytes() == data for p, data in before.items())
    # Only after successful unchanged reuse, exercise custody/dependency denial.
    if family == "form":
        response = next(root.rglob("*.response.json"))
        response.write_bytes(response.read_bytes() + b" ")
        with pytest.raises(ValueError):
            card.claims.decisions[0].reader.model_checks()
    elif family == "preposition-noun":
        from latinitas_cards import latin_library

        changed = tmp_path / "reference.html"
        changed.write_bytes(latin_library.BENNETT_PATH.read_bytes() + b"\n")
        monkeypatch.setattr(latin_library, "BENNETT_PATH", changed)
        with pytest.raises(ValueError):
            card.claims.decisions[0].reader.model_checks()
    elif family in ("noun-genitive", "adjective-noun"):
        import shutil

        from latinitas_cards import acceptance_policy

        package = Path(acceptance_policy.__file__).parent
        copied = tmp_path / "installed-package"
        shutil.copytree(package, copied, ignore=shutil.ignore_patterns("__pycache__"))
        monkeypatch.setattr(acceptance_policy, "__file__", str(copied / "acceptance_policy.py"))
        dependency = copied / ("meaning_language.py" if family == "noun-genitive" else "meaning_evidence.py")
        dependency.write_bytes(dependency.read_bytes() + b"\n")
        with pytest.raises(ValueError):
            card.claims.decisions[0].reader.model_checks()
    else:
        from latinitas_cards.pipeline_orchestration import read_acceptance_model_checks

        changed_assertions = tuple(
            a.model_copy(update={"value": a.value + " false clause"}) if a.id == "rendered" else a
            for a in card.claims.assertions
        )
        with pytest.raises(ValueError, match="complete current card"):
            read_acceptance_model_checks(root, card.binding, card.versions, changed_assertions)


@pytest.mark.parametrize(
    "language,damage,side",
    [
        ("en", "translation", "both"),
        ("de", "translation", "both"),
        ("en", "explanation", "proposal"),
        ("de", "explanation", "counter"),
        ("en", "roles", "counter"),
        ("de", "rendered-extra", "proposal"),
        ("en", "person", "both"),
        ("de", "voice", "both"),
        ("en", "missing-sibling", "counter"),
        ("de", "omitted-reading", "both"),
        ("en", "sense", "both"),
        ("de", "source-sibling", "both"),
    ],
)
def test_wrong_complete_answers_do_not_self_corroborate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str, damage: str, side: str
) -> None:
    import grammar_pipeline_test as fixtures
    from grammar_pipeline_test import configure_pipeline_http, synthetic_pipeline_http

    from latinitas_cards.pipeline_orchestration import read_source_comparison

    profile, objectives = meaning_fixture(tmp_path, language, "verb-noun")
    if damage == "omitted-reading":
        raw = json.loads(objectives.read_text())
        readings = raw["phrases"][0]["reference"]["readings"]
        # Retain a separately supplied equivalent reading as a material sibling.
        second = json.loads(json.dumps(readings[0]))
        second["translations"] = ["das Tor wird vom Jungen geliebt"]
        readings.append(second)
        write(objectives, raw)
    calls: list[dict[str, Any]] = []

    def respond(body: dict[str, Any]) -> bytes:
        wrapper = json.loads(captured_answer(body, paraphrase=bool(calls)))
        current_side = "proposal" if not calls else "counter"
        calls.append(body)
        response = json.loads(wrapper["choices"][0]["message"]["content"])
        if side in (current_side, "both"):
            assertions = {a["id"]: a for a in response["assertions"]}
            if damage == "translation":
                assertions["translation-0"]["value"] = (
                    "the gate loves the boy" if language == "en" else "das Tor liebt den Jungen"
                )
            elif damage == "explanation":
                assertions["explanation-0"]["value"] += " This is true because all Latin nouns are masculine."
            elif damage == "roles":
                assertions["roles-0"]["value"] = canonical(
                    {"subject": "loved entity: boy", "object": "lover: gate", "verb": "present active love"}
                ).decode()
            elif damage in ("rendered-extra", "omitted-reading"):
                rendered = json.loads(assertions["rendered"]["value"])
                if damage == "rendered-extra":
                    rendered["answer"] += "\nThis phrase is always natural Latin."
                else:
                    lines = rendered["answer"].split("\n")
                    rendered["answer"] = "\n".join(lines[:2] + lines[4:])
                assertions["rendered"]["value"] = canonical(rendered).decode()
            elif damage in ("person", "voice", "source-sibling"):
                construction = json.loads(assertions["construction"]["value"])
                verb = construction["components"]["verb"]
                if damage == "source-sibling":
                    verb["entry"]["forms"][0]["features"]["number"] = "plural"
                else:
                    verb["form"]["features"][damage] = "1" if damage == "person" else "passive"
                assertions["construction"]["value"] = canonical(construction).decode()
            elif damage == "sense":
                assertions["translation-0"]["selected_sense"] = canonical(
                    {"subject": "boy", "object": "love", "verb": "gate"}
                ).decode()
            else:
                response["assertions"] = [a for a in response["assertions"] if a["id"] != "roles-0"]
        wrapper["choices"][0]["message"]["content"] = canonical(response).decode()
        return canonical(wrapper)

    monkeypatch.setattr(fixtures, "provider_answer", respond)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = []
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        run = load_run(root)
        card = current_cards(profile, run.profile, run.objectives, root)[0][0]
        proof = read_source_comparison(root, card)
        assert proof is not None and proof["relation"] != "equivalent", proof
        if damage == "translation":
            comparison = next(
                c for c in json.loads(canonical(proof))["comparisons"] if c["assertion_id"] == "translation-0"
            )
            assert comparison["relation"] == "contradiction"
            assert comparison["check"]["objections"] == ["bounded_meaning_contradiction"]
        checks, _ = card.claims.decisions[0].reader.model_checks()
        assert any(c.result != "pass" for c in checks)
        assert report["file_export"]["exported"] == 0 and len(sent) == 2


@pytest.mark.parametrize("language", ["en", "de"])
def test_all_six_present_cells_have_bound_meaning_evidence(tmp_path: Path, language: str) -> None:
    profile, objectives = meaning_fixture(tmp_path, language)
    data = json.loads((tmp_path / "inputs.json").read_text())
    entry = data["entries"][0]
    cells = (
        ("amō", "1", "singular", "I love", "ich liebe"),
        ("amās", "2", "singular", "you love", "du liebst"),
        ("amat", "3", "singular", "he/she/it loves", "er/sie/es liebt"),
        ("amāmus", "1", "plural", "we love", "wir lieben"),
        ("amātis", "2", "plural", "you (plural) love", "ihr liebt"),
        ("amant", "3", "plural", "they love", "sie lieben"),
    )
    for form in entry["forms"]:
        form["source_role"] = "other"
    prototype = entry["forms"][0]
    for surface, person, number, _, _ in cells[1:]:
        entry["forms"].append(
            {
                **prototype,
                "key": "cell-" + person + "-" + number,
                "text": surface,
                "features": {**prototype["features"], "person": person, "number": number},
            }
        )
    deck = tmp_path / "deck.csv"
    deck.write_text(
        deck.read_text().replace("amāvī — amātus", "amāvī — amātus — " + " — ".join(c[0] for c in cells[1:]))
    )
    rebind(profile, data)
    output = tmp_path / "workflow"
    result = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    run = load_run(root)
    cards, _ = current_cards(profile, run.profile, run.objectives, root)
    observed = {}
    for card in cards:
        claims = {a.id: a for a in card.claims.assertions}
        surface = json.loads(claims["target"].value)["surface"]
        if surface == "amāre":
            assert any(c.result != "pass" for c in Analysis.read_json(card.analysis_path.read_bytes()).checks)
            continue
        assert all(c.result == "pass" for c in Analysis.read_json(card.analysis_path.read_bytes()).checks)
        observed[surface] = claims["translation"].value
    assert observed == {c[0]: c[3 if language == "en" else 4] for c in cells}
    assert result["file_export"]["exported"] == 0


@pytest.mark.parametrize("language", ["en", "de"])
@pytest.mark.parametrize("damage", ["compatible", "context", "overlapping-sense", "source-sibling", "caveat"])
def test_bounded_meaning_does_not_qualify_unproved_claims(tmp_path: Path, language: str, damage: str) -> None:
    profile, objectives = meaning_fixture(tmp_path, language)
    raw = json.loads(objectives.read_text())
    objective = raw["forms"][0]
    data = json.loads((tmp_path / "inputs.json").read_text())
    entry = data["entries"][0]
    if damage == "compatible":
        objective["translations"] = ["I feel love" if language == "en" else "ich empfinde Liebe"]
    elif damage == "context":
        objective["context"] = "Additional context requires its own interpretation."
    elif damage == "overlapping-sense":
        entry["senses"][0]["key"] = objective["sense"] = "affection"
    elif damage == "source-sibling":
        sibling = entry["forms"][1]
        sibling["source_role"] = "other"
        sibling["features"]["number"] = "plural"
    else:
        objective["caveats"].append("This wording is always natural Latin.")
    rebind(profile, data)
    write(objectives, raw)
    output = tmp_path / "workflow"
    result = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    run = load_run(root)
    cards, _ = current_cards(profile, run.profile, run.objectives, root)
    card = next(c for c in cards if c.objective.object_key == "production")
    checks = {c.id: c for c in Analysis.read_json(card.analysis_path.read_bytes()).checks}
    if damage == "context":
        assert checks["target"].result != "pass"
    else:
        assert checks["target"].result == "pass"
    assert checks["rendered"].result != "pass"
    assert result["file_export"]["exported"] == 0


@pytest.mark.parametrize("language", ["en", "de"])
@pytest.mark.parametrize("family", ["preposition-noun", "noun-genitive", "adjective-noun", "verb-noun"])
def test_phrase_context_is_displayed_and_required(tmp_path: Path, language: str, family: str) -> None:
    profile, objectives = meaning_fixture(tmp_path, language, family)
    for mode in ("bound", "missing", "changed"):
        raw = json.loads(objectives.read_text())
        evidence = raw["phrases"][0]["reference"]["evidence"]
        if mode == "missing":
            evidence[:] = [e for e in evidence if e["source"] != "constructed-exercise-context"]
        elif mode == "changed":
            evidence.append(
                dict(
                    source="constructed-exercise-context",
                    reference="development/changed-context",
                    version="1",
                    text="A different unverified setting.",
                )
            )
        write(objectives, raw)
        output = tmp_path / mode
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        run = load_run(root)
        card = current_cards(profile, run.profile, run.objectives, root)[0][0]
        checks = {c.id: c for c in Analysis.read_json(card.analysis_path.read_bytes()).checks}
        assert checks["construction"].result == "pass"
        if mode == "bound":
            assert card.prompt.endswith("\n" + CONTEXTS[family][language])
            assert all(c.result == "pass" for c in checks.values())
        else:
            assert checks["rendered"].result != "pass"
        assert report["file_export"]["exported"] == 0


@pytest.mark.parametrize("language", ["en", "de"])
@pytest.mark.parametrize("case", ["compatible", "conflict", "ambiguous", "unknown"])
def test_finite_meaning_diagnostics_distinguish_unproved_claims(tmp_path: Path, language: str, case: str) -> None:
    profile, objectives = meaning_fixture(tmp_path, language)
    raw = json.loads(objectives.read_text())
    primary = "I love" if language == "en" else "ich liebe"
    other = "we love" if language == "en" else "wir lieben"
    translations = {
        "compatible": ["I feel love" if language == "en" else "ich empfinde Liebe"],
        "conflict": [other],
        "ambiguous": [primary, other],
        "unknown": ["An uninterpreted utterance."],
    }
    raw["forms"][0]["translations"] = translations[case]
    write(objectives, raw)
    output = tmp_path / "workflow"
    report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
    root = output / "run"
    run = load_run(root)
    card = next(
        c
        for c in current_cards(profile, run.profile, run.objectives, root)[0]
        if c.objective.object_key == "production"
    )
    checks = {c.id: c for c in Analysis.read_json(card.analysis_path.read_bytes()).checks}
    assert checks["target"].result == "pass"
    expected = {
        "compatible": "compatible_alternatives",
        "conflict": "contradiction",
        "ambiguous": "ambiguous",
        "unknown": "unresolved",
    }[case]
    assert checks["translation"].objections == ("bounded_meaning_" + expected,)
    assert checks["translation"].result != "pass" and checks["rendered"].result != "pass"
    assert report["file_export"]["exported"] == 0


@pytest.mark.parametrize("language", ["en", "de"])
@pytest.mark.parametrize("case", ["compatible_alternatives", "contradiction", "ambiguous", "unresolved"])
def test_public_comparison_distinguishes_meaning_with_correct_source_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str, case: str
) -> None:
    import grammar_pipeline_test as fixtures
    from grammar_pipeline_test import configure_pipeline_http, synthetic_pipeline_http

    from latinitas_cards.pipeline_orchestration import read_source_comparison

    def respond(body: dict[str, Any]) -> bytes:
        wrapper = json.loads(captured_answer(body))
        response = json.loads(wrapper["choices"][0]["message"]["content"])
        assertions = {a["id"]: a for a in response["assertions"]}
        primary = "I love" if language == "en" else "ich liebe"
        other = "we love" if language == "en" else "wir lieben"
        values = {
            "compatible_alternatives": "I feel love" if language == "en" else "ich empfinde Liebe",
            "contradiction": other,
            "ambiguous": primary + " / " + other,
            "unresolved": "An uninterpreted utterance.",
        }
        assertions["translation"]["value"] = values[case]
        assertions["translation"]["alternatives"] = [other] if case == "ambiguous" else []
        wrapper["choices"][0]["message"]["content"] = canonical(response).decode()
        return canonical(wrapper)

    monkeypatch.setattr(fixtures, "provider_answer", respond)
    profile, objectives = meaning_fixture(tmp_path, language)
    data = json.loads((tmp_path / "inputs.json").read_text())
    data["entries"][0]["forms"] = data["entries"][0]["forms"][:1]
    rebind(profile, data)
    with synthetic_pipeline_http() as (url, sent, _):
        configure_pipeline_http(profile, url)
        settings = json.loads(profile.read_text())
        settings["selection"]["entries"] = ["v"]
        write(profile, settings)
        output = tmp_path / "workflow"
        report = call("run", "--profile", str(profile), "--objectives", str(objectives), "--output", str(output))
        root = output / "run"
        run = load_run(root)
        card = current_cards(profile, run.profile, run.objectives, root)[0][0]
        proof = read_source_comparison(root, card)
        assert proof is not None and proof["relation"] != "equivalent", proof
        comparisons = {c["assertion_id"]: c for c in json.loads(canonical(proof))["comparisons"]}
        assert comparisons["source"]["relation"] == comparisons["target"]["relation"] == "equivalent"
        translation = comparisons["translation"]
        assert translation["relation"] == ("unresolved" if case == "ambiguous" else case)
        assert "bounded_meaning_" + case in translation["check"]["objections"]
        assert report["file_export"]["exported"] == 0 and len(sent) == 2
    assert read_source_comparison(root, card) == proof
