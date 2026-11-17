"""Source-bound analysis is inspectable evidence, never export authority."""

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from latinitas_cards.acceptance import load_acceptance, load_stage, reusable_stage, save_acceptance, stage_artifact
from latinitas_cards.acceptance_policy import BOOTSTRAP
from latinitas_cards.acceptance_records import Assertion, Versions, digest
from latinitas_cards.claim_review import Evidence
from latinitas_cards.deck_evidence import DeckAnalysis, DeckTask, SourceCell, analyze_construction, analyze_deck
from latinitas_cards.latin_library import LibraryReading, LocalStanza, ProcessingUnavailable
from latinitas_cards.phrase_inputs import POS, Features, Pattern, Role, Sense
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.sources import CanonicalSourceRecord, SourceProvenance

PROFILE = DeckProfile.default(
    note_type="Vocabulary",
    lexical_entry_field="Lemma",
    principal_parts_field="Parts",
    meaning_field="Meaning",
    source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
)
VERSIONS = Versions(
    producer="test/1",
    analyzer="synthetic/1",
    proposal_prompt=None,
    counter_prompt=None,
    checker="deck-evidence/1",
    comparison="none/1",
    policy="v050-disabled",
    policy_sha256=digest(BOOTSTRAP),
    proposal_model=None,
    counter_model=None,
)
ABL = Features(case="ablative", number="singular", gender="masculine")
DAT = Features(case="dative", number="singular", gender="masculine")


class Library:
    version = "synthetic/1"
    ambiguity_complete = True
    readings: tuple[LibraryReading, ...] = (
        LibraryReading(lemma="amicus", pos="noun", features=ABL),
        LibraryReading(lemma="amicus", pos="noun", features=DAT),
    )

    def analyze(self, text: str) -> tuple[LibraryReading, ...]:
        assert text == "amicō"
        return self.readings

    def senses(self, lemma: str, language: str) -> tuple[Sense, ...]:
        return (
            Sense(
                key="friend",
                meaning="Freund",
                evidence=(Evidence("synthetic-lexicon", "amicus.friend", "1", "Freund"),),
            ),
        )


def source(**fields: str) -> CanonicalSourceRecord:
    return CanonicalSourceRecord(
        "csv",
        "Vocabulary",
        {"ID": "17", "Lemma": "amicus", "Parts": "amicō", "Meaning": "Freund", **fields},
        SourceProvenance(Path("deck.csv"), "row 2"),
        "17",
    )


def task(**changes: object) -> DeckTask:
    return DeckTask.model_validate(
        {
            "form": SourceCell(field="Parts", start=0, end=5),
            "meaning": SourceCell(field="Meaning", start=0, end=6),
            "pos": "noun",
            "features": ABL,
            "source_role": "other",
            "selected_sense": "friend",
            "preposition": "cum",
            "extra_assertions": (),
            **changes,
        }
    )


def run(root: Path, request: DeckTask | None = None, library: Library | None = None) -> DeckAnalysis:
    return analyze_deck(
        root,
        record=source(),
        profile=PROFILE,
        source_scope="deck",
        task=request or task(),
        versions=VERSIONS,
        library=library or Library(),
    )


def test_government_disambiguates_without_blanket_source_authority(tmp_path: Path) -> None:
    result = run(tmp_path)
    assert result.state == "corroborated"
    assert result.checks[0].result == "pass"
    assert "dative" in " ".join(result.checks[0].alternatives)
    assert result.decision.state == "withheld"
    assert result.decision.reason == "policy_not_enabled"
    report = stage_artifact(tmp_path, result.stage)
    assert isinstance(report, dict)
    assert report["scope"] == "mapped morphology and lexical sense plausibility; no translation entailment"
    assert report["library_version"] == "synthetic/1"
    assert report["source"]["meaning"] == "Freund"
    assert report["checks"][0]["result"] == "pass"


@pytest.mark.parametrize(
    ("selected", "state", "reason"),
    [
        (task(preposition="ad"), "contradiction", "government"),
        (task(preposition=None), "unresolved", "ambiguous"),
        (
            task(features=Features(case="accusative", number="singular", gender="masculine")),
            "contradiction",
            "morphology",
        ),
        (task(preposition="in"), "unresolved", "ambiguous_government"),
    ],
)
def test_asymmetric_government_and_ambiguity(tmp_path: Path, selected: DeckTask, state: str, reason: str) -> None:
    result = run(tmp_path, selected)
    assert result.state == state
    assert reason in " ".join(result.checks[0].objections)
    assert result.decision.state == "withheld"


def test_missing_analyzer_is_not_linguistic_rejection(tmp_path: Path) -> None:
    class Missing(Library):
        def analyze(self, text: str) -> tuple[LibraryReading, ...]:
            raise ProcessingUnavailable("local models absent")

    result = run(tmp_path, library=Missing())
    assert result.state == "unavailable"
    assert result.checks[0].result == "unresolved"
    assert result.decision.state == "failed"
    assert result.decision.reason == "processing_unavailable"
    report = stage_artifact(tmp_path, result.stage)
    assert isinstance(report, dict)
    assert report["reason"] == "local models absent"


def test_unsupported_and_other_assertions_are_explicit(tmp_path: Path) -> None:
    class Empty(Library):
        readings = ()

    unsupported = run(tmp_path / "empty", library=Empty())
    assert unsupported.state == "unsupported"
    assert unsupported.checks[0].result == "not_applicable"
    extra = tuple(
        Assertion(
            id=kind,
            kind=kind,
            value="unsupported sentence",
            selected_sense="Freund" if kind == "translation" else None,
            alternatives=(),
            evidence=(Evidence("source", "Meaning", "1", "Freund"),),
        )
        for kind in ("translation", "segmentation", "explanation")
    )
    result = run(tmp_path / "extra", task(extra_assertions=extra))
    assert result.checks[0].result == "pass"
    assert [c.result for c in result.checks[2:]] == ["not_applicable"] * 3
    assert result.decision.state == "withheld"


def test_mapping_and_source_identity_cannot_be_repaired(tmp_path: Path) -> None:
    for record, request in [
        (source(ID="different"), task()),
        (source(), task(form=SourceCell(field="Meaning", start=0, end=6))),
        (source(), task(form=SourceCell(field="Parts", start=0, end=99))),
        (source(), task(source_role="ppp")),
    ]:
        with pytest.raises(ValueError):
            analyze_deck(
                tmp_path,
                record=record,
                profile=PROFILE,
                source_scope="deck",
                task=request,
                versions=VERSIONS,
                library=Library(),
            )


def test_changed_mapping_task_or_library_invalidates_binding(tmp_path: Path) -> None:
    first = run(tmp_path)
    second = run(tmp_path, task(preposition=None))
    assert first.decision.binding.task_sha256 != second.decision.binding.task_sha256
    assert first.stage.stage_id != second.stage.stage_id
    with pytest.raises(ValueError, match="analyzer version"):
        run(tmp_path, library=type("Changed", (Library,), {"version": "synthetic/2"})())


def test_incomplete_analyzer_hypothesis_cannot_reject_or_disambiguate(tmp_path: Path) -> None:
    class Tagger(Library):
        ambiguity_complete = False
        readings = (LibraryReading(lemma="amicus", pos="noun", features=DAT),)

    for index, readings in enumerate((Tagger.readings, (LibraryReading(lemma="amicus", pos="noun", features=ABL),))):
        tagger = Tagger()
        tagger.readings = readings
        result = run(tmp_path / str(index), library=tagger)
        assert result.state == "unresolved"
        assert result.checks[0].result == "unresolved"


def test_incomplete_features_and_partial_source_words_never_pass(tmp_path: Path) -> None:
    class Partial(Library):
        readings = (LibraryReading(lemma="amicus", pos="noun", features=Features(case="ablative")),)

    result = run(tmp_path, library=Partial())
    assert result.state == "unresolved"
    with pytest.raises(ValueError, match="splits a word"):
        run(tmp_path, task(form=SourceCell(field="Parts", start=0, end=4)))


def test_library_only_control_and_actual_missing_local_resources(tmp_path: Path) -> None:
    class Unambiguous(Library):
        readings = (LibraryReading(lemma="amicus", pos="noun", features=ABL),)

    positive = run(tmp_path / "positive", task(preposition=None), Unambiguous())
    assert positive.state == "corroborated"
    assert positive.checks[0].result == "pass"
    assert positive.decision.state == "withheld"
    library = LocalStanza(tmp_path / "no-installed-resources")
    unavailable = analyze_deck(
        tmp_path / "unavailable",
        record=source(),
        profile=PROFILE,
        source_scope="deck",
        task=task(),
        versions=VERSIONS.model_copy(update={"analyzer": library.version}),
        library=library,
    )
    assert unavailable.state == "unavailable"
    assert unavailable.decision.reason == "processing_unavailable"


def test_ppp_requires_actual_confirmed_role_not_fourth_position(tmp_path: Path) -> None:
    record = source(Lemma="amō", Parts="amō — amāre — amāvī — amātus", Meaning="lieben")
    features = Features(case="nominative", number="singular", gender="masculine")
    selected = task(
        form=SourceCell(field="Parts", start=22, end=28),
        pos="adjective",
        features=features,
        preposition=None,
        source_role="ppp",
        selected_sense="love",
    )

    class Participle(Library):
        def senses(self, lemma: str, language: str) -> tuple[Sense, ...]:
            return (
                Sense(
                    key="love", meaning="lieben", evidence=(Evidence("synthetic-lexicon", "amō.love", "1", "lieben"),)
                ),
            )

        def analyze(self, text: str) -> tuple[LibraryReading, ...]:
            assert text == "amātus"
            return (LibraryReading(lemma="amō", pos="adjective", features=features),)

    for role in ("supine", "ppp"):
        profile = DeckProfile.default(
            note_type="Vocabulary",
            lexical_entry_field="Lemma",
            principal_parts_field="Parts",
            meaning_field="Meaning",
            source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
            principal_part_roles=("present_1s", "present_infinitive", "perfect_1s", role),
        )
        if role == "supine":
            with pytest.raises(ValueError, match="source role"):
                analyze_deck(
                    tmp_path,
                    record=record,
                    profile=profile,
                    source_scope="deck",
                    task=selected,
                    versions=VERSIONS,
                    library=Participle(),
                )
        else:
            result = analyze_deck(
                tmp_path,
                record=record,
                profile=profile,
                source_scope="deck",
                task=selected,
                versions=VERSIONS,
                library=Participle(),
            )
            assert result.state == "corroborated"


@pytest.mark.parametrize("end", [5, 6])
def test_decomposed_word_cannot_supply_an_absent_source_form(tmp_path: Path, end: int) -> None:
    class AnySurface(Library):
        def analyze(self, text: str) -> tuple[LibraryReading, ...]:
            return self.readings

    with pytest.raises(ValueError, match="splits a word"):
        analyze_deck(
            tmp_path,
            record=source(Parts="amico\u0304rum"),
            profile=PROFILE,
            source_scope="deck",
            task=task(form=SourceCell(field="Parts", start=0, end=end)),
            versions=VERSIONS,
            library=AnySurface(),
        )


def test_rich_text_lemma_is_semantic_text_not_a_contradiction(tmp_path: Path) -> None:
    result = analyze_deck(
        tmp_path,
        record=source(Lemma="<i>amicus</i>"),
        profile=PROFILE,
        source_scope="deck",
        task=task(),
        versions=VERSIONS,
        library=Library(),
    )
    assert result.state == "corroborated"


def local_resources(root: Path) -> Path:
    root.mkdir()
    (root / "la").mkdir()
    (root / "resources.json").write_text("{}")
    return root


def test_unreadable_resources_produce_public_unavailable_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = local_resources(tmp_path / "resources")
    read = Path.read_bytes

    def unreadable(path: Path) -> bytes:
        if path == resources / "resources.json":
            raise PermissionError("resource unreadable")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    library = LocalStanza(resources)
    result = analyze_deck(
        tmp_path / "store",
        record=source(),
        profile=PROFILE,
        source_scope="deck",
        task=task(),
        versions=VERSIONS.model_copy(update={"analyzer": library.version}),
        library=library,
    )
    assert result.state == "unavailable"
    assert result.decision.reason == "processing_unavailable"


def test_resource_replacement_invalidates_execution(tmp_path: Path) -> None:
    resources = local_resources(tmp_path / "resources")
    library = LocalStanza(resources)
    (resources / "resources.json").write_text('{"changed":true}')
    with pytest.raises(ProcessingUnavailable, match="resource.*changed"):
        library.analyze("amō")


def test_stanza_perfect_features_survive_conversion_without_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resources = local_resources(tmp_path / "resources")
    word = SimpleNamespace(
        text="amāvī", lemma="amō", upos="VERB", feats="Tense=Perf|Mood=Ind|Voice=Act|Person=1|Number=Sing"
    )

    def pipeline(**options: object) -> Callable[[str], SimpleNamespace]:
        assert options["download_method"] is None
        assert options["use_gpu"] is False
        return lambda text: SimpleNamespace(sentences=[SimpleNamespace(words=[word])])

    monkeypatch.setitem(sys.modules, "stanza", SimpleNamespace(Pipeline=pipeline))
    readings = LocalStanza(resources).analyze("amāvī")
    assert readings[0].features == Features(
        tense="perfect", mood="indicative", voice="active", person="1", number="singular"
    )


@pytest.mark.parametrize(
    ("selected_sense", "gloss", "outcome"),
    [
        ("friend", "Freund", "pass"),
        ("enemy", "Freund", "fail"),
        ("friend", "Bekannter", "unresolved"),
        (None, "Freund", "unresolved"),
    ],
)
def test_sense_plausibility_is_separate_from_morphology(
    tmp_path: Path, selected_sense: str | None, gloss: str, outcome: str
) -> None:
    class MeaningLibrary(Library):
        def senses(self, lemma: str, language: str) -> tuple[Sense, ...]:
            return (
                Sense(key="friend", meaning="Freund", evidence=(Evidence("lexicon", "amicus.friend", "1", "Freund"),)),
            )

    result = analyze_deck(
        tmp_path,
        record=source(Meaning=gloss),
        profile=PROFILE,
        source_scope="deck",
        task=task(selected_sense=selected_sense, meaning=SourceCell(field="Meaning", start=0, end=len(gloss))),
        versions=VERSIONS,
        library=MeaningLibrary(),
    )
    assert result.checks[0].result == "pass"
    assert result.checks[1].result == outcome
    assert result.decision.state == "withheld"


def test_competing_lemma_and_sense_readings_remain_inspectable(tmp_path: Path) -> None:
    class Ambiguous(Library):
        readings = (*Library.readings, LibraryReading(lemma="other-lemma", pos="noun", features=ABL))

        def senses(self, lemma: str, language: str) -> tuple[Sense, ...]:
            proof = (Evidence("lexicon", "polysemy", "1", "same gloss, different senses"),)
            return (
                Sense(key="friend", meaning="Freund", evidence=proof),
                Sense(key="associate", meaning="Freund", evidence=proof),
            )

    result = run(tmp_path, library=Ambiguous())
    assert result.state == "unresolved"
    assert result.checks[0].result == "unresolved"
    assert result.checks[1].result == "unresolved"
    assert "other-lemma" in " ".join(result.checks[0].alternatives)
    assert "associate" in " ".join(result.checks[1].alternatives)


def constituent(
    root: Path, identity: str, surface: str, lemma: str, pos: POS, features: Features, sense: Sense
) -> DeckAnalysis:
    from dataclasses import replace

    class Component(Library):
        def analyze(self, text: str) -> tuple[LibraryReading, ...]:
            assert text == surface
            return (LibraryReading(lemma=lemma, pos=pos, features=features),)

        def senses(self, checked_lemma: str, language: str) -> tuple[Sense, ...]:
            assert checked_lemma == lemma
            return (sense,)

    return analyze_deck(
        root,
        record=replace(
            source(ID=identity, Lemma=lemma, Parts=surface, Meaning=sense.meaning), source_identity=identity
        ),
        profile=PROFILE,
        source_scope="deck",
        versions=VERSIONS,
        library=Component(),
        task=task(
            form=SourceCell(field="Parts", start=0, end=len(surface)),
            meaning=SourceCell(field="Meaning", start=0, end=len(sense.meaning)),
            pos=pos,
            features=features,
            preposition=None,
            selected_sense=sense.key,
        ),
    )


@pytest.mark.parametrize("family", ["preposition-noun", "noun-genitive", "adjective-noun", "verb-noun"])
@pytest.mark.parametrize("wrong", [False, True])
def test_four_family_construction_evidence(tmp_path: Path, family: str, wrong: bool) -> None:
    proof = (Evidence("synthetic-lexicon", "lexical constraints", "1", "selected sense/valency"),)
    girl = Sense(key="girl", meaning="Mädchen", evidence=proof)
    friend = Sense(key="friend", meaning="Freund", evidence=proof)
    good = Sense(key="good", meaning="gut", evidence=proof)
    love = Sense(
        key="love", meaning="lieben", valency={"subject": "nominative", "object": "accusative"}, evidence=proof
    )
    noun = Features(case="nominative", number="singular", gender="feminine")
    analyses: dict[str, DeckAnalysis] = {}
    roles: dict[str, Role] = {}

    def add(
        role: str, surface: str, lemma: str, pos: POS, features: Features, sense: Sense, required: Features
    ) -> None:
        analyses[role] = constituent(tmp_path, role, surface, lemma, pos, features, sense)
        roles[role] = Role(pos=pos, senses=(f"{role}/{sense.key}",), requires=required)

    if family == "preposition-noun":
        add("object", "amicō", "amicus", "noun", ABL, friend, Features(case="ablative"))
    elif family == "noun-genitive":
        add("head", "puella", "puella", "noun", noun, girl, Features())
        add(
            "dependent",
            "amicō" if wrong else "amīcī",
            "amicus",
            "noun",
            ABL if wrong else Features(case="genitive", number="singular", gender="masculine"),
            friend,
            Features(case="genitive"),
        )
    elif family == "adjective-noun":
        add("head", "puella", "puella", "noun", noun, girl, Features())
        add(
            "modifier",
            "bonus" if wrong else "bona",
            "bonus",
            "adjective",
            Features(case="nominative", number="singular", gender="masculine") if wrong else noun,
            good,
            Features(),
        )
    else:
        add(
            "subject",
            "puella" if wrong else "puellae",
            "puella",
            "noun",
            noun if wrong else Features(case="nominative", number="plural", gender="feminine"),
            girl,
            Features(case="nominative"),
        )
        add(
            "verb",
            "amant",
            "amō",
            "verb",
            Features(person="3", number="plural", tense="present", mood="indicative", voice="active"),
            love,
            Features(person="3"),
        )
        add(
            "object",
            "amicum",
            "amicus",
            "noun",
            Features(case="accusative", number="singular", gender="masculine"),
            friend,
            Features(case="accusative"),
        )
    pattern = Pattern.model_validate(
        {
            "key": "construction",
            "family": family,
            "roles": roles,
            "function_words": ("ad" if wrong else "cum",) if family == "preposition-noun" else (),
        }
    )
    result = analyze_construction(
        tmp_path, pattern=pattern, components={r: a.stage.stage_id for r, a in analyses.items()}, versions=VERSIONS
    )
    assert result.state == ("contradiction" if wrong else "corroborated")
    assert result.checks[0].method == "rule"
    assert result.checks[0].result == ("fail" if wrong else "pass")
    assert result.decision.state == "withheld"


@pytest.mark.parametrize("preposition", ["cum", "ad", None])
def test_persisted_graph_preserves_checks_and_objections(tmp_path: Path, preposition: str | None) -> None:
    result = run(tmp_path, task(preposition=preposition))
    path = tmp_path / "decision.json"
    save_acceptance(path, result.decision)
    restored = load_acceptance(path)
    assert restored == result.decision
    for identity in restored.stage_ids:
        stage = load_stage(tmp_path, identity)
        assert (
            reusable_stage(
                tmp_path, identity, binding=restored.binding, versions=VERSIONS, input_sha256=stage.input_sha256
            )
            == stage
        )
    report = stage_artifact(tmp_path, load_stage(tmp_path, result.stage.stage_id))
    assert isinstance(report, dict)
    assert report["checks"] == [c.model_dump(mode="json") for c in result.checks]
    assert report["source"]["task"]["form"] == {"field": "Parts", "start": 0, "end": 5}


def test_actual_mapping_and_source_changes_invalidate_reuse(tmp_path: Path) -> None:
    record = source(Alternate="Freund")
    first = analyze_deck(
        tmp_path, record=record, profile=PROFILE, source_scope="deck", task=task(), versions=VERSIONS, library=Library()
    )
    profile = PROFILE.model_copy(update={"fields": PROFILE.fields.model_copy(update={"meaning_field": "Alternate"})})
    second = analyze_deck(
        tmp_path,
        record=record,
        profile=profile,
        source_scope="deck",
        task=task(meaning=SourceCell(field="Alternate", start=0, end=6)),
        versions=VERSIONS,
        library=Library(),
    )
    assert first.decision.binding.source_identity == second.decision.binding.source_identity
    assert first.decision.binding.source_sha256 == second.decision.binding.source_sha256
    assert first.decision.binding.mapping_sha256 != second.decision.binding.mapping_sha256
    with pytest.raises(ValueError, match="binding_mismatch"):
        reusable_stage(
            tmp_path,
            first.stage.stage_id,
            binding=second.decision.binding,
            versions=VERSIONS,
            input_sha256=first.stage.input_sha256,
        )
    changed = analyze_deck(
        tmp_path,
        record=source(Meaning="Bekannter"),
        profile=PROFILE,
        source_scope="deck",
        task=task(meaning=SourceCell(field="Meaning", start=0, end=9)),
        versions=VERSIONS,
        library=Library(),
    )
    assert changed.decision.binding.source_identity == first.decision.binding.source_identity
    assert changed.decision.binding.source_sha256 != first.decision.binding.source_sha256
    assert changed.decision.binding.extraction_sha256 != first.decision.binding.extraction_sha256


def test_verb_person_and_number_are_not_interchangeable(tmp_path: Path) -> None:
    class Verb(Library):
        def analyze(self, text: str) -> tuple[LibraryReading, ...]:
            assert text == "amās"
            return (
                LibraryReading(
                    lemma="amō",
                    pos="verb",
                    features=Features(
                        person="2", number="singular", tense="present", mood="indicative", voice="active"
                    ),
                ),
            )

    for person, number, expected in [("2", "singular", "pass"), ("3", "plural", "fail")]:
        result = analyze_deck(
            tmp_path,
            record=source(Lemma="amō", Parts="amās"),
            profile=PROFILE,
            source_scope="deck",
            task=task(
                form=SourceCell(field="Parts", start=0, end=4),
                pos="verb",
                preposition=None,
                features=Features.model_validate(
                    {"person": person, "number": number, "tense": "present", "mood": "indicative", "voice": "active"}
                ),
            ),
            versions=VERSIONS,
            library=Verb(),
        )
        assert result.checks[0].result == expected


@pytest.mark.parametrize(
    ("field", "raw", "selected"),
    [
        ("Parts", '<span title="amicō">amicōrum</span>', "amicō"),
        ("Meaning", '<span title="Freund">Feind</span>', "Freund"),
        ("Parts", "<b>amicō</b>rum", "amicō"),
        ("Parts", "&lt;b&gt;amicō&lt;/b&gt;rum", "amicō"),
        ("Parts", "<script>amicō</script>amicōrum", "amicō"),
        ("Parts", "<!--amicō-->amicōrum", "amicō"),
    ],
)
def test_html_metadata_and_cross_tag_prefixes_are_not_source_words(
    tmp_path: Path, field: str, raw: str, selected: str
) -> None:
    span = SourceCell(field=field, start=raw.index(selected), end=raw.index(selected) + len(selected))
    with pytest.raises(ValueError):
        analyze_deck(
            tmp_path,
            record=source(**{field: raw}),
            profile=PROFILE,
            source_scope="deck",
            task=task(**{"form" if field == "Parts" else "meaning": span}),
            versions=VERSIONS,
            library=Library(),
        )


@pytest.mark.parametrize("raw", ["<b>ami</b>cō", "amic&#333;", '<span title="amicōrum">amicō</span>'])
def test_readable_html_words_keep_raw_provenance(tmp_path: Path, raw: str) -> None:
    result = analyze_deck(
        tmp_path,
        record=source(Parts=raw),
        profile=PROFILE,
        source_scope="deck",
        task=task(form=SourceCell(field="Parts", start=0, end=len(raw))),
        versions=VERSIONS,
        library=Library(),
    )
    report = stage_artifact(tmp_path, result.stage)
    assert isinstance(report, dict)
    assert report["source"]["form"] == "amicō"
    assert report["source"]["task"]["form"]["end"] == len(raw)
    assert result.checks[0].result == "pass"
    assert result.stage.binding.source_sha256 == digest(dict(source(Parts=raw).fields))


def test_equivalent_repeated_sense_references_are_not_contradictions(tmp_path: Path) -> None:
    class Repeated(Library):
        def senses(self, lemma: str, language: str) -> tuple[Sense, ...]:
            return super().senses(lemma, language) * 2

    assert run(tmp_path, library=Repeated()).checks[1].result == "pass"


def test_inconsistent_same_key_references_never_pass_in_either_order(tmp_path: Path) -> None:
    proof = (Evidence("synthetic-lexicon", "conflicting valency", "1", "unresolved references"),)
    references = (
        Sense(key="friend", meaning="Freund", valency={"object": "accusative"}, evidence=proof),
        Sense(key="friend", meaning="Freund", valency={"object": "dative"}, evidence=proof),
    )

    class Conflicting(Library):
        references: tuple[Sense, ...] = ()

        def senses(self, lemma: str, language: str) -> tuple[Sense, ...]:
            return self.references

    for index, ordered in enumerate((references, references[::-1])):
        library = Conflicting()
        library.references = ordered
        result = run(tmp_path / str(index), library=library)
        assert result.checks[1].result == "unresolved"
        assert "inconsistent_sense_reference" in result.checks[1].objections


def test_context_dependent_constituent_cannot_be_reused_without_its_context(tmp_path: Path) -> None:
    head = run(tmp_path)
    compatible = analyze_construction(
        tmp_path,
        pattern=Pattern(
            key="cum",
            family="preposition-noun",
            function_words=("cum",),
            roles={
                "object": Role(pos="noun", senses=("17/friend",), requires=Features(case="ablative")),
            },
        ),
        versions=VERSIONS,
        components={"object": head.stage.stage_id},
    )
    assert compatible.checks[0].result == "pass"
    proof = (Evidence("synthetic-lexicon", "girl", "1", "girl"),)
    dependent = constituent(
        tmp_path,
        "dependent",
        "puellae",
        "puella",
        "noun",
        Features(case="genitive", number="singular", gender="feminine"),
        Sense(key="girl", meaning="Mädchen", evidence=proof),
    )
    pattern = Pattern(
        key="genitive",
        family="noun-genitive",
        function_words=(),
        roles={
            "head": Role(pos="noun", senses=("17/friend",)),
            "dependent": Role(pos="noun", senses=("dependent/girl",), requires=Features(case="genitive")),
        },
    )
    result = analyze_construction(
        tmp_path,
        pattern=pattern,
        versions=VERSIONS,
        components={"head": head.stage.stage_id, "dependent": dependent.stage.stage_id},
    )
    assert result.state == "unresolved"
    assert "head: constituent_context_mismatch" in result.checks[0].objections


@pytest.mark.parametrize("voice", ["active", "passive"])
def test_active_valency_cannot_license_passive_constructions(tmp_path: Path, voice: str) -> None:
    proof = (Evidence("synthetic-lexicon", "amō valency", "1", "active transitive sense"),)
    subject = constituent(
        tmp_path,
        "subject",
        "puella",
        "puella",
        "noun",
        Features(case="nominative", number="singular", gender="feminine"),
        Sense(key="girl", meaning="Mädchen", evidence=proof),
    )
    verb = constituent(
        tmp_path,
        "verb",
        "amat" if voice == "active" else "amātur",
        "amō",
        "verb",
        Features.model_validate(
            {"person": "3", "number": "singular", "tense": "present", "mood": "indicative", "voice": voice}
        ),
        Sense(key="love", meaning="lieben", valency={"subject": "nominative", "object": "accusative"}, evidence=proof),
    )
    obj = constituent(
        tmp_path,
        "object",
        "amicum",
        "amicus",
        "noun",
        Features(case="accusative", number="singular", gender="masculine"),
        Sense(key="friend", meaning="Freund", evidence=proof),
    )
    pattern = Pattern(
        key="transitive",
        family="verb-noun",
        function_words=(),
        roles={
            "subject": Role(pos="noun", senses=("subject/girl",), requires=Features(case="nominative")),
            "verb": Role(pos="verb", senses=("verb/love",), requires=Features(person="3")),
            "object": Role(pos="noun", senses=("object/friend",), requires=Features(case="accusative")),
        },
    )
    result = analyze_construction(
        tmp_path,
        pattern=pattern,
        versions=VERSIONS,
        components={"subject": subject.stage.stage_id, "verb": verb.stage.stage_id, "object": obj.stage.stage_id},
    )
    assert result.checks[0].result == ("pass" if voice == "active" else "not_applicable")
