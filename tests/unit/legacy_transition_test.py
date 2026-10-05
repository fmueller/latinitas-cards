"""Explicit guards for legacy note-model transitions.

These tests verify the T-034 contract: legacy per-exercise note models are
never silently reinterpreted, incompatible transitions stay read-only review
outcomes or are rejected outright, an explicitly approved fresh start is the
only supported legacy option and only for disposable data, valuable history
and conflicting personal annotations retain the old collection, and compatible
regeneration of the current model is a separate operation that preserves
identity and history.  A synthetic rehearsal walks the documented fresh-start
checklist, and a drift test pins the published policy statement and inventory.
"""

import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from latinitas_cards.identity import LATINITAS_ID_VERSION, derive_latinitas_id
from latinitas_cards.legacy_transition import (
    CSV_CONSOLIDATION_HISTORY_STATEMENT,
    DestinationNoteEvidence,
    FreshStartApproval,
    LegacyTransitionError,
    classify_legacy_model,
    evaluate_legacy_transition,
    plan_legacy_transition,
)
from latinitas_cards.notes import NOTE_SCHEMA_VERSION
from latinitas_cards.profile import DeckProfile

LEGACY_NOTE_TYPE = "Latinitas Legacy Exercise"
NEW_NOTE_TYPE = "Latinitas Principal Parts Fresh"
DOCS_FILE = Path(__file__).parents[2] / "docs" / "legacy-transition.md"

CURRENT_ID = derive_latinitas_id("source-1", "object-1")
LEGACY_ID = "latinitas-v1-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
LEGACY_SPLIT_GUID = "legacy-split-guid-v1-0123456789abcdef0123456789abcdef"


def _profile(generated_note_type: str = NEW_NOTE_TYPE) -> DeckProfile:
    return DeckProfile.default(
        note_type="Legacy Source",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        generated_note_type=generated_note_type,
    )


def _approval(backup: Path, dedicated_note_type: str = NEW_NOTE_TYPE) -> FreshStartApproval:
    return FreshStartApproval(
        backup_path=backup,
        dedicated_note_type=dedicated_note_type,
        data_is_disposable=True,
        acknowledges_new_schedules=True,
        dedicated_note_type_id="new-model",
    )


@pytest.mark.parametrize("approved_type_id", [None, "wrong-model"])
def test_bound_fresh_start_rejects_unbound_or_mismatched_approved_type(
    verified_backup: Path, approved_type_id: str | None
) -> None:
    from latinitas_cards.destination_state import BoundDestination, schema_contract
    from latinitas_cards.legacy_transition import plan_destination_fresh_start

    original = BoundDestination("old-collection", "profile", {"note_type_id": "old-model"}, {})
    destination = BoundDestination(
        "new-collection", "profile", schema_contract("new-model"), {"scope": "scope", "members": []}
    )
    approval = FreshStartApproval(verified_backup, NEW_NOTE_TYPE, True, True, dedicated_note_type_id=approved_type_id)
    with pytest.raises(LegacyTransitionError, match="approved destination note type ID"):
        plan_destination_fresh_start(
            _profile(),
            [],
            original=original,
            destination=destination,
            selection="fresh_start",
            fresh_start=approval,
        )


@pytest.mark.parametrize("selection", [None, "consolidate", "fresh_start"])
def test_bound_fresh_start_requires_selection_and_separate_destination(
    verified_backup: Path, selection: str | None
) -> None:
    from latinitas_cards.destination_state import BoundDestination, schema_contract
    from latinitas_cards.legacy_transition import plan_destination_fresh_start

    original = BoundDestination("old-collection", "profile", {"note_type_id": "old-model"}, {})
    destination = BoundDestination(
        "old-collection", "profile", schema_contract("new-model"), {"scope": "scope", "members": []}
    )
    with pytest.raises(LegacyTransitionError, match="explicit|unsupported|separate destination"):
        plan_destination_fresh_start(
            _profile(),
            [],
            original=original,
            destination=destination,
            selection=selection,
            fresh_start=_approval(verified_backup),
        )


def test_bound_fresh_start_preserves_original_evidence_and_discloses_schedules(verified_backup: Path) -> None:
    from latinitas_cards.destination_state import BoundDestination, schema_contract
    from latinitas_cards.legacy_transition import plan_destination_fresh_start

    original = BoundDestination(
        "old-collection",
        "profile",
        {"note_type_id": "old-model", "version": "2"},
        {"historical_manifest": ["exercise-id"]},
    )
    destination = BoundDestination(
        "new-collection", "profile", schema_contract("new-model"), {"scope": "scope", "members": []}
    )
    notes = [
        DestinationNoteEvidence(LEGACY_ID, LEGACY_NOTE_TYPE, "2", tags=("personal-tag",), card_count=3),
        DestinationNoteEvidence(LEGACY_ID + "history", LEGACY_NOTE_TYPE, "2", has_review_history=True),
        DestinationNoteEvidence(LEGACY_ID + "personal", LEGACY_NOTE_TYPE, "2", personal_notes="keep me"),
    ]
    before = deepcopy((original, destination, notes))
    backup_bytes = verified_backup.read_bytes()
    plan = plan_destination_fresh_start(
        _profile(),
        notes,
        original=original,
        destination=destination,
        selection="fresh_start",
        fresh_start=_approval(verified_backup),
        legacy_note_types=[LEGACY_NOTE_TYPE],
    )
    assert [review.decision for review in plan.decisions] == ["fresh_start", "retain_and_defer", "retain_and_defer"]
    assert plan.original_destination == "old-collection"
    assert plan.fresh_destination == "new-collection"
    assert "no inherited scheduling" in plan.disclosure
    assert plan.old_collections_retained
    assert (original, destination, notes) == before
    assert verified_backup.read_bytes() == backup_bytes
    with pytest.raises(LegacyTransitionError, match="approval"):
        plan_destination_fresh_start(
            _profile(),
            notes,
            original=original,
            destination=destination,
            selection="fresh_start",
            fresh_start=None,
        )
    destination.schema["note_type_id"] = "old-model"
    with pytest.raises(LegacyTransitionError, match="note type"):
        plan_destination_fresh_start(
            _profile(),
            notes,
            original=original,
            destination=destination,
            selection="fresh_start",
            fresh_start=_approval(verified_backup),
        )


@pytest.fixture()
def verified_backup(tmp_path: Path) -> Path:
    backup = tmp_path / "old-collection.backup.anki2"
    backup.write_bytes(b"synthetic legacy collection bytes")
    return backup


@pytest.mark.parametrize(
    ("first_field", "note_schema", "expected"),
    [
        (CURRENT_ID, NOTE_SCHEMA_VERSION, "current_learning_object"),
        (CURRENT_ID, None, "current_learning_object"),
        (LEGACY_ID, "1", "legacy_per_exercise"),
        (LEGACY_ID, "2", "legacy_per_exercise"),
        (LEGACY_ID, None, "legacy_per_exercise"),
        (LEGACY_SPLIT_GUID, None, "legacy_split_clone"),
        ("", "1", "legacy_per_exercise"),
        (CURRENT_ID, "2", "unrecognized"),
        (LEGACY_ID, NOTE_SCHEMA_VERSION, "unrecognized"),
        (LEGACY_SPLIT_GUID, NOTE_SCHEMA_VERSION, "unrecognized"),
        ("amo", "3", "unrecognized"),
        ("amo", None, "unrecognized"),
    ],
)
def test_classify_legacy_model_matrix(first_field: str, note_schema: str | None, expected: str) -> None:
    assert classify_legacy_model(first_field, note_schema) == expected


def test_current_model_regenerates_in_place_without_fresh_start_approval() -> None:
    evidence = DestinationNoteEvidence(
        first_field=CURRENT_ID,
        note_type=NEW_NOTE_TYPE,
        note_schema=NOTE_SCHEMA_VERSION,
        has_review_history=True,
        card_count=6,
    )
    review = evaluate_legacy_transition(evidence, intended_note_type=NEW_NOTE_TYPE)
    assert review.decision == "compatible_regeneration"
    assert review.model_kind == "current_learning_object"
    assert review.requires == ()
    assert review.first_field == CURRENT_ID


def test_legacy_per_exercise_note_stays_a_no_write_review_outcome() -> None:
    evidence = DestinationNoteEvidence(first_field=LEGACY_ID, note_type=LEGACY_NOTE_TYPE, note_schema="1")
    review = evaluate_legacy_transition(evidence, intended_note_type=NEW_NOTE_TYPE)
    assert review.decision == "legacy_review_required"
    assert review.model_kind == "legacy_per_exercise"
    assert review.requires


def test_legacy_identity_is_never_reinterpreted_as_an_object_identity(tmp_path: Path) -> None:
    backup = tmp_path / "old-collection.backup.anki2"
    backup.write_bytes(b"synthetic legacy collection bytes")
    legacy = DestinationNoteEvidence(first_field=LEGACY_ID, note_schema="1")
    with_fresh_start = evaluate_legacy_transition(
        legacy,
        intended_note_type=NEW_NOTE_TYPE,
        fresh_start=_approval(backup),
    )
    assert with_fresh_start.decision == "fresh_start"
    assert with_fresh_start.first_field == LEGACY_ID
    assert "new identities" in with_fresh_start.reason
    assert LEGACY_ID not in with_fresh_start.reason
    current = evaluate_legacy_transition(
        DestinationNoteEvidence(first_field=CURRENT_ID, note_schema=NOTE_SCHEMA_VERSION),
        intended_note_type=NEW_NOTE_TYPE,
    )
    assert current.decision == "compatible_regeneration"
    assert with_fresh_start.first_field != current.first_field


def test_valuable_review_history_retains_and_defers_even_with_approval(verified_backup: Path) -> None:
    evidence = DestinationNoteEvidence(
        first_field=LEGACY_ID, note_type=LEGACY_NOTE_TYPE, note_schema="1", has_review_history=True
    )
    review = evaluate_legacy_transition(
        evidence, intended_note_type=NEW_NOTE_TYPE, fresh_start=_approval(verified_backup)
    )
    assert review.decision == "retain_and_defer"
    assert CSV_CONSOLIDATION_HISTORY_STATEMENT.split(":")[0] in review.reason


def test_conflicting_personal_annotation_retains_and_defers_even_with_approval(verified_backup: Path) -> None:
    evidence = DestinationNoteEvidence(
        first_field=LEGACY_ID, note_type=LEGACY_NOTE_TYPE, note_schema="1", personal_notes="  mein Merkwort  "
    )
    review = evaluate_legacy_transition(
        evidence, intended_note_type=NEW_NOTE_TYPE, fresh_start=_approval(verified_backup)
    )
    assert review.decision == "retain_and_defer"
    assert "last-write-wins" in review.reason


@pytest.mark.parametrize(
    ("variant", "expected_substrings"),
    [
        ("missing_backup", ("backup",)),
        ("empty_backup", ("backup",)),
        ("legacy_type_collision", ("distinct from every legacy note type", "matches the export note type")),
        ("blank_dedicated_type", ("distinct from every legacy note type", "matches the export note type")),
    ],
)
def test_incomplete_fresh_start_approval_reports_exact_missing_requirements(
    tmp_path: Path,
    variant: str,
    expected_substrings: tuple[str, ...],
) -> None:
    if variant == "missing_backup":
        approval = FreshStartApproval(
            backup_path=tmp_path / "never-created.anki2",
            dedicated_note_type=NEW_NOTE_TYPE,
            data_is_disposable=True,
            acknowledges_new_schedules=True,
        )
    elif variant == "empty_backup":
        empty_backup = tmp_path / "empty.anki2"
        empty_backup.write_bytes(b"")
        approval = FreshStartApproval(
            backup_path=empty_backup,
            dedicated_note_type=NEW_NOTE_TYPE,
            data_is_disposable=True,
            acknowledges_new_schedules=True,
        )
    elif variant == "legacy_type_collision":
        approval = FreshStartApproval(
            backup_path=write_verified_backup(tmp_path),
            dedicated_note_type=LEGACY_NOTE_TYPE,
            data_is_disposable=True,
            acknowledges_new_schedules=True,
        )
    else:
        approval = FreshStartApproval(
            backup_path=write_verified_backup(tmp_path),
            dedicated_note_type="  ",
            data_is_disposable=True,
            acknowledges_new_schedules=True,
        )
    review = evaluate_legacy_transition(
        DestinationNoteEvidence(first_field=LEGACY_ID, note_schema="1"),
        intended_note_type=NEW_NOTE_TYPE,
        fresh_start=approval,
        legacy_note_types=(LEGACY_NOTE_TYPE,),
    )
    assert review.decision == "legacy_review_required"
    assert len(review.requires) == len(expected_substrings)
    for substring in expected_substrings:
        assert any(substring in requirement for requirement in review.requires)


def test_padded_legacy_note_type_names_cannot_bypass_the_distinct_type_gate(verified_backup: Path) -> None:
    review = evaluate_legacy_transition(
        DestinationNoteEvidence(first_field=LEGACY_ID, note_schema="1"),
        intended_note_type=NEW_NOTE_TYPE,
        legacy_note_types=(f"  {NEW_NOTE_TYPE}  ",),
        fresh_start=_approval(verified_backup),
    )
    assert review.decision == "legacy_review_required"
    assert any("distinct from every legacy note type" in requirement for requirement in review.requires)


def test_unconfirmed_disposability_and_schedules_block_the_fresh_start(tmp_path: Path) -> None:
    approval = FreshStartApproval(backup_path=write_verified_backup(tmp_path), dedicated_note_type=NEW_NOTE_TYPE)
    review = evaluate_legacy_transition(
        DestinationNoteEvidence(first_field=LEGACY_ID, note_schema="1"),
        intended_note_type=NEW_NOTE_TYPE,
        fresh_start=approval,
    )
    assert review.decision == "legacy_review_required"
    assert len(review.requires) == 2
    assert any("disposable" in requirement for requirement in review.requires)
    assert any("schedule" in requirement for requirement in review.requires)


def test_backup_gate_documents_its_actual_verification_strength(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("not a backup")
    approval = FreshStartApproval(
        backup_path=unrelated,
        dedicated_note_type=NEW_NOTE_TYPE,
        data_is_disposable=True,
        acknowledges_new_schedules=True,
    )
    assert approval.missing_requirements() == ()


def test_approved_fresh_start_requires_the_matching_dedicated_note_type(verified_backup: Path) -> None:
    approval = _approval(verified_backup, dedicated_note_type="Latinitas Something Else")
    review = evaluate_legacy_transition(
        DestinationNoteEvidence(first_field=LEGACY_ID, note_schema="1"),
        intended_note_type=NEW_NOTE_TYPE,
        fresh_start=approval,
    )
    assert review.decision == "legacy_review_required"
    assert any(NEW_NOTE_TYPE in requirement for requirement in review.requires)


def test_legacy_split_clone_stays_review_only() -> None:
    evidence = DestinationNoteEvidence(first_field=LEGACY_SPLIT_GUID, note_type=LEGACY_NOTE_TYPE)
    review = evaluate_legacy_transition(evidence, intended_note_type=NEW_NOTE_TYPE)
    assert review.decision == "legacy_review_required"
    assert review.model_kind == "legacy_split_clone"
    assert "split-clone" in review.reason


def test_unrecognized_contradictory_evidence_yields_the_inventory_review() -> None:
    review = evaluate_legacy_transition(
        DestinationNoteEvidence(first_field=CURRENT_ID, note_type=NEW_NOTE_TYPE, note_schema="2"),
        intended_note_type=NEW_NOTE_TYPE,
    )
    assert review.decision == "legacy_review_required"
    assert review.model_kind == "unrecognized"
    assert review.requires == ("a reviewed inventory of the destination note model",)


def test_profile_targeting_a_legacy_note_type_is_rejected(verified_backup: Path) -> None:
    with pytest.raises(LegacyTransitionError) as excinfo:
        plan_legacy_transition(
            _profile(generated_note_type=LEGACY_NOTE_TYPE),
            (),
            legacy_note_types=(LEGACY_NOTE_TYPE,),
        )
    assert LEGACY_NOTE_TYPE in str(excinfo.value)


def test_plan_keeps_old_collections_and_separates_decisions(verified_backup: Path) -> None:
    valuable = DestinationNoteEvidence(
        first_field=LEGACY_ID, note_type=LEGACY_NOTE_TYPE, note_schema="1", has_review_history=True
    )
    annotated = DestinationNoteEvidence(
        first_field=f"{LEGACY_ID[:-1]}0", note_type=LEGACY_NOTE_TYPE, note_schema="2", personal_notes="notiz"
    )
    disposable = DestinationNoteEvidence(first_field=f"{LEGACY_ID[:-1]}1", note_type=LEGACY_NOTE_TYPE, note_schema="1")
    current = DestinationNoteEvidence(
        first_field=CURRENT_ID, note_type=NEW_NOTE_TYPE, note_schema=NOTE_SCHEMA_VERSION, has_review_history=True
    )
    plan = plan_legacy_transition(
        _profile(),
        (valuable, annotated, disposable, current),
        legacy_note_types=(LEGACY_NOTE_TYPE,),
        fresh_start=_approval(verified_backup),
    )
    decisions = {review.first_field: review.decision for review in plan.decisions}
    assert decisions[valuable.first_field] == "retain_and_defer"
    assert decisions[annotated.first_field] == "retain_and_defer"
    assert decisions[disposable.first_field] == "fresh_start"
    assert decisions[current.first_field] == "compatible_regeneration"
    assert plan.old_collections_retained is True
    assert plan.intended_note_type == NEW_NOTE_TYPE
    assert plan.legacy_note_types == (LEGACY_NOTE_TYPE,)
    assert plan.review_only is False


def test_review_only_plan_reports_no_write_decisions(verified_backup: Path) -> None:
    valuable = DestinationNoteEvidence(first_field=LEGACY_ID, note_schema="1", has_review_history=True)
    unrecognized = DestinationNoteEvidence(first_field="amo", note_type="Unmapped")
    plan = plan_legacy_transition(
        _profile(),
        (valuable, unrecognized),
        legacy_note_types=(LEGACY_NOTE_TYPE,),
        fresh_start=_approval(verified_backup),
    )
    assert plan.review_only is True
    disposable = DestinationNoteEvidence(first_field=f"{LEGACY_ID[:-1]}3", note_schema="1")
    fresh_start_plan = plan_legacy_transition(
        _profile(),
        (valuable, disposable),
        legacy_note_types=(LEGACY_NOTE_TYPE,),
        fresh_start=_approval(verified_backup),
    )
    assert fresh_start_plan.review_only is False


def test_synthetic_fresh_start_rehearsal_follows_the_documented_checklist(tmp_path: Path) -> None:
    collection = tmp_path / "old-collection.anki2"
    collection.write_bytes(b"synthetic legacy collection bytes")
    original_bytes = collection.read_bytes()
    legacy_types = (LEGACY_NOTE_TYPE,)
    evidences = (
        DestinationNoteEvidence(first_field=LEGACY_ID, note_type=LEGACY_NOTE_TYPE, note_schema="1", card_count=4),
        DestinationNoteEvidence(
            first_field=f"{LEGACY_ID[:-1]}2",
            note_type=LEGACY_NOTE_TYPE,
            note_schema="1",
            personal_notes="behalten",
            has_review_history=True,
        ),
    )

    backup = tmp_path / "old-collection.backup.anki2"
    shutil.copyfile(collection, backup)
    assert backup.is_file() and backup.stat().st_size > 0

    plan = plan_legacy_transition(
        _profile(),
        evidences,
        legacy_note_types=legacy_types,
        fresh_start=_approval(backup),
    )
    decisions = {review.first_field: review.decision for review in plan.decisions}
    assert decisions[evidences[0].first_field] == "fresh_start"
    assert decisions[evidences[1].first_field] == "retain_and_defer"
    assert plan.intended_note_type not in legacy_types
    assert plan.old_collections_retained is True
    assert collection.read_bytes() == original_bytes
    assert backup.read_bytes() == original_bytes


def test_published_docs_pin_the_consolidation_statement_and_inventory() -> None:
    docs = DOCS_FILE.read_text(encoding="utf-8")
    begin = docs.index("<!-- latinitas-consolidation-statement begin -->")
    end = docs.index("<!-- latinitas-consolidation-statement end -->")
    published = " ".join(docs[begin:end].split())
    assert " ".join(CSV_CONSOLIDATION_HISTORY_STATEMENT.split()) in published
    for inventory_name in (
        "tests/unit/cli_test.py",
        "tests/unit/sources_test.py",
        "tests/unit/manifest_test.py",
        "tests/unit/preview_export_test.py",
        "docs/reference-note-type.md",
    ):
        assert inventory_name in docs
    assert f"`{LATINITAS_ID_VERSION}`" in docs
    assert "_write_legacy_database" in docs


def write_verified_backup(tmp_path: Path) -> Path:
    backup = tmp_path / "verified.anki2"
    backup.write_bytes(b"synthetic legacy collection bytes")
    return backup
