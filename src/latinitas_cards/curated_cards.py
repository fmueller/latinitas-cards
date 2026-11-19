"""Deterministic curated objectives; exact human approval, never automatic authority."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from .acceptance_presentation import ACCEPTANCE_FIELDS, Presentation, layout_digest, require_installation, stylesheet
from .acceptance_records import Record, Text, digest
from .cards import CardSlot
from .curated import (
    Approval,
    CuratedInput,
    Item,
    Language,
    Mnemonic,
    matching_decisions,
    mnemonic_fingerprint,
    rule_fingerprint,
    validate_history,
)
from .identity import derive_card_semantic_key, derive_latinitas_id
from .pipeline_profile import PipelineProfile
from .reference_templates import REFERENCE_CARD_CSS, ReferenceCardTemplate

CURATED_RECIPES = ("rule_application", "formation_rule_recall", "component_explanation", "mnemonic_recall")


class RecipeTarget(Record):
    """Explicit finite exercise, referring to intake content rather than new QA answers."""

    key: Text
    item_key: Text
    formation: Text | None
    example: Text | None
    analysis: Literal["supported", "unsupported", "ambiguous"]
    mnemonic_key: Text | None
    mnemonic_side: Literal["front", "back"]
    related_production: tuple[Text, ...]


class RecipePlan(Record):
    format: Literal["latinitas-curated-recipes"]
    version: Literal[1]
    targets: tuple[RecipeTarget, ...]

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len({(t.item_key, t.key) for t in self.targets}) != len(self.targets):
            raise ValueError("duplicate curated target")
        return self


def _template(recipe: str, ordinal: int) -> ReferenceCardTemplate:
    slot = CardSlot(
        semantic_key=recipe + ":curated",
        recipe=recipe,
        role="curated",
        template_name=recipe,
        ordinal=ordinal,
        enabled_field=f"Curated{ordinal}Enabled",
        prompt_field=f"Curated{ordinal}Prompt",
        answer_field=f"Curated{ordinal}Answer",
    )
    return ReferenceCardTemplate(
        slot=slot,
        front=f"{{{{#{slot.enabled_field}}}}}{{{{{slot.prompt_field}}}}}{{{{/{slot.enabled_field}}}}}",
        back=f"{{{{#{slot.enabled_field}}}}}{{{{{slot.answer_field}}}}}{{{{/{slot.enabled_field}}}}}"
        '{{#Personal Notes}}<div class="personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}',
    )


# A separate, explicit new note type, not a migration of existing template ordinals.
CURATED_LAYOUT = tuple(_template(recipe, ordinal) for ordinal, recipe in enumerate(CURATED_RECIPES))
CURATED_FIELDS = (
    "LatinitasID",
    "CardKey",
    *(
        field
        for template in CURATED_LAYOUT
        for field in (template.slot.enabled_field, template.slot.prompt_field, template.slot.answer_field)
    ),
)
TEMPLATE_SHA256 = digest(
    {
        "templates": [asdict(t) for t in CURATED_LAYOUT],
        "css": REFERENCE_CARD_CSS,
        "fields": (*CURATED_FIELDS, "Personal Notes"),
    }
)


@dataclass(frozen=True)
class Settings:
    language: Language
    recipes: tuple[str, ...] = ("rule_application",)
    theme: Literal["light", "dark"] = "light"
    profile: PipelineProfile | None = None
    profile_path: Path | None = None

    @property
    def acceptance_layout(self) -> bool:
        if self.profile is None or self.profile.layout.id != "acceptance-1":
            return False
        layout = self.profile.layout
        require_installation(
            layout.id,
            layout.version,
            layout.template_sha256,
            self.profile_path,
            "curated",
        )
        return True

    @property
    def snapshot(self) -> dict[str, object]:
        return {
            "language": self.language,
            "recipes": self.recipes,
            "theme": self.theme,
            "preset": self.profile.preset if self.profile else None,
            "effective_profile": self.profile.model_dump(mode="json") if self.profile else None,
        }


@dataclass(frozen=True)
class Requirement:
    subject: str
    subject_key: str
    language: str
    fingerprint: str
    status: str
    decision_ids: tuple[str, ...]


@dataclass(frozen=True)
class CuratedCard:
    note_id: str
    card_key: str
    item_key: str
    recipe: str
    front_html: str
    back_html: str
    fingerprint: str
    requirements: tuple[Requirement, ...]
    approval_back_html: str | None = None

    @property
    def reason(self) -> str:
        return next((f"{r.subject}:{r.status}" for r in self.requirements if r.status != "approved"), "accepted")

    def report(self) -> dict[str, object]:
        value = asdict(self)
        if self.approval_back_html is None:
            value.pop("approval_back_html")
        else:
            value["rendered_approval_subject"] = (
                "front_html + approval_back_html; proposed learning appearance, not approval"
            )
        return {**value, "state": "accepted" if self.reason == "accepted" else "withheld", "reason": self.reason}


def _paragraph(label: str, value: str) -> str:
    return f"<p><strong>{escape(label)}</strong> {escape(value)}</p>"


def _render(
    item: Item, target: RecipeTarget, recipe: str, settings: Settings, approved: bool = False
) -> tuple[str, str, tuple[Mnemonic, ...]]:
    de = settings.language == "de"
    if item.rule.language != settings.language:
        raise ValueError("missing_language_variant")
    example = None
    if recipe in ("rule_application", "component_explanation"):
        examples = [e for e in item.examples if e.latin == target.example]
        if len(examples) != 1:
            raise ValueError("missing_or_ambiguous_example")
        example = examples[0]
    mnemonic = None
    if recipe == "mnemonic_recall" or target.mnemonic_key is not None:
        alternatives = [
            m
            for m in item.mnemonics
            if m.language == settings.language and (target.mnemonic_key is None or m.key == target.mnemonic_key)
        ]
        if len(alternatives) != 1:
            raise ValueError("missing_language_variant" if not alternatives else "ambiguous_mnemonic")
        mnemonic = alternatives[0]
    if recipe == "formation_rule_recall":
        if target.formation is None:
            raise ValueError("missing_formation_target")
        if mnemonic is not None and target.mnemonic_side == "front":
            raise ValueError("mnemonic_cue_not_allowed")
        prompt = ("Wie bildet man " if de else "How do you form ") + target.formation + "?"
    elif recipe == "component_explanation":
        assert example is not None
        prompt = (
            (
                "Erkläre Basis/Stamm, Bildungszeichen und Endung in "
                if de
                else "Explain the base/stem, formation markers and ending in "
            )
            + example.latin
            + "."
        )
    elif recipe == "mnemonic_recall":
        assert mnemonic is not None
        prompt = (
            "Welche Regel hilft diese Eselsbrücke zu erinnern? "
            if de
            else "Which rule does this mnemonic help recall? "
        ) + mnemonic.text
    else:
        assert example is not None
        prompt = ("Wende die Regel auf diese Form an: " if de else "Apply the rule to this form: ") + example.latin
    answer = example.explanation if example else item.rule.text
    front = _paragraph("Aufgabe" if de else "Prompt", prompt)
    back = _paragraph("Referenzantwort" if de else "Reference answer", answer)
    if example:
        back += _paragraph("Regel" if de else "Rule", item.rule.text)
    back += _paragraph("Geltungsbereich" if de else "Applicability", item.rule.applicability)
    for caveat in item.rule.caveats:
        back += _paragraph("Einschränkung" if de else "Caveat", caveat)
    evidence = (*item.rule.evidence, *(example.evidence if example else ()))
    for ref in evidence:
        back += _paragraph("Beleg" if de else "Evidence", f"{ref.source} / {ref.reference} / {ref.version}: {ref.text}")
    provenance = item.provenance
    back += _paragraph(
        "Herkunft" if de else "Provenance",
        f"{provenance.document} / {provenance.section} / {provenance.reference} ({provenance.wording})",
    )
    for identity in target.related_production:
        back += _paragraph("Verknüpfte Produktionsübung" if de else "Related production exercise", identity)
    if mnemonic and recipe != "mnemonic_recall":
        aid = _paragraph("Eselsbrücke" if de else "Mnemonic", mnemonic.text)
        if target.mnemonic_side == "front":
            front += aid
        else:
            back += aid
    if mnemonic:
        p = mnemonic.provenance
        back += _paragraph(
            "Herkunft der Eselsbrücke" if de else "Mnemonic provenance",
            f"{p.document} / {p.section} / {p.reference} ({p.wording})",
        )

    def wrap(body: str) -> str:
        colors = "background:#fff;color:#171717" if settings.theme == "light" else "background:#202020;color:#eee"
        return f'<div class="curated-card" lang="{settings.language}" style="{colors};padding:1.25rem">{body}</div>'

    if settings.acceptance_layout:
        if approved:
            checks = ("rule: approved", *(("mnemonic: approved",) if mnemonic else ()), "rendered_card: approved")
        else:
            checks = (
                "Regel und exakte Darstellung benötigen menschliche Freigabe."
                if de
                else "Rule and exact rendering require human approval.",
            )
        back += Presentation(
            derive_card_semantic_key(recipe, target.key),
            "human" if approved else "curated",
            (),
            ("curated", *(f"{e.source} / {e.reference} ({e.version})" for e in evidence)),
            checks,
            (
                "Referenzantwort; Selbstbeurteilung, keine empirische Richtigkeitsquote."
                if de
                else "Reference answer; self-assessment, no empirical accuracy estimate.",
                *item.rule.caveats,
            ),
        ).answer(settings.language)
    return wrap(front), wrap(back), (mnemonic,) if mnemonic else ()


def render_cards(
    data: CuratedInput, plan: RecipePlan, history: tuple[Approval, ...], settings: Settings
) -> tuple[tuple[CuratedCard, ...], tuple[dict[str, str], ...]]:
    validate_history(history)
    items = {item.key: item for item in data.items}
    cards = []
    skipped = []
    selected = tuple(recipe for recipe in settings.recipes if recipe in CURATED_RECIPES)
    processed = 0
    for target in plan.targets:
        if (
            settings.profile
            and settings.profile.selection.entries
            and target.item_key not in settings.profile.selection.entries
        ):
            continue
        item = items.get(target.item_key)
        limit = settings.profile.processing.max_source_units if settings.profile else None
        deferred = limit is not None and processed >= limit
        processed += 1
        for recipe in selected:
            base = {"item_key": target.item_key, "target": target.key, "recipe": recipe}
            try:
                if deferred:
                    raise ValueError("processing_limit")
                if item is None:
                    raise ValueError("missing_item")
                if target.analysis != "supported":
                    raise ValueError(target.analysis)
                front, back, mnemonics = _render(item, target, recipe, settings)
                # Human review binds the exact prospective learning answer, not
                # the neutral administrative preview or approval-history IDs.
                approval_back = (
                    _render(item, target, recipe, settings, approved=True)[1] if settings.acceptance_layout else back
                )
                note_id = derive_latinitas_id("curated", item.key, source_scope=data.source_scope)
                key = derive_card_semantic_key(recipe, target.key)
                rule_fp = rule_fingerprint(item)
                mnemonic_subjects = tuple(("mnemonic", m.key, mnemonic_fingerprint(item, m)) for m in mnemonics)
                fingerprint = digest(
                    {
                        "semantic_key": key,
                        "language": settings.language,
                        "prompt": front,
                        "answer": approval_back,
                        "rule_fingerprint": rule_fp,
                        "mnemonics": tuple(fp for _, _, fp in mnemonic_subjects),
                        "template_sha256": layout_digest("curated") if settings.acceptance_layout else TEMPLATE_SHA256,
                        "presentation": {"theme": settings.theme},
                    }
                )
                subjects = [
                    ("rule", item.key, rule_fp),
                    *mnemonic_subjects,
                    ("rendered_card", key, fingerprint),
                ]
                requirements = []
                for subject, subject_key, fp in subjects:
                    matches = matching_decisions(
                        data.source_scope, item.key, subject, subject_key, settings.language, fp, history
                    )
                    status = "conflict" if len(matches) > 1 else matches[0].decision if matches else "unreviewed"
                    requirements.append(
                        Requirement(
                            subject, subject_key, settings.language, fp, status, tuple(m.approval_id for m in matches)
                        )
                    )
                accepted = all(r.status == "approved" for r in requirements)
                cards.append(
                    CuratedCard(
                        note_id,
                        key,
                        item.key,
                        recipe,
                        front,
                        approval_back if accepted else back,
                        fingerprint,
                        tuple(requirements),
                        approval_back if settings.acceptance_layout else None,
                    )
                )
            except ValueError as error:
                skipped.append({**base, "reason": str(error)})
    # One slot per objective type on an item note. Never pick one of conflicting targets silently.
    slot_counts = Counter((c.note_id, c.recipe) for c in cards)
    conflicts = {slot for slot, count in slot_counts.items() if count > 1}
    for card in cards:
        if (card.note_id, card.recipe) in conflicts:
            skipped.append({"item_key": card.item_key, "recipe": card.recipe, "reason": "conflicting_slot"})
    for recipe in selected:
        if not any(c.recipe == recipe for c in cards) and not any(s["recipe"] == recipe for s in skipped):
            skipped.append({"recipe": recipe, "reason": "missing_selected_content"})
    return tuple(c for c in cards if (c.note_id, c.recipe) not in conflicts), tuple(skipped)


def preview_report(
    data: CuratedInput, plan: RecipePlan, history: tuple[Approval, ...], settings: Settings
) -> dict[str, object]:
    cards, skipped = render_cards(data, plan, history, settings)
    return {
        "source_scope": data.source_scope,
        "effective_configuration": settings.snapshot,
        "cards": [card.report() for card in cards],
        "skipped": skipped,
        "layout": {
            "id": "acceptance-1" if settings.acceptance_layout else "curated-v1",
            "template_sha256": layout_digest("curated") if settings.acceptance_layout else TEMPLATE_SHA256,
            "templates": [asdict(t) for t in CURATED_LAYOUT],
            "fields": (*CURATED_FIELDS, "Personal Notes", *(ACCEPTANCE_FIELDS if settings.acceptance_layout else ())),
            "css": stylesheet() if settings.acceptance_layout else REFERENCE_CARD_CSS,
        },
        "grading": "Selbstbeurteilung; Referenzantworten, keine Zeichenkettengleichheit."
        if settings.language == "de"
        else "Self-assessment; reference answers, not exact-string grading.",
        "summary": "Karten benötigen alle aufgeführten menschlichen Freigaben."
        if settings.language == "de"
        else "Cards require every listed human approval.",
    }


def write_preview(
    path: Path, data: CuratedInput, plan: RecipePlan, history: tuple[Approval, ...], settings: Settings
) -> None:
    cards, _ = render_cards(data, plan, history, settings)
    title = "Kuratierte Karten · Prüfung" if settings.language == "de" else "Curated cards · review"
    body = (
        f"<h1>{title}</h1><p>"
        + ("Prüfansicht, keine Freigabe." if settings.language == "de" else "Inspection view, not approval.")
        + "</p>"
    )
    labels = {
        "rule_application": ("Regelanwendung", "Rule application"),
        "formation_rule_recall": ("Bildungsregel abrufen", "Formation-rule recall"),
        "component_explanation": ("Bestandteile erklären", "Component explanation"),
        "mnemonic_recall": ("Eselsbrücke abrufen", "Mnemonic recall"),
    }
    for card in cards:
        label = labels[card.recipe][0 if settings.language == "de" else 1]
        if settings.language == "de":
            status = "Freigegeben" if card.reason == "accepted" else "Zurückgehalten"
        else:
            status = "Approved" if card.reason == "accepted" else "Withheld"
        body += f"<section><h2>{label}</h2><p>{status} · <code>{escape(card.reason)}</code></p>"
        body += f'<div class="sides"><article>{card.front_html}</article><article>{card.back_html}</article></div>'
        if card.approval_back_html is not None and card.reason != "accepted":
            proposed = (
                "Vorgeschlagene Lernantwort bei Freigabe; dies ist keine Freigabe."
                if settings.language == "de"
                else "Proposed learning answer if approved; this is not approval."
            )
            body += f"<p>{proposed}</p><article>{card.approval_back_html}</article>"
        body += "</section>"
    html = f'<!doctype html><html lang="{settings.language}"><meta charset="utf-8">'
    html += '<meta name="viewport" content="width=device-width,initial-scale=1">'
    html += f"<title>{title}</title><style>{stylesheet() if settings.acceptance_layout else REFERENCE_CARD_CSS}"
    html += "body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:17px/1.5 sans-serif;background:#eee;color:#222}"
    html += ".sides{display:grid;grid-template-columns:1fr 1fr;gap:1rem}article{min-width:0;overflow-wrap:anywhere}"
    html += "section{margin:2rem 0}h2{font-size:1rem}@media(max-width:700px){.sides{grid-template-columns:1fr}}"
    html += f"</style><body>{body}</body></html>"
    with path.open("x") as stream:
        stream.write(html)
