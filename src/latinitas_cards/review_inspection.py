"""Read-only inspection of known field contracts, not an Anki template emulator."""

from __future__ import annotations

import csv
import html
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .cards import TEMPLATE_REGISTRY, CardSlot
from .claim_sidecar import Diagnostic, claim_report
from .form_parsing import PARSING_SLOT, _escape, generate_parsing, parsing_export_answer
from .html_text import source_html_to_text
from .managed_layouts import known_layouts
from .preview_export import PrincipalPartExportResult, prepare_principal_part_export
from .profile import DeckProfile, load_profile
from .review_adapters import ReviewState, parsing_review, principal_review
from .review_packages import read_review_package


@dataclass(frozen=True)
class ReviewWorkspace:
    state: ReviewState
    fields: tuple[Mapping[str, str], ...]
    slots: tuple[CardSlot, ...]
    cards: tuple[dict[str, object], ...]
    profile: DeckProfile
    principal: PrincipalPartExportResult | None = None
    note_type: str | None = None


def card_report(
    fields: Mapping[str, str], slot: CardSlot, claims: Sequence[dict[str, object]], reasons: Sequence[str] = ()
) -> dict[str, object]:
    eligible = fields.get(slot.enabled_field) == "1"
    prompt = fields.get(slot.prompt_field, "") if eligible else ""
    answer = fields.get(slot.answer_field, "") if eligible else ""
    if eligible and slot.recipe == "form_parsing":
        answer = re.sub(r'<span hidden class="parsing-review">.*?</span>', "", answer, flags=re.DOTALL)
        answer = prompt + "<hr>" + answer
    elif eligible:
        parts = re.sub(
            r'<span hidden class="(?:source-extraction|principal-part-review)">.*?</span>',
            "",
            fields.get("Principal Parts", ""),
            flags=re.DOTALL,
        )
        answer += "<br>" + fields.get("Lemma", "") + " · " + parts
        if fields.get("Meaning"):
            answer += " · " + fields["Meaning"]
    if eligible and fields.get("Personal Notes"):
        answer += "<br>" + fields["Personal Notes"]
    return {
        "identity": fields.get("LatinitasID", ""),
        "recipe": slot.recipe,
        "slot": slot.ordinal,
        "semantic_key": slot.semantic_key,
        "eligible": eligible,
        "prompt": source_html_to_text(prompt),
        "answer": source_html_to_text(answer),
        "claims": list(claims),
        "reasons": list(reasons),
    }


def workspace(source: Path, profile: Path | None) -> ReviewWorkspace:
    if profile is None:
        data, state = parsing_review(source)
        results = generate_parsing(data)
        fields = tuple(
            {
                "LatinitasID": r["latinitas_id"],
                "Form": _escape(r["form"]),
                "Context": _escape(r["context"]),
                "ParsingEnabled": "1" if r["eligible"] else "",
                "ParsingPrompt": r["prompt"],
                "ParsingAnswer": parsing_export_answer(r),
            }
            for r in results
        )
        cards = tuple(
            card_report(
                f, PARSING_SLOT, [claim_report(b) for b in state.claims if b.identity == r["latinitas_id"]], r["skips"]
            )
            for f, r in zip(fields, results, strict=True)
        )
        return ReviewWorkspace(state, fields, (PARSING_SLOT,), cards, data.profile)
    effective = load_profile(profile)
    baseline = prepare_principal_part_export(source, effective, profile_path=profile, read_review_sidecar=False)
    state = principal_review(baseline)
    result = prepare_principal_part_export(source, effective, profile_path=profile)
    fields = tuple(dict(n.to_anki_fields()) for n in result.generation.notes)
    cards = tuple(
        card_report(
            f,
            slot,
            [claim_report(b) for b in state.claims if b.identity == f["LatinitasID"]],
            [r.reason for r in comparison.roles if r.status != "present"],
        )
        for f, comparison in zip(fields, result.generation.principal_part_comparisons, strict=True)
        for slot in TEMPLATE_REGISTRY
    )
    return ReviewWorkspace(state, fields, TEMPLATE_REGISTRY, cards, effective, result)


def read_export_csv(
    path: Path, *, note_type: str | None = None, deck: str | None = None
) -> tuple[Mapping[str, str], ...]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines(keepends=True)
    directives: dict[str, str] = {}
    index = 0
    while index < len(lines) and lines[index].startswith("#"):
        key, separator, value = lines[index].rstrip("\r\n").partition(":")
        if not separator or key in directives:
            raise ValueError("invalid CSV directives")
        directives[key] = value
        index += 1
    if note_type is not None and directives.get("#notetype") != note_type:
        raise ValueError("CSV note-type identity differs from current profile")
    if deck is not None and directives.get("#deck") != deck:
        raise ValueError("CSV deck differs from current profile")
    columns = tuple(directives.get("#columns", "").split(","))
    if (
        columns not in {layout.columns for layout in known_layouts()}
        or directives.get("#separator") != "Comma"
        or directives.get("#html") != "true"
    ):
        raise ValueError("unsupported CSV layout; expected documented Latinitas fields/directives")
    records: list[Mapping[str, str]] = []
    try:
        for row in csv.reader(io.StringIO("".join(lines[index:])), delimiter=",", strict=True):
            if len(row) != len(columns):
                raise ValueError("CSV row does not match its declared fields")
            records.append(dict(zip(columns, row, strict=True)))
    except csv.Error as error:
        raise ValueError(f"Malformed CSV: {error}") from error
    return tuple(records)


def inspect_workspace(work: ReviewWorkspace, artifact: Path | None) -> dict[str, object]:
    diagnostics = list(work.state.diagnostics)
    cards = list(work.cards)
    if artifact is not None:
        actual: tuple[Mapping[str, str], ...]
        if artifact.suffix.lower() in {".apkg", ".colpkg"}:
            package = read_review_package(artifact)
            actual = package.fields
            diagnostics.extend(package.diagnostics)
            if package.slots != work.slots:
                diagnostics.append(Diagnostic("slot", "Artifact recipe slots differ from current source"))
            if work.note_type and package.note_types != (work.note_type,):
                diagnostics.append(Diagnostic("identity", "Package note-type differs from current phrase profile"))
        else:
            actual = read_export_csv(
                artifact,
                note_type=work.note_type
                or (work.profile.generated_note_type if work.principal else "Latinitas Contextual Form Parsing v1"),
                deck=work.profile.target_deck,
            )
        expected = {
            f["LatinitasID"]: f
            for f in work.fields
            if any(f.get(s.enabled_field) == "1" for s in work.slots)
            and (
                work.principal is None or f["LatinitasID"] in {n.latinitas_id for n in work.principal.exportable_notes}
            )
        }
        seen: set[str] = set()
        cards = []
        for fields in actual:
            identity = fields["LatinitasID"]
            current = expected.get(identity)
            if identity in seen or current is None:
                diagnostics.append(Diagnostic("identity", f"Duplicate or unmatched identity: {identity}"))
            seen.add(identity)
            if current is not None:
                for field in set(fields) | set(current):
                    value = fields.get(field, "")
                    if field not in {"Tags", "Personal Notes"} and current.get(field, "") != value:
                        diagnostics.append(Diagnostic("drift", f"Drifted field {field} for {identity}"))
                        if any(field in {s.prompt_field, s.answer_field} for s in work.slots) and value:
                            diagnostics.append(Diagnostic("unreviewed", f"Unreviewed asserted content in {field}"))
                        if any(field == s.enabled_field for s in work.slots):
                            diagnostics.append(Diagnostic("slot", f"Drifted eligibility/slot in {field}"))
            for slot in work.slots:
                claims = [claim_report(b) for b in work.state.claims if b.identity == identity]
                reasons = [str(c["reason"]) for c in claims if c["status"] != "accepted"]
                cards.append(card_report(fields, slot, claims, reasons))
        if set(expected) != seen:
            diagnostics.append(Diagnostic("identity", "Export identities differ from current eligible notes"))
    return {"cards": cards, "diagnostics": [asdict(d) for d in diagnostics]}


def inspect_artifact(path: Path) -> dict[str, object]:
    """Artifact provenance is display evidence, never sufficient for a passing check."""
    diagnostics = [
        Diagnostic("unbound", "Current source/profile/sidecar required for check; artifact claims are unverified")
    ]
    if path.suffix.lower() in {".apkg", ".colpkg"}:
        package = read_review_package(path)
        records: tuple[Mapping[str, str], ...] = package.fields
        slots = package.slots
        diagnostics.extend(package.diagnostics)
    else:
        records = read_export_csv(path)
        slots = (PARSING_SLOT,) if records and "ParsingEnabled" in records[0] else TEMPLATE_REGISTRY
    cards = []
    for fields in records:
        claims: list[dict[str, object]] = []
        field = fields.get("ParsingAnswer", fields.get("Principal Parts", ""))
        match = re.search(r'<span hidden class="(parsing-review|principal-part-review)">(.*?)</span>', field, re.DOTALL)
        if match:
            metadata = json.loads(html.unescape(match[2]))
            if match[1] == "parsing-review":
                claims.extend(metadata["claims"])
                claims.extend(metadata["withheld"])
            else:
                for role in metadata["roles"]:
                    claims.extend(role["claims"])
        for slot in slots:
            cards.append(
                card_report(fields, slot, claims, ["Artifact backing evidence is unverified without current source"])
            )
    return {"cards": cards, "diagnostics": [asdict(d) for d in diagnostics]}
