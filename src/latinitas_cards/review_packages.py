"""Bounded offline support for documented legacy-JSON Latinitas package layouts."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import zstandard
from pydantic import BaseModel

from .cards import TEMPLATE_REGISTRY, CardSlot
from .claim_sidecar import Diagnostic
from .form_parsing import PARSING_BACK, PARSING_FIELDS, PARSING_FRONT, PARSING_NOTE_TYPE, PARSING_SLOT
from .reference_templates import REFERENCE_CARD_TEMPLATES, REFERENCE_NOTE_TYPE_FIELDS
from .sources import _pick_collection_database_name, _read_zip_member, read_source_records


class PackageField(BaseModel):
    name: str


class PackageTemplate(BaseModel):
    ord: int
    name: str
    qfmt: str
    afmt: str


class PackageModel(BaseModel):
    name: str
    type: Literal[0] = 0
    flds: tuple[PackageField, ...]
    tmpls: tuple[PackageTemplate, ...]


@dataclass(frozen=True)
class PackageInspection:
    fields: tuple[dict[str, str], ...]
    slots: tuple[CardSlot, ...]
    diagnostics: tuple[Diagnostic, ...]


def read_review_package(path: Path) -> PackageInspection:
    records = read_source_records(path)
    diagnostics: list[Diagnostic] = []
    fields: list[dict[str, str]] = []
    all_slots: tuple[CardSlot, ...] = ()
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as directory:
        name, compressed = _pick_collection_database_name(archive.namelist(), path)
        payload = _read_zip_member(archive, name, path)
        if compressed:
            payload = zstandard.ZstdDecompressor().decompress(payload, max_output_size=512 * 1024 * 1024)
        database = Path(directory) / "collection.anki2"
        database.write_bytes(payload)
        try:
            with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as con:
                row = con.execute("SELECT models FROM col ORDER BY id LIMIT 1").fetchone()
                if row is None or not row[0]:
                    raise ValueError("normalized/protobuf package layouts are unsupported")
                raw_models = json.loads(row[0])
                if not isinstance(raw_models, dict):
                    raise ValueError("package model metadata must be an object")
                models = {int(mid): PackageModel.model_validate(model) for mid, model in raw_models.items()}
                mids = {int(nid): int(mid) for nid, mid in con.execute("SELECT id,mid FROM notes")}
                ordinals: dict[int, list[int]] = {}
                for nid, ordinal in con.execute("SELECT nid,ord FROM cards"):
                    if int(nid) not in mids:
                        diagnostics.append(Diagnostic("identity", "Package has an orphan card"))
                    ordinals.setdefault(int(nid), []).append(int(ordinal))
                for record in records:
                    nid = record.provenance.note_id
                    if nid is None:
                        raise ValueError("package note identity unavailable")
                    model = models.get(mids[nid])
                    if model is None:
                        raise ValueError("package model unavailable")
                    names = tuple(f.name for f in model.flds)
                    slots: tuple[CardSlot, ...]
                    templates: tuple[tuple[int, str, str, str], ...]
                    if names == PARSING_FIELDS and model.name == PARSING_NOTE_TYPE:
                        slots = (PARSING_SLOT,)
                        templates = ((0, PARSING_SLOT.template_name, PARSING_FRONT, PARSING_BACK),)
                    elif names == REFERENCE_NOTE_TYPE_FIELDS:
                        slots = TEMPLATE_REGISTRY
                        templates = tuple(
                            (t.slot.ordinal, t.slot.template_name, t.front, t.back) for t in REFERENCE_CARD_TEMPLATES
                        )
                    else:
                        diagnostics.append(Diagnostic("unsupported", "Unsupported package fields/layout"))
                        continue
                    actual_templates = tuple((t.ord, t.name, t.qfmt, t.afmt) for t in model.tmpls)
                    if actual_templates != templates:
                        diagnostics.append(
                            Diagnostic(
                                "unsupported",
                                "Unsupported package templates; arbitrary JavaScript/cloze/conditions not evaluated",
                            )
                        )
                    if all_slots and all_slots != slots:
                        diagnostics.append(Diagnostic("unsupported", "Mixed Latinitas recipe layouts unsupported"))
                        continue
                    all_slots = slots
                    enabled = sorted(s.ordinal for s in slots if record.fields.get(s.enabled_field) == "1")
                    if sorted(ordinals.get(nid, [])) != enabled:
                        diagnostics.append(Diagnostic("slot", f"Package card identity/slot mismatch for note {nid}"))
                    fields.append(dict(record.fields))
        except (sqlite3.DatabaseError, ValueError, TypeError) as error:
            diagnostics.append(Diagnostic("unsupported", f"Unsupported package layout: {error}"))
    return PackageInspection(tuple(fields), all_slots, tuple(diagnostics))
