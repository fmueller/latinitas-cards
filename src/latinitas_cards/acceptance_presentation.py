"""Opt-in answer presentation; never acceptance authority or an Anki installer."""

from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Literal

from .acceptance_records import canonical, decode_json, digest
from .reference_templates import REFERENCE_CARD_CSS

ACCEPTANCE_FIELDS = ("Acceptance Data", "Evidence Summary", "Acceptance Receipt")
ACCEPTANCE_CSS = """
.acceptance-panel { margin-top:1em; padding:.75em; border:1px solid currentColor;
  text-align:left; font-size:.85em; overflow-wrap:anywhere; }
.acceptance-status { font-weight:600; }
.acceptance-panel summary { cursor:pointer; padding:.4em 0; margin:.6em 0; }
.acceptance-panel summary:focus-visible { outline:3px solid currentColor; outline-offset:3px; }
.acceptance-panel p { margin:.4em 0; }
"""


def layout_digest(family: Literal["grammar", "curated"]) -> str:
    if family == "grammar":
        from .grammar_layout import GRAMMAR_FIELDS, LAYOUT_SHA256

        base = LAYOUT_SHA256
        fields = GRAMMAR_FIELDS
    else:
        from .curated_cards import CURATED_FIELDS, TEMPLATE_SHA256

        base = TEMPLATE_SHA256
        fields = (*CURATED_FIELDS, "Personal Notes")
    return digest(("acceptance-1", family, base, (*fields, *ACCEPTANCE_FIELDS), ACCEPTANCE_CSS))


def _installation(profile_path: Path, family: Literal["grammar", "curated"]) -> tuple[Path, dict[str, object]]:
    path = profile_path.resolve()
    return path.with_name(path.name + ".acceptance-1.json"), {
        "format": "latinitas-layout-installation",
        "version": 1,
        "profile": str(path),
        "family": family,
        "layout": {"id": "acceptance-1", "version": 1, "template_sha256": layout_digest(family)},
    }


def confirm_installation(profile_path: Path, family: Literal["grammar", "curated"]) -> Path:
    from .pipeline_profile import load_pipeline_profile

    profile = load_pipeline_profile(profile_path)
    path, expected = _installation(profile_path, family)
    if profile.layout.model_dump() != expected["layout"]:
        raise ValueError("exact acceptance-1 family migration-plan digest required")
    if path.is_symlink():
        raise ValueError("symlink installation confirmation")
    if path.exists():
        if canonical(decode_json(path.read_bytes())) != canonical(expected):
            raise ValueError("conflicting installation confirmation; inspect and resolve manually")
    else:
        with path.open("xb") as stream:
            stream.write(canonical(expected))
    return path


def require_installation(
    identity: str, version: int, fingerprint: str, profile_path: Path | None, family: Literal["grammar", "curated"]
) -> None:
    if (identity, version, fingerprint) != ("acceptance-1", 1, layout_digest(family)) or profile_path is None:
        raise ValueError("explicit acceptance-1 installation confirmation and exact migration-plan digest required")
    path, expected = _installation(profile_path, family)
    if path.is_symlink() or not path.is_file() or canonical(decode_json(path.read_bytes())) != canonical(expected):
        raise ValueError(
            "installation confirmation required; after manual installation use pipeline confirm-layout --installed"
        )


@dataclass(frozen=True)
class Presentation:
    card_key: str
    acceptance: str
    decision_ids: tuple[str, ...]
    basis: tuple[str, ...]
    checks: tuple[str, ...]
    limitations: tuple[str, ...]

    def answer(self, language: str) -> str:
        de = language == "de"
        labels = {
            "human": "Menschlich geprüft" if de else "Human reviewed",
            "automatic": "Automatisch geprüft" if de else "Automatically checked",
            "mixed": "Menschlich und automatisch geprüft" if de else "Human and automatic checks",
            "none": "Nicht freigegeben" if de else "Not accepted",
            "curated": "Freigabeweg: menschlich" if de else "Approval route: human",
        }
        title = "Warum diese Antwort?" if de else "Why this answer?"
        terms = (
            {
                "source": "Quelle",
                "rule": "Regel",
                "library": "Bibliothek",
                "model": "Modell",
                "curated": "Kuratierte Quelle",
                "human": "Mensch",
                "blind_model": "Unabhängiger Modellcheck",
                "semantic_comparison": "Bedeutungsvergleich",
                "pass": "bestanden",
                "approved": "freigegeben",
                "unreviewed": "ungeprüft",
                "rendered_card": "Kartendarstellung",
                "mnemonic": "Eselsbrücke",
            }
            if de
            else {}
        )
        basis = tuple(terms.get(v, v) for v in self.basis[:6])
        checks = tuple(": ".join(terms.get(part, part) for part in value.split(": ")) for value in self.checks[:6])
        parts = (
            ("Grundlage" if de else "Basis", basis),
            ("Prüfungen" if de else "Checks", checks),
            ("Grenzen" if de else "Limitations", self.limitations),
        )
        body = "".join(f"<p><strong>{label}:</strong> {escape('; '.join(values))}</p>" for label, values in parts)
        return (
            f'<section class="acceptance-panel" lang="{escape(language, quote=True)}">'
            f'<small class="acceptance-status">{labels[self.acceptance]}</small>'
            f"<details><summary>{title}</summary>{body}</details></section>"
        )


def note_tags(cards: tuple[Presentation, ...], inherited: tuple[str, ...]) -> tuple[str, ...]:
    if any(t.casefold().startswith(("latinitas::acceptance::", "latinitas::basis::")) for t in inherited):
        raise ValueError("unowned acceptance namespace collision; explicit destination adoption required")
    origins = {"human" if c.acceptance == "curated" else c.acceptance for c in cards}
    if "none" in origins:
        origin = "incomplete"
    elif len(origins) == 1:
        origin = next(iter(origins))
    else:
        origin = "mixed"
    categories = sorted({b for c in cards for b in c.basis if b in ("source", "rule", "library", "model", "curated")})
    return (f"latinitas::acceptance::{origin}", *(f"latinitas::basis::{b}" for b in categories))


def metadata(cards: tuple[Presentation, ...], language: str, run_id: str) -> dict[str, str]:
    return {
        "Acceptance Data": canonical({"version": 1, "cards": [asdict(c) for c in cards]}).decode(),
        "Evidence Summary": "".join(c.answer(language) for c in cards),
        "Acceptance Receipt": canonical(
            {"version": 1, "run_id": run_id, "card_keys": [c.card_key for c in cards]}
        ).decode(),
    }


def stylesheet() -> str:
    return REFERENCE_CARD_CSS + ACCEPTANCE_CSS
