"""Reader-facing citations and independently inspectable source provenance.

The resolver allowlist is verified against the two pinned Perseus readings.
It identifies an online edition/passage, not the revision of the local XML.
"""

import html
from dataclasses import dataclass

from .canonical import canonical_json
from .corpus_facts import PassageFacts
from .profile_setup import encode_unsafe_controls
from .terminology import card_text


@dataclass(frozen=True)
class _VerifiedReading:
    source_sha256: str
    author: str
    title: str
    citation_units: tuple[str, ...]
    native_ref: str
    passage_hash: str
    url: str
    label_key: str


# Source/edition bindings and exact passage URLs verified in Scaife, 2026-10-31.
# Other references stay nonclickable until independently verified.
_READINGS = {
    "urn:cts:latinLit:phi0448.phi001.perseus-lat2": _VerifiedReading(
        source_sha256="sha256:15ec3ef2a36352742ddb75d4f7da23a143899b6b60ee3a7ca3192d4917ca1568",
        author="Julius Caesar",
        title="De bello Gallico",
        citation_units=("book", "chapter", "section"),
        native_ref="1.1.1",
        passage_hash="sha256:da894d75b462152ae8e76d5b9b199f8247a4aadd8a1fc4e4bc495f65706c686f",
        url="https://scaife.perseus.org/reader/urn:cts:latinLit:phi0448.phi001.perseus-lat2:1.1.1/",
        label_key="caesar",
    ),
    "urn:cts:latinLit:phi0690.phi003.perseus-lat2": _VerifiedReading(
        source_sha256="sha256:e639e6a0e27427ef28031850c4d73aa2b63b9f2cf8a5cf584b37685c52fb874c",
        author="P. Vergilius Maro",
        title="Aeneid",
        citation_units=("book", "line"),
        native_ref="1.11",
        passage_hash="sha256:4c409b51d00330fd430f6d3931e385e97b618b3f0ee632f081bf36aee63bd92e",
        url="https://scaife.perseus.org/reader/urn:cts:latinLit:phi0690.phi003.perseus-lat2:1.11/",
        label_key="virgil",
    ),
}


def _verified_source(facts: PassageFacts) -> _VerifiedReading | None:
    reading = _READINGS.get(facts.work.edition_urn or "")
    if (
        reading
        and facts.header.adapter == "tei/1"
        and facts.header.source.sha256 == reading.source_sha256
        and (facts.work.author, facts.work.title) == (reading.author, reading.title)
    ):
        return reading
    return None


def citation_label(facts: PassageFacts, language: str) -> str:
    work, passage = facts.work, facts.passage
    author, title = work.author, work.title
    reading = _verified_source(facts)
    if reading:
        author = card_text(f"citation_{reading.label_key}_author", language)
        title = card_text(f"citation_{reading.label_key}_work", language)
    units = {"book", "chapter", "section", "line", "verse", "passage"}
    location = ", ".join(
        f"{card_text('citation_' + part.unit, language) if part.unit in units else part.unit} {part.value}"
        for part in passage.citation.parts
    )
    return f"{author or card_text('citation_unknown_author', language)}, {title or work.work}, {location}"


def source_provenance(facts: PassageFacts) -> dict[str, object]:
    header, work, passage = facts.header, facts.work, facts.passage
    missing = [
        f"missing_{name}"
        for name, value in (
            ("author", work.author),
            ("work_title", work.title),
            ("edition", header.edition),
            ("edition_urn", work.edition_urn),
            ("source_url", header.url),
            ("revision", header.revision),
            ("source_license", header.license),
        )
        if not value
    ]
    derived = (
        f"{work.edition_urn}:{passage.native_ref}"
        if work.edition_urn and passage.citation.scheme == "tei-cts"
        else None
    )
    resolver = None
    reading = _verified_source(facts)
    if reading and (
        passage.passage_hash == reading.passage_hash
        and tuple(part.unit for part in passage.citation.parts) == reading.citation_units
        and passage.native_ref == passage.citation.display == reading.native_ref
        and derived is not None
    ):
        resolver = {"url": reading.url, "scope": "edition_passage", "revision_matches_source": False}
    if resolver is None:
        missing.append("no_verified_passage_link")
    return {
        "format": "latinitas-source-provenance",
        "version": 1,
        "header": header.model_dump(mode="json"),
        "work": work.model_dump(mode="json"),
        "passage": {
            "passage_id": passage.passage_id,
            "work": passage.work,
            "native_ref": passage.native_ref,
            "citation": passage.citation.model_dump(mode="json"),
            "passage_hash": passage.passage_hash,
        },
        "derived_cts_reference": derived,
        "resolver": resolver,
        "limitations": missing,
    }


def render_citation(facts: PassageFacts, language: str) -> str:
    provenance = source_provenance(facts)
    label = _escape(citation_label(facts, language))
    resolver = provenance["resolver"]
    if isinstance(resolver, dict):
        label = f'<a href="{_escape(str(resolver["url"]))}">{label}</a>'
    limitations = provenance["limitations"]
    assert isinstance(limitations, list)
    notices = [card_text("citation_" + limitation, language) for limitation in limitations]
    if resolver is not None:
        notices.append(card_text("citation_resolver_revision", language))
    payload = "".join(
        f"\\u{ord(char):04x}" if 0x7F <= ord(char) <= 0x9F else char for char in canonical_json(provenance)
    )
    return (
        label
        + '<details class="corpus-provenance"><summary>'
        + _escape(card_text("citation_provenance", language))
        + "</summary>"
        + "<p>"
        + _escape("; ".join(notices))
        + "</p><pre>"
        + html.escape(payload, quote=True)
        + "</pre></details>"
    )


def _escape(value: str) -> str:
    return html.escape(encode_unsafe_controls(value), quote=True)
