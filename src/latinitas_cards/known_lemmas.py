"""Read-only known-lemma provenance from confirmed deck profiles."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .canonical import digest_bytes
from .corpus_model import latin_fold
from .html_text import source_html_to_text
from .identity import IdentityError, ResolvedSourceIdentity, resolve_source_identity
from .manifest import CsvIdentityManifest, reconcile_csv_manifest
from .profile import load_profile
from .sources import read_source_records


@dataclass(frozen=True, slots=True)
class KnownLemmaSkip:
    location: str
    message: str


@dataclass(frozen=True, slots=True)
class KnownLemmas:
    lemmas: Mapping[str, tuple[ResolvedSourceIdentity, ...]]
    skipped: tuple[KnownLemmaSkip, ...]
    source_digest: str
    profile_digest: str


def load_known_lemmas(
    profile_path: str | Path,
    source_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> KnownLemmas:
    """Extract the first lexical word without allocating identities or saving state.

    CSV sources require their existing scoped identity state. Manifest-backed
    rows without a safe persisted assignment are reported and skipped, just
    like records lacking readable lexical text or a stable identity. Digests
    identify the source and confirmed profile file bytes, not derived content.
    """
    profile_file, source = Path(profile_path), Path(source_path)
    profile = load_profile(profile_file)
    strategy = profile.source_identity.strategy
    skipped: list[KnownLemmaSkip] = []
    records = read_source_records(
        source,
        source_id_field=profile.source_identity.field if strategy == "source_id_field" else None,
        note_type=profile.note_type,
        on_record_error=lambda location, message: skipped.append(KnownLemmaSkip(location, message)),
    )
    scope = None
    assignments: dict[int, str] = {}
    if source.suffix.lower() == ".csv":
        state_file = Path(manifest_path) if manifest_path is not None else Path(f"{source}.latinitas.json")
        if not state_file.is_file():
            raise ValueError("Known-lemma CSV input requires existing source scope state")
        state = CsvIdentityManifest.load(state_file)
        if not state.is_scoped:
            raise ValueError("Known-lemma CSV input requires a committed source scope")
        scope = state.source_scope
        if strategy == "manifest":
            assignments = reconcile_csv_manifest(records, state).identities_by_row
    elif manifest_path is not None or strategy == "manifest":
        raise ValueError("Manifest identity state is supported only for CSV input")

    lemmas: dict[str, list[ResolvedSourceIdentity]] = {}
    for index, record in enumerate(records):
        try:
            identity = resolve_source_identity(
                record, profile, manifest_identity=assignments.get(index), source_scope=scope
            )
            raw = record.fields.get(profile.fields.lexical_entry_field, "")
            words = source_html_to_text(raw).split()
            word = re.sub(r"^\W+|\W+$", "", words[0]) if words else ""
            if not word:
                raise ValueError("lexical entry has no readable first word")
        except (IdentityError, ValueError) as error:
            skipped.append(KnownLemmaSkip(record.provenance.location, str(error)))
            continue
        lemmas.setdefault(latin_fold(word), []).append(identity)

    return KnownLemmas(
        lemmas=MappingProxyType({lemma: tuple(identities) for lemma, identities in lemmas.items()}),
        skipped=tuple(skipped),
        source_digest=digest_bytes(source.read_bytes()),
        profile_digest=digest_bytes(profile_file.read_bytes()),
    )
