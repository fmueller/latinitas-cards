# v0.1.0 Release Readiness

This document records the technical release-candidate boundary. It is not a publication
approval or a native Anki test report.

## Version and candidate status

- Package version in `pyproject.toml`: `0.1.0`.
- Prepared version: `v0.1.0`.
- Prepared tag version: `v0.1.0`.
- Changelog entry: `CHANGELOG.md`, `## [0.1.0]`.
- Status: **candidate awaiting user review**.
- The `v0.1.0` tag and GitHub release do not exist yet. Publication belongs to T-014.

The exact candidate is the single commit containing this preparation and its final checks.
Record that immutable full commit ID in the Taskrail verification and review callback; do
not add a self-referential commit ID to this document.

## Stable scope reviewed

The candidate covers the deck-first path only:

1. inspect a sanitized or user-provided CSV/APKG/COLPKG source;
2. confirm and save a versioned profile with explicit source fields, semantic roles, German
   language tag, target note type/deck, tags, and selected recipes;
3. preview principal-part completion and recognition output; and
4. write repeatable UTF-8 Anki text-import CSV with stable `LatinitasID` first and source
   provenance.

The committed sanitized fixture preserves anonymized derivation from observations recorded
during T-011. It is not the original user deck, and its smoke tests do not revalidate the
original source, private counts, names, or metadata.

## Automated readiness evidence

The final callback must identify the exact candidate commit and attach the results of:

- `uv run ruff check`;
- `uv run mypy`;
- `uv run pytest -v`;
- `mise run check`; and
- the same locked-environment test chain on Python 3.13 and 3.14.

The end-to-end regression test must continue to assert output contents, both selected
recipes, `LatinitasID` first, deterministic bytes across repeats, unchanged source bytes,
and stable identities when managed tags/content change. It must not treat a zero exit code
alone as evidence.

## User review and native Anki boundary

Before publication, the user should perform the remaining manual review in a disposable
Anki collection:

1. Create the dedicated note type named by the profile (`Latinitas Principal Parts`).
2. Create regular fields for every `#columns` name except `Tags`; `Tags` is Anki metadata,
   not a regular field.
3. Create at least one template and enable **Allow HTML in fields**.
4. Import the generated CSV, map `LatinitasID` as the first/matching field, map `Tags` to
   Anki's tags column, and map `Personal Notes` on the first import.
5. Repeat the import with the same note type and matching policy, map `Personal Notes` to
   **Ignore field**, and check that managed fields update without creating duplicate notes.
6. Independently check native rendering, scheduling/deck behavior, and the user's intended
   note-type/template setup.

These are user-owned native-client checks. Browser-rendered generated HTML and automated
CSV assertions do not prove native Anki import, scheduling, or rendering behavior.

## Recovery limitation

Caught process/I/O failures attempt to restore the CSV and manifest pair and report affected
destinations or retained backups when restoration is incomplete. The pair is not durable
against forced termination or power loss between the two replacements: there is no journal
or startup recovery. Manual recovery remains required when that limitation is encountered.

## Dependency and security review notes

The current GitHub Dependabot query (2026-09-25) still reports open alerts [#139](https://github.com/fmueller/latinitas-cards/security/dependabot/139)
and [#140](https://github.com/fmueller/latinitas-cards/security/dependabot/140) for
GitPython in `poetry.lock`, with patched versions 3.1.55 and 3.1.52 respectively. The
candidate has no tracked or working-tree `poetry.lock`, and `GitPython` is absent from both
`pyproject.toml` and the current `uv.lock`. The alerts were not changed or dismissed; this
is evidence that they do not name a dependency in this candidate, not a claim that the
shared alerts are resolved. Recheck them before publication if manifests or packaging
inputs change.

The optional `annotate` and `annotate-gpu` dependency/license inventory is in
[the license compatibility audit](license-compatibility-audit.md). The GPU extra includes
NVIDIA proprietary runtime artifacts and remains highly experimental; unresolved optional
license obligations remain a user review risk.

## Publication boundary

This task prepares and identifies a technical candidate only. It does not create a tag,
GitHub release, package, deployment, or publication claim. T-014 remains the separate,
currently todo publication task and must not be started by this preparation.
