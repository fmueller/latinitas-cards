# Unreleased v0.2.1 integration readiness

**Implementation-only readiness, not publication or native acceptance.** Package,
released changelog and release links remain v0.2.0. No tag, release, deployment,
publication or spec activation is authorized by this record.

## Candidate and provenance

The offline walkthrough ran on 2026-10-11, 16:14–16:33 UTC, against
[accepted main d1beff1](https://github.com/fmueller/latinitas-cards/commit/d1beff10fe8ca7f27486ea751c161468fb8df5a9),
including [T088 af9050e](https://github.com/fmueller/latinitas-cards/commit/af9050e136033a238cbe2c780b2412a8e3f3aab3).
T089 changes documentation and generated Taskrail metadata only; no runtime fix,
production RED/GREEN transition, new production test or visual change is claimed.
The pinned active spec is `specs/v0.2.1.md`, SHA-256
`0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4`.
Taskrail independently returned T089 as the sole eligible next task after `validate`.

Inputs were ordinary UTF-8 CSV plus the documented lexical JSON file, not internal
Python domain-model fixtures: copies of `docs/examples/deck.csv`, `profile.json`,
`deck.csv.latinitas.json`, `phrase-inputs.json`; authored per-fingerprint decisions;
and explicit German recipe references. The vocabulary CSV and identity manifest
remained byte-identical to the examples. The lexical file's SHA-256 was
`ab1a43903aa999e678bd152666b64806811a3dc2e640e8bb7c58bbdce67bab95`.
Parsing started from `docs/examples/form-parsing-cases.json` with sample decisions
removed, then used the shared CLI to accept lemma/case/number and withhold gender.
Principal parts started from a four-row CSV (`amō`, `loquor`, `meminī`, `legō`)
and `setup --non-interactive --confirm`, with explicitly confirmed roles and senses.
The defective row's absent leading infinitive was rejected rather than invented.
Separate PPP and supine profiles checked role-appropriate fourth-form translations.
These authored synthetic judgments are not expert calibration or precision evidence.

## Executed offline matrix

| Lane | Observed result |
| --- | --- |
| Reviewed lexical file | 25 individually specified lexical/pattern decisions; check passed; no analyzer/network/LLM |
| Bounded phrases | `cum amico`, `ad amicum`, `liber amici`, `puella amicum videt`, `puellae amicum vident`, `amicus bonus`; four pattern families |
| Contrasts | Wrong governed case, adjective gender and verb number each rejected selection (exit 1), leaving no output |
| Explicit recipes | Four initial constructions × separately selected translation and recognition = eight exercises; combination and reference gates independently reviewed |
| Corrections and reuse | Authored contextual explanation with separate evidence/review reached the corrected export; unchanged sidecars reused automatically |
| Terminology | Latin/German phrase and parsing canonical claims/reviews/IDs/slots unchanged; German principal labels and form translations checked for both recipes; no translation added to fronts |
| Determinism/check | Phrase and principal reruns byte-identical; current artifact checks exit 0 without diagnostics; source vocabulary/manifest untouched |
| Initial handoff | Eight exported phrase rows mapped to a manually prepared closed schema-18 synthetic destination; CLI capture/adopt passed with explicit membership/ownership |
| Current retained update | Terminology/profile change: eight fixture notes, 12 individually approved operations; emit pending → capture → observe 12, unresolved 0 → reconcile version 3 |
| Other retained payloads | Principal schema 3: three notes/28 fresh approved operations; separate parsing-v1: one note/one fresh approved operation; these lanes stopped at approval, not native import |
| Preservation | Phrase fixture's native IDs, frozen slots, complete card/history/schema/deck tables and Personal Notes unchanged after applying emitted rows to closed SQL storage |
| Negative controls | Tampered CSV and stale lexical evidence checks exit 1; ambiguous nominative/vocative `liber` single-case exercise withheld; old payload receipt rejected; corrected answer requires fresh approval |
| Withdrawal | Empty explicit recipe plan produces eight unsupported retirements; approval exits 1; no withdrawal performed or certified |
| Old style | Closed fixture with historical CSS v2 rejected capture (exit 1) with separately reviewed manual-setup/no-rebinding guidance; current CSS v3 capture passed |

Initial phrase CSV SHA-256 (both reruns):
`88a89ea9e82cf46f03b5a49daafbbd2b25652f7eed11144cd89aa5a4f2538b46`.
Corrected CSV: `204020f6071c548f2ca29ef1a6a2e7b1b295cf2625bc742880b87039d7d09af9`.
Hashes bind this run's paths/profile; they are not portable regeneration expectations.

## Commands and evidence boundary

The [T089 work thread](https://ampcode.com/threads/T-01a12bbc-20a9-738f-85df-4049b0dfd2fa)
records exact invocations, authored files, output, fixture scripts and review dispositions.
The CLI routes used were:

```sh
uv run latinitas-cards setup --input verbs.csv --profile verbs-profile.json --non-interactive --confirm # explicit field/role/recipe overrides in run evidence
uv run latinitas-cards claims list inputs.json --phrase-input --json
uv run latinitas-cards claims record inputs.json --phrase-input --batch input-decisions.json --json
uv run latinitas-cards phrases inspect inputs.json --check --json
uv run latinitas-cards phrases select inputs.json selection-request.json selected.json --json
uv run latinitas-cards phrases generate inputs.json selected.json --json
uv run latinitas-cards phrases recipes inputs.json selected.json recipe-request.json recipes.json --json
uv run latinitas-cards claims record inputs.json --selections selected.json --recipes recipes.json --batch exercise-decisions.json --json
uv run latinitas-cards phrases export inputs.json selected.json recipes.json initial.csv --json
uv run latinitas-cards claims inspect inputs.json --selections selected.json --recipes recipes.json --artifact initial.csv --check --json
uv run --with anki==26.9.3 latinitas-cards managed capture closed.anki2 --selection membership.json --closed-backup --interval-confirmed
uv run latinitas-cards managed adopt snapshot.json --ownership ownership.json --review 'Individually checked fixture origins' --state state.json
uv run latinitas-cards phrases plan inputs.json selected.json recipes.json snapshot.json state.json --json
uv run latinitas-cards managed approve plan.json --operation EXACT-ID/field/EXACT-FIELD --review 'Individually inspected retained payload and footprint'
uv run latinitas-cards managed emit handoff.json --state state.json --output updates.csv
uv run latinitas-cards managed observe observation.json --state state.json --interval-confirmed
uv run latinitas-cards managed reconcile reconciliation.json --state state.json
```

Principal claims use `--profile`; parsing embeds its profile. Both use the same
`claims list/record/inspect` route, `generate --terminology latin|user` or
`form-parsing preview/export --terminology latin|user`. Principal first export
explicitly approved source scope/fresh import. Parsing export explicitly approved
fresh import. Payload plans for these lanes used `managed plan request.json`.

Only destination-fixture preparation used pinned Anki's backend to create closed
storage from exported rows and the documented layout contract. All Latinitas product
operations used the installed CLI; scripts provided stdlib orchestration and set up
external Anki storage, not internal Latinitas model fixtures. The fixture writer and
SQL row application are **not a native CSV import/dialog, scheduling migration,
appearance or restore test**.
An initial harness error used shared phrase `object_key` for membership; the real
guard rejected it. Correct workspace `exercise_key` passed. A live SQLite connection
also correctly blocked capture until checkpointed/closed. Noncanonical `ppp` and
literal `-` were fixture-author errors, corrected to `perfect_passive_participle`
and explicit blank omitted slots; no product failure or runtime patch was inferred.

## Gates and pending acceptance

Initial gates: `uv run ruff check` → all checks passed;
`uv run mypy` → no issues in 106 source files;
`uv run pytest -v` → 1,022 passed; `mise run check` → exit 0.
Taskrail `validate` → state valid; `coverage` → 7/7 areas structurally linked,
6/7 implemented before T089 finalization. Structural coverage is not native coverage.
Final reviewed gates and lifecycle timestamps belong to the T089 task verification
notes and work thread; this initial run is not a substitute for those gates.

Native Linux Anki Desktop `auto` acceptance remains **required and pending**; the
partial Desktop smoke is not sufficient. The maintainer's 2026-10-13 decision
removes AnkiMobile from the v0.2.1 release gate; Mobile remains unverified, not
passed or certified. This supersedes the earlier 2026-10-11 implementation-only
deferral as the current release-gate scope without changing that historical
decision or evidence. The exact dated decisions, partial Desktop smoke and current
Desktop-only checklist in
[morphology native verification](morphology-native-verification.md#automatic-appearance-native-checklist-t067)
are authoritative. Native acceptance has not been run or certified by this record;
no pending result is marked clean/passed/covered here. Browser or backend evidence
cannot substitute for actual native Desktop import/reviewer interaction.
Publication readiness remains conditional on the required Desktop acceptance and
separately authorized release work; this integration-readiness record does not
authorize a tag, release, deployment or publication.

Supported constructions are the four reviewed bounded families, with explicitly
authored German references and finite selections, not general syntax inference,
unrestricted word combinations or automatic translation grading. CSS/templates need
reviewed manual installation. To resolve blocked withdrawal: stop edits/reviews/sync,
retain a recoverable backup, inspect affected notes/cards, review and perform the
manual suspension/withdrawal choice, obtain fresh closed evidence and reconcile
ownership before a new plan/receipt. Omission and retired tags never retire cards.
Old style/template digests require reviewed manual setup in a separately bound
destination; never edit evidence or repurpose scheduled templates to bypass the guard.
