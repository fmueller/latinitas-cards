# v0.2.1 release readiness — 2026-10-14

Release: [v0.2.1](https://github.com/fmueller/latinitas-cards/releases/tag/v0.2.1).
Package metadata is `0.2.1`; the annotated release tag is `v0.2.1`.
Immutable target, exact-commit CI, publication timestamp and fresh-tag installation
evidence are recorded separately in [T101](../planning/tasks/T-101-publish-v0-2-1-annotated-tag-and-github-release.md)
after publication. PyPI, uploaded assets, deployments and v0.3.0 activation remain
excluded. The date is 2026-10-14 in Europe/Zagreb.

## Current candidate and accepted native evidence

Prepared from [accepted planning ce0429e](https://github.com/fmueller/latinitas-cards/commit/ce0429edb016520f4ba37103a04747031c688907),
with unchanged product code from [cac2d7f](https://github.com/fmueller/latinitas-cards/commit/cac2d7ff99a533c39a8bd83a733b4b905c0851c8).
Active spec `specs/v0.2.1.md` SHA-256:
`aabfdf060b1d90074a672b8cd29875677857b3d2ccb1f4455e09ade0e9de5ca4`.
Reference CSS v4 SHA-256:
`18cd5b7d2e5b50f90ac017db225d49f7ff11fa57a35058487300e4b56457945e`.
This candidate changes metadata, tests and records only; no additional native replay
is claimed. The prior [v0.2.0 preparation](https://github.com/fmueller/latinitas-cards/commit/3b3616cef6c948da24d77214f64f2204a7031ba0)
establishes the annotated-tag/GitHub-release method; exact-commit CI precedes tagging.

The [parent acceptance thread](https://ampcode.com/threads/T-01a12a7a-87ff-729a-b72b-b8a98e688203)
accepted fresh replay 3 from the [native replay thread](https://ampcode.com/threads/T-01a1380a-4911-752c-a579-20a15357769c).
See the [current native record](morphology-native-verification.md#accepted-fresh-native-replay-3--2026-10-14)
for the matrix and evidence boundaries. Real Linux Anki/aqt 26.9.3 build 29bb700b,
Python 3.14.2, Qt runtime 6.11.2/xcb exercised 32 actual CSV/reviewer lanes,
112 notes/320 cards, 104 same-card Preferences light → night → light cycles
plus six supplements. There were 2,112 segment checks; minimum context contrast
9.739769:1 and plain Personal Notes 5.741836:1. Fronts were inspected in Light;
representative answers/disclosures covered all three client modes.

The retained-update run observed **54 field/tag operations, not 54 notes**, with
zero unresolved effects. Closed snapshots checked preservation independently of
the media. Isolated native `.colpkg` restore matched three notes, 22 cards and
44 history rows across eight tables. Stock backend 19 cases, stock preparation and
all seven GUI cases passed their intended controls, including expected unmapped
tags (one unresolved) and exact repeat no-op. Screenshots are not table-preservation proof.

The parent independently checked all 54 archive hashes and 4,937 manifest-listed
member hashes. Checkpoint 00 has no member manifest: its archive hash was checked,
but it contributes no member hashes. Durable evidence SHA-256 values:

| Evidence | SHA-256 |
|---|---|
| Inventory | `2c5e03fdf0350661ae4019416d58eaad1b7502538a059583ef20b927db98bc91` |
| Final reports archive (775,377 bytes) | `dd8293856d5dfd3e86f72985dc2af7cfd2a229f19ce35d4b62d155b218a0526c` |
| Final media archive | `3f764f4fb96280a064fd7245f8df9b18e645e1654d8a862668294e67dc30cd26` |

The parent inspected 120 media pages and the 120-second native pointer recording.
Original adversarial queue clean at 4/5 is **separate** from native acceptance:
native run 1 (historical ROUND5) found defects repaired by T097–T099; run 2's runtime
passed but orb/filesystem failure prevented raw-evidence finalization, so it was
not accepted clean; fresh replay 3 was independently accepted with recoverable checkpoints.
The extra scoped `_back` mutant **survived**, and an initial selector failure was
disclosed. The unmutated 1,050-test baseline passed (53.31s). These are report-only,
not a full mutation campaign or efficacy certification; the earlier T099 killed
smoke mutant is distinct.

## Release limits

Linux synthetic acceptance does not verify Mobile, macOS/Windows, sync, live user
collections, arbitrary scheduling or expert language calibration. Mobile was removed
from the gate by the maintainer, not passed. CSS v4 still needs reviewed manual
installation; CSV installs no CSS/templates. Old bindings fail closed: use reviewed
manual setup and fresh capture in a separately bound destination, never rewrite
evidence or repurpose scheduled templates. Changed retained payloads need renewed
individual approval. Structural/lifecycle changes, creation/retirement and automatic
migration remain unsupported; no additional recovery or safety guarantee is made.

## Historical T100 candidate checks — 2026-10-14

The existing release metadata test failed first on `0.2.0 != 0.2.1`, then all
three release-candidate tests passed after the minimal metadata/document update.
Dedicated simplification made no edits. Independent General, Python and Security
reviews and fresh candidate validation returned no concrete findings.
Final `uv run ruff check` passed, `uv run mypy` passed (107 source files),
`uv run pytest -v` passed (1,050 tests, 50.92s), and `mise run check` passed
(including 1,050 tests, 49.71s, formatting and policy-guard suites).
`uv build` produced the 0.2.1 sdist and wheel; an isolated wheel installation
with the locked default export reported version 0.2.1 and passed CLI `--help`,
without urllib3/fsspec installed. This is candidate install evidence, not a
fresh-tag installation or exact-commit remote CI result. Taskrail lifecycle and
final disposition verification are recorded in T100; T101 independently gates
publication on the final tag target's exact CI and fresh-tag installation.

## Current optional annotation advisories — 2026-10-14

Fresh GitHub Dependabot API inspection found four open alerts (three high, one
moderate). Current advisory records confirm the following affected locked versions:

| Advisory | Locked version | Severity / patched version | Exposure |
|---|---|---|---|
| [GHSA-8988-9cw3-xx77](https://github.com/advisories/GHSA-8988-9cw3-xx77) | urllib3 2.7.0 | High / 2.8.0 | HTTPS proxy/target TLS policy mixing; configuration-dependent |
| [GHSA-vxq7-64xx-v4gw](https://github.com/advisories/GHSA-vxq7-64xx-v4gw) | urllib3 2.7.0 | High / 2.8.0 | Unbounded chunk-size line when streaming untrusted chunked responses |
| [GHSA-gh4c-6fx4-qh6g](https://github.com/advisories/GHSA-gh4c-6fx4-qh6g) | urllib3 2.7.0 | Moderate / 2.8.0 | Chunked Deflate trailing bytes can cause a non-progress loop |
| [GHSA-27vj-qcqg-25rc](https://github.com/advisories/GHSA-27vj-qcqg-25rc) | fsspec 2026.2.0 | High / 2026.6.0 | Untrusted Kerchunk/reference JSON templates can execute Python |

Fresh `uv export --locked --no-dev --no-emit-project` and separate `--extra annotate`
and `--extra annotate-gpu` exports exclude urllib3/fsspec from default offline
installation and include both affected versions in each annotation extra.
No dependency versions changed, no alert was dismissed, and no vulnerability-free
claim is made. These exports do not execute annotation or an exploit.

Bounded authoritative source inspection reconfirmed
[Stanza's raw download path](https://github.com/stanfordnlp/stanza/blob/v1.14.0/stanza/resources/common.py#L147-L193)
uses Requests streaming with finite chunks, and
[Requests iteration](https://github.com/psf/requests/blob/v2.33.0/src/requests/models.py#L801-L839)
enables urllib3 content decoding. The chunked/Deflate conditions are therefore
reachable during optional resource acquisition given the relevant server responses;
HTTPS resource hosts do not eliminate these paths. Proxy exploit conditions were
not reproduced or certified absent. The
[fsspec reference parser](https://github.com/fsspec/filesystem_spec/blob/2026.2.0/fsspec/implementations/reference.py#L1005-L1087)
uses unsandboxed Jinja templates. No exposed Kerchunk/reference workflow was found
in the prior bounded [annotation assessment](release-v0.2.0.md#post-publication-advisory-addendum--2026-10-08);
that absence is not universal safety proof. Avoid untrusted resource responses,
proxy-sensitive annotation use and untrusted reference JSON with affected extras.
Remediation remains separate follow-up, not part of this metadata-only candidate.

## Historical integration readiness — 2026-10-11

The following T089 record is historical implementation-only evidence. Its original
pending native status, older CSS evidence and spec hash are not current acceptance.

### Historical candidate and provenance

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

## Historical gates and pending acceptance

Initial gates: `uv run ruff check` → all checks passed;
`uv run mypy` → no issues in 106 source files;
`uv run pytest -v` → 1,022 passed; `mise run check` → exit 0.
Taskrail `validate` → state valid; `coverage` → 7/7 areas structurally linked,
6/7 implemented before T089 finalization. Structural coverage is not native coverage.
Final reviewed gates and lifecycle timestamps belong to the T089 task verification
notes and work thread; this initial run is not a substitute for those gates.

At the time of this historical record, native Linux Anki Desktop `auto` acceptance
remained **required and pending**; the
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
