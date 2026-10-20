# v0.3.0 release evidence — 2026-10-20

**Published:** [GitHub release](https://github.com/fmueller/latinitas-cards/releases/tag/v0.3.0)
and [PyPI 0.3.0](https://pypi.org/project/latinitas-cards/0.3.0/).
The maintainer explicitly authorized T-038 publication on 2026-10-20 and personally
approved the protected `pypi` deployment as `fmueller`.

## Publication verification

- Tagged candidate:
  [`8bc8332`](https://github.com/fmueller/latinitas-cards/commit/8bc83324e9911a1c404ae24aad1c726d37e39af0).
  Its [Build](https://github.com/fmueller/latinitas-cards/actions/runs/39496370685)
  and [CodeQL](https://github.com/fmueller/latinitas-cards/actions/runs/39496370618)
  passed all eight checks before tagging.
- Exactly one annotated `v0.3.0` tag was created under Felix Müller's maintainer
  identity. Remote tag object: `a63f80e52d0888212e7c533003c285956599bb9d`;
  peeled commit: `8bc83324e9911a1c404ae24aad1c726d37e39af0`. The tag was not moved.
- [Release workflow](https://github.com/fmueller/latinitas-cards/actions/runs/39497660383),
  attempt 1: `build`, `verify (3.13)`, `verify (3.14)`, `publish` and `verify-pypi`
  all succeeded. The approval API records `fmueller` approving environment `pypi`.
  Trusted Publishing uploaded the verified build artifacts without rebuilding.
- `verify-pypi` installed exactly `latinitas-cards==0.3.0` from PyPI, ran CLI help,
  and asserted CLTK, Stanza and Torch were absent. The GitHub release was published
  only afterward at 2026-10-20T21:10:26Z, non-draft and non-prerelease.
- Independently downloaded the run's `dist` artifact and compared every filename
  and SHA-256 with the versioned PyPI JSON response. Both files matched exactly;
  neither the build artifact nor the PyPI response contained additional files.

| File | Build and PyPI SHA-256 (identical) |
| --- | --- |
| `latinitas_cards-0.3.0-py3-none-any.whl` | `9b3fe37f367001b4327f815a66ff4675a73af021d59f53a5cd26769d70779dd5` |
| `latinitas_cards-0.3.0.tar.gz` | `03bf9413d6bddefe88fbc17496338f9ca133ced56d3763c8ee099082af87d238` |

The tag points to the tested candidate, not this later evidence commit. Hash
equality establishes file consistency, not a blanket security or license guarantee.
No full mutation efficacy campaign, optional NLP runtime or new native Anki testing
is claimed by this release. Existing platform and compatibility limits below remain.

## Pre-publication preparation — 2026-10-20

This section records the preparation before authorization and publication; its
candidate checks and remote-absence observations are historical, not current state.

The accepted implementation base is
[`fca6842`](https://github.com/fmueller/latinitas-cards/commit/fca68427e41196e7650dfdca8f1a3b78c1455ba1).
Its [Build](https://github.com/fmueller/latinitas-cards/actions/runs/39491087705)
and [CodeQL](https://github.com/fmueller/latinitas-cards/actions/runs/39491088352)
checks passed. Release metadata changes require fresh checks on their own commit
before tagging; these URLs do not verify a later candidate.

The maintainer confirmed the PyPI Trusted Publisher setup, approved the license
audit and confirmed eBible.org latVUC derivation on 2026-10-20; see the dated
[audit and verification limits](license-compatibility-audit.md). Read-only GitHub
API checks confirmed required reviewer `fmueller` and tag-only `v*` deployments for
environment `pypi`. PyPI account configuration is maintainer-reported, not
independently authenticated here. No existing `v0.3.0` tag, GitHub release or
PyPI version was found during preparation.

Completed follow-ups through T-147 are included in T-038's prerequisites; T-128
belongs to an inactive future spec. The full mutation campaign remains unverified:
T-146/T-147 validated baseline collection and one selected mutant, not a full score.
At this stage T-038 remained blocked pending explicit publication authorization.
No tag, upload, deployment approval or GitHub release was performed by preparation;
the later authorization and publication are recorded above.

The sections below retain the original 2026-10-16 walkthrough, its tested revision,
and its historical limitations; they are not fresh executions on this candidate.

## Scope and provenance

Runtime base: [accepted cumulative base](https://github.com/fmueller/latinitas-cards/commit/1db086cae196dafa4a36350bdd354b46e9b66ccc),
containing [T127 schema-export repair](https://github.com/fmueller/latinitas-cards/commit/e2fa794e7dce41e3a09e35ed3b7190b2b366741a).
Pinned `specs/v0.3.0.md` SHA-256:
`fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
Only release metadata, this record, documentation pointers and the existing release
test change here. The lock dependency inventory is unchanged (92 non-project records).

v0.3.0 adds normalized text/USFX corpora, reviewed stable identities, sentence/token
sidecars, patch planning/explicit approval/policy-gated application, corpus claim
decisions, read-only views and effective configuration, and agent-ready diagnostics.
It preserves legacy Anki-facing output. The full Vulgate oracle is a byte-compatibility
check, not a provenance/license approval: at the time of this walkthrough Vulgate
provenance still required maintainer confirmation, now recorded above. Bundled
corpora are excluded from distributions.

## Offline integrated walkthrough

Executed the real `.venv/bin/latinitas-cards` executable from fresh external temporary
directories, not CliRunner or constructed application objects. Inputs came from
committed `tests/fixtures/corpus`, `annotations`, and `policy`; edited text, a labeled
percent/Unicode/apostrophe case, a retract patch and invalid controls were derived
as explicit files. All calls used `--json`. The command appendix records exact argv,
exit codes and artifact SHA-256 snapshots, including setup mistakes and corrections.
Those mistakes were not product defects: config takes `--corpus`, not a positional
corpus/`--annotations`, and an existing claim decision cannot be overwritten with a
conflicting status. Accepted/withheld drift controls therefore use separate copies.

| Lane | Observed result |
| --- | --- |
| Full Vulgate normalize + validate | 0; 35,809 passages, 24,905,456 bytes, committed oracle digest below |
| Plain manifest first/edit/review | 0, then 1 `review_required` without writes, then 0 with explicit reuse/allocation/removal |
| Retired ref | 3 `retired_ref` |
| Labeled fixture and `%2E.é.1` template | 0; native ref `%252E.%C3%A9.1`, raw parts `%2E`, `é`, `1`; validation 0 |
| Straight/curly apostrophes | 0; `d'arma d’arma` spans `[0,1],[1,2],[2,6],[7,8],[8,9],[9,13]`, word/punct/word twice |
| Segment → fixture-morph plan → human approve → apply | all 0; rerun 0 `already_applied`, unchanged bytes |
| Accept morphology; claims inspect + corpus show + annotations inspect --check | all 0; explicit accepted review joined into corpus view |
| Config and policy validate | corrected documented argv both 0; no writes |
| Forbidden corpus claims value / invalid batch | 3 / 1, legacy claims JSON (no CliResult envelope), unchanged decisions |
| Plan/approve retract, then accepted or withheld target decision | old receipt apply 4 `binding_drift`, unchanged sidecar in both independent directories |
| Unrelated decision | apply 0; unchanged completed rerun 0 |
| Agent actor, default policy | approve 5 `actor_refused`; hand-built agent receipt apply 5, unchanged bytes |
| Foreign valid segmentation suffix | apply 4 `binding_drift`, unchanged sidecar |
| One edited passage word | inspect --check 1 `stale`; validate --annotations 1; accepted claim inspect --check 1 `stale` |
| DTD | 3 `unsafe_xml`, no output |
| Mixed fixture partial/re-receipt/no-op | first apply appends 1/pending 1; unchanged retry appends 0; receipt appends 1/pending 0; rerun and different valid re-receipt 0, no duplicate bytes |

The two directories used `TZ=UTC, LC_ALL=C.UTF-8` and
`TZ=Pacific/Honolulu, LC_ALL=C`. Every normalize output (including edited manifest
materialization), segment output, fixture-morph plan, receipt and apply output
was SHA-256 identical across the two directories. Fixed timestamp:
`2026-01-02T03:04:05Z`. Repeated applies intentionally omit it.

| Output | SHA-256 in both directories |
| --- | --- |
| Full Vulgate | `cefcaa8f977df72acd71b76ace7a45ccbe8e81105dd56a86815964e531e0927d` |
| Edited reader | `963ae183ad8c58a65dbd85b706a57f5e56cb32ec42610fe854e703c8a3ec0525` |
| Edited manifest | `668d1e0da8cb55955e0656aec7fec00844b918d56f3db3bac3291e62c9ebf350` |
| Labeled fixture | `437ecf629895f7772b830756e5e5182ddc901b38444704469affa672a6cbd602` |
| Percent/Unicode corpus | `65edeb591fc6cc1ae75f55ef42698e2a7110fc304bbbc59b92f31040bb5a7b76` |
| Apostrophe segmentation | `331ee5bdd2401c2f27ba1694bd77b32a3318084c905fbef28e4e38b3f931f042` |
| Morphology plan | `257ce914bf215785ec024ac5a494a1ff1498bd11d24aa06b6507e47cc433dfcd` |
| Human receipt | `09444f134d10bd08da59592e3270910b29118af02e19ccac24fb97e212e63ccd` |
| Applied annotations | `16148b7a890ee573dece6a3f058359ac979bcbc57758e254aaca6837f876021c` |

## Recovery, concurrency and compatibility boundaries

The [T107 execution record](../planning/tasks/T-107-normalize-plain-text-sources-with-manifest.md#workflow-v3-execution-evidence-2026-10-15)
records interruption before/between/after replacements, first creation, original
approved rerun, byte comparison, allocator/retirement checks and no double allocation.
Its deliberately broken recovery produced 18 failures; restoring recovery passed.
This is recoverable two-file commit, **not atomic visibility across two paths**.
No fault-injection machinery is duplicated by this walkthrough.

[T118](../planning/tasks/T-118-apply-planned-annotation-operations-atomically.md)
records corpus/sidecar/claims-lock contention, competing writers and the exact logical
claims lock under symlinked data (regression initially returned 0 instead of busy/4).
[T119](../planning/tasks/T-119-enforce-destructive-conflicting-and-actor-rules.md)
records synchronized blocking `claims record`, same-plan dependent/conflict safety,
invalid pending anchors and decision drift, plus partial/no-op positive controls.
Its corrected matrix passed 40 cases and final suite passed 2,418. This record cites
that evidence rather than implying this sequential walkthrough re-proves concurrency.

After unshallowing, the required historical golden diff from the addition of
`tests/fixtures/goldens/legacy-vulgate/cases.json` was empty for both legacy-vulgate
and claims-v1 directories. Their characterization tests are included in the full gate.

## Limits and release gates

- `annotate` and conflicting `annotate-gpu` extras remain experimental. The uv CPU
  index pin does not carry over to pip: pip users need the explicit CPU Torch index
  instructions in README. No optional NLP resources/provider endpoint were exercised.
- Corpus, annotation, patch, plan, receipt, policy and CLI-result wire formats are
  version 1 before 1.0; corpus claims are version 2, legacy deck claims remain v1.
  Future compatibility/migrations are not promised by this candidate.
- Supported platforms are Linux and macOS, not Windows. This walkthrough ran Linux
  x86-64 in an orb; it is not fresh macOS evidence.
- No native Anki Desktop or AnkiMobile evidence is claimed: Anki-facing output is
  unchanged and no v0.2.x native acceptance gate is reopened. Historical release
  pointers retain their original evidence and limitations.
- T038 remains blocked maintainer-only. No tag, release, upload, deployment, workflow
  dispatch, spec activation or synthetic adversarial acceptance round occurred here.

## Candidate packaging and validation

`uv sync --locked --dev` warmed the cache (exit 0). On the bumped candidate:

| Exact command | Exit / decisive output |
| --- | --- |
| `UV_OFFLINE=1 bash scripts/check-dist.sh` | 0, distribution inspection passed, version 0.3.0; source wheel rebuilt and inspected |
| `UV_OFFLINE=1 bash scripts/check-install.sh "$(uv python find 3.14)"` | 1: `Because pydantic was not found in the cache and latinitas-cards==0.3.0 depends on pydantic>=2.11.9,<3.0.0` |
| `bash scripts/check-install.sh "$(uv python find 3.14)"` | 0, authorized cache-miss fallback; both distributions passed help/version/normalize/validate, `ok=true` |

Interpreter: CPython 3.14.2, Clang 21.1.4, x86-64 Linux;
`uv python find 3.14` resolved `/home/user/workspace/repo/.venv/bin/python3`.
Both isolated installs explicitly reported optional CLTK/Stanza/Torch distributions
absent. No index upload or PyPI availability is inferred from local installation.

| Local artifact | SHA-256 |
| --- | --- |
| `latinitas_cards-0.3.0-py3-none-any.whl` | `b9e38288a87e1344d8ed218642f3b5128bb737e56c54de567c99489395b86868` |
| `latinitas_cards-0.3.0.tar.gz` | `58709819bee6c00424765bd15920adac079759ab74c3d152901474ac2427d695` |
| Temporary sdist-rebuilt wheel (inspected, then script-cleaned) | `ba7eb862f9385f1f4ba2a624114c908c5769e1f4a4be727fd1e7f5a9dec8cf14` |

These are inspected local artifacts, not publication assets or reproducible-build
claims. Later documentation-only edits do not change the runtime; T038 must build
and verify the exact final publication candidate in package CI.

Metadata TDD: renamed the existing v0.2.1 assertion first; targeted pytest failed
with `AssertionError: assert '0.2.1' == '0.3.0'`, then passed after only the project
and lock versions and candidate documentation changed. Historical assertions and
the unchanged dependency audit remain in that test. Full workflow gate results
and independent review dispositions are recorded in the
[T124 execution notes](../planning/tasks/T-124-validate-integrated-v0-3-0-workflows-offline-and.md#workflow-v3-execution-evidence-2026-10-16).

## Exact CLI transcript

Generated from the captured subprocess transcript; absolute paths identify the
external directories actually used. Exit 2 config and exit 1 conflicting decision
are setup mistakes retained for audit, not acceptance failures.

```text
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/home/user/workspace/repo/data/lat-clementine.usfx.xml' '--adapter' 'usfx' '--corpus-id' 'vulgata-clementina' '--descriptor' '/home/user/workspace/repo/data/lat-clementine.descriptor.json' '--output' '/tmp/t124-walk.qQcfjK/first/vulgata.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'validate' '/tmp/t124-walk.qQcfjK/first/vulgata.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'segment' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--output' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'plan' '/home/user/workspace/repo/tests/fixtures/annotations/fixture-morph.patch.json' '--corpus' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--output' '/tmp/t124-walk.qQcfjK/first/plan.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'approve' '/tmp/t124-walk.qQcfjK/first/plan.json' '--operation' 'sha256:0266cb94f95ad4e83fb13dd50f23c5fd8103bddeda6ed566949e5790597c1bad' '--actor' 'human:reviewer' '--review' 'Checked this exact proposal' '--output' '/tmp/t124-walk.qQcfjK/first/receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/first/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--receipt' '/tmp/t124-walk.qQcfjK/first/receipt.json' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/first/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--receipt' '/tmp/t124-walk.qQcfjK/first/receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/home/user/workspace/repo/data/lat-clementine.usfx.xml' '--adapter' 'usfx' '--corpus-id' 'vulgata-clementina' '--descriptor' '/home/user/workspace/repo/data/lat-clementine.descriptor.json' '--output' '/tmp/t124-walk.qQcfjK/second/vulgata.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'validate' '/tmp/t124-walk.qQcfjK/second/vulgata.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'segment' '/tmp/t124-walk.qQcfjK/second/corpus.jsonl' '--output' '/tmp/t124-walk.qQcfjK/second/annotations.jsonl' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'plan' '/home/user/workspace/repo/tests/fixtures/annotations/fixture-morph.patch.json' '--corpus' '/tmp/t124-walk.qQcfjK/second/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/second/annotations.jsonl' '--output' '/tmp/t124-walk.qQcfjK/second/plan.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'approve' '/tmp/t124-walk.qQcfjK/second/plan.json' '--operation' 'sha256:0266cb94f95ad4e83fb13dd50f23c5fd8103bddeda6ed566949e5790597c1bad' '--actor' 'human:reviewer' '--review' 'Checked this exact proposal' '--output' '/tmp/t124-walk.qQcfjK/second/receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/second/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/second/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/second/annotations.jsonl' '--receipt' '/tmp/t124-walk.qQcfjK/second/receipt.json' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/second/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/second/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/second/annotations.jsonl' '--receipt' '/tmp/t124-walk.qQcfjK/second/receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/first/source.txt' '--adapter' 'text' '--corpus-id' 'reader' '--output' '/tmp/t124-walk.qQcfjK/first/reader.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/first/source.txt' '--adapter' 'text' '--corpus-id' 'reader' '--output' '/tmp/t124-walk.qQcfjK/first/reader.jsonl' '--json'
exit=1
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/first/source.txt' '--adapter' 'text' '--corpus-id' 'reader' '--output' '/tmp/t124-walk.qQcfjK/first/reader.jsonl' '--approve-reuse' '0=p000001' '--approve-allocation' '2' '--approve-removal' 'p000003' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/first/source.txt' '--adapter' 'text' '--corpus-id' 'reader' '--output' '/tmp/t124-walk.qQcfjK/first/reader.jsonl' '--approve-reuse' '2=p000003' '--json'
exit=3
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/home/user/workspace/repo/tests/fixtures/corpus/text/wsl-three.txt' '--adapter' 'text' '--corpus-id' 'reader' '--citation-template' 'wsl:work.section.line' '--output' '/tmp/t124-walk.qQcfjK/first/labeled.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/first/special.txt' '--adapter' 'text' '--corpus-id' 'reader' '--citation-template' 'wsl:work.section.line' '--output' '/tmp/t124-walk.qQcfjK/first/special.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'validate' '/tmp/t124-walk.qQcfjK/first/special.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'segment' '/tmp/t124-walk.qQcfjK/first/special.jsonl' '--output' '/tmp/t124-walk.qQcfjK/first/special.annotations.jsonl' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/second/source.txt' '--adapter' 'text' '--corpus-id' 'reader' '--output' '/tmp/t124-walk.qQcfjK/second/reader.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/second/source.txt' '--adapter' 'text' '--corpus-id' 'reader' '--output' '/tmp/t124-walk.qQcfjK/second/reader.jsonl' '--json'
exit=1
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/second/source.txt' '--adapter' 'text' '--corpus-id' 'reader' '--output' '/tmp/t124-walk.qQcfjK/second/reader.jsonl' '--approve-reuse' '0=p000001' '--approve-allocation' '2' '--approve-removal' 'p000003' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/second/source.txt' '--adapter' 'text' '--corpus-id' 'reader' '--output' '/tmp/t124-walk.qQcfjK/second/reader.jsonl' '--approve-reuse' '2=p000003' '--json'
exit=3
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/home/user/workspace/repo/tests/fixtures/corpus/text/wsl-three.txt' '--adapter' 'text' '--corpus-id' 'reader' '--citation-template' 'wsl:work.section.line' '--output' '/tmp/t124-walk.qQcfjK/second/labeled.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/second/special.txt' '--adapter' 'text' '--corpus-id' 'reader' '--citation-template' 'wsl:work.section.line' '--output' '/tmp/t124-walk.qQcfjK/second/special.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'validate' '/tmp/t124-walk.qQcfjK/second/special.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/second
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'segment' '/tmp/t124-walk.qQcfjK/second/special.jsonl' '--output' '/tmp/t124-walk.qQcfjK/second/special.annotations.jsonl' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'list' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'list' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--fingerprint' 'a93404e6293f5ec8b6d4115f177fbfaf5ae96e5a209430f63bc7c2a5153596c0' '--status' 'accepted' '--reviewer' 'Latin reviewer' '--reason' 'Checked annotation in context' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'inspect' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--check' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'show' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--claims' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl.claims.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'inspect' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--corpus' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--claims' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl.claims.json' '--check' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'config' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--json'
exit=2
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'list' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--fingerprint' 'a93404e6293f5ec8b6d4115f177fbfaf5ae96e5a209430f63bc7c2a5153596c0' '--status' 'accepted' '--reviewer' 'Latin reviewer' '--reason' 'Checked annotation in context' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'inspect' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--check' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'show' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--claims' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl.claims.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'inspect' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--corpus' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--claims' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl.claims.json' '--check' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'config' '--corpus' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'policy' 'validate' '/home/user/workspace/repo/tests/fixtures/policy/mixed.policy.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--fingerprint' 'a93404e6293f5ec8b6d4115f177fbfaf5ae96e5a209430f63bc7c2a5153596c0' '--status' 'accepted' '--reviewer' 'R' '--reason' 'R' '--value' 'x' '--json'
exit=3
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--batch' '/tmp/t124-walk.qQcfjK/first/bad-batch.json' '--json'
exit=1
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--fingerprint' 'a93404e6293f5ec8b6d4115f177fbfaf5ae96e5a209430f63bc7c2a5153596c0' '--status' 'withheld' '--reviewer' 'Latin reviewer' '--reason' 'Checked annotation in context' '--json'
exit=1
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'list' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--fingerprint' 'a93404e6293f5ec8b6d4115f177fbfaf5ae96e5a209430f63bc7c2a5153596c0' '--status' 'accepted' '--reviewer' 'Latin reviewer' '--reason' 'Checked annotation in context' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'inspect' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--check' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'show' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--claims' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl.claims.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'inspect' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--corpus' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--claims' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl.claims.json' '--check' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'config' '--corpus' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'policy' 'validate' '/home/user/workspace/repo/tests/fixtures/policy/mixed.policy.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--fingerprint' 'a93404e6293f5ec8b6d4115f177fbfaf5ae96e5a209430f63bc7c2a5153596c0' '--status' 'accepted' '--reviewer' 'R' '--reason' 'R' '--value' 'x' '--json'
exit=3
cwd=/tmp/t124-walk.qQcfjK/first
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/first/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/first/annotations.jsonl' '--batch' '/tmp/t124-walk.qQcfjK/first/bad-batch.json' '--json'
exit=1
cwd=/tmp/t124-walk.qQcfjK/decision-accepted
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'plan' '/tmp/t124-walk.qQcfjK/first/retract.patch.json' '--corpus' '/tmp/t124-walk.qQcfjK/decision-accepted/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/decision-accepted/annotations.jsonl' '--output' '/tmp/t124-walk.qQcfjK/decision-accepted/accepted.plan.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-accepted
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'approve' '/tmp/t124-walk.qQcfjK/decision-accepted/accepted.plan.json' '--operation' 'sha256:babbdf630ed7994702fdd92951dd6d52678735b8f2a7416f9b5da1dedfeb9f9e' '--actor' 'human:reviewer' '--review' 'Checked exact retraction' '--output' '/tmp/t124-walk.qQcfjK/decision-accepted/accepted.receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-accepted
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/decision-accepted/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/decision-accepted/annotations.jsonl' '--fingerprint' 'a93404e6293f5ec8b6d4115f177fbfaf5ae96e5a209430f63bc7c2a5153596c0' '--status' 'accepted' '--reviewer' 'Latin reviewer' '--reason' 'Checked annotation in context' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-accepted
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/decision-accepted/accepted.plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/decision-accepted/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/decision-accepted/annotations.jsonl' '--receipt' '/tmp/t124-walk.qQcfjK/decision-accepted/accepted.receipt.json' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=4
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'plan' '/tmp/t124-walk.qQcfjK/first/retract.patch.json' '--corpus' '/tmp/t124-walk.qQcfjK/decision-withheld/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/decision-withheld/annotations.jsonl' '--output' '/tmp/t124-walk.qQcfjK/decision-withheld/withheld.plan.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'approve' '/tmp/t124-walk.qQcfjK/decision-withheld/withheld.plan.json' '--operation' 'sha256:babbdf630ed7994702fdd92951dd6d52678735b8f2a7416f9b5da1dedfeb9f9e' '--actor' 'human:reviewer' '--review' 'Checked exact retraction' '--output' '/tmp/t124-walk.qQcfjK/decision-withheld/withheld.receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/decision-withheld/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/decision-withheld/annotations.jsonl' '--fingerprint' 'a93404e6293f5ec8b6d4115f177fbfaf5ae96e5a209430f63bc7c2a5153596c0' '--status' 'withheld' '--reviewer' 'Latin reviewer' '--reason' 'Checked annotation in context' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/decision-withheld/withheld.plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/decision-withheld/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/decision-withheld/annotations.jsonl' '--receipt' '/tmp/t124-walk.qQcfjK/decision-withheld/withheld.receipt.json' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=4
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'plan' '/tmp/t124-walk.qQcfjK/first/retract.patch.json' '--corpus' '/tmp/t124-walk.qQcfjK/decision-withheld/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/decision-withheld/annotations.jsonl' '--output' '/tmp/t124-walk.qQcfjK/decision-withheld/unrelated.plan.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'approve' '/tmp/t124-walk.qQcfjK/decision-withheld/unrelated.plan.json' '--operation' 'sha256:babbdf630ed7994702fdd92951dd6d52678735b8f2a7416f9b5da1dedfeb9f9e' '--actor' 'human:reviewer' '--review' 'Checked exact retraction' '--output' '/tmp/t124-walk.qQcfjK/decision-withheld/unrelated.receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'approve' '/tmp/t124-walk.qQcfjK/decision-withheld/unrelated.plan.json' '--operation' 'sha256:babbdf630ed7994702fdd92951dd6d52678735b8f2a7416f9b5da1dedfeb9f9e' '--actor' 'agent:test' '--review' 'Checked' '--output' '/tmp/t124-walk.qQcfjK/decision-withheld/agent.receipt.json' '--json'
exit=5
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/decision-withheld/unrelated.plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/decision-withheld/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/decision-withheld/annotations.jsonl' '--receipt' '/tmp/t124-walk.qQcfjK/decision-withheld/hand-agent.json' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=5
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/decision-withheld/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/decision-withheld/annotations.jsonl' '--fingerprint' '9fb7e94087742df8e28cdeaa9bf5670e806ca41e6c56d42f9c46509777402547' '--status' 'accepted' '--reviewer' 'Latin reviewer' '--reason' 'Checked annotation in context' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/decision-withheld/unrelated.plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/decision-withheld/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/decision-withheld/annotations.jsonl' '--receipt' '/tmp/t124-walk.qQcfjK/decision-withheld/unrelated.receipt.json' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/decision-withheld
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/decision-withheld/unrelated.plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/decision-withheld/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/decision-withheld/annotations.jsonl' '--receipt' '/tmp/t124-walk.qQcfjK/decision-withheld/unrelated.receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/foreign
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'plan' '/home/user/workspace/repo/tests/fixtures/annotations/fixture-morph.patch.json' '--corpus' '/tmp/t124-walk.qQcfjK/foreign/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/foreign/annotations.jsonl' '--output' '/tmp/t124-walk.qQcfjK/foreign/plan.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/foreign
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'segment' '/tmp/t124-walk.qQcfjK/foreign/corpus.jsonl' '--output' '/tmp/t124-walk.qQcfjK/foreign/annotations.jsonl' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/foreign
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/foreign/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/foreign/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/foreign/annotations.jsonl' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=4
cwd=/tmp/t124-walk.qQcfjK/stale
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'record' '/tmp/t124-walk.qQcfjK/stale/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/stale/annotations.jsonl' '--fingerprint' 'a93404e6293f5ec8b6d4115f177fbfaf5ae96e5a209430f63bc7c2a5153596c0' '--status' 'accepted' '--reviewer' 'R' '--reason' 'Checked' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/stale
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'inspect' '/tmp/t124-walk.qQcfjK/stale/annotations.jsonl' '--corpus' '/tmp/t124-walk.qQcfjK/stale/corpus.jsonl' '--check' '--json'
exit=1
cwd=/tmp/t124-walk.qQcfjK/stale
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'validate' '/tmp/t124-walk.qQcfjK/stale/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/stale/annotations.jsonl' '--json'
exit=1
cwd=/tmp/t124-walk.qQcfjK/stale
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'claims' 'inspect' '/tmp/t124-walk.qQcfjK/stale/corpus.jsonl' '--annotations' '/tmp/t124-walk.qQcfjK/stale/annotations.jsonl' '--check' '--json'
exit=1
cwd=/tmp/t124-walk.qQcfjK/stale
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'corpus' 'normalize' '/tmp/t124-walk.qQcfjK/stale/dtd.xml' '--adapter' 'usfx' '--corpus-id' 'reader' '--output' '/tmp/t124-walk.qQcfjK/stale/unsafe.jsonl' '--json'
exit=3
cwd=/tmp/t124-walk.qQcfjK/partial
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'plan' '/home/user/workspace/repo/tests/fixtures/annotations/mixed-valid.patch.json' '--corpus' '/tmp/t124-walk.qQcfjK/partial/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/partial/annotations.jsonl' '--policy' '/home/user/workspace/repo/tests/fixtures/policy/mixed.policy.json' '--output' '/tmp/t124-walk.qQcfjK/partial/plan.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/partial
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/partial/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/partial/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/partial/annotations.jsonl' '--policy' '/home/user/workspace/repo/tests/fixtures/policy/mixed.policy.json' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/partial
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/partial/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/partial/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/partial/annotations.jsonl' '--policy' '/home/user/workspace/repo/tests/fixtures/policy/mixed.policy.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/partial
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'approve' '/tmp/t124-walk.qQcfjK/partial/plan.json' '--operation' 'sha256:9f417d81631403e1d4a2299464ce6c191d20f18457e60271ade61ae9cf1cf34d' '--actor' 'human:reviewer' '--review' 'Checked exact proposal' '--output' '/tmp/t124-walk.qQcfjK/partial/receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/partial
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/partial/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/partial/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/partial/annotations.jsonl' '--policy' '/home/user/workspace/repo/tests/fixtures/policy/mixed.policy.json' '--receipt' '/tmp/t124-walk.qQcfjK/partial/receipt.json' '--timestamp' '2026-01-02T03:04:05Z' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/partial
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/partial/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/partial/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/partial/annotations.jsonl' '--policy' '/home/user/workspace/repo/tests/fixtures/policy/mixed.policy.json' '--receipt' '/tmp/t124-walk.qQcfjK/partial/receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/partial
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'approve' '/tmp/t124-walk.qQcfjK/partial/plan.json' '--operation' 'sha256:9f417d81631403e1d4a2299464ce6c191d20f18457e60271ade61ae9cf1cf34d' '--actor' 'human:second-reviewer' '--review' 'Reviewed unchanged proposal again' '--output' '/tmp/t124-walk.qQcfjK/partial/new-receipt.json' '--json'
exit=0
cwd=/tmp/t124-walk.qQcfjK/partial
'/home/user/workspace/repo/.venv/bin/latinitas-cards' 'annotations' 'apply' '/tmp/t124-walk.qQcfjK/partial/plan.json' '--corpus' '/tmp/t124-walk.qQcfjK/partial/corpus.jsonl' '--sidecar' '/tmp/t124-walk.qQcfjK/partial/annotations.jsonl' '--policy' '/home/user/workspace/repo/tests/fixtures/policy/mixed.policy.json' '--receipt' '/tmp/t124-walk.qQcfjK/partial/new-receipt.json' '--json'
exit=0

```

### Observed artifact byte digests

Distinct file-name/SHA-256 snapshots from those calls (including inputs and controls).

```text
accepted.plan.json  e1b1c9a2f08c9304b70b6d9f863c7296d614139b51faed3ad82d8c3daf8bae4b
accepted.receipt.json  4f1bdc2a8f87d35b9adc15dacd37e87681ba85c88344783fa75e953489f3eb6c
annotations.jsonl  16148b7a890ee573dece6a3f058359ac979bcbc57758e254aaca6837f876021c
annotations.jsonl  32b29ffda2d2a271cbc9de3ea9a50f79b550e712c770b88219bffbee775a8283
annotations.jsonl  55793989427969de2270d8374b74345b3bfacb07a43755325de38bbc1598342e
annotations.jsonl  cb822ae209d8905721a582b3e271582f8da76f9226a1c017ccf167afb617ce4e
annotations.jsonl  e2b54c72f6ae19fb838824fb2253491bcfa6f5c1e2a192bdd930c0c21b882f47
annotations.jsonl  fcf222b066850ff4e5c3ed1556f2bbb4093ea2366fc01bafd3bded9bea77a252
bad-batch.json  f3ab7173b4084e7a374b63d35071a3cd349c20dedc51a31ab07b6d7d32ad1f39
corpus.jsonl  514a636f9133e10991757f4be917e757b837bed11710ad8dd571f2c61b9ad41f
corpus.jsonl  e1b20bca0560eb1c95348f92447f061e21c00d99436458e41a715da3a1a08453
corpus.jsonl.claims.json  10837b4d9b453fd2e2917c66a7149c3672e489734ff4b68a50e2ab5212531520
corpus.jsonl.claims.json  33da7dca7c91a57ff9ad353a604b9e1991e021a280e617f440bd9d69ffb8e67a
corpus.jsonl.claims.json  ca41f92f5882904c3722bb16cfa50d3649134872f536acbc514401df5a5dec37
corpus.jsonl.claims.json  ce2a16b80ea6197de04b3b85e2ea4c43822a5a0c28758876b8a7277ca3ae40d2
dtd.xml  61747af6179fdb14a1977ee2c3ad98d2a263eaa7f80dec25de8e3ecab55d859f
hand-agent.json  c5e2639d405494f2ecef2f6309b80542b0a08d0194aa90d6c805c32465b0eafb
labeled.jsonl  437ecf629895f7772b830756e5e5182ddc901b38444704469affa672a6cbd602
new-receipt.json  9910f3df3029cfe4ecd43d49c2aad2ed6dcd4dffeece1f5bd6657e59f74d525b
plan.json  23c37bae37236083a45997eae8ef7045e09143ca37c06fa1441574580a176af2
plan.json  257ce914bf215785ec024ac5a494a1ff1498bd11d24aa06b6507e47cc433dfcd
plan.json  8e115f3c0c71b9f0b3129c4bed952e47b8733366b134bd903391ec3b344dd3a0
reader.jsonl  963ae183ad8c58a65dbd85b706a57f5e56cb32ec42610fe854e703c8a3ec0525
reader.jsonl  d6b785eaa5c6ca8846708ddf2abdd6616d14033dbf133ed0f85b55276ceb9ff0
reader.manifest.json  4700b87d033bbfd732da01ed69e048000f42d816990ec702b29cb3a58ffedf63
reader.manifest.json  668d1e0da8cb55955e0656aec7fec00844b918d56f3db3bac3291e62c9ebf350
receipt.json  09444f134d10bd08da59592e3270910b29118af02e19ccac24fb97e212e63ccd
receipt.json  26179770b141ee1c78bd99c464c669800aa9de30c8a3d040c9c500b3b500ee33
retract.patch.json  25f7dff57d5e7070e07a1512bb5d4dcf17a0d6a1c64e368176d742ae93141259
source.txt  0d62f24ba941b5b83b0c0428153649549800efa7feb5baebc58c32807451d9ba
source.txt  d1f89121fc1ef32e342817ddb85b1784665511eda8700438532d60d455f817cd
special.annotations.jsonl  331ee5bdd2401c2f27ba1694bd77b32a3318084c905fbef28e4e38b3f931f042
special.jsonl  65edeb591fc6cc1ae75f55ef42698e2a7110fc304bbbc59b92f31040bb5a7b76
special.txt  dbde8b2999dba2760790dffd47b6edb7479e46d1407fcf48f1893690995e4eba
unrelated.plan.json  fa88cd88c30fa6d980e673a1d6e37160ef1e724771f2269b18341932e66b7d19
unrelated.receipt.json  0899625ff542910093103d7c1e5aa7e1dd129072ed14067fa55d784b6b6fefa6
vulgata.jsonl  cefcaa8f977df72acd71b76ace7a45ccbe8e81105dd56a86815964e531e0927d
withheld.plan.json  e1b1c9a2f08c9304b70b6d9f863c7296d614139b51faed3ad82d8c3daf8bae4b
withheld.receipt.json  4f1bdc2a8f87d35b9adc15dacd37e87681ba85c88344783fa75e953489f3eb6c
```
