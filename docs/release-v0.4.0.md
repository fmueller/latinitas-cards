# v0.4.0 publication — 2026-11-01

**Published:** [GitHub release](https://github.com/fmueller/latinitas-cards/releases/tag/v0.4.0)
and [PyPI 0.4.0](https://pypi.org/project/latinitas-cards/0.4.0/).

## Publication receipt

- Released candidate:
  [`3d51ee6`](https://github.com/fmueller/latinitas-cards/commit/3d51ee680ea68ef9d521970178706cc3eb977d57).
  Exact-head [Build](https://github.com/fmueller/latinitas-cards/actions/runs/41289777048)
  and [CodeQL](https://github.com/fmueller/latinitas-cards/actions/runs/41289776580)
  passed all eight checks, including Python 3.13/3.14 tests and packaging.
- One annotated `v0.4.0` tag was pushed. Tag object
  `4e05ec66bf23f8f4e012b83767293ccc6b527f82` peels to the candidate above;
  the tag was not moved or recreated. Runtime source, project version and lock
  remained unchanged from implementation base `bf1358a` during preparation.
- [Release run 41290520404](https://github.com/fmueller/latinitas-cards/actions/runs/41290520404),
  attempt 1, passed all five jobs: `build`, `verify (3.13)`, `verify (3.14)`,
  `publish` and `verify-pypi`. The attempt-specific jobs endpoint and completed
  job logs confirmed the final result; earlier reads briefly lacked the last
  job and returned a PyPI 404, so those preliminary results were not accepted.
- The integration's normal deployment approval POST returned HTTP 403. No
  approval retry or protection bypass was attempted. The maintainer then
  reported personally approving `pypi`; the pending deployment cleared and
  publication proceeded under the unchanged required-reviewer/tag-only rules.
  The approval-history API returned an empty list, so this record does not
  claim that API independently identified the human approver.
- `publish` used OIDC and the same verified `dist` artifact, without rebuilding.
  [Artifact 13077152205](https://github.com/fmueller/latinitas-cards/actions/runs/41290520404/artifacts/13077152205)
  was downloaded before approval; both package hashes matched the build logs.
  Both actual workflow packages contain the tagged README byte-for-byte as
  their Markdown long description.
- `verify-pypi` installed exactly `latinitas-cards==0.4.0` from PyPI, ran CLI
  help and passed the CLTK/Stanza/PyTorch absence assertions. Only afterward,
  both PyPI files were independently downloaded and hashed. Their exact
  filename set and SHA256 values equal the reviewed workflow artifact and
  [PyPI JSON metadata](https://pypi.org/pypi/latinitas-cards/0.4.0/json):

| File | SHA256: artifact = PyPI JSON = downloaded PyPI bytes |
| --- | --- |
| [Wheel](https://files.pythonhosted.org/packages/cd/12/2b984256fa2d84c1b270fd9542c80b953e173191aef71904d31c0211eef4/latinitas_cards-0.4.0-py3-none-any.whl) | `b549cf4850b199b79462f95d5090145f2cbecb04d6c2c9ed782953df99692403` |
| [Source distribution](https://files.pythonhosted.org/packages/dd/5e/d5b48ec7df677a46ca36c06920962085bcecaf506bcc9031ac0c6943f64a/latinitas_cards-0.4.0.tar.gz) | `817c671513e75b7650369ec20af9173d214472e1bf3e6abb98caa39c8caa04de` |

Every published wheel/sdist matched, with no extra files. The non-draft,
non-prerelease GitHub release was created only after that comparison, at
2026-11-01T09:45:06Z. No TestPyPI, manual upload, dispatch, republish, dependency
change or protection change was used.

The maintainer explicitly accepted the existing GitHub-reported low-severity
[Dependabot alert 146](https://github.com/fmueller/latinitas-cards/security/dependabot/146)
risk before tagging. The integration received HTTP 403 reading its details;
the affected package, manifest and runtime exposure remain unverified. This
is a scoped risk acceptance, not a fix, dismissal or assertion of no exposure.
The dependency and alert state were not changed.

Preparation validation on the released candidate passed the exact Ruff, mypy,
pytest chain: 3,232 passed, seven skipped, one existing serializer warning
(585.40 seconds). The main push's full pre-push suite also passed (584.27
seconds). Independent general, release/prose and adversarial review resolved
the adjacent installation-documentation test's obsolete v0.3.0 expectations;
no findings remained. The installed-wheel README setup/preview/export/repeat
flow ran on an isolated supplied APKG, with unchanged source and identical
repeat CSV. CommonMark rendering and all 50 README repository links/anchors
passed. These preparation checks are distinct from the final evidence-commit
validation and CI.

The follow-up evidence change passed the full Ruff, mypy and pytest chain:
3,232 passed, seven skipped, one existing serializer warning (593.63 seconds).
Dedicated simplification and independent general/prose/adversarial reviews
reported no remaining findings. Evidence-commit CI belongs to that later main
revision, not the unchanged released tag.

## Authorization and preparation

The maintainer authorized the T-173 gate and v0.4.0 publication on 2026-11-01,
including release preparation, one annotated tag, protected-environment approval
where permitted, and a publication evidence commit. The prerequisites and
ordered checks follow [the runbook](releasing.md).
The maintainer confirmed the current PyPI Trusted Publisher and approved the
bounded README/docs refresh before continuing the release. At preparation time,
no v0.4.0 tag or distribution existed. T-173 completion requires final evidence
validation as well as publication, not authorization alone.

## Publication preflight, 2026-11-01

- Fetched `origin/main` equals implementation base `bf1358a`; the worktree was
  initially clean. Taskrail 0.4.0 validation passed with `specs/v0.4.0.md` active
  and T-173 the only open task on that spec. Future-spec tasks remain excluded.
- Read-only checks found no remote `v0.4.0` tag or GitHub release. The exact
  PyPI 0.4.0 JSON endpoint returned 404; the project reported 0.3.0 as latest.
- GitHub environment `pypi` requires reviewer `fmueller`, permits self-review,
  disables administrator bypass, and has exactly one deployment policy:
  `v*` of type `tag`, with no branch policy. These settings were not changed.
- The [license audit](license-compatibility-audit.md#bundled-corpus-provenance)
  records the maintainer's 2026-10-20 Vulgate descriptor and eBible.org latVUC
  derivation confirmations. The current bundled XML hash still matches the
  audit. This is not an independent upstream byte comparison or clearance of
  optional NLP distribution obligations.
- The maintainer confirmed the current PyPI Trusted Publisher configuration:
  tuple is owner `fmueller`, repository `latinitas-cards`, workflow `release.yml`,
  environment `pypi`. This is current maintainer confirmation, not independent
  access to private PyPI configuration or a test upload.

## Scope and provenance

The accepted implementation base is
[`bf1358a`](https://github.com/fmueller/latinitas-cards/commit/bf1358acea8eebc50b69c8013a2942ce2a718208),
including readable localized source details and nested technical provenance.
Release preparation changes documentation and release records, not runtime code.
Package and project lock metadata are 0.4.0. The final tag candidate and its
exact-head Build/CodeQL results are recorded above; the implementation base
is not the release candidate's commit identity.

v0.4.0 adds corpus-first generation with four reviewed context/translation
recipes, deterministic ranked budgets and sticky selections, a bounded
Perseus/EpiDoc TEI adapter, aligned translation patch import, and German/English
generated content. Reference translations guide self-assessment; they are not
exact-string grading keys. The [binding spec](../specs/v0.4.0.md),
[corpus walkthrough](corpus-first-cards.md) and
[CLI contract](agent-cli-contract.md) define the supported inputs and boundaries.

## Offline walkthrough and checks

The T-172 execution record in
[its task notes](../planning/tasks/T-172-validate-integrated-v0-4-0-workflows-offline-and.md)
records commands, results, review dispositions and the final independent
checkpoint identity. Local transcripts and generated fixture inputs/outputs
are retained in the owning checkout's orb. These records distinguish the
initial blocked attempt from the resumed candidate validation.

The walkthrough uses a disposable directory outside the checkout, the
documented public CLI and stipulated synthetic assertions on
`Marcus amicum videt.`. It covers:

- TEI and plain-text normalization, validation and segmentation;
- an aligned English translation through `annotations import-translations`,
  plus anchored `latinitas-morph/1`, token-gloss and phrase-translation facts,
  all through annotation plan, explicit approval and application;
- individual `claims record` decisions, corpus profile validation, selection
  and preview for all four recipes on both adapters;
- known-lemma input with matching source identities in reports and previews,
  ranking changes without identity, eligibility, card-field or deck mutation;
- deselect/restore, initial CSV export, `inspect --check`, corpus managed-plan
  request and an English deck-profile preview;
- negative controls for applied but unreviewed facts, changed source context,
  tampered CSV, unknown TEI elements and an unsupported French corpus profile.

Offline managed planning uses an explicitly synthetic destination snapshot and
reviewed ownership fixture. It proves the CLI handoff contract, **not** a native
collection capture, import, scheduling or preservation result. Real snapshots
must come from the [closed-backup capture/adoption flow](managed-capture-and-adoption.md).

T-154 characterization goldens remain byte-identical to their introducing commit
[`681cf22`](https://github.com/fmueller/latinitas-cards/commit/681cf22e672ce595efeac418d859509569bc9807).
T-174's authorized filtered claims-v1 updates are a different fixture set; no
goldens are regenerated here. The no-grading test inventories source AST
equality checks involving translation payloads, with exact reviewed exceptions
for validation, conflict and drift checks and injected grading controls. It is
a bounded architectural guard, not arbitrary whole-program dataflow proof.

## Readiness evidence and limits

The [Linux Anki Desktop acceptance kit](native-v040-acceptance.md) defines the
required rows; formal gate disposition remains maintainer-owned. Browser previews,
synthetic snapshots, backend fixtures, CSV inspection and passing automated tests
do not substitute for actual native evidence. The gate
requires manual corpus-v1 setup, German/English four-recipe import and re-import,
first-field matching with Personal Notes unmapped, actual fronts/answers in
light/dark/auto with light-night-light toggles, English deck-first samples,
and real Perseus Caesar BG/Aeneid smoke runs. AnkiMobile is outside that gate
and, together with real scheduled collections/history, explicitly out of scope
for v0.4.0 release verification and readiness, not a blocker. Both remain
unverified, not compatible or schedule-preserving by assertion. The required
Desktop checks use isolated synthetic profiles without personal accounts or sync.

### Cumulative acceptance evidence and revision boundaries

The earlier native acceptance revision is
[`041989c`](https://github.com/fmueller/latinitas-cards/commit/041989c54f32ed3a0ebc0481fd81776f2c368396),
recorded by the [native acceptance owner](https://ampcode.com/threads/T-01a1919f-f44c-7114-99d0-a46e95a25537).
That retained evidence records official Anki Desktop 26.09.3 on Debian 12 Wayland,
342 inspected states, 12 synthetic notes and 19 new/unanswered cards. It includes
import/update/repeat, Personal Notes and identity checks, and native theme toggles.
It is reused evidence, not a fresh native rerun or proof about real review history.

The [T-183 implementation and acceptance record](https://ampcode.com/threads/T-01a1947b-9810-7027-aaff-212d65be4cda)
binds the final presentation to implementation commit `bf1358a` and accepted
product tree `c54166a5bb878d95437dee8841aedc694d07d3d7` (planning held at
its base). Official Anki Desktop 26.09.3 inspected 24 final-CSS states:
Caesar/Aeneid × German/English × three disclosure states × light/dark.
Two actual-input clips and event receipts exercised pointer, Space, Enter and
Tab, including neighboring Replay Audio; light-dark-light toggles passed.
Actual imports updated eight German and eight English notes without creating
notes. The quiescent audit covered all 16 notes/cards: nine localized fields,
23 other exported fields and tags exact, canonical JSON and hrefs equal,
IDs and all scheduling columns unchanged, and zero revlog rows. Citation HTML
was entity-normalized by Anki, so byte identity is not claimed. Personal Notes
were empty, not a test of preserving nonempty user notes. CSS differed only by
the native editor's trimmed final newline.

T-183's full local chain passed 3,232 tests with seven skips and one existing
serializer warning. All eight checks passed on `bf1358a`:
[Build](https://github.com/fmueller/latinitas-cards/actions/runs/41259760087)
and [CodeQL](https://github.com/fmueller/latinitas-cards/actions/runs/41259759965).
The earlier adversarial campaign was run on `6ce6095`, not `bf1358a`:
320 public CLI calls, 388 checks and 204 supplemental tests reported clean.
The [later film record](https://ampcode.com/threads/T-01a1947c-0432-75a9-98a1-41a66aef1d19)
reports a fresh 113-second film on `bf1358a`, including real Scaife clickouts;
decode, frame and ASR checks are not human listening or a new full acceptance run.
These are cumulative, scoped receipts, not universal testing on one revision.

Real-source smoke tests used PerseusDL `canonical-latinLit` revision
[`8b536bd`](https://github.com/PerseusDL/canonical-latinLit/tree/8b536bdf3ed1f43494055efe112433b4a58eab67):

| Source | SHA256 of unmodified XML | Normalized passages |
| --- | --- | ---: |
| `data/phi0448/phi001/phi0448.phi001.perseus-lat2.xml` | `15ec3ef2a36352742ddb75d4f7da23a143899b6b60ee3a7ca3192d4917ca1568` | 2,150 |
| `data/phi0690/phi003/phi0690.phi003.perseus-lat2.xml` | `e639e6a0e27427ef28031850c4d73aa2b63b9f2cf8a5cf584b37685c52fb874c` | 9,843 |

Both `corpus normalize --adapter tei` and `corpus validate --json` exited 0.
Use valid corpus IDs `phi0448-phi001` and `phi0690-phi003` (dots are not allowed).
Fresh downloads of both pinned XML files on 2026-11-01 matched these hashes;
normalization and validation rerun on implementation base `bf1358a` again
exited 0 with 2,150 and 9,843 passages respectively. The earlier
independent XML extraction matched every canonical passage and reference. The
Aeneid has 9,897 source lines: 54 editorially deleted lines are omitted from
canonical passages with warnings and retained in notes. Source gaps at Caesar
8.55.2 and Aeneid 12.732 also warn; neither passage is dropped.

The corpus walkthrough and native/annotation fixtures use stipulated synthetic
German/English translations, not an identified external translated edition.
Mock fixture licence metadata is not rights evidence for a real edition.
The downloaded Latin XML remains local test data, not bundled or redistributed:
Caesar's header declares CC BY-SA 4.0; Aeneid has no licence element, so retain
the repository's CC BY-SA 4.0 default and varying-rights caveat rather than infer
blanket public domain. Future real parallel translations require their own
translator, edition, source and rights review. This scoped review does not grant
redistribution rights or certify linguistic accuracy.

These observations supply scoped test evidence; publication authorization is
recorded separately above. Future release specs
do not yet inherit a complete candidate-bound media/independent-QA approval process;
that policy work needs a separately approved change, not an implied skill update.

Other limits remain:

- Per-occurrence annotations need current accepted claims. Applying a patch is
  not linguistic acceptance; confidence and producer kind never grant it.
- Source corrections retain passage identity. Retained exercises whose context
  changes require review; changed folded targets produce different exercise
  identities. Obsolete stored claim decisions need the documented reviewed
  sidecar migration; there is no general CLI pruning command.
- Known lemmas affect ranking and informational provenance only. Without them,
  every word counts as unknown, so tight unknown-word limits may filter all cards.
- TEI is fail-closed outside its documented subset; verse fragments do not
  automatically become complete sentence exercises, and cross-passage sentence
  joining is unsupported. v0.3.0 rejects TEI corpora written by this candidate.
- Initial CSV import can create notes; managed CSV updates cannot create or
  retire notes, change templates or alter sibling structure. Unsupported
  structural plans require separately reviewed manual resolution.
- A deck-profile language change stales its reviews. A corpus target-language
  change creates different identities and needs a new selection, not automatic
  retirement of the previous language's notes.
- Optional CLTK/Stanza/PyTorch resources and live model endpoints are not part
  of offline acceptance. No new optional-runtime, Windows, native-client,
  mutation-effectiveness or general linguistic-quality claim is made here.

Release publication, platform CI and the existing license/provenance review
remain separate obligations under the [release procedure](releasing.md) and
[license compatibility audit](license-compatibility-audit.md).
