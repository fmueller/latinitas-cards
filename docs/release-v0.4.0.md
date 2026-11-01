# v0.4.0 release preparation — 2026-11-01

This is an **unreleased candidate**. The published package remains v0.3.0.
The maintainer authorized the T-173 gate and v0.4.0 publication on 2026-11-01,
including release preparation, one annotated tag, protected-environment approval
where permitted, and a publication evidence commit. Publication still requires
the prerequisites and ordered checks in [the runbook](releasing.md).
The maintainer confirmed the current PyPI Trusted Publisher and approved the
bounded README/docs refresh before continuing the release. At preparation time,
no v0.4.0 tag or distribution has been published. T-173 completion requires
successful publication and the follow-up evidence below, not authorization alone.

## Publication preflight, 2026-11-01

- Fetched `origin/main` equals implementation base `bf1358a`; the worktree was
  initially clean. Taskrail 0.4.0 validation passed with `specs/v0.4.0.md` active
  and T-173 the only open task on that spec. Future-spec tasks remain excluded.
- Read-only checks found no remote `v0.4.0` tag or GitHub release. The exact
  PyPI 0.4.0 JSON endpoint returned 404; the project reports 0.3.0 as latest.
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
exact-head Build/CodeQL results must be recorded after preparation is committed;
the implementation base is not a claim about that future candidate's CI.

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
