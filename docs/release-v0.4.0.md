# v0.4.0 candidate readiness — 2026-10-31

This is an **unreleased candidate**, not publication approval or native Anki
acceptance. The published package remains v0.3.0. T-173 is the blocked,
maintainer-only release gate; this preparation does not unblock it, create a
tag, publish a release or upload a distribution. Its date may be corrected by
the maintainer before tagging.

## Scope and provenance

The accepted runtime base is
[`3a20e48`](https://github.com/fmueller/latinitas-cards/commit/3a20e48c7ad1bdee8550422482f85610b8ee8f80),
including the known-lemma report provenance repair and the dependency-only
advancement. This candidate changes release metadata, documentation and tests,
not runtime code. Package and project lock metadata are 0.4.0; the non-project
dependency inventory stays unchanged from that base.

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

### Candidate-bound verification, 2026-10-31

The fetched `origin/main` candidate is
[`041989c`](https://github.com/fmueller/latinitas-cards/commit/041989c54f32ed3a0ebc0481fd81776f2c368396),
identical to the [native-tested revision](https://ampcode.com/threads/T-01a1919f-f44c-7114-99d0-a46e95a25537).
That retained evidence records official Anki Desktop 26.09.3 on Debian 12 Wayland,
342 inspected states, 12 synthetic notes and 19 new/unanswered cards. It includes
import/update/repeat, Personal Notes and identity checks, and native theme toggles.
It is reused evidence, not a fresh native rerun or proof about real review history.

Real-source smoke tests used PerseusDL `canonical-latinLit` revision
[`8b536bd`](https://github.com/PerseusDL/canonical-latinLit/tree/8b536bdf3ed1f43494055efe112433b4a58eab67):

| Source | SHA256 of unmodified XML | Normalized passages |
| --- | --- | ---: |
| `data/phi0448/phi001/phi0448.phi001.perseus-lat2.xml` | `15ec3ef2a36352742ddb75d4f7da23a143899b6b60ee3a7ca3192d4917ca1568` | 2,150 |
| `data/phi0690/phi003/phi0690.phi003.perseus-lat2.xml` | `e639e6a0e27427ef28031850c4d73aa2b63b9f2cf8a5cf584b37685c52fb874c` | 9,843 |

Both `corpus normalize --adapter tei` and `corpus validate --json` exited 0.
Use valid corpus IDs `phi0448-phi001` and `phi0690-phi003` (dots are not allowed).
Independent XML extraction matched every canonical passage and reference. The
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

These observations supply test evidence, not human readiness approval or separate
permission to publish. T-173 remains blocked and unchanged. Future release specs
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
