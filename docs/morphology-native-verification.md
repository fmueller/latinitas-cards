# Morphology native presentation gate

Native acceptance was completed on **2026-10-06** on Anki Desktop 26.09.2 and on
AnkiMobile 25.09 (iPhone 16 Pro Max, iPad Air M4); see the
[native acceptance run](#native-acceptance-run--2026-10-06) and its limitations.
The 2026-10-05 Desktop run below is the earlier partial record. Neither T-053
browser captures nor T-061 backend import checks count as presentation evidence.

## Executed partial Desktop check — 2026-10-05

Source baseline: [accepted main 3086b7c](https://github.com/fmueller/latinitas-cards/commit/3086b7c7d4dc59a1b549b0e1a3f65750b99ca9e9).
Fetch confirmed that origin/main contains this commit and the checkout matched
it before Taskrail selected T-062. T-053 was completed; the active spec was
`specs/v0.2.0.md#morphology-themes`.

Environment: Linux x86_64 orb Desktop, Anki Desktop **26.09.3** from
`aqt==26.9.3`, Python 3.14.2, Qt xcb on Xwayland display :0. This is the actual
Anki reviewer, not Chrome/mobile emulation. No physical touch hardware was used.

Availability probes found no installed Anki/aqt/PyQt6, `ideviceinfo`,
`idevice_id`, `xcrun`, `adb`, or USB device bus. The orb is Linux; the available
`amp orb simulator` command is for a Mac orb and is not an iPhone/iPad here.
No native AnkiMobile device/client/version is available in this checkout.
`amp orb desktop ensure` started the Desktop. The first disposable Anki launch
failed because Qt xcb libraries were missing; `ldd` identified the missing
libraries. Installing libxcb-cursor0, libxcb-image0, libxcb-keysyms1,
libxcb-render-util0, libxcb-xkb1 and libxkbcommon-x11-0 enabled the GUI.

The executed commands were:

```bash
uv run --with aqt==26.9.3 python -c \
  'import aqt, anki.buildinfo; print(aqt.__file__, anki.buildinfo.version)'
amp orb desktop ensure
DISPLAY=:0 QT_QPA_PLATFORM=xcb uv run --with aqt==26.9.3 \
  anki -b "$work/anki" -p T062Disposable
DISPLAY=:0 QT_QPA_PLATFORM=xcb T062_BASE="$work/anki" \
  T062_ARTIFACTS="$captures" uv run --with aqt==26.9.3 \
  python /tmp/t062-desktop-smoke.py
```

`work` was a new temporary directory. Initial language/prefs dialogs were
acknowledged; Anki created the disposable `User 1` profile. The smoke script
refused a nonempty collection. It copied the reference fields, ten frozen
templates and CSS through Anki's model API into this empty profile, then added
one synthetic note through the native API. No live/user collection, sync,
provisioning command, destination journal or scheduled template was touched.
This API seeding is **not** the documented manual GUI setup/CSV delivery test.

The fixture used existing `review_claim`/`assess_claim` gates for two stipulated
accepted perfect claims (explanation and segmentation), and
`compare_principal_parts`/`render_cards` for both existing recipes. It did not
infer or approve corpus claims. The role status sequence was independently
asserted as absent, absent, present, withheld. Four native cards were created
at frozen ordinals 2, 4, 7, 9. Only ordinal **2**, perfect completion, was
actually displayed. Creation of recognition cards is not presentation proof.

The script entered the native reviewer and captured the front, invoked native
`reviewer._showAnswer()`, captured the revealed answer with `QWidget.grab()`,
and closed Anki normally. Exit status was 0 and output was
`T062 native Desktop front/answer captures written; no Mobile or complete matrix claim`.
This is programmatic native reveal, not a tested keyboard/touch reveal gesture.

Both captures were inspected. The front showed the perfect completion blank
and the Latin cue `Supinum: amātum`. The muted/light/static answer showed
`amāvī`, the accepted perfect split/explanation, and the four-role comparison:
present and infinitive explicitly absent, perfect accepted, supine withheld.
The core answer and comparison were visible without expansion. Text was legible,
the withheld row wrapped without clipping, and no overlap was observed at the
1100×900 window request. This does not establish both recipes' Latin-first
acceptance or readability on other clients/sizes.

## Evidence binding

The execution thread retains the inspected front/answer PNGs, the smoke script
and its JSON report: [T-062 evidence](https://ampcode.com/threads/T-01a10a92-901d-7165-ba89-279460e43b32).
The report records actual card IDs/ordinals and the two claim fingerprints.
Hashes bind this run, not stable native IDs across reruns.

| Bound content/setup or artifact | SHA-256 |
|---|---|
| `cards.py` | `1933af8413902e3967c608a7446434b94dd55197f9ed84a43bcf7b1538296e98` |
| `profile.py` | `3145d640ccc7faea7c54d7501b0bd5193acd9d04e0e01bf5db66cb7a3d14c990` |
| `principal_relationships.py` | `ca6411b6a6d2ca538ea198e54f9c5f1740caa6ffec8443db0554933e8d5aa192` |
| `claim_review.py` | `6a2ef2bdf769d1826490a418495c5d24bf7168764d916d7eab768955c59a233c` |
| `reference_templates.py` | `727d6c205d00227e99970a8b94f10fa2fde60dcfb9203de43b90ca10591bfc44` |
| Reference CSS v2 | `a038b778ed678d932d44ec27714d387e9d5b786675a1c5d4d1e2f1d10649d752` |
| `docs/reference-note-type.md` | `537c3fee549fc9a11ccc90f9dfcb85e1b368fa7b5ef15ce701d29368ee18ecdb` |
| Front PNG | `d539253d72d61322583569687a70674f1524e74141bd6faa1f88a45cbe4e110b` |
| Answer PNG | `f7ae222f0344c3fbce6bdbc1ed47b58d6ac793556a1faba54ef864c88de6385c` |
| Smoke script | `a9dbcbb4fb21e8440b352f275dabee1f1f2a40a0f2adade97bb5a23891129151` |
| JSON report | `3d2d9ffe492f4e4f88497fb4beb345193a9816320ecf5e26ff7cace8cc42d630` |


## Native acceptance run — 2026-10-06

The maintainer ran the checks by hand on their own clients. Source code matched
[be41c4d](https://github.com/fmueller/latinitas-cards/commit/be41c4d); later commits
changed only specs and tasks. Hashes of the bound source files are unchanged from the
table above. `form_parsing.py` is
`08bfbe2678d1f241edd14657074c8c72a89f0c7f7d4eaf5aee2b938cfc79cf5e` and
`docs/contextual-form-parsing.md` before this update was
`59f3e4da9d7e1f43bfb3c91ddde6f5c64c070d175be5900ff4b87e2ecc0b9f5e`.

| Client | Version | Device / OS |
|---|---|---|
| Anki Desktop (secondary) | 26.09.2 (bb0dd6d1) | Linux desktop |
| AnkiMobile (primary) | 25.09 | iPhone 16 Pro Max, iOS 26 |
| AnkiMobile (primary) | 25.09 | iPad Air M4, iPadOS 26 (reported as iOS 26) |

### Setup and delivery

`scripts/build-native-acceptance-kit.py` (run with `anki==26.9.3`) built the kit.
It contains a carrier `.apkg` with the reference note types (fields, frozen templates and
reference CSS v2 taken from the authoritative contract) and exporter CSVs produced
through `prepare_principal_part_export`/`write_principal_part_csv` and the
`form-parsing export` CLI. Claims are stipulated fixture reviews
(`native-test fixture`), not calibrated analysis. Per maintainer decision 2026-10-06
the hand-typed GUI note-type setup was waived, and the carrier package delivered the
note types. A disposable-backend dry run passed before handoff: carrier import, every
CSV mapped (HTML, tags column, no missing note type), expected counts, and identical
card IDs/ordinals after the retheme update.

The maintainer backed up the collection and imported the carrier and CSVs A–F, G and H
through the native Desktop dialog: update existing, note-type match scope, Personal
Notes unmapped. The content lived in a separate `Latinitas Test` deck tree in the main
profile. It reached both iOS devices through AnkiWeb sync after deck A was reimported in
its original muted/light/static form.

| Kit file | Content | SHA-256 |
|---|---|---|
| `0-note-types-carrier.apkg` | both note types + 2 placeholders | `a325c221c37f4c51220f3a164eaa23d572cbe1ce6c33f9f930866f30ff7a6e8c` |
| `A-principal-parts.csv` | muted/light/static; amō, moneō, ferō, dīcō; 4 notes, 28 cards | `2b7902a0ffe2307715bf592a095ddc7eae19d0033c6c56087fd9a72ad700b7b3` |
| `A2-retheme-update.csv` | same objects, monochrome/dark/disclosure | `de162fc2cf6daabf389a746fa1a04de885caacecdc5a5999cc69cc68bee1a66d` |
| `B-principal-parts.csv` | amō muted/dark/static, 8 cards | `0e6febc84c288a82df28f0cc7ec02cfd5399680e32c8b09edb41467f8c2eb8d3` |
| `C-principal-parts.csv` | amō monochrome/light/static | `a06c2cbcd77b7c4960c4530ba9d0d10f28e915247a3547bf8837631180fb84a7` |
| `D-principal-parts.csv` | amō monochrome/dark/static | `d830f69ab9d6fc1a5c83640f16a0cf5078690897a70e4b1a83ad61f65c1cf690` |
| `E-principal-parts.csv` | amō muted/light/disclosure | `a7db416764569ea42a82e2c56b2a5d84e3f78c258622524f3fe232f1f6951de5` |
| `F-principal-parts.csv` | amō monochrome/dark/disclosure | `50bf7df8a3f2303fb7caf731e8360ae96baf389ddfa13181e234da67b9325c7e` |
| `G-form-parsing.csv` | parsing muted/light; 2 cards | `1950bd6320f60a5dad6a89f1b3c64de56a3580fcc11baf98e0592a5bf4e47db7` |
| `H-form-parsing.csv` | parsing monochrome/dark; 2 cards | `9044f9cbbe8c85b8c0ed42e34533f7465df3572bac6b0256e3707a8cb6ab7a6c` |

The generator run had SHA-256
`9bfc3260d566bfb371134b829ad4093a6b40956be425df57d6073cff989844f6`. The committed
script differs only by `ruff format` and its usage line.

### Observed results

All checks passed, as reported by the maintainer.

| Check | Desktop 26.09.2 | iPhone | iPad |
|---|---|---|---|
| Latin-first completion and recognition fronts | pass | pass | pass |
| Real reveal (keyboard on Desktop, tap on iOS); core answer, compact comparison and four-role section readable | pass | pass | pass |
| No clipping or horizontal scroll; portrait and landscape | — | pass | pass |
| amō four roles with accepted splits/explanations; moneō supine withheld and no present split; ferō suppletive, no split; dīcō supine absent, coarse `dīx- \| -ī` | pass | pass | pass |
| muted/dark, monochrome/light, monochrome/dark readable; monochrome understandable without colour | pass | pass | pass |
| Disclosure: collapsed on reveal, core and compact comparison visible, opens and closes by click/touch (E, F) | pass | pass | pass |
| Client dark mode (Anki dark theme / iOS dark mode) on A, D, F readable | pass | pass | pass |
| Parsing G/H: 2 cards each, puellae without unreviewed gender, amāvit six features, no card for withheld *Rosam puellae dat.* | pass | pass | pass |
| Retheme update: 4 updated / 0 new, still 28 cards with the same identities, new theme rendered, Personal Notes kept | pass | — | — |

The maintainer raised three follow-ups, now specified for v0.2.1 and not defects of this
gate: CLI claim review and inspection (T-064), configurable Latin or user-language
terminology (T-065), and reviewed form translations (T-066).

### Fallback decision

Native `<details>/<summary>` disclosure works on every tested client: Desktop 26.09.2
and AnkiMobile 25.09 on iPhone and iPad. Disclosure is therefore a **verified option**
for those clients. **Static stays the profile default** because it is the readable
choice for clients and versions not tested here. No JavaScript was added.

### Limitations

- No native screenshots were captured for this run (maintainer decision); evidence is
  the reported observations above.
- The maintainer observed and reported results; there is no automated capture of the
  touch interactions.
- Content is synthetic, with stipulated reviews. Hand-typed GUI setup was waived in
  favour of the carrier package.
- Only the listed client versions and devices are covered; AnkiDroid and the web
  client are untested. Repeat affected checks when templates, CSS, answer markup or
  recipes change for a release candidate.
