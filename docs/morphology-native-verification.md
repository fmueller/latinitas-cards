# Morphology native presentation gate

Native acceptance remains **blocked**, not complete. The required primary AnkiMobile
iPhone/iPad acceptance has not been executed. Neither T-053 browser captures
nor T-061 native backend import checks close this presentation gate.

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

## Open native acceptance and fallback decision

- Record real AnkiMobile version and actual iPhone **and** iPad devices. On each,
  exercise reveal and touch expansion, core/compact visibility, accepted further
  explanation and explicit absent/withheld rows, muted/monochrome light/dark.
- Finish Desktop recognition/completion, both themes and appearances, real
  reveal gestures and details/summary closed/open interactions. The partial
  static completion capture above closes none of these remaining matrix cells.
- Exercise the documented manual reference CSS/template setup and native CSV
  import with preview parity; compare claims, withholding, eligibility, stable
  note/card identities, recipe keys and frozen template slots across settings.
  Do not use this API-created fixture or T-061 backend results as that proof.
- Retain **static as the unverified-client default**. Desktop static has only
  this limited observed reveal/readability evidence; Mobile static is untested.
  No native details/summary decision is possible yet. Do not enable disclosure
  as verified, call static a universal compatibility fix, or add untested JS.
- Bind every remaining capture/interaction to client/version/device, tested
  content/profile/setup and source hashes. Repeat affected checks for release
  candidate changes. T-060 owns affected retests after its later recipe changes;
  that ownership does not waive this blocked native gate.

Required unavailable checks remain open. Obtain native iPhone/iPad access and
resume this same task; do not mark T-062 complete or claim release readiness.
