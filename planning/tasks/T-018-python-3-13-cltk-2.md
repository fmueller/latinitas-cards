---
id: T-018-python-3-13-cltk-2
title: Require Python 3.13 and migrate to CLTK 2
status: completed
priority: high
spec_ref: specs/v0.1.0.md#uv-toolchain
dependencies: []
updated_at: "2026-09-23T20:42:13Z"
---

# T-018-python-3-13-cltk-2 Require Python 3.13 and migrate to CLTK 2

## Description

Raise the minimum Python to 3.13 (`specs/v0.1.0.md#uv-toolchain`) so the experimental
`annotate` extra can move to CLTK 2.x. CLTK 1.x caps Python below 3.13 and CLTK 2.x requires
3.13+, so the two changes must land together. CLTK 2.x drops NLTK and the other heavy
dependencies from its core and makes Stanza an opt-in extra.

## Acceptance

- `requires-python` starts at 3.13; classifiers, mypy `python_version`, `.python-version`,
  and the CI test matrix match the supported range.
- The `annotate` extra depends on `cltk[stanza]` 2.x and `uv.lock` no longer contains NLTK.
- `annotate` uses the CLTK 2 `NLP` API; `upos` holds the UD tag, `morph_features` is written
  in UD `Key=Value|...` form, and `xpos` stays as an empty column because CLTK 2 does not
  expose it. The obsolete manual Stanza model download is removed.
- Unit tests cover the CLTK 2 word-to-row mapping without importing CLTK.
- README, AGENTS.md, and the spec state the new Python range and the annotation output change.
- ruff, mypy, and pytest pass on every supported Python version; `annotate` produces analyses
  with the extra installed.

## Verification Notes

- Verify run 2026-09-23T20:42:13Z passed: ruff, format, mypy, and pytest (66 tests) pass on
  Python 3.14 and 3.13. With the `cltk[stanza]` 2.5.1 extra on 3.13, `annotate` writes UD
  `upos` and FEATS values with an empty `xpos`. `uv.lock` no longer contains NLTK, GitPython,
  spaCy, or gensim.

## Implementation Notes

- 2026-09-23T20:42:13Z: verification pass
