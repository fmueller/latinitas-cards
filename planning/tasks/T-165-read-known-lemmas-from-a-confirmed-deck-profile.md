---
id: T-165-read-known-lemmas-from-a-confirmed-deck-profile
title: Read known lemmas from a confirmed deck profile
status: completed
priority: medium
spec_ref: specs/v0.4.0.md#ranked-card-budgets
dependencies:
    - T-162-define-the-corpus-generation-profile-and-corpus
updated_at: "2026-10-29T19:31:15Z"
---

# T-165-read-known-lemmas-from-a-confirmed-deck-profile Read known lemmas from a confirmed deck profile

## Description

Implements Fixed Contracts **Known-lemma input**: load any supported confirmed deck
profile and its source read-only, extract and fold one lemma per record (HTML stripped,
punctuation trimmed), report unreadable records, and expose lemma-to-deck-source-identity
links and digests for ranking and the report. Never writes deck files or state.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Fixtures with `dīcō, dīcere`, HTML markup and macrons yield the folded lemma
  `dico`; unreadable records are reported and skipped.
- CSV (with scope state) and APKG fixtures produce identical lemma sets for identical
  content; source and profile digests are reported; source files are byte-unchanged.

## Implementation Notes

- Start only when `taskrail status --json` shows active spec `specs/v0.4.0.md`; otherwise
  stop without `taskrail start` and report that v0.4.0 is not active.
- Follow-up rule (as in v0.3.0): (a) a defect in completed work this task builds on is
  fixed inside this task with a regression test; (b) follow-ups are filed only for work
  no remaining v0.4.0 task needs, with `--priority medium` or `high`, never `low`, and
  never with `--follow-up` on T-172; (c) a follow-up needing a maintainer decision (for
  example a Fixed Contracts disagreement) is created and then blocked with the reason
  "maintainer decision: ...".

## Verification Notes

- Implemented `known_lemmas.load_known_lemmas` as a read-only domain API for
  later ranking consumers; no ranking, selection, or preview command added.
  Extraction follows the Fixed Contract literally: source HTML text, first
  whitespace word, edge non-word trimming, then `latin_fold`. Comma-separated
  text without whitespace is not split into alternative lemmas.
- TDD: the initial focused test failed because the API did not exist, then four
  cases passed. Independent review exposed eager-reader failures for isolated
  malformed CSV rows and empty package GUIDs. Four new regression cases failed
  before the opt-in adapter recovery fix, then passed. Existing reader defaults
  remain strict. Nine helper cases and the source/identity tests pass (34 total).
  In-memory mutations blanking lexical diagnostics or fabricating missing IDs
  each failed the new mixed-record guard; restored behavior passed.
- Two dedicated code-simplifier passes recommended no changes. The independent
  conventional-review coordinator used medium-risk general, Python, security /
  provenance, and test specialists, relayed as checkout-local sibling agents.
  A separate adversarial verifier confirmed the reader defect, then challenged
  the fix and overall outcome. The coordinator verified all dispositions and
  approved the unchanged code subject to final validation. No unresolved findings
  or follow-ups remain.
- Independent acceptance QA used real saved profiles, scoped CSV, APKG, and
  COLPKG fixtures through the public API: 60 checks passed. Literal expected
  lemmas covered decorated/macron entries, HTML/entity/attribute boundaries,
  punctuation and no-space comma traps, and repeated lemmas with distinct IDs.
  CSV/package lemma sets matched. Reordered manifest survivors kept their
  persisted IDs and scope despite malformed physical rows; unresolved rows were
  skipped. Mixed unreadable records retained valid neighbors and useful locations
  and reasons. Raw source/profile SHA-256 digests matched independent expectations.
  Every successful and fatal-path invocation preserved input bytes, modes,
  modification times, and file inventory, including scope/profile error paths.
- QA command: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/t165_acceptance_qa.py`;
  the disposable harness and fixtures were removed after execution. Acceptance
  candidate: base `eaed8d6549cb4f01255bce61e58564b4c092924b` plus SHA-256 files:
  - `src/latinitas_cards/sources.py`:
    `293ed47da4edbe7d78daad698b436114bef7af03a7606e1204f617458ce3adeb`
  - `src/latinitas_cards/known_lemmas.py`:
    `20e625c2b2fce82842dd01b7f0501fa84c2eb51943c7d2cee930a75fdfd52f74`
  - `tests/unit/known_lemmas_test.py`:
    `fba840a157da026212bb9413568310149a2ce574a2f1dc28820c90f6c792abf6`
- Limits: independent QA used legacy SQLite package containers, not compressed
  modern ones (existing adapter tests cover those). Schema-1 profiles requiring
  explicit terminology upgrade remain rejected by the existing profile loader,
  as documented and pinned by existing tests; no automatic upgrade was added.
  Ranking/select and their downstream report journey belong to later tasks.
- Final post-review chain: `uv run ruff check` passed; `uv run mypy` passed
  (190 source files); `uv run pytest -v` passed with 3066 passed, 7 skipped,
  and one existing Pydantic serializer warning in 488.02 seconds. Reviewed code
  hashes above remained unchanged. Taskrail verify run: 2026-10-29T19:31:15Z.
- 2026-10-29T19:31:15Z: verification pass
- 2026-10-29T19:31:15Z: Implemented and independently accepted read-only known-lemma helper; final mandatory chain passed after review and fixes. See verification notes and verify run timestamp.
