---
id: T-136-reject-segment-aliases-to-resolved-default
title: Reject segment aliases to resolved default-manifest DATA locks
status: completed
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-17T09:18:05Z"
---

# T-136-reject-segment-aliases-to-resolved-default Reject segment aliases to resolved default-manifest DATA locks

## Description

Final E2E Round 5 at `ee6cdb97510169db89e193a02f13aacce4434fcb` confirms
segmentation misses the resolved lock of a symlinked default-manifest DATA file.
The Sidecar Annotation Model requires writer aliases to manifests/locks to be
rejected. Normalizing through this explicit default-manifest symlink succeeds;
no arbitrary custom-manifest discovery is involved.

Severity medium: reserved-path alias acceptance and forbidden file creation.
Priority high follows the round's filing instruction. Held reserved bytes/inode
and normalizer exclusion remain intact; no concurrent corruption or detached
held-lock claim is made.

Dedupe: explicit reopening overlap with completed T-131/T-129 and T-113. The
original 126-case segment matrix still passes; the newly exercised default
manifest DATA symlink exposes an omitted resolved reservation. T-134 reports
the related apply path, but segment has its own writer input-protection list.
T-133 is apply-only, T-130/T-132 claims-only, and future T-128 unrelated. This
standalone task has no dependencies or follow-up to T-124 or the other findings.

### Self-contained reproduction

Run from the checkout. Every path is disposable and the manifest is the known
default, not a custom unknown path:

```sh
root=$(mktemp -d)
cli="$PWD/.venv/bin/latinitas-cards"
printf 'Canis currit.\nFemina librum legit.\n' > "$root/source.txt"
"$cli" corpus normalize "$root/source.txt" --adapter text --corpus-id segment --output "$root/corpus.jsonl" --json
mv "$root/corpus.manifest.json" "$root/manifest-data.json"
ln -s manifest-data.json "$root/corpus.manifest.json"
"$cli" corpus normalize "$root/source.txt" --adapter text --corpus-id segment --output "$root/corpus.jsonl" --json
rm "$root/manifest-data.json.lock"
ln -s manifest-data.json.lock "$root/hop"
ln -s hop "$root/side.jsonl.lock"
"$cli" annotations segment "$root/corpus.jsonl" --output "$root/side.jsonl" --timestamp 2026-01-02T03:04:05Z --json
test -e "$root/manifest-data.json.lock" && echo 'Unexpected reserved file created'
wc -l "$root/side.jsonl"
rm -rf "$root"
```

Expected segment exit 3/alias, absent reserved lock and absent sidecar. Actual
exit 0/appended 2, newly created zero-byte reserved lock and a three-line
sidecar. Existing output-hardlink variant exits 0/appended 2 under a real held
manifest lock. Existing output-lock symlink gives 4/busy held and 0/appended 2
released rather than 3/alias in both cases. Ordinary output succeeds.

## Acceptance

- Reject output/output-lock aliases to the resolved default-manifest DATA lock,
  including existing held/released and dangling paths, with 3/alias/no write.
- Keep original T-129/T-131 matrix, normalizer exclusion and ordinary controls
  passing; preserve reserved bytes/inodes and absent reservations.
- Preserve existing sidecar behavior and exact logical claims locks. No unknown
  manifest discovery, product redesign or future T-128 work is implied.

## Verification Notes

- Round 5 bundle `round5-evidence.tar.gz` includes the independently authored
  `round5/coordinator/segment-manifest-replay.py`, exact generated inputs,
  before/after hashes, safe environment, DEVNULL, timeouts and raw CLI streams.
  Fresh replay: 20 CLI calls, 36 assertions, 14 failures. All positive controls,
  held inode/bytes and real normalizer exclusion/retry assertions pass.
- Source check: `commands/annotations.py` passes only the logical default
  manifest lock to `append_records`; normalization resolves manifest DATA
  before constructing its acquired lock. This is separate from apply's list.

## Implementation Notes

### Scope and ownership

- Accepted cumulative base and fetched `origin/main`:
  `9257e3530a3ec62fdcc6ce1a4e77b3618f17bcf7`, containing T-134
  `ed74d5bc9bddb6d2e91d111aa82f0fc737b80234` and T-135
  `fa8a9926bca3d01801232dd57bcfae33b3721abe` (ancestry checked after
  `git fetch --quiet --unshallow origin`). No source-checkout transfer.
- `git fetch origin main`; `git status --short`; `git rev-parse HEAD origin/main`;
  `sha256sum specs/v0.3.0.md`; `mise exec -- taskrail validate`;
  `mise exec -- taskrail next --json`; `mise exec -- taskrail status` established
  the exact pin `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`,
  valid state, sole eligible high todo T-136 and zero active tasks.
- Rechecked pin/selector/status immediately before
  `mise exec -- taskrail start T-136-reject-segment-aliases-to-resolved-default`.
  This cycle owns only T-136. T-038 stays blocked; T-128 stays unstarted.
- `mise exec -- lefthook install` installed hooks. Git identity is Felix Müller
  (maintainer). No attribution trailers. Version/spec/release gates unchanged.
- Inspected normalization's `targets` and `writer_locks(targets)`, then
  `atomic_io.writer_locks`: DATA resolves before `.lock` derivation. Segment's
  existing `append_records` validates output and its lock against protected paths
  before acquisition. Added only the resolved default-manifest reservation to
  that owner list; no helper, suffix contract, apply refactor or lock redesign.

### Independent public reproduction and strict TDD

- Before production edits, subprocess CLI regression matrix:
  `uv run pytest tests/unit/annotations_command_test.py -k
  resolved_default_manifest_lock -v --tb=short`:
  **25 failed, 3 passed, 2 skipped, 67 deselected in 116.86s**. All failures
  were wrong alias outcomes, not setup failures. Existing direct output-lock
  spellings already alias manifest DATA (three passes); absent hardlinks are
  impossible (two skips).
- Matrix independently normalizes two text passages, moves default manifest
  DATA behind a supported symlink, then successfully normalizes again. It
  covers held/released/absent reservations, output/output-lock, direct,
  symlink/hardlink/chained/directory aliases. It checks exact 3/alias,
  directory entries, bytes/inodes, symlink targets, absent reservations,
  real subprocess normalizer exclusion before/after, successful release retry,
  ordinary append=2 and idempotent skip=2 controls.
- Separate `uv run python -` public CLI replay (stdlib subprocess, DEVNULL,
  timeout=30, disposable TemporaryDirectory): initial and DATA-symlink
  normalization 0; held normalizer 4/busy before and after; held output
  hardlink 0/appended=2; held output-lock symlink 4/busy; released symlink
  0/appended=2; released normalizer 0. Held reservation bytes and inode stayed
  equal; only the selected output hardlink detached. Dangling output-lock
  chain returned 0/appended=2, created empty reserved lock and three-line
  sidecar. Ordinary output returned 0/appended=2. No held-inode detachment
  or exclusion bypass is claimed: medium reserved-path acceptance severity.
- GREEN after the one-line owner change:
  `uv run pytest tests/unit/annotations_command_test.py
  tests/unit/annotation_apply_test.py tests/unit/atomic_io_test.py -v`:
  **228 passed, 7 skipped in 218.52s**. Existing T-129/T-131 tests and T-134
  apply matrix remain unchanged. `uv run ruff format
  tests/unit/annotations_command_test.py` formatted only the new test snippet.
- Retrieved the accepted Round 5 bundle into disposable external storage to
  replay its original `round5/policy/adversarial.py` (with colocated harness)
  and `round5/coordinator/segment-manifest-replay.py`, without changing their
  semantics. This is regression validation, not a new acceptance round.

### Workflow gates

- Initial exact `uv run ruff check` → `uv run mypy` → `uv run pytest -v`:
  ruff clean; mypy 169 source files; **2621 passed, 7 skipped, 1 warning in
  356.13s**. Warning is the existing Pydantic serializer warning in
  `test_writer_rejects_float_without_replacing`. Full/browser suites sequential.
- Dedicated simplifier Task loaded personal `code-simplifier`; removed only
  the diagnostic print in the new test. Parent reread resulting diff; accepted
  that simplification, no rejected suggestions. Its
  `uv run pytest -q tests/unit/annotations_command_test.py`: **95 passed,
  2 skipped**; changed-file Ruff check/format check and `git diff --check` passed.
- Independent read-only Task lanes, launched separately in parallel, each
  loaded personal `code-reviewer` and reviewer-index mapping:
  General → `ecc/agents/code-reviewer.md` (no companion);
  Security → security-reviewer, security-review, common security rules;
  Database → database-reviewer, postgres-patterns, database-migrations;
  Python → python-reviewer, python-patterns. Database applied filesystem
  persistence/flock reasoning only, not inapplicable SQL advice.
- Lane selection: General mandatory; filesystem alias trust/data-integrity
  risk requires Security; persistent writer exclusion requires Database;
  changed Python source/tests requires Python. Three specialists meet budget.
  Framework/UI/ML/RAG/healthcare/network lanes omitted: not materially affected.
- General verbatim: "No concrete task-relevant findings."
  Security verbatim: "No concrete task-relevant findings."
  Database verbatim: "No concrete task-relevant findings."
  Python verbatim: "No concrete task-relevant findings."
  General/Database independently passed `git diff --check`; other lanes used
  read-only source review. None ran full/browser suites.
- Fresh read-only candidate-validation Task loaded `code-reviewer` and mapped
  references, independently inspected diff and ownership. Verbatim:
  "VALIDATED — empty candidate set." and "No new finding was necessary."
  It passed `git diff --check` and two targeted regression cases. Its first
  selector matched no tests; corrected collected parameter IDs passed (2).
  Zero candidates, rejected IDs, duplicates, deferrals or unresolved findings;
  therefore no finding-fix TDD was required. One review/disposition cycle.
- Exact unchanged external replay commands:
  `python3 /tmp/t136-replay/round5/policy/adversarial.py`:
  `{"checks":1575,"commands":750,"failures":[]}` (original 126-case matrix).
  `python3 /tmp/t136-replay/round5/coordinator/segment-manifest-replay.py`:
  `{"commands": 20, "assertions": 36, "failures": 0}`. Resolved-lock hardlink
  and output-lock aliases now 3/alias held/released; dangling chain 3/alias
  with absent reservation/sidecar; ordinary output still appends two. All
  before/after hashes, held bytes/inodes, busy-before/after and retry controls
  pass. Scripts kept outside tracked checkout; no new testing-round launch.
- Hosted CI on the accepted base is separately reported failing Python
  3.13/3.14 Unit Tests; unrelated diagnosis/dependency workers remain read-only
  until overseer release. Local checks here do not assert hosted CI green.
- Fresh final exact `uv run ruff check` → `uv run mypy` → `uv run pytest -v`:
  ruff clean; mypy 169 source files; **2621 passed, 7 skipped, 1 existing
  warning in 315.40s**. Then sequential `mise run check` passed: formatting,
  mypy, **2621 passed, 7 skipped, 1 existing warning in 322.04s**, all commit,
  push-message, author, mutation, distribution and publish-guard suites.
  No failed mandatory chain required restart. No full suites overlapped.
- Fresh final read-only Task loaded `code-reviewer` in disposition-verification
  mode and mapped references. Verbatim: "Disposition verification: zero
  dispositions." and "One review cycle; no new or unresolved task-relevant
  issues found." It inspected the actual owner/alias/lock paths, diff,
  acceptance and evidence; `git diff --check` passed. No edits or lifecycle
  writes by reviewers. No deferrals or follow-up fixes necessary.
- Green supplemental replay retained held hardlink reservation inode 1569665
  and output-lock-symlink reservation inode 1569673, both with exact original
  header bytes. Normalizer returned 4 before/after rejected aliases and 0
  after release. All absent-reservation and no-mutation assertions passed.
- Tested source SHA256:
  `e104867b523c7ce21f4fb44b624485e6705a33cc36a6faf432fbe0dc5ab5fb9c`;
  tested regression-file SHA256:
  `1bde4317720d1a6e32cd0334dc0f09f2a982cfa807e4221aca1ea9959db67f3a`.
- Before lifecycle finalization, repeated `git fetch origin main`,
  `git rev-parse HEAD origin/main`, `sha256sum specs/v0.3.0.md`,
  `mise exec -- taskrail validate`, `mise exec -- taskrail status`,
  `mise exec -- taskrail next --json`, and task frontmatter inspection:
  exact accepted base/pin unchanged, state valid, T-136 sole active owner
  and selected task, T-038 blocked/T-128 untouched. No mismatch.
- 2026-10-17T09:17:56Z: verification pass

### Lifecycle and delivery

- After all gates, ran `mise exec -- taskrail verify
  T-136-reject-segment-aliases-to-resolved-default --result pass --summary
  'Resolved default-manifest DATA reservation protected before contention;
  public RED/GREEN, original 126-case replay and 20-call supplemental replay
  passed; simplifier, four independent lanes, candidate validation, final
  disposition verifier, exact ruff/mypy/pytest chain and mise run check passed.'`.
  Verification passed at **2026-10-17T09:17:56Z**; no ignored evidence paths
  are committed.
- Repeated pin, `mise exec -- taskrail validate`, status, selector and task
  frontmatter checks before `mise exec -- taskrail complete
  T-136-reject-segment-aliases-to-resolved-default`. Completed only T-136.
- Post-completion `mise exec -- taskrail validate` passed;
  `mise exec -- taskrail status`: **134 done, 0 active, 1 blocked, 1 todo**.
  `mise exec -- taskrail next --json`: **no active-spec eligible task**;
  T-128 was explicitly skipped as v0.8.0 work and remains unstarted.
  T-038 remains the blocked maintainer-only release gate. No second task.
- Delivery is a focused Conventional Commit under the installed hooks and
  maintainer identity, followed by authorized non-force `origin/main` push
  after fetch/containment reconciliation. Exact delivered/tested commit and
  remote containment are reported in the completion callback after success.
  No release/tag/publication/deployment/manual workflow or infrastructure
  actions; no claim of acceptance-round cleanliness or hosted CI success.
