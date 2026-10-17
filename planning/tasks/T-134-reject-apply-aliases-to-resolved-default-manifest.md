---
id: T-134-reject-apply-aliases-to-resolved-default-manifest
title: Reject apply aliases to resolved default-manifest DATA locks
status: completed
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-17T08:17:36Z"
---

# T-134-reject-apply-aliases-to-resolved-default-manifest Reject apply aliases to resolved default-manifest DATA locks

## Description

Final synthetic E2E Round 5 confirms an unresolved part of the Sidecar
Annotation Model writer-alias contract at revision
`ee6cdb97510169db89e193a02f13aacce4434fcb`. A default manifest whose DATA is a
symlink is supported by normalization, but apply does not protect the resolved
manifest DATA lock. This is not discovery of an arbitrary custom manifest.

Severity: medium filesystem-contract violation; priority high as requested for
confirmed round findings. No concurrent corruption or held-lock detachment was
demonstrated. The reserved inode, bytes and normalizer exclusion remain intact.

Dedupe: reopening overlap with completed T-133 and T-118 is explicit. T-133
protects the logical default-manifest lock but misses its resolved DATA lock.
T-129/T-131 own segment paths; T-130/T-132 own duplicate claims. T-128 is future
v0.8.0 memory work and unrelated. No dependency or follow-up to T-124.

### Self-contained reproduction

Run from the checkout with its installed CLI. This uses disposable synthetic
data, an actual kernel lock, and only supported CLI commands:

```python
import fcntl
import json
import os
import subprocess
import tempfile
from pathlib import Path

cli = str(Path(".venv/bin/latinitas-cards").resolve())
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    def run(*args):
        result = subprocess.run(
            [cli, *map(str, args), "--json"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(result.returncode, result.stdout)
        return result.returncode

    source, corpus = root / "input.txt", root / "corpus.jsonl"
    source.write_text("Puella rosas portat.\nSenex aquam bibit.\n")
    normalize = ["corpus", "normalize", source, "--adapter", "text", "--corpus-id", "replay", "--output", corpus]
    assert run(*normalize) == 0
    manifest = corpus.with_suffix(".manifest.json")
    data = root / "manifest-storage.json"
    manifest.rename(data)
    manifest.symlink_to(data.name)
    assert run(*normalize) == 0
    reserved = Path(str(data) + ".lock")
    header = b'{"format":"latinitas-annotations","record":"header","version":1}\n'
    reserved.write_bytes(header)
    sidecar = root / "side.jsonl"
    os.link(reserved, sidecar)
    passage = next(row for line in corpus.read_text().splitlines() if (row := json.loads(line))["record"] == "passage")
    patch, plan = root / "patch.json", root / "plan.json"
    patch.write_text(
        json.dumps(
            {
                "format": "latinitas-annotation-patch",
                "version": 1,
                "operations": [
                    {
                        "operation": "add",
                        "layer": "translation",
                        "anchor": {key: passage[key] for key in ("passage_id", "passage_hash")},
                        "value": {"language": "en", "text": "The girl carries roses."},
                        "producer": {"kind": "agent", "name": "replay", "version": "1"},
                        "created": "2026-01-02T03:04:05Z",
                    }
                ],
            }
        )
    )
    assert run("annotations", "plan", patch, "--corpus", corpus, "--sidecar", sidecar, "--output", plan) == 0
    with reserved.open("rb") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inode = reserved.stat().st_ino
        assert run(*normalize) == 4
        actual = run("annotations", "apply", plan, "--corpus", corpus, "--sidecar", sidecar)
        assert run(*normalize) == 4
        assert reserved.stat().st_ino == inode and reserved.read_bytes() == header
        print("Expected 3/alias, actual:", actual)
    assert run(*normalize) == 0
```

Actual apply: exit 0, appended 1; sidecar changes from the one-line header to
header plus annotation. Expected: exit 3/alias and no mutation. Existing
output-lock symlink/hardlink/chain aliases instead give 4/busy while held and
0/appended after release. Dangling output-lock chains create the previously
absent resolved manifest lock and sidecar. Ordinary destinations succeed.

## Acceptance

- Reject aliases to the resolved lock of a known default-manifest DATA symlink
  with 3/alias, before contention or writes, for held/released and absent cases.
- Preserve reserved inode/bytes, real normalizer exclusion, absent reservations,
  and ordinary apply/reapply controls; preserve exact logical claims locking.
- Cover direct reachable spellings plus symlink/hardlink/chain/directory paths
  without inventing custom-manifest discovery or a new suffix contract.

## Verification Notes

- Round 5 evidence: `ROUND5-REPORT.md` and `round5-evidence.tar.gz`, delivered
  with the testing thread. Bundle contains the self-contained independent
  manifest replay and exact inputs/argv/exits/checks under `round5/policy` and
  `round5/coordinator`. Both fresh executions produced 80 commands, 142 checks,
  and 54 failures against unchanged 3/alias/no-mutation expectations.
- Source cross-check: apply protects `Path(f"{manifest}.lock")`; normalizer's
  `writer_locks` resolves manifest DATA before appending `.lock`. The resolved
  reservation is not in apply's protected list. No fix was attempted.

## Implementation Notes

Historical Round 5 delivery: unstarted, findings only, with no product or test
changes in that round. The following records the subsequent T-134 fix cycle.

### Workflow-v3 audit

1. **Understand / ownership.** Fetched origin/main and established accepted
   cumulative base `f6cc0a0d7d50c535c56ef5223f466a795f2f94b4` (findings/state
   after tested product `ee6cdb97510169db89e193a02f13aacce4434fcb`). No product
   transfer from the overseer's old local main. SHA256 of specs/v0.3.0.md:
   `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
   `mise exec -- taskrail validate` returned `state valid`; `next --json`
   selected this high/todo task; status reported 131 done, zero active,
   one blocked, four todo. Repeated pin/validate/selector/status immediately
   before `taskrail start`; this cycle owns only T-134.
   Inspected Sidecar Annotation Model and Fixed Contracts (writer aliases,
   manifest recovery and apply safety), apply's preflight and post-lock checks,
   append_records, reject_aliases, writer_locks and normalization's source owner.
   Normalization locks resolved DATA; claims deliberately lock the logical path.
2. **Strict RED / GREEN.** Ran the task's standalone subprocess reproduction
   before production changes, with a genuinely normalized default manifest DATA
   symlink. Normalize succeeded before and after moving DATA to
   manifest-storage.json. Held resolved-lock hardlink output: apply exited 0,
   appended 1, while normalization exited 4 before and after; reserved bytes and
   inode were unchanged, and normalization exited 0 after release. This confirms
   the alias-contract failure, not held-inode detachment or exclusion bypass.
   Extended the existing public CLI reserved-writer matrix before production:
   `uv run pytest tests/unit/annotation_apply_test.py -k
   'reserved_writer_alias and resolved-manifest' -q` produced
   `7 failed, 6 passed, 1 skipped, 114 deselected`.
   Failures: hardlink output; symlink/hardlink/chain/directory output-lock;
   absent and absent-chain output-lock. Held output-lock aliases returned
   4/busy, released aliases and dangling reservations returned 0, not 3/alias.
   Direct and symlink-resolved .lock outputs were already rejected by suffix;
   hardlinks expose why suffix checking alone is insufficient.
   Minimal source fix: include `Path(f"{manifest.resolve()}.lock")` in the
   existing protected list. This list reaches sidecar and all derived-lock
   preflight checks, post-acquisition sidecar checks and append_records checks.
   No new lock acquisition, helper, suffix rule or custom-manifest discovery.
   GREEN apply suite: `123 passed, 5 skipped`; rerunning the valid standalone
   reproduction now returns 3/alias, with normalize 4 before/after while held,
   reserved inode/bytes intact and normalize 0 after release.
   Extended that disposable valid-manifest replay with an ordinary destination:
   first apply returned 0/appended 1, reapply returned 0/already_applied true,
   appended 0 with byte-identical retry; reserved inode/bytes remained intact.
3. **Initial checks.** First mandatory chain stopped on E501 in the expanded
   test decorator. Ran the existing Ruff formatter, reread the edit, restarted
   the entire exact chain: `uv run ruff check` (all checks passed),
   `uv run mypy` (169 source files), `uv run pytest -v`
   (`2593 passed, 5 skipped, 1 warning in 230.79s`). Full suites are sequential.
   Existing policy, receipt, partial, no-op, timestamp, reapply, claims-lock and
   competing-writer tests remain in that suite. Matrix asserts byte snapshots,
   stable reserved inode and absent reservations; it uses placeholder DATA
   because apply protects known paths without parsing manifest contents. The
   separate standalone replay supplies the valid supported normalization control.
4. **Simplify.** Dedicated Task loaded personal code-simplifier; factored the
   repeated manifest-hardlink condition into one test-local predicate used three
   times. Accepted this test-only simplification after diff inspection. No
   production/changelog simplification warranted. Its focused apply suite passed
   123/5; parent's matrix rerun passed 65/5 (58 deselected).
5. **Independent lanes / candidate validation.** Four separate parallel read-only
   Tasks loaded personal code-reviewer and reviewer-index. General loaded ECC
   code-reviewer plus common code-review/security/testing guidance. Python loaded
   python-reviewer and python-patterns. Security loaded security-reviewer,
   security-review, its cloud-security companion and common security/testing.
   Database loaded database-reviewer, postgres-patterns and database-migrations;
   applied only relevant filesystem locking/persistence guidance (no SQL work).
   General is mandatory; Python covers implementation, Security covers path trust
   and data loss, Database covers exclusion/persistence. Three specialist lanes
   meet the default budget. No web/API framework, NLP/ML, RAG, network, healthcare
   or other-language lane applies to this path-only Python CLI change.
   Each lane's verbatim conclusion: **"No concrete task-relevant findings."**
   General/Database diff checks passed; Python independently ran the matrix:
   65 passed, 5 skipped. A fresh candidate-validation Task loaded code-reviewer,
   confirmed zero concrete candidates and independently passed that same matrix.
   It corrected Security's spurious unavailable-base-ending-a7 limitation:
   actual and requested base are the exact f6cc revision above, and actual diff
   review used that base. This report metadata mistake does not affect scope.
6. **Dispositions.** No validated product findings, rejected candidate IDs,
   deduplications or deferrals. Candidate validator's bookkeeping observation:
   "The task file's `Implementation Notes` still says **\"Unstarted\"** and
   **\"no product or test changes\"** (`planning/tasks/T-134-…md:131–133`), while
   the task is in progress and the implementation and tests are modified. This
   is stale task bookkeeping, not a product finding; it should be updated before
   the task is verified or completed."
   Resolved by retaining that sentence's Round 5 historical meaning and adding
   this subsequent implementation audit. No behavior fix or test is appropriate
   for this historical-note clarification; no finding-fix TDD cycle was needed.
7. **Final checks / fresh disposition verifier.** After simplification and
   dispositions, reran exact `uv run ruff check` -> `uv run mypy` ->
   `uv run pytest -v`: all checks passed, 169 files, 2593 passed/5 skipped/
   1 warning in 220.39s. Then `mise run check` exited 0: Ruff, 391 files already
   formatted, mypy 169 files, pytest 2593 passed/5 skipped/1 warning in 219.47s,
   commit/push/author/mutation-config/distribution/publish-guard policy suites.
   The warning is the existing Pydantic serializer warning in the deliberate
   float-rejection test, not a suppressed failure. Full suites ran sequentially.
   Fresh read-only Task loaded code-reviewer in disposition-verification mode,
   inspected actual diff and final logs, and independently passed diff checking.
   Verbatim outcome: **"Bookkeeping disposition: RESOLVED."** and
   **"Product findings: none."** No unresolved or newly introduced concrete
   task-relevant issues. One review/disposition cycle; no second cycle required.
   All final gates and independent verification preceded lifecycle finalization.
8. **Scope limits.** T-135 docs and T-136 segment are separate and unstarted.
   T-038 remains blocked maintainer-only; T-128 future-v0.8 memory is untouched.
   No spec, priority, dependency, acceptance gate, version, release/tag, deployment,
   manual workflow, provider, secret or infrastructure change. No universal race,
   corruption-safety, macOS/native or NLP proof is claimed. Five adversarial rounds
   remain consumed NOT CLEAN; this cycle launched no testing round or persistent
   internal thread, and does not reset the count or start rounds 6–8.

### Lifecycle and delivery preparation

- Rechecked the exact spec hash, valid state, active-spec path, one active owner
  and selector T-134 immediately before each verify/complete lifecycle write.
  Verification passed at 2026-10-17T08:17:28Z; completion at
  2026-10-17T08:17:36Z followed all code gates and fresh reviewer verification.
- Post-completion validate: `state valid`; status: 132 done, zero active,
  one blocked, three todo. Observed next: T-135, high/todo, not started.
  T-136 remains todo; no second task was claimed.
- Installed hooks using `mise exec -- lefthook install` before committing;
  git identity is Felix Müller <felix@fmueller.io>. Focused delivery uses a
  conventional T-134 commit without attribution trailers and a non-force push
  to origin/main; remote reconciliation/containment is reported in the callback.

- 2026-10-17T08:17:28Z: verification pass
- 2026-10-17T08:17:36Z: Implemented and independently reviewed; verification pass at 2026-10-17T08:17:28Z after all final gates. T-134 only; T-135/T-136 remain separate.
