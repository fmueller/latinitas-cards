---
id: T-131-reject-segmentation-outputs-aliasing-the-default
title: Reject segmentation outputs aliasing the default manifest lock
status: completed
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-16T18:03:17Z"
---

# T-131-reject-segmentation-outputs-aliasing-the-default Reject segmentation outputs aliasing the default manifest lock

## Description

Round 2 synthetic CLI testing independently confirmed that `annotations segment`
can replace the default manifest lock, detaching a held lock inode and allowing
normalization to proceed while the original descriptor remains locked. This
violates the alias rejection contract in the referenced sidecar model section.
Severity: medium data-integrity/serialization risk; repair priority high.

Tested product revision: `fd9a677ad9cc8ff43efac09b036a5ddafcbb20a5`.
Spec SHA256: `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
Linux / Python 3.14.2. Run this self-contained reproducer from the checkout with
`uv run python`; all CLI data lives in a disposable external directory:

```python
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile

cli = str(Path(".venv/bin/latinitas-cards").resolve())
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / "source.txt").write_text("Mater filium amat.\n")

    def run(*args):
        result = subprocess.run(
            [cli, *args, "--json"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )
        print(result.returncode, result.stdout.strip())
        return result.returncode

    normalize = (
        "corpus",
        "normalize",
        "source.txt",
        "--adapter",
        "text",
        "--corpus-id",
        "lane-test",
        "--output",
        "corpus.jsonl",
    )
    assert run(*normalize) == 0
    lock = root / "corpus.manifest.json.lock"
    lock.write_text('{"format":"latinitas-annotations","record":"header","version":1}\n')
    segment = (
        "annotations",
        "segment",
        "corpus.jsonl",
        "--timestamp",
        "2026-01-02T03:04:05Z",
        "--output",
    )
    assert run(*segment, "ordinary.jsonl") == 0
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(*normalize) == 4
        before = os.fstat(held.fileno()).st_ino
        print("segment expected 3; actual:", run(*segment, str(lock)))
        print("inode preserved:", before == lock.stat().st_ino)
        print("held normalize expected 4; actual:", run(*normalize))
    assert run(*normalize) == 0
```

Expected exits: `0, 0, 4, 3, 4, 0`, unchanged lock bytes/inode and an `alias`
diagnostic. Actual exits: `0, 0, 4, 0, 0, 0`; the header-only lock becomes an
annotation sidecar and normalization bypasses the still-held old inode.

## Acceptance

- Segment rejects direct, symlink, hard-link, chained and directory aliases to
  the default manifest lock with exit 3 / `alias`, preserving bytes and inode.
- Protect absent reserved paths as well as existing valid-header lock files.
- A real held manifest lock still excludes normalization after the rejected
  output attempt. Ordinary sidecar output and released-lock retries succeed.
- Preserve completed T-129 corpus and logical claims-lock protections, including
  symlinked claims data. Do not change locking semantics or require discovery
  of arbitrary custom manifest paths that are not supplied to segment.

## Verification Notes

- Confirmed on 2026-10-16 by the lane harness, separate six-command replay and
  coordinator replay. Policy/lock harness: 442 commands, 758 assertions, 19
  failures, all this omission. Corpus and logical claims-lock retests pass.
- Coordinator observed held inode 1572155 replaced by inode 1572173.
  Before SHA256: `a66fa0f09d1f85c0a0113fd32a79ea9c1d39631afc69097342e216e4677f5b05`.
  After SHA256: `df8b1eedfc8ce8a42bebcf70307459051741c4c2ccc17a66479c3ba4dc9a8916`.
- Direct/directory aliases bypass exclusion. Other alias forms detach their
  selected path, but the original manifest lock continues excluding writers.
- Evidence is retained in the Round 2 report/bundle: policy lane command log,
  assertion ledger and independent manifest-lock replay. No implementation.
- Deduplication searched all 130 prior tasks. T-129 explicitly covers corpus
  and claims locks and now passes; this is the remaining default manifest-lock
  omission, not a hidden dismissal of a resurfaced bug. T-113's enumerated
  acceptance excludes this path; T-107 owns normalization's manifest locks;
  T-117/T-118 own plan/apply, not segment. T-130 is unrelated duplicate claims.
  T-038 remains blocked and T-128 remains future v0.8.0 work. No dependencies
  or follow-up relationship to completed T-124 are introduced.

## Implementation Notes

`commands/annotations.py` passes default manifest data, corpus locks and the
logical claims lock to `append_records`, but omits the default manifest lock.
`annotation_sidecar.append_records` protects only these explicit inputs and its
own output lock. `commands/corpus.py` locks both corpus and default manifest.

## T-131 Workflow Audit

### Understanding and ownership

- Accepted cumulative base: `a35519b776a61b23a41126f324a41c171b4a3762`;
  previously tested product: `fd9a677ad9cc8ff43efac09b036a5ddafcbb20a5`.
- Before start: fetched origin/main equaled HEAD and the accepted base; clean
  worktree, valid Taskrail state, zero active tasks, T-131 sole eligible high
  todo. Spec SHA256 matched the pin in Description. Only T-131 was started.
- Source owner is segment's protected-path list. Added the default manifest
  lock there, using existing resolved-path/same-inode alias rejection before
  output creation, locking or replacement. Inspected normalizer lock targets,
  append output-lock protection, corpus logical/resolved locks and exact
  logical claims lock. No new helper, lock semantics or custom discovery.

### Independent public CLI reproduction and TDD

- Ran the self-contained Description reproducer before production changes:
  exits `0, 0, 4, 0, 0, 0`; held inode `1569708` became `1569714`.
  SHA256 changed from
  `a66fa0f09d1f85c0a0113fd32a79ea9c1d39631afc69097342e216e4677f5b05`
  to `df8b1eedfc8ce8a42bebcf70307459051741c4c2ccc17a66479c3ba4dc9a8916`.
- RED: `uv run pytest tests/unit/annotations_command_test.py -k
  'manifest and (held_reserved or absent_reserved)' -q`: 10 failed,
  57 deselected; segment returned 0 instead of 3, before the source fix.
- GREEN: `uv run pytest tests/unit/annotations_command_test.py -q`:
  67 passed. Held-lock cases cover direct, symlink, hardlink, chained and
  directory aliases for manifest/corpus/claims/symlinked-claims; absent
  reserved cases also cover the output's lock alias. Assertions check alias
  diagnostics, unchanged bytes/inodes, real exclusion, ordinary output and
  released normalization. Missing files cannot have hardlink aliases.
- Replayed the same Description CLI after the fix: `0, 0, 4, 3, 4, 0`;
  held inode stayed `1570176`, SHA256 stayed
  `a66fa0f09d1f85c0a0113fd32a79ea9c1d39631afc69097342e216e4677f5b05`.
  Only direct/directory replacement bypassed the original exclusion before
  the fix; symlink/hardlink replacement detached the selected alias only.

### Simplification and independent review

- Dedicated Task loaded `code-simplifier`: no edits; direct protected-path
  addition and explicit public CLI matrix needed no added abstraction.
- Separate read-only Tasks loaded `code-reviewer`, reviewer-index and each
  dedicated ECC lane: General (code-reviewer); Security (security-reviewer,
  security-review companion/common security rules); Database
  (database-reviewer, postgres-patterns/database-migrations companions,
  relevant filesystem-lock concurrency only); Python (python-reviewer,
  python-patterns plus common review/testing guidance).
- General mandatory; Security covers filesystem alias/data-integrity risk;
  Database covers persistent flock serialization; Python covers changed
  language/test contracts. Three specialist lanes meet the default budget.
  Framework, ML/RAG, healthcare and network lanes omitted: not affected.
- Each lane's verbatim conclusion: "No concrete task-relevant findings."
  Database independently ran alias/lock tests: 30 passed, 37 deselected.
- Fresh candidate-validation Task loaded `code-reviewer`, checked the actual
  diff and relevant references: "No concrete task-relevant findings."
  It independently ran the same 30 cases successfully. Empty candidate set
  validated: no rejected IDs, deferred findings or finding-fix TDD needed.

### Validation and limits

- Initial exact `uv run ruff check` → `uv run mypy` → `uv run pytest -v`:
  all passed; mypy 169 files, pytest 2516 passed, one existing Pydantic
  serialization warning. Full suites run sequentially for browser isolation.
- Final exact chain passed again: ruff clean, mypy 169 files, pytest
  2516 passed in 188.63 seconds. Then `mise run check` passed, including
  formatting, 2516 tests in 183.83 seconds and every policy-guard suite.
- Fresh disposition-verification Task loaded `code-reviewer` in disposition
  mode plus General/Security/Database/Python references and companions:
  "No concrete task-relevant findings." Empty dispositions validated;
  no unresolved/new issues. One review cycle; no finding-fix cycle needed.
- Unreleased changelog extends existing lock-alias entry; v0.3.0 release
  remains unchanged. No custom manifests discovered, locking redesigned,
  spec/priority/dependency/gate changes or release actions performed.
- T-038 remains blocked for the maintainer; T-128 stays unstarted future
  v0.8.0 work. This cycle does not run round 3 or select another task.
- 2026-10-16T18:03:08Z: verification pass
- 2026-10-16T18:03:17Z: Verified pass 2026-10-16T18:03:08Z after required gates and independent review; durable workflow audit records reproduction and disposition evidence.
