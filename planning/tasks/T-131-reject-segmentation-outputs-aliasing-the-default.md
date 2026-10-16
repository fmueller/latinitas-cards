---
id: T-131-reject-segmentation-outputs-aliasing-the-default
title: Reject segmentation outputs aliasing the default manifest lock
status: todo
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-16T17:35:23Z"
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
