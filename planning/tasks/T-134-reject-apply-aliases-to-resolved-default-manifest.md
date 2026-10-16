---
id: T-134-reject-apply-aliases-to-resolved-default-manifest
title: Reject apply aliases to resolved default-manifest DATA locks
status: todo
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-16T20:45:54Z"
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

Unstarted. Finding-only delivery; no product or test changes in this round.
