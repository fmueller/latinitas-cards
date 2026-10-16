---
id: T-133-reject-annotation-apply-aliases-to-reserved-writer
title: Reject annotation apply aliases to reserved writer locks
status: todo
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-16T19:37:41Z"
---

# T-133-reject-annotation-apply-aliases-to-reserved-writer Reject annotation apply aliases to reserved writer locks

## Description

Round 4 synthetic CLI testing at a542fd09105b551b2f73199c85b1d0c1518e57f2
found that `annotations apply` accepts aliases to reserved writer locks, contrary
to `specs/v0.3.0.md#sidecar-annotation-model`. The spec requires new writers to
reject outputs aliasing manifests or locks directly, by symlink or hard link.

An output hardlinked to the default manifest lock returns 0 and appends one
annotation while that lock is held. An output-lock symlink to the same existing
lock returns 4/busy while held, then 0 after release, instead of 3/alias.
Dangling output-lock symlinks to the default manifest lock or a symlinked
corpus's separate logical `.lock` return 0 and create the absent reserved path.

Severity: medium filesystem-contract violation; high repair priority. No held
lock inode detachment or normalizer exclusion bypass was demonstrated: atomic
replacement detaches the output hardlink only, and the reserved inode and bytes
remain intact. Do not describe this as proven concurrent data corruption.

### Standalone reproduction

Run from the repository with its installed CLI. This uses only synthetic data
in a disposable external directory and public commands; no application imports.
The reserved file's contents deliberately form a valid empty annotation stream
to distinguish alias refusal from incidental parse failure.

```python
import json
import subprocess
import tempfile
from pathlib import Path

cli = str(Path(".venv/bin/latinitas-cards").resolve())
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    def run(*args):
        process = subprocess.run(
            [cli, *map(str, args), "--json"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(process.returncode, process.stdout.strip())
        return process.returncode

    source, corpus = root / "latin.txt", root / "corpus.jsonl"
    side, patch, plan = root / "notes.jsonl", root / "patch.json", root / "plan.json"
    source.write_text("Puella rosas portat.\nSenex aquam bibit.\n")
    assert run("corpus", "normalize", source, "--adapter", "text", "--corpus-id", "repro", "--output", corpus) == 0
    passage = next(
        row for line in corpus.read_text().splitlines() if (row := json.loads(line)).get("record") == "passage"
    )
    reserved = root / "corpus.manifest.json.lock"
    reserved.unlink()
    Path(str(side) + ".lock").symlink_to(reserved.name)
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
                        "producer": {"kind": "agent", "name": "repro", "version": "1"},
                        "created": "2026-01-02T03:04:05Z",
                    }
                ],
            }
        )
    )
    assert run("annotations", "plan", patch, "--corpus", corpus, "--sidecar", side, "--output", plan) == 0
    run("annotations", "apply", plan, "--corpus", corpus, "--sidecar", side)
    print("reserved exists:", reserved.exists(), "sidecar exists:", side.exists())
```

Expected exits: 0, 0, **3/alias**; reserved and sidecar remain absent. Actual:
0, 0, **0**, appended 1; both files exist. The absent reserved file becomes an
empty lock file (0 bytes); the new sidecar contains its header and annotation.
An ordinary independent sidecar control returns 0 and appends the same operation.

## Acceptance

- Reject apply output and output-lock aliases to known reserved writer paths,
  including the default manifest lock and logical/resolved corpus locks, with
  exit 3/alias before creating or replacing those paths.
- Cover existing/absent direct, symlink, hardlink, chained and directory aliases
  where applicable; preserve held inode, bytes and real writer exclusion.
- Keep normal apply, retry/no-op, corpus/claims/sidecar locking and receipt/policy
  behavior unchanged. No discovery of arbitrary unknown custom manifests.
- Preserve the passing T-129/T-131 segmentation and T-130/T-132 claims replays.
- Run the mandatory validation chain after implementation.

## Verification Notes

- Round 4 tested exact accepted revision and pinned v0.3.0 spec SHA256
  fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704.
- The policy/locks matrix produced 20 failed assertions for this single finding;
  all 126 mandatory prior reserved-lock cases passed (750 CLI invocations,
  1,575 assertions). No timeout or internal-70 occurred.
- Independent coordinator reproduction used different Latin text and producer,
  with 12 CLI calls: hardlink and absent-output-lock cases both return 0 instead
  of 3; ordinary output succeeds. Held manifest bytes/inode remain unchanged,
  normalizer returns 4 before and after apply and 0 after lock release.
- Reproduction harness initially selected the source record by line number;
  that KeyError was a harness error, corrected to select `record == passage`.
  Original failed harness is retained in the round evidence, not hidden.
- Evidence delivery contains ROUND4-REPORT.md, a reproducible bundle, full argv,
  stdin/timeouts, outputs, input captures, hashes and failure classifications.
  No product implementation or new verification-complete transition is included.

## Implementation Notes

Source correlation: `annotation_apply.py` protects corpus, claims, default
manifest data and locks it acquires, but omits the default manifest lock and
the separate logical corpus lock. Its suffix check catches direct `.lock`
outputs but not a normally named hardlink or output-lock alias.

Dedupe searched all task files, including completed T-129 through T-132 and
future T-128. T-129/T-131 own segmentation, which passes at this revision;
T-130/T-132 own claim resolution. T-118's completed acceptance explicitly
requires apply lock-alias protection: this is an uncovered acceptance defect,
not a claim that T-118 lacked that requirement. Reopening overlap is explicit;
this standalone high-priority task records the new counterexample. T-113's
general writer requirement overlaps; T-119's policy revalidation is unaffected.
T-128 concerns future dense-markup retention, not alias protection. No task
dependency or follow-up to T-124, release gate or spec change is requested.
