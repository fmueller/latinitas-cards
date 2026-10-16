---
id: T-129-reject-segmentation-outputs-aliasing-corpus-and
title: Reject segmentation outputs aliasing corpus and claims locks
status: todo
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-16T16:17:10Z"
---

# T-129-reject-segmentation-outputs-aliasing-corpus-and Reject segmentation outputs aliasing corpus and claims locks

## Description

Round 1 synthetic CLI testing found that `annotations segment` accepts reserved
corpus and logical claims lock paths as its output. This violates
`specs/v0.3.0.md#sidecar-annotation-model`: new writers must reject aliases to
corpora, sidecars and locks. Severity: medium data-integrity/serialization risk;
priority high because a selected output can detach a held writer lock.

Tested revision: `8acdba8e15940e90529c9af80be3609eb3530b49` on Linux,
Python 3.14.2. Spec SHA256:
`fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.

Minimal reproduction, using an external temporary directory and the installed
`latinitas-cards` executable (all subprocesses have empty stdin):

```python
import fcntl, os, pathlib, subprocess, tempfile

with tempfile.TemporaryDirectory() as temporary:
    d = pathlib.Path(temporary)
    source, corpus = d / 'source.txt', d / 'corpus.jsonl'
    source.write_text('Mater filium amat.\n')
    def run(*args):
        p = subprocess.run(['latinitas-cards', *map(str, args), '--json'],
            cwd=d, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=20)
        print(p.returncode, p.stdout)
        return p.returncode
    normal = ['corpus', 'normalize', source, '--adapter', 'text',
              '--corpus-id', 'lane-test', '--output', corpus]
    assert run(*normal) == 0
    run('annotations', 'segment', corpus, '--output',
        str(corpus) + '.claims.json.lock',
        '--timestamp', '2026-01-02T03:04:05Z')
    lock = pathlib.Path(str(corpus) + '.lock')
    lock.write_text('{"format":"latinitas-annotations",'
                    '"record":"header","version":1}\n')
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(*normal) == 4
        inode = os.fstat(held.fileno()).st_ino
        run('annotations', 'segment', corpus, '--output', lock,
            '--timestamp', '2026-01-02T03:04:05Z')
        print('lock replaced:', inode != lock.stat().st_ino)
        run(*normal)
```

Expected: both segment calls exit 3 `alias`, create/replace no reserved path,
and the final normalizer still exits 4 `busy`. Actual: segment exits 0 for both;
the held lock inode changes at its pathname and final normalization exits 0.
Ordinary sidecar output succeeds as positive control. POSIX lock contents are
opaque; the header seed does not violate a documented lock-content contract.
The absent claims-lock case requires no manually seeded lock contents.

## Acceptance

- Reject outputs aliasing corpus and exact logical claims lock paths directly,
  through symlinks and through hard links, before mutation; preserve normal
  sidecar operation and existing CLI diagnostic precedence.
- Reproduce with a real held lock and subprocesses: rejected segmentation must
  preserve the inode, bytes and subsequent writer exclusion.
- Cover an absent logical claims-lock destination, existing lock files and a
  normal output control. Do not remove or replace persistent locks.
- No redesign of lock primitives, future memory work, or release-gate changes.

## Verification Notes

- Discovered and independently replayed on 2026-10-16. Main lanes 5/6 harness:
  251 subprocesses, 435 assertions, ten failures all this finding. Direct,
  symlink and hard-link aliases were accepted; demonstrated exclusion bypass
  is the direct-path variant. Six-command focused replay confirmed it twice.
- Header-only lock SHA256 before:
  `a66fa0f09d1f85c0a0113fd32a79ea9c1d39631afc69097342e216e4677f5b05`;
  replaced lock SHA256 after:
  `df8b1eedfc8ce8a42bebcf70307459051741c4c2ccc17a66479c3ba4dc9a8916`.
- Dedupe included all 128 pre-existing tasks. Completed T-113 owns segment
  aliases but does not cover this reserved-lock omission. T-118's apply
  protections passed. T-084 fixed the analogous phrase-selection path, not
  segment. T-119, T-038 and future T-128 do not cover this finding.
- Investigation only: no fix, implementation lifecycle transition or release.

## Implementation Notes

`commands/annotations.py` passes corpus, claims data and default manifest to
`append_records`, but omits corpus/claims lock paths. The sidecar writer adds
only its own lock, so the bad output acquires `.lock.lock` and replaces `.lock`.
