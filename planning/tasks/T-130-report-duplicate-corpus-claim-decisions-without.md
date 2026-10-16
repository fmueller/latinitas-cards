---
id: T-130-report-duplicate-corpus-claim-decisions-without
title: Report duplicate corpus claim decisions without internal errors
status: todo
priority: high
spec_ref: specs/v0.3.0.md#agent-ready-cli-contract
dependencies: []
updated_at: "2026-10-16T16:17:10Z"
---

# T-130-report-duplicate-corpus-claim-decisions-without Report duplicate corpus claim decisions without internal errors

## Description

Round 1 synthetic CLI testing found an internal error when a persisted version-2
claims file contains two identical decisions. All three joined readers crash
into exit 70 instead of reporting controlled input/conflict diagnostics.
Severity: medium availability/diagnostic failure on an external input; priority
high for the agent-ready contract. No acceptance bypass or mutation observed.

Contract: `specs/v0.3.0.md#agent-ready-cli-contract` and the version-2 joins in
`#sidecar-annotation-model`. Exit 70 is for unexpected internal failures; review
conflicts must be ignored/reported without corrupting accepted/open semantics.
The fix must select a diagnostic from the closed registry, or deliberately
extend its documented schema/registry, rather than leak a legacy string.

Tested revision: `8acdba8e15940e90529c9af80be3609eb3530b49`, Linux,
Python 3.14.2. Spec SHA256:
`fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.

Reproduce with the installed executable in an external temporary directory:

```python
import json, pathlib, subprocess, tempfile

with tempfile.TemporaryDirectory() as temporary:
    d = pathlib.Path(temporary)
    (d / 'source.txt').write_text('Mater filium amat.\n')
    def run(*args):
        p = subprocess.run(['latinitas-cards', *args, '--json'], cwd=d,
            stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=30)
        print(p.returncode, p.stdout)
        return json.loads(p.stdout)
    run('corpus', 'normalize', 'source.txt', '--adapter', 'text',
        '--corpus-id', 'probe', '--output', 'corpus.jsonl')
    run('annotations', 'segment', 'corpus.jsonl', '--output', 'sidecar.jsonl',
        '--timestamp', '2026-01-02T03:04:05Z')
    report = run('claims', 'list', 'corpus.jsonl',
                 '--annotations', 'sidecar.jsonl')
    run('claims', 'record', 'corpus.jsonl', '--annotations', 'sidecar.jsonl',
        '--fingerprint', report['claims'][0]['fingerprint'],
        '--status', 'accepted', '--reviewer', 'Tester', '--reason', 'Checked')
    commands = [
        ['corpus', 'show', 'corpus.jsonl', '--annotations', 'sidecar.jsonl'],
        ['corpus', 'validate', 'corpus.jsonl', '--annotations', 'sidecar.jsonl'],
        ['annotations', 'inspect', 'sidecar.jsonl', '--corpus', 'corpus.jsonl',
         '--check'],
    ]
    for command in commands:
        run(*command, '--claims', 'corpus.jsonl.claims.json')
    path = d / 'corpus.jsonl.claims.json'
    stored = json.loads(path.read_text())
    stored['decisions'].append(stored['decisions'][0])
    path.write_text(json.dumps(stored))
    for command in commands:
        run(*command, '--claims', 'corpus.jsonl.claims.json')
```

Actual: single-decision controls exit 0/0/0; duplicate input returns 70/70/70,
one envelope with `internal` and `result: null`. Expected: a controlled
diagnostic identifying the conflicting input, with the duplicated decision
not treated as accepted; never exit 70. The exact controlled exit depends on
the chosen documented registry entry, not an assumed legacy `conflict` code.
All readers preserve corpus, sidecar and claims bytes and mtimes. This probe
deliberately edits external persisted input; normal `claims record` correctly
rejects conflicting write batches and is not alleged to produce this file.

## Acceptance

- All three readers handle duplicate persisted version-2 decisions without an
  internal failure or traceback, preserve read-only behavior and produce one
  valid CLI envelope with a documented diagnostic/exit.
- Conflicting decisions remain ignored, not accepted. A single valid decision
  remains visible and accepted as the positive control.
- Preserve existing stale/copied-binding/version rejection and legacy-v1
  claim output/exit exceptions; add coverage through public commands.
- Do not redesign claim identity or alter publication/future-spec gates.

## Verification Notes

- Observed 2026-10-16: 114 real CLI subprocesses, 261 checks; three failures,
  one per joined reader, all this issue. Independent six-command control/repro
  repeated by the coordinator: 0/0/0 versus 70/70/70. Every reader preserved
  file hashes, mtimes and directory membership. No timeouts.
- Deduplicated against all 128 pre-existing tasks, including completed and
  future work. T-125's completed G-1 was an invalid-corpus join failure; this
  reproduces on a valid corpus with current anchors and duplicate decisions.
  T-115's deeply nested JSON finding is distinct. T-103 owns the diagnostic
  registry but has no matching duplicate-decision finding. Not T-038/T-128.
- Investigation only; no implementation or lifecycle transition performed.

## Implementation Notes

`corpus_claims._resolve` emits legacy `Diagnostic("conflict", ...)` for the
duplicate. `join_corpus_decisions` calls `Code(finding.code)`, but the closed
enum has no `conflict` member. Its ValueError becomes the internal envelope.
