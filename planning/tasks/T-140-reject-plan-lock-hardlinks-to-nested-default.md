---
id: T-140-reject-plan-lock-hardlinks-to-nested-default
title: Reject plan lock hardlinks to nested default-manifest DATA reservations
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#policy-gated-agent-automation
dependencies: []
updated_at: "2026-10-17T13:18:18Z"
---

# T-140-reject-plan-lock-hardlinks-to-nested-default Reject plan lock hardlinks to nested default-manifest DATA reservations

## Description

Round 6 at 0e6afbcf9d9f33493d7426b461a4e6473f0cd9f9 confirms a
medium-severity filesystem reservation contract violation in `annotations plan`.
When the known default manifest is a DATA symlink into a nested directory,
a plan's output lock hardlinked to that resolved manifest lock is not rejected
as an alias. Held: actual exit 4/busy, expected 3/alias. Released: actual exit
0 and a plan is written, expected 3/alias with no output. The manifest DATA
symlink is supported by real normalization; this requires no arbitrary custom
manifest discovery. Reserved inode/bytes and normalizer exclusion remain
intact: no exclusion bypass, inode detachment or corruption was demonstrated.

Overlap: completed T-117 owns plan/receipt artifact alias rejection. Completed
T-134 and T-136 fixed resolved default-manifest DATA reservations in apply and
segment respectively, not this separate plan writer. All existing tasks,
including completed ones, were searched; this nested plan-lock hardlink case
is not an existing open finding. Do not reopen already-fixed segment/apply
behavior or require discovery of unspecified custom manifests.

## Acceptance

- With a known default manifest DATA symlink into a nested directory, reject
  the plan output-lock hardlink to its resolved lock before acquisition:
  exit 3/alias held and released, no plan created, reservation unchanged.
- Preserve normalizer exclusion while held, successful normalization after
  release, and independent plan/apply and byte-identical plan retry controls.
- Cover ordinary symlink/chain and dangling aliases as passing controls; these
  already reject correctly in the round-6 probe. Direct output hardlinks must
  also report alias rather than merely exists (currently exit 3/exists).
- Run mandatory ruff/mypy/pytest validation plus focused real subprocess
  lock tests without changing the spec or expanding custom-manifest discovery.

## Verification Notes

Executable reproduction from repository root (temporary local files only):

```bash
uv run python - <<'PY'
import fcntl, json, os, pathlib, subprocess, tempfile
cli = str(pathlib.Path('.venv/bin/latinitas-cards').resolve())
def run(args):
    p = subprocess.run([cli, *map(str, args), '--json'],
                       stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, timeout=30)
    print(p.returncode, p.stdout.strip())
    return p
with tempfile.TemporaryDirectory() as td:
    d = pathlib.Path(td)
    source, corpus, side = d/'source.txt', d/'corpus.jsonl', d/'side.jsonl'
    source.write_text('Mater filium amat.\nPuer librum legit.\n')
    norm = ['corpus', 'normalize', source, '--adapter', 'text',
            '--corpus-id', 'plan-probe', '--output', corpus]
    assert run(norm).returncode == 0
    (d/'storage').mkdir()
    manifest, data = corpus.with_suffix('.manifest.json'), d/'storage/data.json'
    manifest.rename(data)
    manifest.symlink_to('storage/data.json')
    assert run(norm).returncode == 0
    passage = next(r for r in map(json.loads, corpus.read_text().splitlines())
                   if r['record'] == 'passage')
    patch, output = d/'patch.json', d/'selected-plan.json'
    patch.write_text(json.dumps({'format': 'latinitas-annotation-patch',
        'version': 1, 'operations': [{'operation': 'add', 'layer': 'translation',
        'anchor': {k: passage[k] for k in ('passage_id', 'passage_hash')},
        'value': {'language': 'en', 'text': 'mother'},
        'producer': {'kind': 'agent', 'name': 'fixture', 'version': '1'},
        'created': '2026-01-02T03:04:05Z'}]}))
    reserved = pathlib.Path(str(data) + '.lock')
    os.link(reserved, pathlib.Path(str(output) + '.lock'))
    args = ['annotations', 'plan', patch, '--corpus', corpus,
            '--sidecar', side, '--output', output]
    before = reserved.stat().st_ino, reserved.read_bytes()
    with reserved.open('ab') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(norm).returncode == 4
        run(args)  # Actual 4/busy; expected 3/alias.
        assert run(norm).returncode == 4
        assert not output.exists()
    run(args)  # Actual 0, writes plan; expected 3/alias and absent output.
    print('plan created:', output.exists())
    assert (reserved.stat().st_ino, reserved.read_bytes()) == before
    assert run(norm).returncode == 0
PY
```

Fresh four-shape CLI probe executed twice: 32 calls and 53 assertions per
run; only the hardlink shape produced five failed assertions. Symlink, chain
and dangling controls passed; ordinary plan/apply passed. An independently
authored direct output-hardlink probe returns 3/exists (not alias), preserves
output/reserved bytes and held normalizer exclusion. This is a secondary
diagnostic violation, not evidence of a write through a held reservation.

## Implementation Notes

`annotation_plans.write_document` scans parent directories of logical inputs
for locks but does not include the known manifest's resolved DATA lock when
it resides elsewhere. Follow the explicit known default-manifest identity;
avoid relying on an ambient-directory scan for that reservation.
Findings-only filing; no task start, fixes, dependency or gate changes.
