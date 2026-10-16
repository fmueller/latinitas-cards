---
id: T-129-reject-segmentation-outputs-aliasing-corpus-and
title: Reject segmentation outputs aliasing corpus and claims locks
status: completed
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-16T16:50:33Z"
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
    source, corpus = d / "source.txt", d / "corpus.jsonl"
    source.write_text("Mater filium amat.\n")

    def run(*args):
        p = subprocess.run(
            ["latinitas-cards", *map(str, args), "--json"],
            cwd=d,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )
        print(p.returncode, p.stdout)
        return p.returncode

    normal = ["corpus", "normalize", source, "--adapter", "text", "--corpus-id", "lane-test", "--output", corpus]
    assert run(*normal) == 0
    run(
        "annotations",
        "segment",
        corpus,
        "--output",
        str(corpus) + ".claims.json.lock",
        "--timestamp",
        "2026-01-02T03:04:05Z",
    )
    lock = pathlib.Path(str(corpus) + ".lock")
    lock.write_text('{"format":"latinitas-annotations","record":"header","version":1}\n')
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(*normal) == 4
        inode = os.fstat(held.fileno()).st_ino
        run("annotations", "segment", corpus, "--output", lock, "--timestamp", "2026-01-02T03:04:05Z")
        print("lock replaced:", inode != lock.stat().st_ino)
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

### Workflow-v3 implementation evidence (2026-10-16)

1. Understand: fetched origin/main; HEAD and remote both matched accepted base
   e63d7663f82f5c3307a86545ea8cdd96d320c89e. Round1 product revision was
   8acdba8e15940e90529c9af80be3609eb3530b49; no product transfer was needed.
   Spec SHA256 remained
   fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704.
   Taskrail validate passed; next --json selected exactly this high/todo task
   with zero active tasks. Started this task only via mise exec -- taskrail.
   Independently inspected normalizer writer_locks, append_records,
   reject_aliases and record_corpus_batch. Claims locks suffix the logical
   claims data path, not its symlink target (also checked T-118 regression).
2. RED: public CLI subprocesses in external temporary directories with empty
   stdin, real held flocks, direct/symlink/hardlink aliases of corpus/claims
   locks (including symlinked claims data) and absent logical claims lock:
   uv run pytest tests/unit/annotations_command_test.py
   -k 'held_reserved_lock or absent_logical' -q -s: 10 failed, 37 deselected.
   All segment commands wrongly exited 0 instead of alias/3.
   Exact task replay confirmed exits 0 normalize, 0 absent claims-lock output,
   4 normalize under held corpus lock, 0 segment over lock, 0 normalize.
   Held inode 1569733 changed to 1569740; SHA256 changed from
   a66fa0f09d1f85c0a0113fd32a79ea9c1d39631afc69097342e216e4677f5b05 to
   df8b1eedfc8ce8a42bebcf70307459051741c4c2ccc17a66479c3ba4dc9a8916.
   GREEN: extended segment's existing append_records protected list with
   logical/resolved corpus locks and exact logical claims lock; no lock API
   change, no early validation or precedence change. Targeted 47 passed.
   Exact after replay: exits 0 normalize, 3 absent claims-lock output (absent
   afterward), 4 held normalize, 3 segment over lock, 4 held normalize,
   0 ordinary sidecar. Inode stayed 1569636 and bytes retained SHA256
   a66fa0f09d1f85c0a0113fd32a79ea9c1d39631afc69097342e216e4677f5b05.
   No persistent locks removed/replaced; tests additionally check subsequent
   subprocess nonblocking exclusion for all nine held-lock cases.
3. Initial chain: ruff and mypy passed; pytest had 2487 pass / 1 fail because
   install_docs_test required Unreleased to stay empty after v0.3.0. Adapted
   that stale assertion to heading order for the required Unreleased fix note;
   version stays 0.3.0 and release/maintainer gates remain unchanged.
   Restarted uv run ruff check -> uv run mypy -> uv run pytest -v:
   all passed; 169 source files, 2488 tests, one existing serializer warning.
4. Dedicated simplifier Task loaded code-simplifier; removed only debug/hash
   printing from the held-lock test. Accepted; substantive assertions kept.
   No other suggestions/rejections. Targeted annotations + install docs:
   50 passed. Parent inspected resulting diff.
5. Separate parallel read-only Task reviews: General, Security, Database
   (filesystem persistence/concurrency), Python. Each loaded code-reviewer,
   reviewer-index, its dedicated ECC reviewer; Security loaded security-review
   and linked cloud-infrastructure-security; Database loaded postgres-patterns
   and database-migrations; Python loaded python-patterns. General has no
   companions. Three specialists cover security/integrity, locking and Python;
   framework/SQL/ML/UI lanes omitted as inapplicable (no framework/API changes).
   Each lane's verbatim finding conclusion:
   "No concrete task-relevant findings."
   Security diff-check passed; Database targeted 10 passed; Python targeted
   50 passed. Parent did not overlap full/browser suites.
6. Fresh candidate-validation Task loaded code-reviewer plus lane references:
   "No concrete task-relevant findings."
   "0 candidates"; no validation/rejection/deduplication or fixes required.
   Disposition: zero findings, zero deferrals, no regression-TDD fix needed.
   T-130 stays separate todo; T-128 unstarted; T-038 blocked. No round2 run.
7. Post-review exact ruff/mypy/pytest chain passed again (2488 tests). Initial
   mise run check then failed at ruff format --check: supplied Python snippets
   in T-129 and untouched T-130 were unformatted. Independently confirmed the
   same two failures from pristine HEAD copies, using locked Ruff 0.16.8.
   First fresh disposition verifier loaded code-reviewer and lane references,
   found zero unresolved code findings, but correctly retained this gate blocker.
   No taskrail verify/complete was run while failing. Overseer explicitly
   authorized formatter-only corrections to both snippets, preserving semantics
   and historical evidence. Ran existing uv run ruff format on those two files;
   inspected full snippet diffs, confirmed identical Python ASTs before/after
   for both, and identical T-130 text outside the snippet. This is formatting
   only, not T-130 product/lifecycle/acceptance work. No gate/config changes.
   Restarted exact uv run ruff check -> uv run mypy -> uv run pytest -v:
   all pass, 169 source files, 2488 passed / one existing serializer warning.
   Sequential mise run check: pass, 385 files already formatted, 2488 tests
   passed, commit/push/author/mutation/config/distribution/publish guards pass
   (distribution checks: 54 cases). git diff --check passed.
   Fresh final disposition-verification Task loaded code-reviewer in final
   mode; verdict: "Final disposition: PASS"; "0 candidates and 0 deferrals";
   "I found no new task-relevant issue." Formatter blocker resolved. One code
   review cycle, no finding-fix cycles. All gates passed before lifecycle close.
8. Taskrail verify passed at 2026-10-16T16:50:26Z; completed through the CLI at
   2026-10-16T16:50:33Z. Validate passed; 127 done, zero active, one blocked,
   two todo. Observed next --json: T-130 (high/todo), NOT started. T-038 remains
   maintainer-blocked and T-128 remains future-spec todo. Spec pin unchanged.
   Delivery limited to focused non-force main push; no release/tag/publication,
   deployments, manual workflow triggers, settings or infrastructure changes.
- 2026-10-16T16:50:26Z: verification pass
- 2026-10-16T16:50:33Z: Verified pass at 2026-10-16T16:50:26Z after all gates and independent final disposition PASS; held persistent locks retain inode, bytes and exclusion.
