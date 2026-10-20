---
id: T-145-patch-optional-fsspec-dependency-for-advisory-144
title: Patch optional fsspec dependency for advisory 144
status: completed
priority: high
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies: []
updated_at: "2026-10-20T18:42:57Z"
---

# T-145-patch-optional-fsspec-dependency-for-advisory-144 Patch optional fsspec dependency for advisory 144

## Description

Remediate Dependabot alert 144 (GHSA-27vj-qcqg-25rc) in the optional NLP
dependency lock under the v0.3.0 PyPI packaging contract. Alert 144 is unrelated
to completed Taskrail T-144. Pin the active spec to SHA256
fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704.

## Acceptance

- Lock fsspec to a compatible patched release (at least 2026.6.0) without
  unrelated dependency upgrades or default NLP additions.
- Preserve package metadata, optional CPU/GPU separation, no-publish guards,
  historical audit bytes, and the maintainer-only blocked release gate.
- Refresh only current advisory-related inventory evidence and verify safe
  public CLI/package workflows without installing NLP or executing exploits.
- Complete strict TDD, simplification, independent review and the sequential
  Ruff, mypy, pytest and mise gates before Taskrail verification/completion.

## Verification Notes

- Strict TDD: extended the existing public packaging metadata test with a
  GHSA-27vj-qcqg-25rc version boundary. Its targeted run failed on the old
  lock: `assert (2026, 2, 0) >= (2026, 6, 0)`. After the targeted resolver
  update and inventory fingerprint refresh, all three packaging tests passed.
- Mandatory `uv run ruff check` then `uv run mypy` then `uv run pytest -v`
  passed: no lint/type errors; 2633 passed, 7 skipped, one Pydantic serialization
  warning in the existing `test_writer_rejects_float_without_replacing` test.
- `bash scripts/check-dist.sh` passed; Python 3.14 wheel and sdist installs
  passed `scripts/check-install.sh`: help/version and corpus normalize/validate
  (`ok=true`), with optional NLP distributions absent.
- Dedicated `code-simplifier` Task: no changes; direct assertions and inventory
  edits already minimal. `git diff --check` passed.
- Independent `code-reviewer` Tasks selected General and Security (including
  security-review companion). Both: "No concrete task-relevant findings."
  Python lane omitted: no application Python changed, only existing packaging
  assertions; database/framework/ML lanes omitted: no such behavior changed.
  Fresh candidate validation: "No validated findings." Fresh disposition
  verification: "No concrete task-relevant findings." No fixes or deferrals;
  one review cycle. Reviewers did not independently fetch wheel metadata.
- Full `mise run check` passed, including the second sequential full pytest
  run and all commit, author, mutation, distribution and publish guard suites.
- Rebased without conflicts onto the coordinator provenance handoff
  a821468d4db7a9c6492bee1163b0f4057161f01d. Combined exact Ruff/mypy/pytest
  chain passed (2633 passed, 7 skipped, one known warning), followed by a
  passing full `mise run check` (same pytest counts and all guard suites).
- Fresh integration review at rebase confirmed byte-identical corpus provenance,
  T-038 and corpus provenance test. It flagged this stale verification note:
  "Combined provenance-base validation pending the coordinator handoff";
  fixed by recording the delivered base and actual validation state here.

## Implementation Notes

- Initial remote main: 3a1ff60fcd0555582cc9f4c36ae4d3e777719485.
- Coordinate main serialization with the provenance owner; no push before the
  exact provenance revision handoff and its hosted CI passing. The advisory
  commit does not edit T-038 or corpus provenance. A separately authorized
  documentation commit records the later explicit maintainer derivation answer;
  it does not change the blocked release gate or authorize publication.
- Current alert metadata was read through the GitHub API without attack
  examples. Observed optional exposure: default inverse tree and default/dev
  exports omit fsspec; both annotation exports include it through Torch and
  Hugging Face. First-party source/tests contain no affected API references;
  this is not a categorical safety claim about third-party call paths.
- `uv lock --upgrade-package fsspec==2026.6.0` changed only fsspec among 92
  package records. PyPI requires fsspec>=2023.5.0 for huggingface-hub 1.31.0
  and fsspec>=0.8.5 for torch 2.14.1; the targeted patch resolves without
  project metadata changes. No optional NLP or exploit execution occurred.
- Exact wheel PEP 658 metadata: Python >=3.10, BSD-3-Clause, LICENSE; no
  package license or supported-Python change. Historical audit SHA256 remains
  1ab38cb0f042aefcfef1d63efb4fa47689c92b64ee044e1ddd8baaaa110de137.
- 2026-10-20T18:42:57Z: verification pass
- 2026-10-20T18:42:57Z: Optional dependency advisory remediated; all sequential gates and reviews passed; no release authorization or T038 lifecycle change.
