# Maintainer PyPI release runbook

This is a procedure for the maintainer, not authorization to execute it. The first
PyPI release is planned for v0.3.0, only after the T-038 maintainer release gate;
this document does not assert current PyPI availability. T-038 stays blocked until
all other v0.3.0 tasks are complete and the maintainer authorizes publication.
Later releases reuse this workflow; no v0.2.x native acceptance gate is reopened.

## Prerequisites (maintainer-owned)

- Confirm the PyPI project `latinitas-cards` and its pending Trusted Publisher:
  owner `fmueller`, repository `latinitas-cards`, workflow `release.yml`,
  environment `pypi`. Use Trusted Publishing, not credentials in the repository.
- Confirm GitHub environment `pypi` has a required reviewer and tag-only deployment
  protection. Only the `publish` job receives this environment and `id-token: write`.
- Confirm Vulgate descriptor values and whether `data/lat-clementine.usfx.xml`
  derives from the eBible.org latVUC file. Record the confirmed provenance and
  descriptor values in `docs/license-compatibility-audit.md`.
- Confirm all other v0.3.0 work is complete before unblocking T-038. Confirm Build
  (`build.yml`) and CodeQL, where enabled in repository settings, are green for
  the exact candidate commit. Confirm version/changelog readiness, including the
  matching `pyproject.toml` version and release notes.

A missing prerequisite blocks the gate naming that item. Do not bypass an approval
or attempt publication to discover whether configuration is correct.

## Ordered release and evidence sequence

1. Record the candidate commit, successful Build and applicable CodeQL run URLs,
   version/changelog review, prerequisite confirmations and maintainer authorization.
   The candidate must be on `main`; inspect the exact revision to be tagged.
2. Create and push exactly one annotated `vX.Y.Z` tag for that candidate and version.
   The committed `.github/workflows/release.yml` triggers only on pushed tags matching
   `v[0-9]+.[0-9]+.[0-9]+`; there is no manual workflow trigger.
3. Wait for `build` and `verify` to pass. `build` checks tag/version equality and
   ancestry on `origin/main`, builds and inspects distributions, records SHA256
   hashes in the job summary, and uploads the `dist` artifact. `verify` installs
   those artifacts on Python 3.13 and 3.14 outside the checkout and checks the CLI
   and absence of default NLP extras. Save the run URL and artifact hashes.
4. Review that evidence, then approve the `pypi` deployment using the required
   environment reviewer. This is the approval mechanism, not a manual upload.
5. Observe `publish`: it downloads the same `dist` artifact and uses
   `pypa/gh-action-pypi-publish`, pinned by SHA in `release.yml`, with Trusted
   Publishing OIDC. Do not rebuild or substitute files between verification and upload.
6. Wait for `verify-pypi` to pass. It retries installation of the exact tagged
   version from PyPI for up to 10 minutes, runs `--help`, and asserts CLTK, Stanza
   and PyTorch are absent. This propagation retry is not permission to retry publishing.
7. Independently verify PyPI file hashes equal the build artifacts for every wheel
   and sdist, recording filenames, SHA256 values and the comparison outcome. Stop on
   a mismatch; a successful install alone is insufficient release evidence.
8. Only after these checks, publish the GitHub release for the same tag with concise
   user-facing notes. Record its URL and the PyPI version URL.
9. Make a follow-up evidence commit recording the candidate/tag, workflow results,
   environment approval, file-hash comparison, release URLs and remaining limitations.
   Complete T-038 only with the required maintainer evidence, not merely a green build.

## Prohibitions and failure handling

No TestPyPI, API tokens, manual uploads, tag rewrites or blind retries. Never move
or recreate a published tag. A published version cannot be re-uploaded; inspect
workflow logs and PyPI state before any recovery decision. If a version or files
already exist, do not upload replacements. Block and report the exact failed step
or missing prerequisite, retain the evidence, and let the maintainer decide whether
a corrected candidate and new version/tag are needed. Do not manually trigger a
workflow, bypass environment protection or treat this runbook as release permission.
