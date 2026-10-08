# v0.2.0 Release Readiness

Prepared package and annotated tag versions: `0.2.0` and `v0.2.0`.
Publication follows v0.1.1: local gates, exact-commit Python 3.13/3.14 CI,
one annotated tag, public GitHub release, fresh-tag locked installation, and
separate publication evidence. No PyPI publishing, uploaded assets or deployment.
The completed spec remains `specs/v0.2.0.md`; v0.2.1 is not activated.

## Readiness evidence

The accepted implementation is
[`e13bef1`](https://github.com/fmueller/latinitas-cards/commit/e13bef1a5643160db9e59a19ce4ecbf49df865e6).
The [restarted adversarial round-three report](https://ampcode.com/threads/T-01a11941-f440-76bf-a0c6-28c0a83d0871)
found no confirmed defect within its synthetic CLI/native-backend scope. Its
103 assertions across 83 managed CLI calls, 69 capture cases, journal-failure
probe, and native subset-import checks are prior evidence, not fresh release runs.
The 926-test result was from round two; round three did not repeat the full suite.
Release preparation changes metadata, documentation and metadata tests only.

A fresh release-preparation run on 2026-10-08 passed:

```bash
uv run --with anki==26.9.3 python scripts/check-managed-anki.py --output-dir "$work/evidence"
```

All 17 native backend cases passed on Linux x86_64/glibc 2.36, Python 3.14.2,
Anki 26.9.3. They retain complete card/review-log rows and Personal Notes during
selected content/tag updates, no-op import, partial application, recovery and
anchored-absence versus never-anchored creation refusal. See the
[fixture and exact transport settings](managed-csv-native-verification.md).
The report records accepted HEAD and hashes the actual application sources;
only release metadata was edited. Script SHA-256:
`2be32b1bd9617db35f03d577cd7fd34301c7fce2ec63b2129239f499d1a59f14`.
Report SHA-256:
`45f2d6c46c7f5e15b10575e459a0688461de0e028837164db57f1cdb65935d6a`.

The [actual Desktop 26.09.3 import-dialog gate](managed-csv-desktop-verification.md)
separately verified mapped content and tag-only addition/removal/final removal,
no-op and deliberately unmapped-tag failure. This release does not claim that
backend tests repeat GUI interaction or establish arbitrary-client compatibility.

The [2026-10-06 morphology native acceptance](morphology-native-verification.md)
is **maintainer-reported**, limited to a synthetic four-verb kit on Desktop
26.09.2 and AnkiMobile 25.09 (iPhone 16 Pro Max/iPad Air M4), with no screenshots
for that run. The **2026-10-07** maintainer waiver applies only to the additional
sanitized representative-deck native gate. It is not universal certification,
proof of CSS migration on scheduled notes, or a waiver of all native checks.

The initial release-preparation `uv run ruff check`, `uv run mypy`, and
`uv run pytest -v` passed: 90 source files and 926 tests on 2026-10-08.
Final post-review checks, the full `mise run check` gate and exact-commit CI must
pass before tagging. Publication,
CI run links and fresh-tag install evidence are recorded separately in the
publication task after those actions occur; this document is not proof of them.

## Practical limitations

- Managed delivery is offline/manual CSV import, not live Anki integration.
  Keep Anki closed during capture, use a full recoverable backup, reacquire
  destination evidence and prevent intervening edits. Backup recoverability and
  freshness are operator attestations, not a file-transport lock or proof.
- Capture supports the closed schema-18 contract and retains private byte-bound
  evidence; it does not certify arbitrary backups, media recovery or sync.
- Only compatible existing-note content and reconciled tags can be applied.
  Slot/front/back/guard, structural/lifecycle, identity, CSS and template migration
  are unsupported. A separately approved fresh start gets new scheduling.
  Keep Personal Notes unmapped; skipped effects remain unresolved, not successful.
- Automatic linguistic claim calibration is disabled. Explanations, segmentation
  and fourth-role labels need explicit manual review through the Python API;
  CLI-only decks omit unreviewed claims. Form-parsing exercises require review,
  not automatic acceptance or an accuracy certification.
- Static comparison remains the default and unverified-client fallback. Native
  presentation evidence stays within the reported clients/fixtures above.
- Optional annotation requires CLTK/Stanza models and has the unresolved network
  exposure below. Neither annotation extra is required for supported offline
  workflows or the release installation smoke check.

## Current dependency-advisory assessment — 2026-10-08

Fresh GitHub Dependabot API inspection reports three open urllib3 alerts:

| Alert | Advisory | Severity | Conditions |
|---|---|---|---|
| [#141](https://github.com/fmueller/latinitas-cards/security/dependabot/141) | GHSA-8988-9cw3-xx77 | High | HTTPS proxy and target TLS policies can be mixed; exploitability depends on effective proxy/TLS configuration. |
| [#142](https://github.com/fmueller/latinitas-cards/security/dependabot/142) | GHSA-vxq7-64xx-v4gw | High | Streaming an untrusted chunked response can buffer an unbounded chunk-size line. |
| [#143](https://github.com/fmueller/latinitas-cards/security/dependabot/143) | GHSA-gh4c-6fx4-qh6g | Moderate | Chunked Deflate streaming with trailing encoded bytes can loop without progress. |

Locked urllib3 2.7.0 is affected; all three list 2.8.0 as the patched version.
Fresh `uv export --locked --no-dev --no-emit-project` excludes requests/urllib3;
both `--extra annotate` and `--extra annotate-gpu` include requests 2.33.0 and
urllib3 2.7.0. No dependency versions were changed or alerts dismissed.

The application invokes CLTK's Latin Stanza backend during annotation.
[CLTK 2.5.1's Stanza process](https://github.com/cltk/cltk/blob/33e1653331fc2499e3f2f5da45237d87db0313cc/src/cltk/stanza/processes.py#L45-L63)
uses `REUSE_RESOURCES`: missing resources/models can trigger downloads.
[Stanza 1.14.0](https://github.com/stanfordnlp/stanza/blob/v1.14.0/stanza/resources/common.py#L147-L193)
uses `requests.get(..., stream=True)` and `iter_content(chunk_size=131072)`
for the resource index and raw-Requests model download path. In
[Requests 2.33.0](https://github.com/psf/requests/blob/v2.33.0/src/requests/models.py#L801-L839),
iteration calls urllib3 streaming with `decode_content=True`. Therefore the
chunked-response and Deflate paths are concretely reachable during optional
resource acquisition, conditional on server responses. HTTPS default resource
hosts reduce ordinary attack exposure but do not remove affected code paths.
CLTK/Stanza do not explicitly configure custom `proxy_ssl_context` or HTTPS
forwarding; Requests can inherit environment proxies. The proxy advisory's
specific exploit conditions were not reproduced, and are not certified absent.

These are real optional-runtime risks, not default offline-workflow dependencies.
The release is bounded to those offline workflows; annotation extras are **not
recommended for untrusted network responses or proxy-sensitive use** until
urllib3 is separately updated and those paths verified. Optional-resource
network behavior and exploit conditions were assessed from authoritative source,
not tested by installing CLTK or contacting a malicious server. Remediation
remains follow-up; this release does not claim the repository is vulnerability-free.
