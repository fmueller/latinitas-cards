# v0.1.1 Release Readiness

Prepared package and annotated tag versions: `0.1.1` and `v0.1.1`.
The owner requested this GitHub release on 2026-10-04. Publication follows the
v0.1.0 process: final checks, exact-commit CI, annotated tag, GitHub release,
fresh-tag installation, and separate publication evidence in T-047.
No PyPI publication, uploaded binary assets, or deployment is part of this release.

## Readiness evidence

The active `specs/v0.1.1.md` authored-note implementation is complete. Five
fix/test cycles ended with cycles 4 and 5 passing without new confirmed defects.
The final implementation gate passed 627 tests and mypy checked 65 source files.
See the [final independent test report](https://ampcode.com/threads/T-01a108a3-d401-7414-9012-36cdb6c6c1ff).
Release preparation changes metadata and documentation, not application behavior.

Fresh synthetic CSVs were imported through Anki 26.9.3's native Linux backend:
first import created eight notes/cards, re-extraction added one, and edited
re-import retained nine. Existing Anki note IDs, LatinitasIDs, and card IDs
remained stable; nonempty Personal Notes survived unmapped re-import and reopen.
All three kinds (`vocab`, `form`, `qa`) had their native card HTML inspected,
including mixed newline handling. This is not a desktop import-dialog test,
other-version/platform certification, or human approval of a learner's content.

## Practical limitations

- Import into Anki is manual. Back up before updates; keep Personal Notes unmapped.
  Destination-only manual tags are not guaranteed to survive managed updates.
- Native Anki can strip NUL characters and normalize Unicode to NFC.
- Unknown kind filters can select nothing. Review the preview before exporting.
- Catchable staging failures are tested; hard crashes, power loss, network
  filesystems, and general simultaneous exports have no recovery guarantee.
- Import only files reported by the current successful export: older selections'
  files are intentionally not deleted. Grammar and translations need human review.

## Dependency advisory assessment

On 2026-10-04 GitHub reports open urllib3 alerts
[#141](https://github.com/fmueller/latinitas-cards/security/dependabot/141),
[#142](https://github.com/fmueller/latinitas-cards/security/dependabot/142), and
[#143](https://github.com/fmueller/latinitas-cards/security/dependabot/143): two
high and one moderate. Locked urllib3 2.7.0 is affected; the listed fix is 2.8.0.
The HTTPS proxy TLS configuration issue and response-stream resource-exhaustion
issues are relevant when optional annotation dependencies perform network requests.

`uv export --locked --no-dev --no-emit-project` excludes urllib3 and requests;
adding `--extra annotate` includes requests 2.33.0 and urllib3 2.7.0. Thus these
are real optional-runtime exposures, not vulnerabilities in the default offline
authored-note path. Neither annotation extra is recommended for untrusted network
responses or proxy-sensitive use until separately updated and verified. Alerts
are not dismissed or claimed resolved; dependency remediation remains follow-up.
