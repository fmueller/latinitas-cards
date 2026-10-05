---
id: T-063-escape-unsafe-unicode-controls-in-managed-cli-json
title: Escape unsafe Unicode controls in managed CLI JSON review output
status: completed
priority: medium
spec_ref: specs/v0.2.0.md#managed-update-plans
dependencies: []
updated_at: "2026-10-05T07:50:16Z"
---

# T-063-escape-unsafe-unicode-controls-in-managed-cli-json Escape unsafe Unicode controls in managed CLI JSON review output

## Description

Adversarial synthetic round 1 reproduced unsafe terminal output in the
implemented managed-plan review CLI at revision
`2289f45e07b8e1dcae25e4c381635445aeea78c6`.
This belongs to the spec's inspectable, machine-readable managed update plans.

A proposed Meaning containing U+202E/U+202C bidi controls and U+009B C1 CSI
passes the supported JSON request boundary. `managed plan` exits successfully
and emits those characters literally to stdout. Bidi controls can reorder
displayed review text; C1 handling depends on the terminal. No terminal code
execution, destination mutation, or native preservation failure is claimed.

`commands/managed.py` echoes `_encoded(result)` on successful plan, approval,
emission, observation, and reconciliation paths. `destination_state._encoded`
uses `ensure_ascii=False` for canonical persistence/hash encoding. JSON escapes
C0 characters but not these Unicode controls. The error path already uses
`encode_unsafe_controls`; `commands/form_parsing.py` uses ASCII JSON for safe
preview. Do not change canonical persisted bytes or plan identities merely to
make the terminal representation safe.

Deduplication: no matching open Taskrail task or GitHub issue was found.
T-058's resolved SEC-1 covered tag-cell validation, not success-output fields.
T-060's resolved SEC-1 covered contextual preview, not managed commands.

## Acceptance

- Successful managed CLI JSON responses contain no literal unsafe terminal
  controls, including C1, bidi overrides/isolates, and Unicode line separators,
  from proposed/destination field values, profile metadata, or review text.
- JSON decoding retains the exact original strings. Plans, fingerprints,
  approvals, selected operations, and persisted journal semantics remain
  unchanged; presentation escaping must not double-escape the actual data.
- Regression tests invoke the CLI with asymmetric hostile and ordinary Unicode
  values, assert exact stdout bytes and decoded values, and exercise shared
  success-output paths (at least plan and approval), not only errors or tags.
- Existing selected-field approval, emission, observation, and native
  preservation checks remain green. Native presentation/release blockers
  T-060, T-062, and T-038 remain unchanged by this terminal-only fix.

## Verification Notes

Reproduce from a clean checkout with only a disposable synthetic collection:

```bash
evidence=$(mktemp -d)
uv run --with anki==26.9.3 python scripts/check-managed-anki.py \
  --output-dir "$evidence/native" > "$evidence/native.log"
uv run python - "$evidence/native/before-snapshot.json" \
  "$evidence/request.json" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
from latinitas_cards.destination_state import DestinationSnapshot, adopt

snapshot = DestinationSnapshot(Path(sys.argv[1]).read_text())
ownership = {
    n["identity"]: {
        "source_tags": [], "configured_tags": [],
        "keep_tags": n["tags"], "keep_fields": [],
    }
    for n in snapshot.payload["notes"]
}
baseline = adopt(snapshot, ownership, "synthetic reproduction adoption")
note = snapshot.payload["notes"][0]
hostile = "safe\u202eRLO\u202c\u009b31mred\u009b0m"
request = {
    "binding": baseline["binding"], "snapshot": snapshot.payload,
    "baseline": baseline, "effective_profile": {},
    "proposals": [{
        "identity": note["identity"],
        "fields": dict(note["fields"], Meaning=hostile),
    }],
}
Path(sys.argv[2]).write_text(json.dumps(request))
result = subprocess.run(
    ["uv", "run", "latinitas-cards", "managed", "plan", sys.argv[2]],
    capture_output=True,
)
print("exit", result.returncode)
print("raw_RLO", b"\xe2\x80\xae" in result.stdout)
print("raw_C1_CSI", b"\xc2\x9b" in result.stdout)
decoded = json.loads(result.stdout)
print("roundtrip", decoded["requests"][0]["fields"]["Meaning"] == hostile)
PY
```

Actual: `exit 0`, `raw_RLO True`, `raw_C1_CSI True`, `roundtrip True`.
Expected: successful valid JSON, both raw-control checks False, roundtrip True.
The original reproduction used independently authored moneo/navigo objects,
asymmetric native schedules, manual tags, and Personal Notes; this shorter
replay uses the committed disposable backend harness for convenient acquisition.

Round-1 baseline checks: `uv run ruff check` passed; `uv run mypy` reported
no issues in 87 source files; `uv run pytest -v` reported 837 passed.
The native backend harness passed its 16 scenarios on Anki 26.9.3. These checks
do not fix or cover the reproduced terminal defect and do not certify GUI or
AnkiMobile presentation.

## Implementation Notes

All five managed success-output paths now use ASCII JSON presentation, matching
the existing form-parsing convention. Canonical serialization, hashing,
persistence, domain values, operation selection, and approvals remain unchanged.

Strict RED: the Unicode review regression failed its literal-control exclusion
assertion with exit 0 and exact decoded plan data. GREEN: 32 managed plan/application
tests passed, including asymmetric proposed Meaning, retained destination Lemma,
profile metadata, and approval review with every C1 control, bidi formatting and
isolate controls, Unicode separators, and ordinary accented/Greek/CJK text.
Exact stdout bytes and decoded values are checked alongside independent canonical
SHA-256 parity and selected-field approval verification.

Initial and final validation: `uv run ruff check` passed; `uv run mypy` reported
no issues in 87 source files; `uv run pytest -v` reported 838 passed. The standalone
synthetic native replay reported exit 0, raw_RLO False, raw_C1_CSI False, and
roundtrip True. The unchanged Anki 26.9.3 backend harness passed all 16 scenarios;
this is not native GUI or AnkiMobile presentation certification.

The dedicated code-simplifier loaded its skill, made no changes, and passed all
18 plan tests. Separate General, Python, and Security code-reviewer lanes each
returned verbatim: "No concrete task-relevant findings." General loaded the ECC
code reviewer; Python loaded python-reviewer and python-patterns; Security loaded
security-reviewer, security-review, and common security guidance. Database/framework
lanes were omitted because no persistence or framework behavior changed. Fresh
candidate validation found no candidates or rejected IDs; fresh disposition
verification returned the same no-findings conclusion. No fixes, deferrals, or
follow-up tasks were required. T-060, T-062, and T-038 gates remain unchanged.

CLI example: raw U+202E/U+202C/U+009B review characters become visible JSON
`\u202e`/`\u202c`/`\u009b` escapes; decoding restores the original strings.
- 2026-10-05T07:50:05Z: verification pass
- 2026-10-05T07:50:16Z: Terminal-only JSON escaping delivered after strict RED/GREEN, independent General/Python/Security review, candidate/disposition verification, final ruff/mypy/838 pytest and 16 Anki backend scenarios; verification 2026-10-05T07:50:05Z. Native presentation/release blockers unchanged.
