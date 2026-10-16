---
id: T-135-correct-obsolete-legacy-usfx-hardening
title: Correct obsolete legacy USFX hardening documentation
status: todo
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies: []
updated_at: "2026-10-16T20:45:55Z"
---

# T-135-correct-obsolete-legacy-usfx-hardening Correct obsolete legacy USFX hardening documentation

## Description

Round 5 at `ee6cdb97510169db89e193a02f13aacce4434fcb` confirms that
`docs/normalized-corpus.md` says legacy Vulgate hardening is pending and those
commands are not yet resource-bounded XML ingestors. The active USFX Normalization
contract and actual supported CLI disagree: legacy readers already reject DTDs
and enforce the XML size/depth limits.

Severity low: inaccurate user-facing documentation, not missing runtime guards.
Priority high follows the final-round finding-filing instruction, not a claim of
high runtime severity. Dedupe across existing tasks found implementation overlap
with completed T-112 and T-110, but no open correction of this stale statement.
Future T-128 memory overhead is explicitly distinct from these XML input limits.
There is no dependency or follow-up to T-124.

### Self-contained reproduction

From the checkout, use only disposable synthetic input:

```sh
root=$(mktemp -d)
printf 'Front,Back\nMater,mother\nabsens,absent\n' > "$root/words.csv"
printf '<usfx><book id="Z"><c id="2"/><v id="3"/>Mater amat.<ve/></book></usfx>' > "$root/valid.xml"
{ printf '<!DOCTYPE usfx>'; cat "$root/valid.xml"; } > "$root/dtd.xml"
.venv/bin/latinitas-cards preview --input "$root/words.csv" --usfx "$root/valid.xml"
.venv/bin/latinitas-cards preview --input "$root/words.csv" --usfx "$root/dtd.xml"
rm -rf "$root"
```

Actual: valid input exits 0; DTD input exits 3 and reports that DTD and entities
are not allowed. Expected runtime behavior is exactly that; expected docs must
describe implemented guards rather than pending work. Byte effect: none in the
checkout. Generate/cloze rejection controls preserve pre-existing output
sentinels; successful whole-cell controls match immutable citation-bearing
goldens. Fresh depth 64/65 and size 64 MiB/+1 controls independently demonstrate
the limits rather than inferring resource bounds solely from DTD rejection.

## Acceptance

- Documentation accurately describes implemented legacy DTD/entity, input-size
  and depth rejection while preserving legacy projection/output compatibility.
- Do not imply a total-memory bound, resolve T-128, change runtime behavior, or
  introduce a safety promise for legacy HTML output.

## Verification Notes

- Final-round bundle `round5-evidence.tar.gz`, `round5/usfx-package` contains
  exact final-control commands/assertions and the boundary/golden harness.
  `final-controls.py` uses independently generated inputs and literal expected
  output cells. DTD negative and valid positive inputs run through preview,
  generate and cloze. No confirmed runtime failure accompanies this finding.
- Compare `docs/normalized-corpus.md` lines 145–146 against active spec USFX
  Normalization, which explicitly says XML hardening applies to legacy commands.

## Implementation Notes

Unstarted. No documentation correction or product change made in Round 5.
