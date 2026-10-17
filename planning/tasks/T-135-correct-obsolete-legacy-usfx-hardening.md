---
id: T-135-correct-obsolete-legacy-usfx-hardening
title: Correct obsolete legacy USFX hardening documentation
status: in_progress
priority: high
spec_ref: specs/v0.3.0.md#usfx-normalization
dependencies: []
updated_at: "2026-10-17T08:24:59Z"
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

Historical Round 5 state: unstarted; no documentation correction or product
change was made in that round. The following records the subsequent T-135 cycle.

### Workflow-v3 understanding and scope

- Accepted cumulative base: `ed74d5bc9bddb6d2e91d111aa82f0fc737b80234`.
  `git fetch --quiet origin main; git rev-parse HEAD origin/main` confirmed both
  refs at that base before work; no source-overseer local-main transfer.
- `sha256sum specs/v0.3.0.md` confirmed the pinned digest
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
- `mise exec -- taskrail validate` returned `state valid`;
  `mise exec -- taskrail next --json` selected this high/todo task;
  `mise exec -- taskrail status` reported zero active before start. Rechecked
  pin, selector and ownership before writes, then ran
  `mise exec -- lefthook install` and
  `mise exec -- taskrail start T-135-correct-obsolete-legacy-usfx-hardening`.
- Source ownership: legacy command routes in `cli.py` call
  `usfx_adapter.legacy_projection`; its `_EventReader(legacy=True)` shares the
  stat/stream byte count, depth count and expat rejection handlers. The active
  USFX Normalization contract explicitly includes these legacy guards.
- Only the obsolete paragraph in `docs/normalized-corpus.md` changes product
  documentation. Accepted-input projection/output compatibility is retained;
  DTD/entity declarations and external entities, input over 64 MiB and element
  depth over 64 are rejected. The text explicitly disclaims total-memory bounds.
  No HTML-safety claim, runtime change, T-128 resolution or broader cleanup.
- Documentation-only: no artificial prose TDD or exact-wording tests. Existing
  behavior controls establish the claims; no behavior test needed adding.

### Initial controls and simplification

- `uv run pytest -v tests/unit/usfx_adapter_test.py tests/unit/legacy_goldens_test.py`:
  `37 passed in 27.06s`. Existing evidence includes DTD/entity/depth/size rejection
  across legacy routes, accepted depth 64 and exactly 64 MiB, differential
  projection and immutable whole-output legacy goldens.
- Dedicated Task loaded `code-simplifier`, shortened the paragraph and ran
  focused tests: `37 passed`. Accepted its sentence structure; rejected its
  broader "references" wording because predefined references are not rejected,
  restoring precise "external entities" before independent review.
- Post-simplification command:
  `git diff --check; uv run pytest -q tests/unit/usfx_adapter_test.py tests/unit/legacy_goldens_test.py`:
  clean diff and `37 passed in 22.73s`.
- Standalone real CLI controls used the exact command below. The initial attempt
  incorrectly reused Front/Back for cloze and failed with
  `KeyError: "Form column 'form' not found. Available: ['Front', 'Back']"`.
  Corrected the disposable fixture, not runtime code; rerun passed all six cases.
  Positive content is a smoke assertion, not a substitute for full-output goldens.

```sh
uv run python - <<'PY'
import subprocess, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as root:
 p=Path(root); (p/'words.csv').write_text('Front,Back\nMater,mother\nabsens,absent\n'); (p/'forms.csv').write_text('form\nMater\n')
 xml='<usfx><book id="Z"><c id="2"/><v id="3"/>Mater amat.<ve/></book></usfx>'
 for kind,prefix in [('valid',''),('dtd','<!DOCTYPE usfx>')]:
  corpus=p/f'{kind}.xml'; corpus.write_text(prefix+xml)
  for route in ['preview','generate','cloze']:
   out=p/f'{route}.csv'; out.write_text('SENTINEL\n')
   args=['.venv/bin/latinitas-cards',route,'--input',str(p/('forms.csv' if route=='cloze' else 'words.csv')),'--corpus' if route=='cloze' else '--usfx',str(corpus)]
   if route!='preview': args+=['--output',str(out)]
   result=subprocess.run(args,capture_output=True,text=True)
   assert result.returncode==(0 if kind=='valid' else 3),(args,result.stdout,result.stderr)
   if kind=='dtd':
    assert 'unsafe_xml' in result.stderr,(result.stdout,result.stderr)
    assert out.read_text()=='SENTINEL\n'
   elif route!='preview': assert 'Mater' in out.read_text(),out.read_text()
   else: assert 'Mater' in result.stdout,result.stdout
   print(f'{route} {kind}: exit {result.returncode}; '+('unsafe_xml; output sentinel preserved' if kind=='dtd' else 'Mater positive content verified'))
PY
```

Verbatim successful control output:

```text
preview valid: exit 0; Mater positive content verified
generate valid: exit 0; Mater positive content verified
cloze valid: exit 0; Mater positive content verified
preview dtd: exit 3; unsafe_xml; output sentinel preserved
generate dtd: exit 3; unsafe_xml; output sentinel preserved
cloze dtd: exit 3; unsafe_xml; output sentinel preserved
```

### Independent reviews and dispositions

- Separate parallel read-only General and Security Tasks loaded `code-reviewer`
  and `references/reviewer-index.md`. General loaded ECC `agents/code-reviewer.md`
  (no mapped companions). Security loaded ECC `agents/security-reviewer.md`,
  `skills/security-review/REFERENCE.md`, its cloud-infrastructure companion and
  common security/code-review rules. Security selected because input-hardening
  claims can overpromise; General covers acceptance, scope and evidence.
- Omitted Python/framework lanes: no Python or framework code changes. Omitted
  Database and other domain lanes: no schema, persistence, concurrency or other
  specialist invariant changed. One specialist is within the default budget.
- General verbatim conclusion: `No concrete task-relevant findings.`
- Security verbatim conclusion: `No concrete task-relevant findings.`
- Fresh read-only candidate-validation Task loaded `code-reviewer` and inspected
  actual diff, source, spec and tests. Verbatim: `Validation conclusion: zero
  candidates.` and `No contradiction was evidenced in the actual diff or
  surrounding source, spec, and tests.` No candidate IDs to reject/deduplicate;
  no findings to fix or defer. It could not invoke bare taskrail; parent uses
  `mise exec -- taskrail` and independently confirms lifecycle state.
- These are workflow-v3 task-local reviews, not adversarial rounds 6–8. Five
  prior NOT CLEAN rounds remain consumed; no round counter reset or worker round.

### Final checks

- Executed sequentially:
  `uv run ruff check && uv run mypy && uv run pytest -v`, then `mise run check`.
  First chain: `All checks passed!`; `Success: no issues found in 169 source
  files`; `2593 passed, 5 skipped, 1 warning in 199.27s (0:03:19)`.
- `mise run check` exited 0: ruff passed; `391 files already formatted`; mypy
  passed for 169 source files; `2593 passed, 5 skipped, 1 warning in 193.08s
  (0:03:13)`; commit/push message, author, mutation results/configuration,
  distribution inspection (54 cases) and publish-guard synthetic checks passed.
  These are local checks, not publication or a manually triggered workflow.
- Both suites report the existing Pydantic serialization warning in
  `corpus_io_test.py::test_writer_rejects_float_without_replacing`; no changed
  runtime code or new test was needed. No gate failure or chain restart.
- `git diff --check` passed. Spec digest unchanged. No source, test, fixture,
  spec, priority, dependency, release or environment settings were changed.
- Fresh read-only final disposition-verification Task loaded `code-reviewer`,
  inspected the actual diff, audit, spec, source and tests, and ran Taskrail
  validation plus diff checks. Verbatim: `No concrete task-relevant findings.`;
  `Validation conclusion: zero candidates.`; `No contradiction was evidenced in
  the actual diff or surrounding source, spec, and tests.`; `No finding IDs,
  fixes, or deferrals.` It relied on the parent's full-suite results, not a new
  suite run. One task-local review/disposition cycle; no outstanding findings.
