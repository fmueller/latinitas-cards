---
id: T-239-optimize-evidence-validation-and-integration-ci-cost
title: Optimize evidence validation and integration CI cost
status: completed
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-238-implement-isolated-research-classification-and-fidelity
updated_at: "2026-11-21T20:45:46Z"
---

# T-239-optimize-evidence-validation-and-integration-ci-cost Optimize evidence validation and integration CI cost

## Description

Reduce the cost of the v0.5.0 shared-evidence integration journeys and CI without
weakening acceptance, recursive proof validation or same-process tamper detection.
This is the maintainer-requested performance follow-up before remaining release
gates, under specs/v0.5.0.md#shared-evidence-and-automatic-acceptance.

Accepted base is b28d9394475e5634ddef7cc66d0033f6b383affa. Its final local suite
passed 4076 tests, skipped 7, in 4613.76 seconds. Exact-head CI integration used
5965.01 seconds on Python 3.13 and 4147.18 seconds on Python 3.14. Those are
historical observations, not controlled causal estimates for T238's 16 new cases.
The expensive model journeys use local synthetic HTTP, not hosted LLM inference.

A profiled accepted T237 German verb-noun comparison/reuse journey passed in
302.08 seconds under instrumentation: 3849 Bennett-section loads/extractions,
40802 HTML-to-text calls, 416 lexicon constructions and 416 implementation-digest
calculations. Nested costs overlap; do not sum them or treat instrumented time as
an uninstrumented baseline. Measure on the actual accepted T238 base.

Use engineering judgment and diminishing returns. Consider reducing repeated
verified reference parsing/proof work, trimming redundant test cases to useful
happy paths and edges, and bounded parallel CI execution. No exhaustive tuning
or benchmark framework is requested. Any test removal follows the installed
mutation workflow; its missing adapter needs separate setup approval. Review
redundancies and proceed with runtime/CI improvements while that approval is
pending rather than silently installing setup or weakening coverage.

Follow-up derived from T-238-implement-isolated-research-classification-and-fidelity's verification or discovery.

## Acceptance

- Record matched uninstrumented before/after measurements for representative
  expensive public comparison/reuse and research classify/save/read journeys.
  Report commands, exact source/environment identities, wall times, test counts
  added/removed and limits. Compare final exact-head CI with accepted-base CI.
- Deliver worthwhile reductions without changing linguistic results, complete
  assertion inventories, per-slot reasons, acceptance authority, policy state,
  or German v0.4 characterization goldens. Unsupported and stale evidence still
  fail closed. Do not cache by path alone or trust a stored accepted flag.
- Verify evidence mutation, deletion, replacement and restoration within one
  process, including changed reference/source/output/request commitments and
  implementation/registry identity. Cover proof-edge/index corruption,
  foreign-record/partial-proof denial and exact restored positive reuse at
  appropriate public boundaries. Reuse existing tests where possible.
- Review costly duplicate parametrizations and repeated higher-level checks.
  Retain distinct semantic, integrity, denominator and authority failure modes;
  document why any proposed reduction preserves them. Execute actual pruning
  only through an approved, calibrated mutation adapter and deletion probes.
- Evaluate CI parallelism against shared files, reference mutations, ports,
  process state and run-store isolation. Implement a simple bounded approach
  when evidence supports it; otherwise explain the measured tradeoff. Preserve
  Python 3.13/3.14 coverage, stage failure gating, aggregate checks and execution
  of every retained test. No skips, retries or relaxed checks to manufacture green.
- Complete TDD where applicable, simplification, independent conventional review,
  disposition verification and independent integrated acceptance on the frozen
  candidate. Run final Ruff, mypy and full pytest in repository-required order,
  then supported Taskrail verify/complete. Commit/push directly to main under
  maintainer identity with installed hooks and no attribution trailers. Exact-head
  Build and CodeQL success are required before ownership handback.
- No hosted inference, reference acquisition, study/protected-answer access,
  policy enablement, native Anki qualification, publication or deployment belongs
  to this task. Do not select a successor; parent owns the remaining release gates.

## Verification Notes

- Accepted T238 base: b28d9394475e5634ddef7cc66d0033f6b383affa. Final executable
  diff SHA256: 5488d0a7e170372f268108e817d40b1cb30ffeda3f00fd62e483df996e19314d.
- Final ordered Ruff -> mypy -> full pytest completed 2026-11-21T20:01:01Z:
  Ruff clean, mypy clean across 249 files, 4077 passed / 7 unchanged skips /
  1 existing serialization warning in 4103.14 pytest seconds (4104.59 wall,
  4072.45 CPU seconds). Formatting and mutation configuration/result guards pass.
- Independent frozen-candidate acceptance completed 2026-11-21T20:40:34Z:
  two-worker loadfile integration run passed 2902 / skipped 7 in 2181.31 pytest
  seconds (2181.80 wall, 3846.79 CPU seconds). All 2909 collection, serial,
  parallel and JUnit nodes/results match, with zero omissions or duplicates.
  Supplemental actual-store checks passed two current positives, two foreign
  research-read denials and two foreign production-export denials, without writes
  or model calls. Executable identity remained unchanged.
- Matched final Python 3.14.7 / pytest 9.2.0 / Pydantic 2.14.1 environment,
  accepted archived source/tests versus candidate, command pytest -q --durations=2
  selecting German verb-noun public comparison/reuse and research classify/read:
  two passed each, 351.33 -> 242.95 wall seconds (30.85% reduction),
  350.24 -> 241.95 CPU seconds. This is not whole-suite causal attribution.
  Final serial whole-suite and parallel integration timings are different scopes.
- Independent general, integrity, CI/cost and claims review, scoped simplifier,
  disposition verification and challenges A/B/C support local readiness. Unicode
  normalization and restored public analyze reuse have distinguishing red/green
  probes. The missing Uncomment 3.7.0 hygiene tool is a disclosed limit.
- Exact-head hosted Build/CodeQL and Python 3.13 qualification remain delivery
  gates at this local verification checkpoint; handback waits for terminal CI.

## Implementation Notes

- Cache only immutable HTML results by exact text and strict-record syntax bytes
  by exact input/field set. Keep original canonical normalization and fresh model
  validation; current file reads/digests, proof reconstruction and authority stay
  outside reuse. Entry-count bounds are not byte-memory limits.
- Two CI pytest processes use loadfile scheduling after an isolation audit of
  temporary package/reference copies, port-zero servers, stores and process state.
  Preserve both Python versions, every retained test and all stage/aggregate gates.
- Add one tiny public Unicode unit regression; extend existing strict-record,
  reference restoration and representative saved-unscorable cases. Remove zero
  cases or parametrizations and add no skips. Distinct semantic/language matrices
  were retained because narrow reader mutation evidence does not prove redundancy.
- Maintainer-approved scoped mutmut 3.8.1 setup is documented in mutation/setup.md;
  3.8.0 failed pytest 9.2 statistics collection. Cold calibration measured 17 reader
  mutants: initial 12 killed / 5 survived in 560.13 seconds, post-extension 16 killed /
  1 diagnostic-wrapper equivalent in 424.29 seconds, no timeouts or regressions.
  The unchanged population excludes 278 unmeasured module mutants. No deletion
  probe was justified or run. Calibration predates the shared-reader Unicode
  correction and does not qualify the changed cache modules. Broad diff mutation
  is outside this bounded calibration/budget; no broad adequacy is claimed.
- No hosted inference, new reference requests, protected answers, enabled policies,
  native Anki qualification, publication or deployment. Charged campaign totals
  remain four development attempts and three reference requests. No successor owned.
- 2026-11-21T20:45:37Z: verification pass
- 2026-11-21T20:45:46Z: Implemented and independently accepted locally on frozen executable diff 5488d0a7e170372f268108e817d40b1cb30ffeda3f00fd62e483df996e19314d. Verification passed 2026-11-21T20:45:37Z after final ordered chain, independent full two-worker integrations, cumulative review and disposition verification. Ownership continues through direct-main delivery and exact-head Build/CodeQL; no successor selected.
