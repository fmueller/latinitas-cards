---
id: T-193-orchestrate-inspectable-stages-with-bounded-resume
title: Orchestrate inspectable stages with bounded resume
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-192-select-full-eligible-output-with-resumable
updated_at: "2026-11-02T23:10:11Z"
---

# T-193-orchestrate-inspectable-stages-with-bounded-resume Orchestrate inspectable stages with bounded resume

## Description

P50-4 focused command/profile for ingest, analysis, proposal, blind check, decision,
selection and file export. Own orchestration/resume, not adapter internals; do not
grow cli.py. Follow approved contracts and spec execution/validation requirements.

## Acceptance

- One approved-name entry point emits files plus report/stage data; each stage remains
  separately usable/inspectable. File creation never applies changes to Anki.
  LLM-off supports library/rule/human inputs without promising identical coverage.
- Validate persisted completed stages. Interruption/resume does not duplicate accepted
  artifacts or repeat valid calls. Unknown/failed calls remain unaccepted until reconciled,
  with finite limits and cancellation.
- Changed source/checker/configuration invalidates affected reuse; valid unrelated work
  survives. Config errors, outages, malformed results and exhausted limits are explicit
  outcomes without fallback.
- Hosted permission, data-scope display and locality survive resume/profile changes;
  permission for one endpoint/scope cannot silently authorize a new one.

## Verification Notes

Drive CLI interruption after success and during an unknown call, resume and inspect
artifacts/requests. Change one binding; compare reuse. Check offline/no-network,
cancellation and permission controls. Run the mandatory chain.

## Implementation Notes
