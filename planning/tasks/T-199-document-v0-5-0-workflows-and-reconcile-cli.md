---
id: T-199-document-v0-5-0-workflows-and-reconcile-cli
title: Document v0.5.0 workflows and reconcile CLI schemas
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-198-present-acceptance-evidence-and-morphology-without
updated_at: "2026-11-02T23:10:11Z"
---

# T-199-document-v0-5-0-workflows-and-reconcile-cli Document v0.5.0 workflows and reconcile CLI schemas

## Description

Own README/workflows, schema/agent CLI docs, runnable examples and native acceptance
kit for P50-2 through P50-6, not publication. Follow spec execution/validation rules.

## Acceptance

- Examples cover library-only/offline and authorized LLM stages, inspected/resumed runs,
  selection/batches/reports, optional review, curated extraction/approval, initial CSV
  import and supported managed update/ineligibility handoff.
- Reconcile command/profile/schema versions, diagnostic/outcome registry and machine
  examples with runtime. Explain processing/output limits, no implicit output quota,
  lower LLM-off coverage, resources and declared data scope.
- Explain upgrades/rollback, policy withdrawal/manual resolution, user ownership and
  opt-in templates. Omission is not retirement; configuration is not hosted permission.
- Prepare disposable Linux Desktop kit with expected results for both languages,
  light/dark/collapsed/expanded/keyboard, mixed statuses, import/update/withdrawal and
  scheduling/history checks. Exclude Mobile/existing scheduled-collection claims.
- Keep user-facing pending changes under Unreleased. State limitations/exclusions,
  not unmeasured accuracy or unexecuted native/model results.

## Verification Notes

Execute examples with disposable fixtures and controlled/no-network transports;
compare docs to actual registry/schema outputs. Native steps stay planned until
executed. Run documentation checks and mandatory chain after any code changes.

## Implementation Notes
