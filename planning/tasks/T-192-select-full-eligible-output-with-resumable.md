---
id: T-192-select-full-eligible-output-with-resumable
title: Select full eligible output with resumable optional batches
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#grammar-exercise-library
dependencies:
    - T-191-extend-translation-first-forms-and-controlled
updated_at: "2026-11-02T23:10:11Z"
---

# T-192-select-full-eligible-output-with-resumable Select full eligible output with resumable optional batches

## Description

P50-3 deterministic vocabulary/pattern/recipe selection and suppression. Own batch
continuation under approved processing/output limits; existing phrase max_exercises
is not an implicit new output quota. Follow spec execution/validation requirements.

## Acceptance

- With sufficient processing limits, export all eligible selected material after
  duplicate/redundant suppression, including more than 20 cards. Pattern generation
  remains bounded, not an exhaustive Cartesian product.
- Optional card/batch limits change selection, not linguistic acceptance. Persist
  deterministic continuation, identities and effective options; later batches reuse
  valid checks without repeating provider work solely for batching.
- Respect approved limit precedence and filters. Changed linguistic inputs invalidate
  affected decisions; unchanged reruns and batch changes preserve valid results/keys.
- Expose suppression, accepted-unselected and deferred source work separately for
  reports, without inventing counts of undiscovered candidates.

## Verification Notes

Use a greater-than-20 eligible set, small batches/continuation, cross-batch duplicates,
processing-limit interruption and source invalidation. Compare public identities and
captured transport counts. Run the mandatory validation chain.

## Implementation Notes
