---
id: T-229-design-reading-support-and-optional-passage
title: Design reading support and optional passage recipes
status: todo
priority: medium
spec_ref: specs/v0.5.5.md#reading-driven-exercises
dependencies: []
updated_at: "2026-11-20T04:09:52Z"
---

# T-229-design-reading-support-and-optional-passage Design reading support and optional passage recipes

## Description

Resolve the technical design for the approved reading-support, optional passage-search
and optional translation-comparison refinements in `specs/v0.5.5.md#reading-driven-exercises`.
The spec wording is applied; this task tracks the remaining design, not implementation
permission or activation of v0.5.5.

## Acceptance

- Specify support-versus-target representation and front-side answer-leakage checks.
- Specify passage-search answer spans and handling of multiple defensible examples.
- Specify translation contrasts, including two valid readings and labelled incorrect contrasts.
- Reuse shared assertion acceptance, provenance, schemas and identity contracts; identify
  any required adapters and approval prerequisites without creating a parallel engine.
- Map the added spec acceptance cases to design/evaluation scenarios before implementation.

## Verification Notes

- Spec refinement is documentation-only; validate Taskrail structure and whitespace.
- Technical design and its evaluation scenarios remain pending.

## Implementation Notes

The refinements were motivated by a university course programme and sample written exam.
No course-specific schema or preset subsystem is added. Existing saved selections and
vocabulary/construction filters remain the course-stage mechanism.
