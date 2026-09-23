---
id: T-019-annotate-cpu-gpu-extras
title: CPU-only annotate extra with experimental GPU variant
status: todo
priority: medium
spec_ref: specs/v0.1.0.md#existing-experimental-capabilities
dependencies:
    - T-018-python-3-13-cltk-2
updated_at: "2026-09-23T20:37:22Z"
---

# T-019-annotate-cpu-gpu-extras CPU-only annotate extra with experimental GPU variant

## Description

Make the `annotate` extra install CPU-only PyTorch by default and add a highly experimental
`annotate-gpu` extra for CUDA PyTorch (`specs/v0.1.0.md#existing-experimental-capabilities`).
The CUDA wheels are several GB and fail at import on machines without matching CUDA libraries.

## Acceptance

- `uv sync --extra annotate` resolves PyTorch from the PyTorch CPU index and installs no
  NVIDIA packages.
- `uv sync --extra annotate-gpu` resolves CUDA PyTorch; the two extras are declared as
  conflicting so they cannot be installed together.
- The default install and CI stay free of PyTorch.
- README documents both extras, labels `annotate-gpu` as highly experimental, and notes that
  the CPU/GPU index selection applies to uv workflows, not plain pip installs.
- ruff, mypy, and pytest pass; `annotate` produces analyses with the CPU extra.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
