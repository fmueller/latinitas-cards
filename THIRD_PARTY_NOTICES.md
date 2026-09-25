# Third-Party Notices

Latinitas Cards (`latinitas-cards`) is licensed under GPL-3.0-or-later. This notice is
scoped to the exact `uv.lock` resolved on 2026-09-25: 93 `[[package]]` records, including
the project itself, or 92 third-party package records (91 normalized names because CPU and
PyPI CUDA builds are separate `torch` records). The complete version-by-version inventory
and evidence are in [the dependency-license audit](docs/license-compatibility-audit.md).

## Stable runtime dependencies

The default runtime resolves these direct packages from `uv.lock`:

- `pydantic` 2.13.5 — MIT
- `pandas` 2.3.3 — BSD-3-Clause
- `rich` 15.0.0 — MIT
- `click` 8.4.2 — BSD-3-Clause
- `typer` 0.27.2 — MIT
- `zstandard` 0.25.0 — BSD-3-Clause

Their transitive runtime packages and the development group are recorded individually in
the audit. Development-only packages are notices for source distributions and development
environments, not additional stable runtime promises.

## Optional experimental annotation dependencies

The `annotate` extra is CPU-only on non-macOS platforms and resolves `cltk` 2.5.1 with its
`stanza` extra, `stanza` 1.14.0, and PyTorch `2.14.0+cpu` from the PyTorch CPU index. The
macOS marker uses PyPI PyTorch `2.14.0`. The `annotate-gpu` extra resolves PyPI PyTorch
`2.14.0` and the following CUDA-related records; it conflicts with the CPU extra:

- `cuda-bindings` 13.4.1
- `cuda-pathfinder` 1.8.1
- `cuda-toolkit` 13.0.3.0
- `nvidia-cublas` 13.1.1.3
- `nvidia-cuda-cupti` 13.0.85
- `nvidia-cuda-nvrtc` 13.0.88
- `nvidia-cuda-runtime` 13.0.96
- `nvidia-cudnn-cu13` 9.24.0.43
- `nvidia-cufft` 12.0.0.61
- `nvidia-cufile` 1.15.1.6
- `nvidia-curand` 10.4.0.35
- `nvidia-cusolver` 12.0.4.66
- `nvidia-cusparse` 12.6.3.3
- `nvidia-cusparselt-cu13` 0.8.1
- `nvidia-nccl-cu13` 2.30.7
- `nvidia-nvjitlink` 13.4.52
- `nvidia-nvshmem-cu13` 3.4.5
- `nvidia-nvtx` 13.0.85
- `triton` 3.8.0

CLTK's upstream project identifies its license as MIT; Stanza is Apache-2.0; PyTorch
artifacts declare a composite Apache-2.0/BSD/BSL/MIT expression; and the NVIDIA runtime
artifacts declare NVIDIA proprietary license references in their wheel metadata, except
`nvidia-nvtx`, which declares Apache 2.0. `cuda-toolkit` does not declare a license in its
artifact core metadata and is recorded as unresolved in the audit rather than guessed.
Do not treat the GPU extra as GPL-compatible by default or as a stable v0.1.0 workflow
requirement.

## Distribution obligations and boundaries

Dependencies are resolved as separate packages and are not vendored into this repository.
If a future wheel, application bundle, container, or other artifact redistributes any of
these dependencies or their native libraries, include the corresponding license files and
notices, preserve copyright and attribution terms, and review the NVIDIA CUDA EULA and
proprietary package terms before distribution. In particular, do not replace
`LicenseRef-NVIDIA-Proprietary` with MIT/BSD language.

The optional CLTK/Stanza/PyTorch stack can download language models at runtime; those model
artifacts are outside this Python-package inventory and require their own license review.
The project does not bundle those models. See the audit for source links, exact lock
markers, package-manager distribution scope, and unresolved obligations.
