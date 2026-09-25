# Dependency License and Distribution Audit

This project is licensed under **GPL-3.0-or-later** (`pyproject.toml` and `LICENSE`). This
is an engineering inventory and distribution-risk review, not legal advice or a blanket
compatibility opinion.

## Snapshot and method

- Version snapshot: `v0.1.0` candidate preparation on 2026-09-25.
- Dependency source of truth: `uv.lock`, which currently has 93 package records (`[[package]]`),
  including the editable `latinitas-cards` project record and 92 third-party package
  records. The third-party records have 91 normalized names because `torch` has separate
  CPU-index and PyPI records.
- Resolution scopes were checked with locked, non-installing exports for the default
  runtime (`main`), `dev`, `annotate`, and `annotate-gpu`. The CPU and GPU extras conflict
  in `pyproject.toml`; they are not intended to be installed together.
- License evidence was taken from the exact locked wheel core metadata where available,
  including PEP 658 metadata from PyPI and the PyTorch CPU index, and from the package
  license files/upstream repositories linked below. No multi-gigabyte CUDA or PyTorch
  wheel was installed for this audit.
- An artifact metadata value of `UNSTATED` was not silently converted into a permissive
  license. Where an upstream license file provides a clear source-backed answer it is
  named; otherwise the row remains unresolved.

## Compatibility conclusion

The stable default runtime is composed primarily of permissive licenses, but this audit
does **not** assert that every locked dependency is automatically GPLv3-compatible. The
optional annotation graph contains copyleft packages (`udapi` and `udtools`), a composite
PyTorch distribution with many third-party notices, and NVIDIA proprietary CUDA runtime
artifacts. `cuda-toolkit` does not declare a license in its exact wheel core metadata.
Those optional and distribution-specific obligations require review before bundling or
redistributing them.

The repository does not vendor these packages. A package-manager installation and a future
self-contained artifact are different distribution cases: a bundled wheel, application,
container, or native-library distribution must carry the relevant notices and license texts
and comply with the applicable terms.

## Exact locked inventory

`main`, `dev`, `annotate`, and `annotate-gpu` in the scope column mean that the package
appears in the corresponding locked export. `annotate` means the CPU-index build on
non-macOS where the lock markers select it; `annotate-gpu` means the PyPI CUDA build.

| Package | Version | Scope | License evidence from the locked artifact or upstream source |
| --- | --- | --- | --- |
| `annotated-doc` | `0.0.4` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `annotated-types` | `0.7.0` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE`, metadata expression unstated |
| `anyio` | `4.14.2` | annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `ast-serialize` | `0.11.2` | dev | MIT; wheel `LICENSE` and `crates/LICENSE` |
| `certifi` | `2026.1.4` | annotate, annotate-gpu | MPL-2.0; wheel `LICENSE` |
| `charset-normalizer` | `3.4.4` | annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `click` | `8.4.2` | main, dev, annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE.txt` |
| `cltk` | `2.5.1` | annotate, annotate-gpu | MIT; upstream [CLTK license](https://github.com/cltk/cltk/blob/master/LICENSE), wheel ships `LICENSE` |
| `colorama` | `0.4.6` | main, dev, annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE.txt`, metadata expression unstated |
| `coverage` | `7.16.1` | dev | Apache-2.0; wheel `LICENSE.txt` and `NOTICE.txt` |
| `cuda-bindings` | `13.4.1` | annotate-gpu | Apache-2.0; wheel `LICENSE` |
| `cuda-pathfinder` | `1.8.1` | annotate-gpu | Apache-2.0; wheel `LICENSE` |
| `cuda-toolkit` | `13.0.3.0` | annotate-gpu | **Unresolved:** exact wheel core metadata declares no license or license file |
| `emoji` | `2.15.0` | annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE.txt`, metadata expression unstated |
| `filelock` | `3.24.3` | annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `fsspec` | `2026.2.0` | annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE` |
| `h11` | `0.16.0` | annotate, annotate-gpu | MIT; wheel `LICENSE.txt` |
| `hf-xet` | `1.6.0` | annotate, annotate-gpu | Apache-2.0; wheel `LICENSE` |
| `httpcore` | `1.0.9` | annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE.md` |
| `httpx` | `0.28.1` | annotate, annotate-gpu | BSD-3-Clause; upstream project license, wheel metadata omits a license file |
| `huggingface-hub` | `1.31.0` | annotate, annotate-gpu | Apache-2.0; wheel `LICENSE` |
| `idna` | `3.19` | annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE.md` |
| `iniconfig` | `2.1.0` | dev | MIT; wheel `LICENSE` |
| `jinja2` | `3.1.6` | annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE.txt`, metadata expression unstated |
| `libcst` | `1.9.0` | dev | MIT with identified PSF-2.0 and Apache-2.0 portions; wheel license text |
| `librt` | `0.15.0` | dev | MIT; wheel `LICENSE` |
| `linkify-it-py` | `2.2.0` | dev | MIT; wheel `LICENSE` |
| `markdown-it-py` | `4.0.0` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE` and `LICENSE.markdown-it`, metadata expression unstated |
| `markupsafe` | `3.0.3` | annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE.txt` |
| `mdit-py-plugins` | `0.6.1` | dev | MIT; wheel `LICENSE`, metadata expression unstated |
| `mdurl` | `0.1.2` | main, dev, annotate, annotate-gpu | MIT; upstream project license, artifact metadata omits a license field |
| `mpmath` | `1.3.0` | annotate, annotate-gpu | BSD; wheel `LICENSE` |
| `mutmut` | `3.8.0` | dev | BSD-3-Clause; wheel `LICENSE` |
| `mypy` | `2.3.1` | dev | MIT; wheel `LICENSE` plus typeshed license notices |
| `mypy-extensions` | `1.1.0` | dev | MIT; wheel `LICENSE` |
| `networkx` | `3.6.1` | annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE.txt` |
| `numpy` | `2.5.3` | main, dev, annotate, annotate-gpu | BSD-3-Clause, 0BSD, MIT, Zlib, CC0-1.0; wheel license files |
| `nvidia-cublas` | `13.1.1.3` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-cuda-cupti` | `13.0.85` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-cuda-nvrtc` | `13.0.88` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-cuda-runtime` | `13.0.96` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-cudnn-cu13` | `9.24.0.43` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-cufft` | `12.0.0.61` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-cufile` | `1.15.1.6` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-curand` | `10.4.0.35` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-cusolver` | `12.0.4.66` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-cusparse` | `12.6.3.3` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-cusparselt-cu13` | `0.8.1` | annotate-gpu | NVIDIA Proprietary Software; exact wheel metadata, upstream [cuSPARSELt](https://developer.nvidia.com/cusparselt) |
| `nvidia-nccl-cu13` | `2.30.7` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-nvjitlink` | `13.4.52` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-nvshmem-cu13` | `3.4.5` | annotate-gpu | `LicenseRef-NVIDIA-Proprietary`; wheel `License.txt` |
| `nvidia-nvtx` | `13.0.85` | annotate-gpu | Apache-2.0; wheel `License.txt` |
| `packaging` | `25.0` | dev, annotate, annotate-gpu | Apache-2.0 OR BSD-2-Clause; wheel `LICENSE`, `LICENSE.APACHE`, and `LICENSE.BSD` |
| `pandas` | `2.3.3` | main, dev, annotate, annotate-gpu | BSD-3-Clause; pandas upstream license, artifact expression unstated |
| `pandas-stubs` | `3.0.0.260204` | dev | BSD-3-Clause; wheel `LICENSE` |
| `pathspec` | `1.1.1` | dev | MPL-2.0; wheel `LICENSE`, metadata expression unstated |
| `platformdirs` | `4.11.8` | dev, annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `pluggy` | `1.6.0` | dev | MIT; wheel `LICENSE` |
| `protobuf` | `6.33.5` | annotate, annotate-gpu | 3-Clause BSD; upstream package metadata |
| `pydantic` | `2.13.5` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `pydantic-core` | `2.46.5` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `pygments` | `2.21.0` | main, dev, annotate, annotate-gpu | BSD-2-Clause; wheel `AUTHORS` and `LICENSE` |
| `pytest` | `9.1.1` | dev | MIT; wheel `LICENSE` |
| `python-dateutil` | `2.9.0.post0` | main, dev, annotate, annotate-gpu | Dual Apache-2.0 or BSD-3-Clause; wheel `LICENSE` |
| `python-dotenv` | `1.2.3` | annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE` |
| `pytz` | `2025.2` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE.txt` |
| `pyyaml` | `6.0.3` | dev, annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `pyyaml-ft` | `8.0.0` | dev | MIT; wheel `LICENSE` |
| `regex` | `2026.2.19` | annotate, annotate-gpu | Apache-2.0 AND CNRI-Python; wheel `LICENSE.txt` |
| `requests` | `2.33.0` | annotate, annotate-gpu | Apache-2.0; wheel `LICENSE` and `NOTICE` |
| `rich` | `15.0.0` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `ruff` | `0.16.8` | dev | MIT; wheel `LICENSE` |
| `setproctitle` | `1.3.7` | dev | BSD-3-Clause; wheel `LICENSE` |
| `setuptools` | `84.0.0` | annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `shellingham` | `1.5.4` | main, dev, annotate, annotate-gpu | ISC; wheel `LICENSE` |
| `six` | `1.17.0` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `stanza` | `1.14.0` | annotate, annotate-gpu | Apache-2.0; wheel `LICENSE`, upstream [Stanza license](https://github.com/stanfordnlp/stanza/blob/main/LICENSE) |
| `sympy` | `1.14.0` | annotate, annotate-gpu | BSD; wheel `LICENSE` |
| `termcolor` | `3.3.0` | annotate, annotate-gpu | MIT; wheel `COPYING.txt` |
| `textual` | `8.2.8` | dev | MIT; wheel `LICENSE` |
| `torch` | `2.14.0` | annotate-gpu and macOS annotate | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT; wheel license files, upstream [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| `torch` | `2.14.0+cpu` | non-macOS annotate | Same PyTorch composite expression; CPU-index wheel license files |
| `tqdm` | `4.67.3` | annotate, annotate-gpu | MPL-2.0 AND MIT; wheel `LICENCE` |
| `triton` | `3.8.0` | annotate-gpu | MIT; wheel `LICENSE` |
| `typer` | `0.27.2` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `typing-extensions` | `4.15.0` | main, dev, annotate, annotate-gpu | PSF-2.0; wheel `LICENSE` |
| `typing-inspection` | `0.4.2` | main, dev, annotate, annotate-gpu | MIT; wheel `LICENSE` |
| `tzdata` | `2025.2` | main, dev, annotate, annotate-gpu | Apache-2.0; wheel `LICENSE` and `licenses/LICENSE_APACHE` |
| `udapi` | `0.5.2` | annotate, annotate-gpu | GPL-3.0-or-later; wheel `LICENSE` |
| `udtools` | `0.2.8` | annotate, annotate-gpu | GPL-2.0-or-later; wheel `LICENSE.txt` |
| `urllib3` | `2.7.0` | annotate, annotate-gpu | MIT; wheel `LICENSE.txt` |
| `zstandard` | `0.25.0` | main, dev, annotate, annotate-gpu | BSD-3-Clause; wheel `LICENSE` |

## Source links and obligations

- [CLTK license](https://github.com/cltk/cltk/blob/master/LICENSE) — MIT.
- [Stanza license](https://github.com/stanfordnlp/stanza/blob/main/LICENSE) — Apache-2.0.
- [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) and [PyTorch
  package page](https://pypi.org/project/torch/2.14.0/) — composite expression and bundled
  third-party license files.
- [NVIDIA CUDA license/EULA materials](https://docs.nvidia.com/cuda/eula/index.html) and
  the exact NVIDIA wheel `License.txt` files — proprietary terms apply to the runtime
  artifacts identified above; the PyPI metadata is not treated as MIT/BSD.
- The remaining rows cite the exact wheel license files or upstream project metadata in the
  table. Re-run the locked exports and refresh this audit whenever `uv.lock` changes.

MIT/BSD/ISC/PSF/Apache/MPL notices still need to be preserved in redistributed artifacts;
MPL and GPL-family packages carry their own source/notice obligations. The NVIDIA entries
require separate review of the EULA and redistribution terms. The unresolved `cuda-toolkit`
metadata is a limitation of this audit and must not be filled with an assumption.
