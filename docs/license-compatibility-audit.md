# License Compatibility Audit (GPL-3.0-or-later)

This project is licensed under **GPL-3.0-or-later** (`pyproject.toml` + `LICENSE`).

## Scope and method

- Dependency source of truth: `poetry.lock` (all resolved packages in `main` and `dev` groups).
- License metadata source: PyPI JSON metadata for each locked package version, plus installed wheel metadata when PyPI metadata was incomplete.
- Compatibility baseline: whether each dependency license is generally compatible with GPLv3 when distributed together in a Python application.

> Note: This is an engineering compliance review, not legal advice.

## Compatibility result

All currently locked dependencies are under licenses that are **GPLv3-compatible** (mostly MIT/BSD/Apache-2.0/ISC/PSF-2.0/MPL-2.0/Unlicense).

## Locked dependency inventory and license review

| Package | Version | Group(s) | License | GPLv3-compatible? | Notes |
|---|---:|---|---|---|---|
| annotated-types | 0.7.0 | main | MIT | Yes | Permissive. |
| cfgv | 3.4.0 | dev | MIT | Yes | Permissive. |
| click | 8.3.0 | main | BSD-3-Clause | Yes | Permissive. |
| colorama | 0.4.6 | main,dev | BSD | Yes | Permissive BSD family. |
| distlib | 0.4.0 | dev | PSF-2.0 | Yes | GPL-compatible free software license. |
| exceptiongroup | 1.3.1 | dev | MIT | Yes | Permissive. |
| filelock | 3.19.1 | dev | Unlicense | Yes | GPL-compatible public-domain style dedication. |
| identify | 2.6.14 | dev | MIT | Yes | License from package metadata field. |
| iniconfig | 2.1.0 | dev | MIT | Yes | Permissive. |
| librt | 0.7.8 | dev | MIT | Yes | Permissive. |
| markdown-it-py | 4.0.0 | main | MIT | Yes | Permissive. |
| mdurl | 0.1.2 | main | MIT | Yes | Permissive. |
| mypy | 1.19.1 | dev | MIT | Yes | Permissive. |
| mypy-extensions | 1.1.0 | dev | MIT | Yes | License confirmed from installed LICENSE file. |
| nodeenv | 1.9.1 | dev | BSD | Yes | Permissive BSD family. |
| numpy | 2.2.6 | main | BSD | Yes | Permissive BSD family. |
| numpy | 2.3.3 | main,dev | BSD | Yes | Alternate locked version for other env markers. |
| packaging | 25.0 | dev | Apache-2.0 or BSD | Yes | Apache-2.0 is GPLv3-compatible. |
| pandas | 2.3.3 | main | BSD | Yes | Permissive BSD family. |
| pandas-stubs | 3.0.0.260204 | dev | BSD | Yes | Dev-only stubs. |
| pathspec | 0.12.1 | dev | MPL-2.0 | Yes | MPL-2.0 is GPLv3-compatible (file-level copyleft). |
| platformdirs | 4.4.0 | dev | MIT | Yes | Permissive. |
| pluggy | 1.6.0 | dev | MIT | Yes | Permissive. |
| pre-commit | 4.5.1 | dev | MIT | Yes | Permissive. |
| pydantic | 2.12.5 | main | MIT | Yes | Permissive. |
| pydantic-core | 2.41.5 | main | MIT | Yes | Permissive. |
| pygments | 2.19.2 | main,dev | BSD | Yes | Permissive BSD family. |
| pytest | 9.0.2 | dev | MIT | Yes | Dev-only. |
| python-dateutil | 2.9.0.post0 | main | Apache-2.0 or BSD | Yes | Apache-2.0 is GPLv3-compatible. |
| pytz | 2025.2 | main | MIT | Yes | Permissive. |
| pyyaml | 6.0.3 | dev | MIT | Yes | Dev-only here. |
| rich | 13.9.4 | main | MIT | Yes | Permissive. |
| ruff | 0.15.2 | dev | MIT | Yes | Dev-only. |
| shellingham | 1.5.4 | main | ISC | Yes | Permissive. |
| six | 1.17.0 | main | MIT | Yes | Permissive. |
| tomli | 2.4.0 | dev | MIT | Yes | Dev-only for older Python. |
| typer | 0.19.2 | main | MIT | Yes | Permissive. |
| typing-extensions | 4.15.0 | main,dev | PSF-2.0 | Yes | GPL-compatible free software license. |
| typing-inspection | 0.4.2 | main | MIT | Yes | Permissive. |
| tzdata | 2025.2 | main | Apache-2.0 | Yes | GPLv3-compatible. |
| virtualenv | 20.34.0 | dev | MIT | Yes | Dev-only. |
| zstandard | 0.25.0 | main | BSD-3-Clause | Yes | Permissive. |

## Are you currently respecting dependency license terms?

Short answer: **mostly yes**, with one release-hardening improvement recommended.

### What you already do correctly

- Project license is clearly declared as GPL-3.0-or-later in package metadata and includes a GPL text file.
- Dependencies are pulled as separate Python packages (not vendored into this repository), which usually means each package carries its own license metadata in installation artifacts.

### What to add before release (recommended)

1. Keep a **third-party notices** file in the repo/release artifact listing runtime dependencies and their licenses.
2. If you distribute as a **self-contained binary/container image** with bundled site-packages, include third-party license texts/notices in that artifact (or a documented link packaged alongside it).
3. Re-run this audit whenever `poetry.lock` changes.

## Practical obligations by license family in this dependency set

- **MIT/BSD/ISC/PSF/Unlicense**: preserve copyright + license notice in redistributions.
- **Apache-2.0**: preserve license text and NOTICE handling where applicable.
- **MPL-2.0** (`pathspec`, dev dependency): file-level copyleft; if modified MPL-covered files are redistributed, source of those files must remain available under MPL.

In your current setup (no vendored third-party source in-repo), these obligations are typically satisfied by standard package-manager distribution, but adding explicit notices improves release readiness and auditability.
