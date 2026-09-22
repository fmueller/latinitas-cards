#!/usr/bin/env bash
# Run every discovered mutant and fail if execution or result collection fails.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
uv run mutmut run
uv run mutmut results --all true | bash "$script_dir/check-mutation-results.sh"
