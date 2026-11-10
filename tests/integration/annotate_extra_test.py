import subprocess
import sys

import pandas as pd
import pytest

import latinitas_cards.cli as cli_mod


def _block_cltk_import(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes `import cltk` raise ImportError.
    monkeypatch.setitem(sys.modules, "cltk", None)


def test_cli_module_imports_without_loading_cltk() -> None:
    code = "import sys, latinitas_cards.cli; sys.exit(1 if 'cltk' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_cltk_is_installed_detects_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_cltk_import(monkeypatch)
    assert cli_mod._cltk_is_installed() is False


def test_annotate_without_cltk_names_annotate_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_cltk_import(monkeypatch)
    monkeypatch.setattr(cli_mod, "_cltk_is_installed", lambda: False)

    with pytest.raises(RuntimeError, match=r"uv sync --extra annotate"):
        cli_mod.annotate_with_cltk(pd.DataFrame({"form": ["amo"]}), form_column="form")


def test_annotate_with_broken_cltk_reports_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_cltk_import(monkeypatch)
    monkeypatch.setattr(cli_mod, "_cltk_is_installed", lambda: True)

    with pytest.raises(RuntimeError, match=r"CLTK is installed but failed to import") as excinfo:
        cli_mod.annotate_with_cltk(pd.DataFrame({"form": ["amo"]}), form_column="form")

    assert "--extra annotate" not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)
