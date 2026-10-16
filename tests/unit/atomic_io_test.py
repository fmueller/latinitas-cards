"""Writer safety and recoverable multi-output staging contracts."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from latinitas_cards.atomic_io import (
    atomic_replace,
    reject_aliases,
    replace_staged,
    stage_bytes,
    write_artifact,
    writer_lock,
    writer_locks,
)
from latinitas_cards.cli_contract import Code, ContractError


@pytest.mark.parametrize("form", ["direct", "symlink", "hardlink"])
def test_reject_aliases_without_writes(tmp_path: Path, form: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"original")
    output = tmp_path / "output"
    if form == "direct":
        output = source
    elif form == "symlink":
        output.symlink_to(source)
    else:
        os.link(source, output)
    with pytest.raises(ContractError) as error:
        write_artifact(output, b"new", [source])
    assert error.value.diagnostic.code == Code.ALIAS
    assert source.read_bytes() == output.read_bytes() == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == (["source"] if form == "direct" else ["output", "source"])


def test_artifact_noop_and_refusal(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    write_artifact(output, b"one", [])
    before = output.stat()
    write_artifact(output, b"one", [])
    assert output.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ContractError) as error:
        write_artifact(output, b"two", [])
    assert error.value.diagnostic.code == Code.EXISTS
    assert output.read_bytes() == b"one"


def test_busy_and_multilock_cleanup(tmp_path: Path) -> None:
    first, later = tmp_path / "a", tmp_path / "z"
    later.write_bytes(b"old")
    with writer_lock(later):
        with pytest.raises(ContractError) as error:
            write_artifact(later, b"new", [])
        assert error.value.diagnostic.code == Code.BUSY
        with pytest.raises(ContractError) as error, writer_locks([later, first]):
            pytest.fail("contended multi-lock entered")
        assert error.value.diagnostic.code == Code.BUSY
        with writer_lock(first):
            pass
    with writer_locks([later, first]):
        for path in (first, later):
            with pytest.raises(ContractError), writer_lock(path):
                pytest.fail("lock not held")
    assert later.read_bytes() == b"old"
    assert (tmp_path / "a.lock").exists() and (tmp_path / "z.lock").exists()


def test_lock_order_is_lexical_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    acquired: list[str] = []
    original = fcntl.flock

    def observe(fd: int, operation: int) -> None:
        if operation & fcntl.LOCK_EX:
            acquired.extend(
                path.name for path in tmp_path.glob("*.lock") if os.path.samestat(os.fstat(fd), path.stat())
            )
        original(fd, operation)

    monkeypatch.setattr(fcntl, "flock", observe)
    with writer_locks([tmp_path / "z", tmp_path / "a"]):
        pass
    assert acquired == ["a.lock", "z.lock"]


def test_failed_replace_preserves_old_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "artifact"
    output.write_bytes(b"old")

    def fail(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(ContractError) as error:
        atomic_replace(output, b"new")
    assert error.value.diagnostic.code == Code.UNREADABLE
    assert output.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [output]


def test_stage_both_before_replace_and_fsync_parents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import stat

    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"old a")
    second.write_bytes(b"old b")
    synced = []
    original = os.fsync

    def observe(fd: int) -> None:
        synced.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        original(fd)

    monkeypatch.setattr(os, "fsync", observe)
    with stage_bytes(first, b"new a") as a, stage_bytes(second, b"new b") as b:
        assert synced == ["file", "file"]
        assert first.read_bytes() == b"old a" and second.read_bytes() == b"old b"
        replace_staged(first, a)
        replace_staged(second, b)
    assert synced == ["file", "file", "dir", "dir"]
    assert first.read_bytes() == b"new a" and second.read_bytes() == b"new b"
    assert set(tmp_path.iterdir()) == {first, second}


def test_filesystem_failures_are_input_rejection(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "file"
    operations: list[Callable[[], object]] = [
        lambda: atomic_replace(missing, b"x"),
        lambda: write_artifact(missing, b"x", []),
    ]
    for operation in operations:
        with pytest.raises(ContractError) as error:
            operation()
        assert error.value.exit_code == 3
        assert error.value.diagnostic.code == Code.UNREADABLE
    reject_aliases(tmp_path / "absent", [tmp_path / "other-absent"])


def test_resolution_failure_is_input_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: Path, strict: bool = False) -> Path:
        raise OSError("cannot resolve path")

    monkeypatch.setattr(Path, "resolve", fail)
    with pytest.raises(ContractError) as error:
        write_artifact(tmp_path / "file", b"new", [])
    assert error.value.diagnostic.code == Code.UNREADABLE
