"""POSIX writer primitives shared by the new contract commands only."""

import fcntl
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .cli_contract import Code, ContractError


@contextmanager
def _filesystem(path: Path) -> Iterator[None]:
    try:
        yield
    except OSError as error:
        raise ContractError(Code.UNREADABLE, str(error), path=str(path)) from error


def reject_aliases(output: Path, protected: Iterable[Path]) -> None:
    """Reject lexical, symlink and inode aliases, including absent lexical paths."""
    with _filesystem(output):
        for path in protected:
            if output.resolve() == path.resolve() or (output.exists() and path.exists() and output.samefile(path)):
                raise ContractError(Code.ALIAS, "Output aliases a protected path", path=str(output))


@contextmanager
def writer_locks(paths: Iterable[Path]) -> Iterator[None]:
    """Hold all target locks in lexical resolved-path order; never remove lock files."""
    targets = list(paths)
    with _filesystem(targets[0] if targets else Path(".")), ExitStack() as stack:
        resolved = sorted({path.resolve() for path in targets})
        locks = [Path(f"{path}.lock") for path in resolved]
        for target in resolved:
            reject_aliases(target, locks)
        for index, lock in enumerate(locks):
            reject_aliases(lock, locks[:index])
        for lock in locks:
            stream = stack.enter_context(lock.open("ab"))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ContractError(Code.BUSY, "Writer lock is held", path=str(lock)) from error
        yield


@contextmanager
def writer_lock(path: Path) -> Iterator[None]:
    with writer_locks([path]):
        yield


@contextmanager
def stage_bytes(path: Path, data: bytes) -> Iterator[Path]:
    """Stage and fsync bytes beside the target; clean up on all exit paths."""
    with _filesystem(path):
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            staged = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                staged.unlink(missing_ok=True)
                raise
        try:
            yield staged
        finally:
            staged.unlink(missing_ok=True)


def replace_staged(path: Path, staged: Path) -> None:
    """Replace a staged file and durably record the replacement in its directory."""
    with _filesystem(path):
        os.replace(staged, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_replace(path: Path, data: bytes) -> None:
    """Replace one file; callers advancing existing state must hold its writer lock."""
    with stage_bytes(path, data) as staged:
        replace_staged(path, staged)


def write_artifact(path: Path, data: bytes, protected: Iterable[Path]) -> None:
    """Create an immutable artifact, accepting byte-identical reruns only."""
    protected = tuple(protected)
    with _filesystem(path):
        lock = Path(f"{path.resolve()}.lock")
    reject_aliases(path, [*protected, lock])
    reject_aliases(lock, protected)
    with writer_lock(path), _filesystem(path):
        reject_aliases(path, [*protected, lock])
        try:
            previous = path.read_bytes()
        except FileNotFoundError:
            atomic_replace(path, data)
        else:
            if previous != data:
                raise ContractError(Code.EXISTS, "Output exists with different bytes", path=str(path))
