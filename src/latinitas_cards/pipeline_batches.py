"""Run-local, receipt-bound CSV publication and crash recovery; no acceptance authority."""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from .acceptance_records import Hash, Record, canonical, digest


class CardRef(Record):
    note_id: str
    card_key: str


class Receipt(Record):
    format: Literal["latinitas-selection-receipt"]
    version: Literal[1]
    run_id: Hash
    request_id: Hash
    selection_sha256: Hash
    previous_receipt_id: Hash | None
    reconciled_receipt_id: Hash | None
    ordered_card_refs_sha256: Hash
    emitted_cards: tuple[CardRef, ...]
    output: str
    output_sha256: Hash
    result: dict[str, object]


class Publication(Record):
    receipt: Receipt
    csv: str
    staged_output: str
    device: int
    inode: int


class Index(Record):
    format: Literal["latinitas-selection-index"] = "latinitas-selection-index"
    version: Literal[1] = 1
    requests: dict[str, str]
    snapshot: str | None


@contextmanager
def run_lock(root: Path) -> Iterator[None]:
    fd = os.open(root / "selection.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("run_busy") from None
        yield
    finally:
        os.close(fd)


def _atomic(path: Path, payload: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=".selection-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Batches:
    """Caller holds the run lock throughout eligibility checks and publication."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root / "selection"
        if self.root.is_symlink():
            raise ValueError("symlink selection store")
        self.root.mkdir(exist_ok=True)
        self.run_id = run_id
        path = self.root / "index.json"
        if path.is_symlink():
            raise ValueError("symlink selection index")
        self.index = Index.read_json(path.read_bytes()) if path.exists() else Index(requests={}, snapshot=None)

    def recover(self, snapshot: str) -> None:
        pending = self.root / "pending.json"
        if pending.is_symlink():
            raise ValueError("symlink pending publication")
        if not pending.exists():
            return
        publication = Publication.read_json(pending.read_bytes())
        receipt = publication.receipt
        staged = self._stage(publication)
        output = Path(receipt.output)
        if (output.exists() or output.is_symlink()) and (output.is_symlink() or not output.samefile(staged)):
            raise ValueError("batch output conflict; staged publication retained")
        receipt_id = digest(receipt)
        if self.index.requests.get(receipt.request_id) == receipt_id:
            self.read(receipt_id)
            pending.unlink()
            staged.unlink()
            _sync_directory(self.root)
            _sync_directory(output.parent)
            return
        if receipt.selection_sha256 != snapshot:
            if output.exists():
                output.unlink()
                _sync_directory(output.parent)
            os.replace(pending, self.root / (digest(publication) + ".aborted"))
            _sync_directory(self.root)
            staged.unlink()
            _sync_directory(output.parent)
            return
        self._finish(publication)

    def read(self, receipt_id: str) -> Receipt:
        receipt = self._read(receipt_id)
        pending = [receipt.previous_receipt_id, receipt.reconciled_receipt_id]
        seen = {receipt_id}
        while pending:
            ancestor_id = pending.pop()
            if ancestor_id is None or ancestor_id in seen:
                continue
            seen.add(ancestor_id)
            ancestor = self._read(ancestor_id)
            pending.extend((ancestor.previous_receipt_id, ancestor.reconciled_receipt_id))
        return receipt

    def _read(self, receipt_id: str) -> Receipt:
        if len(receipt_id) != 64 or any(c not in "0123456789abcdef" for c in receipt_id):
            raise ValueError("invalid receipt ID")
        path = self.root / (receipt_id + ".json")
        if path.is_symlink() or not path.exists():
            raise ValueError("stale or foreign predecessor receipt")
        receipt = Receipt.read_json(path.read_bytes())
        if (
            digest(receipt) != receipt_id
            or receipt.run_id != self.run_id
            or self.index.requests.get(receipt.request_id) != receipt_id
        ):
            raise ValueError("stale or foreign predecessor receipt")
        output = Path(receipt.output)
        if (
            output.is_symlink()
            or not output.is_file()
            or hashlib.sha256(output.read_bytes()).hexdigest() != receipt.output_sha256
        ):
            raise ValueError("batch output hash mismatch")
        return receipt

    def request(
        self, snapshot: str, limit: int | None, layout: str, predecessor: str | None, reconcile: str | None
    ) -> tuple[str, Receipt | None, tuple[CardRef, ...]]:
        if predecessor is not None and reconcile is not None:
            raise ValueError("choose continuation or reconciliation, not both")
        request_id = digest((self.run_id, snapshot, predecessor, limit, layout, reconcile))
        if request_id in self.index.requests:
            return request_id, self.read(self.index.requests[request_id]), ()
        prior_id = predecessor or reconcile
        previous = self.read(prior_id) if prior_id is not None else None
        if predecessor and previous is not None and previous.selection_sha256 != snapshot:
            raise ValueError("stale selection snapshot; explicit reconciliation required")
        if self.index.snapshot is not None and self.index.snapshot != snapshot and not reconcile and not predecessor:
            raise ValueError("changed selection snapshot; explicit reconciliation required")
        return request_id, None, previous.emitted_cards if previous else ()

    def publish(self, receipt: Receipt, payload: bytes) -> dict[str, object]:
        output = Path(receipt.output)
        fd, name = tempfile.mkstemp(prefix=".latinitas-batch-", dir=output.parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            stat = os.fstat(stream.fileno())
        _sync_directory(output.parent)
        publication = Publication(
            receipt=receipt, csv=payload.decode("utf-8"), staged_output=name, device=stat.st_dev, inode=stat.st_ino
        )
        _atomic(self.root / "pending.json", canonical(publication))
        return self._finish(publication)

    def _stage(self, publication: Publication) -> Path:
        staged = Path(publication.staged_output)
        if (
            publication.receipt.run_id != self.run_id
            or hashlib.sha256(publication.csv.encode("utf-8")).hexdigest() != publication.receipt.output_sha256
            or staged.parent != Path(publication.receipt.output).parent
            or not staged.name.startswith(".latinitas-batch-")
            or staged.is_symlink()
            or not staged.is_file()
        ):
            raise ValueError("invalid staged batch")
        stat = staged.stat()
        if (stat.st_dev, stat.st_ino) != (publication.device, publication.inode) or hashlib.sha256(
            staged.read_bytes()
        ).hexdigest() != publication.receipt.output_sha256:
            raise ValueError("invalid staged batch")
        return staged

    def _finish(self, publication: Publication) -> dict[str, object]:
        receipt = publication.receipt
        staged = self._stage(publication)
        output = Path(receipt.output)
        receipt_id = digest(receipt)
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.samefile(staged):
                raise ValueError("batch output conflict; staged publication retained")
        else:
            os.link(staged, output)
            _sync_directory(output.parent)
        path = self.root / (receipt_id + ".json")
        receipt_bytes = canonical(receipt)
        if path.is_symlink() or (path.exists() and path.read_bytes() != receipt_bytes):
            raise ValueError("receipt conflict")
        _atomic(path, receipt_bytes)
        index = Index(
            requests={**self.index.requests, receipt.request_id: receipt_id}, snapshot=receipt.selection_sha256
        )
        _atomic(self.root / "index.json", canonical(index))
        self.index = index
        (self.root / "pending.json").unlink(missing_ok=True)
        _sync_directory(self.root)
        staged.unlink()
        _sync_directory(output.parent)
        return self.result(receipt)

    @staticmethod
    def result(receipt: Receipt) -> dict[str, object]:
        return {**receipt.result, "receipt_id": digest(receipt)}
