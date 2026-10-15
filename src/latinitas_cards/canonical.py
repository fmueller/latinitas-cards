"""Canonical JSON and SHA-256 for new versioned documents (not legacy digests)."""

import hashlib
import json


def canonical_json(value: object) -> str:
    def reject_floats(part: object) -> None:
        if isinstance(part, float):
            raise ValueError("canonical JSON forbids floats")
        if isinstance(part, dict):
            for key, item in part.items():
                reject_floats(key)
                reject_floats(item)
        elif isinstance(part, (list, tuple)):
            for item in part:
                reject_floats(item)

    reject_floats(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def digest(value: object) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))
