"""Crash-resistant append-only JSONL telemetry for agent calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
from threading import Lock
from typing import Any


_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "apikey",
        "authorization",
        "authorizationheader",
        "headers",
        "payload",
        "requestbody",
        "requestheaders",
        "requestpayload",
    }
)
_LOCKS_GUARD = Lock()
_PATH_LOCKS: dict[str, Lock] = {}
_BEARER_PATTERN = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]+",
    flags=re.IGNORECASE,
)
_AUTHORIZATION_VALUE_PATTERN = re.compile(
    r"(\bauthorization\b[\"']?\s*[:=]\s*[\"']?)"
    r"(?!\[REDACTED\])[^\"'\s,;}\]]+",
    flags=re.IGNORECASE,
)
_REDACTION_MARKER = "[REDACTED]"


def _normalized_field_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _assert_no_sensitive_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and _normalized_field_name(key) in _SENSITIVE_FIELD_NAMES:
                raise ValueError(f"sensitive telemetry field is forbidden: {key}")
            _assert_no_sensitive_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_sensitive_fields(item)


def _sanitize_string(value: str, known_secrets: tuple[str, ...]) -> str:
    sanitized = value
    for secret in sorted(known_secrets, key=len, reverse=True):
        sanitized = sanitized.replace(secret, _REDACTION_MARKER)
    sanitized = _BEARER_PATTERN.sub(f"Bearer {_REDACTION_MARKER}", sanitized)
    sanitized = _AUTHORIZATION_VALUE_PATTERN.sub(
        rf"\1{_REDACTION_MARKER}",
        sanitized,
    )
    return sanitized


def sanitize_telemetry_value(
    value: Any,
    *,
    known_secrets: Sequence[str] = (),
) -> Any:
    """Return a recursively value-sanitized, JSON-compatible copy."""
    secrets = tuple(
        secret
        for secret in known_secrets
        if type(secret) is str and secret
    )
    if isinstance(value, Mapping):
        return {
            key: sanitize_telemetry_value(item, known_secrets=secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            sanitize_telemetry_value(item, known_secrets=secrets)
            for item in value
        ]
    if type(value) is str:
        return _sanitize_string(value, secrets)
    return value


def _thread_lock_for(path: Path) -> Lock:
    normalized = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(normalized, Lock())


class _ProcessFileLock:
    """Small cross-platform advisory lock backed by a sibling lock file."""

    def __init__(self, path: Path):
        self._path = path
        self._file = None

    def __enter__(self):
        self._file = open(self._path, "a+b")
        self._file.seek(0)
        if os.name == "nt":
            import msvcrt

            if os.fstat(self._file.fileno()).st_size == 0:
                self._file.write(b"\0")
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.seek(0)
            msvcrt.locking(self._file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._file is not None
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None


class AppendOnlyTelemetry:
    """Append one fully serialized and durable JSON object per line.

    Serialization and sensitive-field validation finish before the output file is
    opened, so either a complete line is appended or the file remains unchanged.
    A process lock plus a per-path thread lock keeps concurrent writers from
    interleaving records.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        known_secrets: Sequence[str] = (),
    ):
        self.path = Path(path)
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._secrets_lock = Lock()
        self._known_secrets: set[str] = set()
        for secret in known_secrets:
            self.register_secret(secret)

    def register_secret(self, secret: str) -> None:
        """Register a value that must be redacted from every future record."""
        if type(secret) is not str or not secret:
            raise ValueError("telemetry secrets must be non-empty exact strings")
        with self._secrets_lock:
            self._known_secrets.add(secret)

    def append(self, record: Mapping[str, Any]) -> None:
        if type(record) is not dict:
            raise TypeError("telemetry record must be an exact dictionary")
        _assert_no_sensitive_fields(record)
        with self._secrets_lock:
            known_secrets = tuple(self._known_secrets)
        sanitized_record = sanitize_telemetry_value(
            record,
            known_secrets=known_secrets,
        )
        serialized = json.dumps(
            sanitized_record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        encoded = (serialized + "\n").encode("utf-8")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        thread_lock = _thread_lock_for(self.path)
        with thread_lock, _ProcessFileLock(self._lock_path):
            with open(self.path, "ab") as stream:
                written = stream.write(encoded)
                if written != len(encoded):
                    raise OSError("telemetry append was incomplete")
                stream.flush()
                os.fsync(stream.fileno())


__all__ = ["AppendOnlyTelemetry", "sanitize_telemetry_value"]
