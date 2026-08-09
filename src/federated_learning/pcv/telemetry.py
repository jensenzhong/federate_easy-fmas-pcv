"""Crash-resistant append-only JSONL telemetry for agent calls."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
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

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    def append(self, record: Mapping[str, Any]) -> None:
        if type(record) is not dict:
            raise TypeError("telemetry record must be an exact dictionary")
        _assert_no_sensitive_fields(record)
        serialized = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        encoded = (serialized + "\n").encode("utf-8")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        thread_lock = _thread_lock_for(self.path)
        with thread_lock, _ProcessFileLock(self._lock_path):
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0),
                0o600,
            )
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("telemetry append made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


__all__ = ["AppendOnlyTelemetry"]
