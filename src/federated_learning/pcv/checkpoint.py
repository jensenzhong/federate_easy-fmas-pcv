"""Exact, atomic checkpoints for pause-and-resume FMAS experiments."""

from __future__ import annotations

import errno
import os
import random
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1
CHECKPOINT_FILENAME = "last_complete.pt"

_DIRECTORY_FSYNC_SUPPORTED = os.name != "nt" and hasattr(os, "O_DIRECTORY")
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    value
    for value in (
        errno.EINVAL,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOSYS", None),
    )
    if value is not None
)

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "last_complete_round",
        "freeze_id",
        "method",
        "training_seed",
        "llm_rep",
        "global_model_state",
        "server_optimizer_state",
        "previous_weights",
        "best_validation",
        "best_model_state",
        "rng_state",
        "partition_sha256",
        "config_sha256",
        "prompt_hashes",
    }
)


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointFormatError(CheckpointError, ValueError):
    """A checkpoint or checkpoint request has an invalid schema."""


class CheckpointRestoreError(CheckpointError):
    """Restoration failed after validation; live state was rolled back."""


class ResumeApprovalRequired(CheckpointError, PermissionError):
    """Resume was not explicitly approved by the user."""


class ResumeMismatchError(CheckpointError):
    """Checkpoint identity does not exactly match the requested run."""


def _plain_tensor(value: torch.Tensor) -> torch.Tensor:
    """Clone tensor storage and erase Parameter/tensor-subclass identity."""

    return value.as_subclass(torch.Tensor).detach().clone(memory_format=torch.preserve_format)


def _deep_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _plain_tensor(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise CheckpointFormatError("object-dtype NumPy arrays are unsafe in checkpoints")
        if value.dtype.kind not in "biufc":
            raise CheckpointFormatError(f"unsupported NumPy array dtype: {value.dtype}")
        copied = np.array(value, copy=True, order="K", subok=False)
        if not copied.dtype.isnative:
            copied = copied.astype(copied.dtype.newbyteorder("="), copy=False)
        try:
            return _plain_tensor(torch.from_numpy(copied))
        except (TypeError, ValueError, RuntimeError) as exc:
            raise CheckpointFormatError(f"unsupported NumPy array dtype: {value.dtype}") from exc
    if isinstance(value, np.generic):
        return _deep_clone(value.item())
    if isinstance(value, Mapping):
        cloned = {}
        for key, item in value.items():
            if type(key) not in (bool, int, float, str):
                raise CheckpointFormatError(f"unsupported checkpoint mapping key: {type(key).__name__}")
            cloned[key] = _deep_clone(item)
        return cloned
    if isinstance(value, tuple):
        return tuple(_deep_clone(item) for item in value)
    if isinstance(value, list):
        return [_deep_clone(item) for item in value]
    if type(value) in (type(None), bool, int, float, str):
        return value
    raise CheckpointFormatError(f"unsupported unsafe checkpoint value: {type(value).__name__}")


def _encode_numpy_rng_state(state: tuple[Any, ...]) -> dict[str, Any]:
    if type(state) is not tuple or len(state) != 5:
        raise CheckpointFormatError("NumPy RNG state has an invalid shape")
    bit_generator, keys, position, has_gauss, cached_gaussian = state
    if not isinstance(keys, np.ndarray) or keys.dtype.hasobject:
        raise CheckpointFormatError("NumPy RNG keys must be a numeric array")
    return {
        "bit_generator": str(bit_generator),
        "keys": torch.tensor(keys.astype(np.int64, copy=True), dtype=torch.int64),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _decode_numpy_rng_state(state: Mapping[str, Any]) -> tuple[Any, ...]:
    keys = state["keys"]
    return (
        state["bit_generator"],
        keys.detach().cpu().numpy().astype(np.uint32, copy=True),
        state["position"],
        state["has_gauss"],
        state["cached_gaussian"],
    )


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and already-initialized CUDA RNGs."""

    cuda_initialized = bool(torch.cuda.is_initialized())
    cuda_device_count = int(torch.cuda.device_count())
    cuda_state = None
    # Checking initialization is deliberately used instead of availability:
    # checkpointing a CPU run must not initialize a CUDA context.
    if cuda_initialized:
        cuda_state = [_plain_tensor(state) for state in torch.cuda.get_rng_state_all()]
        if len(cuda_state) != cuda_device_count:
            raise CheckpointFormatError(
                "CUDA RNG state count does not match the captured CUDA device count"
            )
    return {
        "python": _deep_clone(random.getstate()),
        "numpy": _encode_numpy_rng_state(np.random.get_state()),
        "torch_cpu": _plain_tensor(torch.get_rng_state()),
        "cuda_initialized": cuda_initialized,
        "cuda_device_count": cuda_device_count,
        "torch_cuda": cuda_state,
    }


def _validate_rng_state(state: Any) -> None:
    if not isinstance(state, Mapping):
        raise CheckpointFormatError("rng_state must be a mapping")
    expected = {
        "python",
        "numpy",
        "torch_cpu",
        "cuda_initialized",
        "cuda_device_count",
        "torch_cuda",
    }
    if set(state) != expected:
        raise CheckpointFormatError("rng_state fields do not match schema")

    try:
        validator = random.Random()
        validator.setstate(_deep_clone(state["python"]))
    except Exception as exc:
        raise CheckpointFormatError("rng_state.python is invalid") from exc

    try:
        numpy_state = state["numpy"]
        expected_numpy = {
            "bit_generator",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }
        if type(numpy_state) is not dict or set(numpy_state) != expected_numpy:
            raise ValueError("NumPy RNG state fields do not match schema")
        if type(numpy_state["bit_generator"]) is not str:
            raise ValueError("NumPy bit generator must be a string")
        keys = numpy_state["keys"]
        if (
            type(keys) is not torch.Tensor
            or keys.device.type != "cpu"
            or keys.dtype != torch.int64
            or keys.ndim != 1
        ):
            raise ValueError("NumPy RNG keys must be a one-dimensional int64 CPU tensor")
        if bool(torch.any(keys < 0).item()) or bool(torch.any(keys > np.iinfo(np.uint32).max).item()):
            raise ValueError("NumPy RNG keys exceed uint32 range")
        if type(numpy_state["position"]) is not int:
            raise ValueError("NumPy RNG position must be an integer")
        if type(numpy_state["has_gauss"]) is not int:
            raise ValueError("NumPy RNG has_gauss must be an integer")
        if type(numpy_state["cached_gaussian"]) is not float:
            raise ValueError("NumPy cached Gaussian must be a float")
        validator_np = np.random.RandomState()
        validator_np.set_state(_decode_numpy_rng_state(numpy_state))
    except Exception as exc:
        raise CheckpointFormatError("rng_state.numpy is invalid") from exc

    cpu_state = state["torch_cpu"]
    if type(cpu_state) is not torch.Tensor or cpu_state.device.type != "cpu":
        raise CheckpointFormatError("rng_state.torch_cpu must be a plain CPU tensor")
    try:
        validator_torch = torch.Generator(device="cpu")
        validator_torch.set_state(cpu_state)
    except Exception as exc:
        raise CheckpointFormatError("rng_state.torch_cpu is invalid") from exc

    cuda_state = state["torch_cuda"]
    cuda_initialized = state["cuda_initialized"]
    cuda_device_count = state["cuda_device_count"]
    if type(cuda_initialized) is not bool:
        raise CheckpointFormatError("rng_state.cuda_initialized must be a boolean")
    if type(cuda_device_count) is not int or cuda_device_count < 0:
        raise CheckpointFormatError("rng_state.cuda_device_count must be a non-negative integer")
    if cuda_initialized:
        if type(cuda_state) is not list:
            raise CheckpointFormatError("initialized CUDA RNG states must be a list")
        if len(cuda_state) != cuda_device_count:
            raise CheckpointFormatError("CUDA RNG state count does not match cuda_device_count")
        if any(
            type(item) is not torch.Tensor
            or item.device.type != "cpu"
            or item.dtype != torch.uint8
            or item.ndim != 1
            for item in cuda_state
        ):
            raise CheckpointFormatError("CUDA RNG states must be one-dimensional uint8 CPU tensors")
    elif cuda_state is not None:
        raise CheckpointFormatError("uninitialized CUDA must have no CUDA RNG states")


def _validate_cuda_topology_for_restore(state: Mapping[str, Any]) -> None:
    current_initialized = bool(torch.cuda.is_initialized())
    current_device_count = int(torch.cuda.device_count())
    if state["cuda_initialized"] is not current_initialized:
        raise CheckpointRestoreError(
            "CUDA topology mismatch: initialization state differs from checkpoint"
        )
    if state["cuda_device_count"] != current_device_count:
        raise CheckpointRestoreError(
            "CUDA topology mismatch: device count differs from checkpoint"
        )
    if current_initialized:
        current_states = torch.cuda.get_rng_state_all()
        if type(current_states) is not list or len(current_states) != current_device_count:
            raise CheckpointRestoreError(
                "CUDA topology mismatch: current RNG state count differs from device count"
            )
        for device_index, (saved_state, current_state) in enumerate(
            zip(state["torch_cuda"], current_states)
        ):
            if (
                type(current_state) is not torch.Tensor
                or current_state.device.type != "cpu"
                or current_state.dtype != torch.uint8
                or current_state.ndim != 1
            ):
                raise CheckpointRestoreError(
                    f"CUDA RNG state format is invalid for current device {device_index}"
                )
            if saved_state.shape != current_state.shape or saved_state.numel() != current_state.numel():
                raise CheckpointRestoreError(
                    f"CUDA RNG state shape/numel mismatch for device {device_index}"
                )


def _set_rng_state_unchecked(state: Mapping[str, Any]) -> None:
    random.setstate(_deep_clone(state["python"]))
    np.random.set_state(_decode_numpy_rng_state(state["numpy"]))
    torch.set_rng_state(_plain_tensor(state["torch_cpu"]).cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all([_plain_tensor(item).cpu() for item in state["torch_cuda"]])


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore every captured RNG exactly, rolling back on a setter failure."""

    _validate_rng_state(state)
    _validate_cuda_topology_for_restore(state)
    before = capture_rng_state()
    try:
        _set_rng_state_unchecked(state)
    except Exception:
        try:
            _set_rng_state_unchecked(before)
        except Exception:
            pass
        raise


def _state_from_source(
    *,
    object_value: Any,
    explicit_state: Mapping[str, Any] | None,
    object_name: str,
    getter_name: str,
) -> Mapping[str, Any]:
    if (object_value is None) == (explicit_state is None):
        raise CheckpointFormatError(
            f"provide exactly one of {object_name} and {object_name}_state"
        )
    if explicit_state is not None:
        if not isinstance(explicit_state, Mapping):
            raise CheckpointFormatError(f"{object_name}_state must be a mapping")
        return explicit_state
    getter = getattr(object_value, getter_name, None)
    if not callable(getter):
        raise CheckpointFormatError(f"{object_name} must provide {getter_name}()")
    state = getter()
    if not isinstance(state, Mapping):
        raise CheckpointFormatError(f"{object_name}.{getter_name}() must return a mapping")
    return state


def build_checkpoint_payload(
    *,
    round_index: int,
    freeze_id: str,
    method: str,
    training_seed: int,
    llm_rep: int,
    previous_weights: Mapping[str, Any],
    best_validation: Mapping[str, Any],
    best_model_state: Mapping[str, torch.Tensor],
    partition_sha256: str,
    config_sha256: str,
    prompt_hashes: Mapping[str, str],
    model: Any = None,
    global_model_state: Mapping[str, torch.Tensor] | None = None,
    server_optimizer: Any = None,
    server_optimizer_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete schema-v1 payload with no mutable source aliases."""

    if type(round_index) is not int or round_index < 0:
        raise CheckpointFormatError("last_complete_round must be a non-negative integer")
    model_state = _state_from_source(
        object_value=model,
        explicit_state=global_model_state,
        object_name="global_model",
        getter_name="state_dict",
    )
    optimizer_state = _state_from_source(
        object_value=server_optimizer,
        explicit_state=server_optimizer_state,
        object_name="server_optimizer",
        getter_name="get_optimizer_state",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "last_complete_round": round_index,
        "freeze_id": freeze_id,
        "method": method,
        "training_seed": training_seed,
        "llm_rep": llm_rep,
        "global_model_state": _deep_clone(model_state),
        "server_optimizer_state": _deep_clone(optimizer_state),
        "previous_weights": _deep_clone(previous_weights),
        "best_validation": _deep_clone(best_validation),
        "best_model_state": _deep_clone(best_model_state),
        "rng_state": capture_rng_state(),
        "partition_sha256": partition_sha256,
        "config_sha256": config_sha256,
        "prompt_hashes": _deep_clone(prompt_hashes),
    }
    _validate_payload(payload)
    return payload


def _require_exact_int(payload: Mapping[str, Any], field: str, *, non_negative: bool = False) -> None:
    value = payload[field]
    if type(value) is not int or (non_negative and value < 0):
        qualifier = "non-negative " if non_negative else ""
        raise CheckpointFormatError(f"{field} must be a {qualifier}integer")


def _require_nonempty_string(payload: Mapping[str, Any], field: str) -> None:
    if type(payload[field]) is not str or not payload[field]:
        raise CheckpointFormatError(f"{field} must be a non-empty string")


def _validate_tensor_state(value: Any, field: str) -> None:
    if not isinstance(value, Mapping):
        raise CheckpointFormatError(f"{field} must be a mapping")
    if any(type(key) is not str or type(tensor) is not torch.Tensor for key, tensor in value.items()):
        raise CheckpointFormatError(f"{field} must map strings to plain tensors")


def _validate_safe_value(value: Any, path: str = "checkpoint") -> None:
    if type(value) in (type(None), bool, int, float, str, torch.Tensor):
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _validate_safe_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) not in (bool, int, float, str):
                raise CheckpointFormatError(
                    f"{path} has an unsupported mapping key: {type(key).__name__}"
                )
            _validate_safe_value(item, f"{path}[{key!r}]")
        return
    raise CheckpointFormatError(f"{path} contains unsupported unsafe type {type(value).__name__}")


def _validate_payload(payload: Any) -> None:
    if type(payload) is not dict:
        raise CheckpointFormatError("checkpoint payload must be a plain dictionary")
    _validate_safe_value(payload)
    if set(payload) != _REQUIRED_FIELDS:
        missing = sorted(_REQUIRED_FIELDS.difference(payload))
        extra = sorted(set(payload).difference(_REQUIRED_FIELDS))
        raise CheckpointFormatError(f"checkpoint fields do not match schema (missing={missing}, extra={extra})")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise CheckpointFormatError(f"schema_version must be {SCHEMA_VERSION}")
    _require_exact_int(payload, "last_complete_round", non_negative=True)
    _require_exact_int(payload, "training_seed")
    _require_exact_int(payload, "llm_rep", non_negative=True)
    for field in ("freeze_id", "method", "partition_sha256", "config_sha256"):
        _require_nonempty_string(payload, field)
    _validate_tensor_state(payload["global_model_state"], "global_model_state")
    _validate_tensor_state(payload["best_model_state"], "best_model_state")
    for field in ("server_optimizer_state", "previous_weights", "best_validation"):
        if not isinstance(payload[field], Mapping):
            raise CheckpointFormatError(f"{field} must be a mapping")
    prompt_hashes = payload["prompt_hashes"]
    if not isinstance(prompt_hashes, Mapping) or any(
        type(key) is not str or type(value) is not str or not value
        for key, value in prompt_hashes.items()
    ):
        raise CheckpointFormatError("prompt_hashes must map strings to non-empty strings")
    _validate_rng_state(payload["rng_state"])


def _checkpoint_target(directory_or_path: os.PathLike[str] | str) -> Path:
    candidate = Path(directory_or_path)
    return candidate if candidate.name == CHECKPOINT_FILENAME else candidate / CHECKPOINT_FILENAME


def _fsync_directory(directory: Path) -> None:
    if not _DIRECTORY_FSYNC_SUPPORTED:
        return
    descriptor = None
    try:
        descriptor = os.open(directory, _DIRECTORY_OPEN_FLAGS)
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def save_checkpoint(
    directory_or_path: os.PathLike[str] | str,
    payload: Mapping[str, Any],
) -> Path:
    """Durably save to a same-directory temp and replace last_complete.pt."""

    _validate_payload(payload)
    target = _checkpoint_target(directory_or_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(_deep_clone(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def load_checkpoint(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Safely load one regular-file descriptor onto CPU and validate it."""

    checkpoint_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(checkpoint_path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CheckpointFormatError(f"checkpoint path is not a regular file: {checkpoint_path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            payload = torch.load(
                handle,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        _validate_payload(payload)
    except FileNotFoundError:
        raise
    except CheckpointFormatError:
        raise
    except Exception as exc:
        raise CheckpointFormatError(f"could not load checkpoint {checkpoint_path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return dict(payload)


_EXPECTED_METADATA = {
    "requested_freeze_id": "freeze_id",
    "requested_method": "method",
    "requested_training_seed": "training_seed",
    "requested_llm_rep": "llm_rep",
    "requested_partition_sha256": "partition_sha256",
    "requested_config_sha256": "config_sha256",
    "requested_prompt_hashes": "prompt_hashes",
}


def _metadata_equal(checkpoint_value: Any, requested_value: Any) -> bool:
    """Compare run identity without Python's cross-type numeric equality."""

    if isinstance(checkpoint_value, Mapping):
        if not isinstance(requested_value, Mapping):
            return False
        if set(checkpoint_value) != set(requested_value):
            return False
        return all(
            type(key) is str
            and key in requested_value
            and _metadata_equal(value, requested_value[key])
            for key, value in checkpoint_value.items()
        )
    return type(checkpoint_value) is type(requested_value) and checkpoint_value == requested_value


def _load_validated_resume_payload(
    *,
    resume_checkpoint: os.PathLike[str] | str,
    requested_freeze_id: str | None,
    requested_method: str | None,
    requested_training_seed: int | None,
    requested_llm_rep: int | None,
    requested_partition_sha256: str | None,
    requested_config_sha256: str | None,
    requested_prompt_hashes: Mapping[str, str] | None,
) -> tuple[dict[str, Any], int]:
    requested = {
        "requested_freeze_id": requested_freeze_id,
        "requested_method": requested_method,
        "requested_training_seed": requested_training_seed,
        "requested_llm_rep": requested_llm_rep,
        "requested_partition_sha256": requested_partition_sha256,
        "requested_config_sha256": requested_config_sha256,
        "requested_prompt_hashes": requested_prompt_hashes,
    }
    missing = [name for name, value in requested.items() if value is None]
    if missing:
        raise CheckpointFormatError(f"full resume requires {', '.join(missing)}")

    payload = load_checkpoint(resume_checkpoint)
    for requested_name, checkpoint_name in _EXPECTED_METADATA.items():
        if not _metadata_equal(payload[checkpoint_name], requested[requested_name]):
            raise ResumeMismatchError(f"resume mismatch for {checkpoint_name}")
    return payload, payload["last_complete_round"] + 1


def validate_resume(
    *,
    user_approved_resume: bool,
    resume_checkpoint: os.PathLike[str] | str | None = None,
    checkpoint_freeze_id: str | None = None,
    requested_freeze_id: str | None = None,
    requested_method: str | None = None,
    requested_training_seed: int | None = None,
    requested_llm_rep: int | None = None,
    requested_partition_sha256: str | None = None,
    requested_config_sha256: str | None = None,
    requested_prompt_hashes: Mapping[str, str] | None = None,
) -> int | None:
    """Authorize and validate either legacy freeze-only or full resume input."""

    if user_approved_resume is not True:
        raise ResumeApprovalRequired("resume requires user_approved_resume=True")

    legacy_inputs = checkpoint_freeze_id is not None
    if legacy_inputs:
        if resume_checkpoint is not None or any(
            value is not None
            for value in (
                requested_method,
                requested_training_seed,
                requested_llm_rep,
                requested_partition_sha256,
                requested_config_sha256,
                requested_prompt_hashes,
            )
        ):
            raise CheckpointFormatError("legacy resume arguments cannot be mixed with full resume arguments")
        if requested_freeze_id is None:
            raise CheckpointFormatError("requested_freeze_id is required in legacy resume mode")
        if checkpoint_freeze_id != requested_freeze_id:
            raise ResumeMismatchError("resume mismatch for freeze_id")
        return None

    if resume_checkpoint is None:
        raise CheckpointFormatError("resume_checkpoint path is required for full resume")
    _, start_round = _load_validated_resume_payload(
        resume_checkpoint=resume_checkpoint,
        requested_freeze_id=requested_freeze_id,
        requested_method=requested_method,
        requested_training_seed=requested_training_seed,
        requested_llm_rep=requested_llm_rep,
        requested_partition_sha256=requested_partition_sha256,
        requested_config_sha256=requested_config_sha256,
        requested_prompt_hashes=requested_prompt_hashes,
    )
    return start_round


def restore_checkpoint(
    *,
    model: Any,
    server_optimizer: Any,
    user_approved_resume: bool,
    resume_checkpoint: os.PathLike[str] | str,
    requested_freeze_id: str,
    requested_method: str,
    requested_training_seed: int,
    requested_llm_rep: int,
    requested_partition_sha256: str,
    requested_config_sha256: str,
    requested_prompt_hashes: Mapping[str, str],
) -> int:
    """Validate first, then transactionally restore model, optimizer, and RNG."""

    if user_approved_resume is not True:
        raise ResumeApprovalRequired("resume requires user_approved_resume=True")
    payload, start_round = _load_validated_resume_payload(
        resume_checkpoint=resume_checkpoint,
        requested_freeze_id=requested_freeze_id,
        requested_method=requested_method,
        requested_training_seed=requested_training_seed,
        requested_llm_rep=requested_llm_rep,
        requested_partition_sha256=requested_partition_sha256,
        requested_config_sha256=requested_config_sha256,
        requested_prompt_hashes=requested_prompt_hashes,
    )

    _restore_loaded_payload(
        payload=payload,
        model=model,
        server_optimizer=server_optimizer,
    )
    return int(start_round)


def _restore_loaded_payload(
    *,
    payload: Mapping[str, Any],
    model: Any,
    server_optimizer: Any,
) -> None:
    """Restore one already validated payload with rollback on any mutation failure."""

    model_loader = getattr(model, "load_state_dict", None)
    optimizer_loader = getattr(server_optimizer, "load_optimizer_state", None)
    if not callable(model_loader) or not callable(optimizer_loader):
        raise CheckpointFormatError("restore targets must provide state loading methods")
    model_before = _deep_clone(model.state_dict())
    optimizer_before = _deep_clone(server_optimizer.get_optimizer_state())
    rng_before = capture_rng_state()
    try:
        model_loader(_deep_clone(payload["global_model_state"]))
        optimizer_loader(_deep_clone(payload["server_optimizer_state"]))
        restore_rng_state(payload["rng_state"])
    except Exception as exc:
        rollback_failures = []
        for label, rollback in (
            ("model", lambda: model_loader(model_before)),
            ("optimizer", lambda: optimizer_loader(optimizer_before)),
            ("RNG", lambda: restore_rng_state(rng_before)),
        ):
            try:
                rollback()
            except Exception as rollback_exc:
                rollback_failures.append(f"{label}: {rollback_exc}")
        detail = f"checkpoint restore failed: {exc}"
        if rollback_failures:
            detail += f"; rollback failures: {', '.join(rollback_failures)}"
        raise CheckpointRestoreError(detail) from exc


def restore_training_checkpoint(
    *,
    model: Any,
    server_optimizer: Any,
    user_approved_resume: bool,
    resume_checkpoint: os.PathLike[str] | str,
    requested_freeze_id: str,
    requested_method: str,
    requested_training_seed: int,
    requested_llm_rep: int,
    requested_partition_sha256: str,
    requested_config_sha256: str,
    requested_prompt_hashes: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    """Restore once and return the remaining engine state from that same safe load."""

    if user_approved_resume is not True:
        raise ResumeApprovalRequired("resume requires user_approved_resume=True")
    payload, start_round = _load_validated_resume_payload(
        resume_checkpoint=resume_checkpoint,
        requested_freeze_id=requested_freeze_id,
        requested_method=requested_method,
        requested_training_seed=requested_training_seed,
        requested_llm_rep=requested_llm_rep,
        requested_partition_sha256=requested_partition_sha256,
        requested_config_sha256=requested_config_sha256,
        requested_prompt_hashes=requested_prompt_hashes,
    )
    _restore_loaded_payload(
        payload=payload,
        model=model,
        server_optimizer=server_optimizer,
    )
    return int(start_round), {
        "previous_weights": _deep_clone(payload["previous_weights"]),
        "best_validation": _deep_clone(payload["best_validation"]),
        "best_model_state": _deep_clone(payload["best_model_state"]),
        "last_complete_round": int(payload["last_complete_round"]),
    }


__all__ = [
    "CHECKPOINT_FILENAME",
    "SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointFormatError",
    "CheckpointRestoreError",
    "ResumeApprovalRequired",
    "ResumeMismatchError",
    "build_checkpoint_payload",
    "capture_rng_state",
    "load_checkpoint",
    "restore_checkpoint",
    "restore_training_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "validate_resume",
]
