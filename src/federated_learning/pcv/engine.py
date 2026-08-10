"""Transactional one-round execution for the FMAS-PCV study.

The engine deliberately owns no network client and reads no credentials.  Agent
and training work enter through explicit injected collaborators, which also
makes the transaction boundary independently testable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import copy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from typing import Any

import torch

from ..server_optimizers import FedAvgServerOptimizer, FedYogiServerOptimizer
from .agents import DeepSeekCallError
from .candidates import (
    build_anchor_candidates,
    build_deterministic_candidates,
    deduplicate_candidates,
    weighted_average_state,
)
from .checkpoint import (
    build_checkpoint_payload,
    capture_rng_state,
    restore_rng_state,
    save_checkpoint,
)
from .gate import select_with_gate
from .schemas import CandidateAction, CandidateDecision, ClientTelemetry, LocalCandidateVote
from .telemetry import AppendOnlyTelemetry, sanitize_telemetry_value
from .voting import aggregate_candidate_votes


FORMAL_METHODS = frozenset(
    {
        "FEDAVG_STRICT",
        "FEDYOGI_STRICT",
        "DPCV_FEDYOGI",
        "SA_PCV_FEDYOGI",
        "FMAS_PCV_FEDYOGI",
    }
)
_FMAS_PROPOSERS = (
    "performance_proposer",
    "stability_proposer",
    "balance_proposer",
)
_SAFE_AGENT_FAILURE_CATEGORIES = frozenset(
    {"authentication", "connection", "http", "schema"}
)
_SAFE_AGENT_ROLES = frozenset(
    {
        "preflight",
        "diagnostic",
        "performance_proposer",
        "stability_proposer",
        "balance_proposer",
        "critic",
        "coordinator",
        "single_proposer",
        "orchestrator",
    }
)


def _plain_tensor_clone(value: torch.Tensor) -> torch.Tensor:
    """Clone a runtime tensor without retaining a tensor-subclass channel."""

    base = value if type(value) is torch.Tensor else value.as_subclass(torch.Tensor)
    return base.detach().clone()


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _plain_tensor_clone(value)
    if isinstance(value, Mapping):
        return {_clone(key): _clone(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone(item) for item in value)
    if type(value) in (type(None), bool, int, float, str, bytes, Path):
        return value
    return copy.deepcopy(value)


def _clone_tensor_state(
    value: Mapping[str, torch.Tensor],
    *,
    callback_boundary: bool = False,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("model state must be a non-empty mapping")
    cloned: dict[str, torch.Tensor] = {}
    for name, tensor in value.items():
        if type(name) is not str or not name:
            raise TypeError("model state keys must be non-empty exact strings")
        if callback_boundary and type(tensor) is not torch.Tensor:
            raise TypeError("callback model state values must be exact torch.Tensor instances")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("model state values must be tensors")
        cloned[name] = _plain_tensor_clone(tensor)
    return cloned


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(_clone(value))


class ExperimentRuntimeError(RuntimeError):
    """Sanitized description of a non-agent round failure."""

    category = "runtime"

    def __init__(self, exception_type: str, message: str):
        self.exception_type = str(exception_type)
        self.role = "engine"
        # This object may be shown interactively, but the durable pause report
        # intentionally records only the exception type (never arbitrary text).
        self.detail = str(message)
        super().__init__(f"{self.exception_type}: {self.detail}")


class ExperimentPaused(RuntimeError):
    """Fail-stop signal: the round was rolled back and execution must stop."""

    def __init__(
        self,
        failure: DeepSeekCallError | ExperimentRuntimeError,
        report_path: str | os.PathLike[str],
        *,
        rollback_errors: Sequence[str] = (),
        report_error: str | None = None,
        report_persisted: bool = True,
    ):
        if type(failure) not in (DeepSeekCallError, ExperimentRuntimeError):
            raise TypeError("failure must be an exact sanitized engine failure")
        self.failure = failure
        self.report_path = Path(report_path)
        self.rollback_errors = tuple(rollback_errors)
        self.report_error = report_error
        self.report_persisted = bool(report_persisted)
        super().__init__(f"experiment paused after {failure.category} failure")


class PauseReportPersistenceError(RuntimeError):
    """All built-in pause-report targets were unavailable."""

    def __init__(self, attempted_path: Path, error_types: Sequence[str]):
        self.attempted_path = attempted_path
        self.error_types = tuple(error_types)
        super().__init__("pause report could not be persisted")


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    method: str
    decision: CandidateDecision
    candidate_ids: tuple[str, ...]
    stronger_anchor_id: str
    aggregate_mape: Mapping[str, float]
    agent_call_count: int
    optimizer_telemetry: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregate_mape", _frozen_mapping(self.aggregate_mape))
        object.__setattr__(
            self,
            "optimizer_telemetry",
            _frozen_mapping(self.optimizer_telemetry),
        )

    @property
    def requested_candidate_id(self) -> str:
        return self.decision.requested_candidate_id

    @property
    def selected_candidate_id(self) -> str:
        return self.decision.selected_candidate_id

    @property
    def gate_status(self) -> str:
        return self.decision.gate_status


@dataclass
class _Preview:
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    telemetry: dict[str, Any]


@dataclass
class _RoundWork:
    round_index: int
    agent_calls: list[str] = field(default_factory=list)
    telemetry_records: list[dict[str, Any]] = field(default_factory=list)
    diagnostic: Mapping[str, Any] = field(default_factory=dict)
    critique: Mapping[str, Any] = field(default_factory=dict)
    telemetry_payload: Mapping[str, Any] = field(default_factory=dict)
    candidates: tuple[CandidateAction, ...] = ()
    previews: dict[str, _Preview] = field(default_factory=dict)
    votes: tuple[LocalCandidateVote, ...] = ()
    aggregate_mape: dict[str, float] = field(default_factory=dict)
    stronger_anchor_id: str = ""


@dataclass
class _RuntimeSnapshot:
    global_state: dict[str, torch.Tensor]
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    previous_weights: dict[str, float]
    best_validation: dict[str, Any]
    best_model_state: dict[str, torch.Tensor]
    last_complete_round: int
    telemetry_records: list[dict[str, Any]]
    pending_telemetry: list[dict[str, Any]]
    rng_state: dict[str, Any]
    artifacts: dict[Path, bytes | None]
    telemetry: "_TelemetrySnapshot | None"


@dataclass(frozen=True)
class _TelemetrySnapshot:
    path: Path
    existed: bool
    offset: int


class PCVEngine:
    """Execute one complete PCV round as an all-or-nothing transaction.

    ``agent_orchestrator`` is an explicit staged role caller for FMAS.  It must
    be callable (or provide ``call``) and is invoked for the six frozen roles.
    ``single_agent`` is a separate two-call SA dependency; a six-role FMAS
    orchestrator is never silently reused as SA.
    """

    def __init__(
        self,
        *,
        method: str,
        model: Any,
        server_optimizer: Any,
        sample_counts: Mapping[str, int],
        train_clients: Callable[..., Mapping[str, Mapping[str, torch.Tensor]]],
        collect_telemetry: Callable[..., Mapping[str, ClientTelemetry]],
        evaluate_candidates: Callable[..., Any],
        training_config: Mapping[str, Any] | None = None,
        training_seed: int = 0,
        fedyogi_lr_scale: float = 1.0,
        fedyogi_clip_norm: float | None = 1.0,
        previous_weights: Mapping[str, float] | None = None,
        best_validation: Mapping[str, Any] | None = None,
        best_model_state: Mapping[str, torch.Tensor] | None = None,
        last_complete_round: int = 0,
        agent_orchestrator: Any = None,
        single_agent: Any = None,
        gate_selector: Callable[..., CandidateDecision] = select_with_gate,
        telemetry_sink: Any = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        checkpoint_writer: Callable[..., Any] | None = None,
        pause_report_path: str | os.PathLike[str] | None = None,
        pause_report_writer: Callable[..., Any] | None = None,
        freeze_id: str = "development",
        llm_rep: int = 0,
        partition_sha256: str = "development-partition",
        config_sha256: str = "development-config",
        prompt_hashes: Mapping[str, str] | None = None,
    ) -> None:
        if method not in FORMAL_METHODS:
            raise ValueError(f"invalid formal method: {method!r}")
        if type(last_complete_round) is not int or last_complete_round < 0:
            raise ValueError("last_complete_round must be a non-negative exact integer")
        if type(training_seed) is not int:
            raise TypeError("training_seed must be an exact integer")
        for name, callback in (
            ("train_clients", train_clients),
            ("collect_telemetry", collect_telemetry),
            ("evaluate_candidates", evaluate_candidates),
            ("gate_selector", gate_selector),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        if not callable(getattr(model, "state_dict", None)) or not callable(
            getattr(model, "load_state_dict", None)
        ):
            raise TypeError("model must provide state_dict/load_state_dict")
        if not callable(getattr(server_optimizer, "get_optimizer_state", None)) or not callable(
            getattr(server_optimizer, "load_optimizer_state", None)
        ):
            raise TypeError("server_optimizer must provide state get/load methods")

        counts = dict(sample_counts)
        if not counts or any(
            type(client_id) is not str
            or not client_id
            or type(count) is not int
            or count <= 0
            for client_id, count in counts.items()
        ):
            raise ValueError("sample_counts must map client IDs to positive exact integers")
        if len(counts) > 20:
            raise ValueError("candidate weight bounds support at most twenty clients")

        self.method = method
        self.model = model
        self.server_optimizer = server_optimizer
        self.sample_counts = counts
        self.train_clients = train_clients
        self.collect_telemetry = collect_telemetry
        self.evaluate_candidates = evaluate_candidates
        self.training_config = _clone(training_config or {})
        self.training_seed = training_seed
        self.fedyogi_lr_scale = float(fedyogi_lr_scale)
        self.fedyogi_clip_norm = fedyogi_clip_norm
        self.agent_orchestrator = agent_orchestrator
        self.single_agent = single_agent
        self.gate_selector = gate_selector
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        if checkpoint_writer is not None and not callable(checkpoint_writer):
            raise TypeError("checkpoint_writer must be callable")
        if checkpoint_writer is not None and self.checkpoint_path is None:
            raise ValueError(
                "checkpoint_writer requires checkpoint_path so its durable output "
                "belongs to the round transaction"
            )
        self.checkpoint_writer = checkpoint_writer
        if pause_report_writer is not None and not callable(pause_report_writer):
            raise TypeError("pause_report_writer must be callable")
        self.pause_report_writer = pause_report_writer
        if pause_report_path is None:
            pause_report_path = (
                self.checkpoint_path.parent / "PAUSED.json"
                if self.checkpoint_path is not None
                else Path("PAUSED.json")
            )
        self.pause_report_path = Path(pause_report_path)
        if self.pause_report_path == self.checkpoint_path:
            raise ValueError("pause report and checkpoint paths must differ")

        self.telemetry_sink = self._validated_telemetry_sink(telemetry_sink)
        self._telemetry_path = (
            None
            if self.telemetry_sink is None
            else Path(self.telemetry_sink.path).resolve(strict=False)
        )
        if self._telemetry_path is not None and self._telemetry_path in {
            (
                None
                if self.checkpoint_path is None
                else self.checkpoint_path.resolve(strict=False)
            ),
            self.pause_report_path.resolve(strict=False),
        }:
            raise ValueError("telemetry path must be separate from checkpoint and pause report")

        self.freeze_id = str(freeze_id)
        self.llm_rep = int(llm_rep)
        self.partition_sha256 = str(partition_sha256)
        self.config_sha256 = str(config_sha256)
        self.prompt_hashes = dict(prompt_hashes or {"engine": "no-agent-prompts"})

        self.global_state = _clone_tensor_state(model.state_dict())
        model.load_state_dict(_clone_tensor_state(self.global_state))
        self.previous_weights = self._validated_initial_weights(previous_weights)
        self.best_validation = _clone(best_validation or {"mape": None, "round": None})
        self.best_model_state = _clone_tensor_state(
            self.global_state if best_model_state is None else best_model_state
        )
        self.last_complete_round = last_complete_round
        self.telemetry_records: list[dict[str, Any]] = []
        self.pending_telemetry: list[dict[str, Any]] = []
        self._active_work: _RoundWork | None = None

    @staticmethod
    def _validated_telemetry_sink(value: Any) -> AppendOnlyTelemetry | None:
        if value is None:
            return None
        if type(value) is not AppendOnlyTelemetry:
            raise TypeError(
                "telemetry_sink must be an exact path-backed AppendOnlyTelemetry "
                "transactional sink"
            )
        if not callable(getattr(value, "append", None)):
            raise TypeError("transactional telemetry sink must provide append()")
        path = Path(value.path).resolve(strict=False)
        if path.exists():
            metadata = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("telemetry path must be a local regular file")
        elif path.parent.exists() and not path.parent.is_dir():
            raise ValueError("telemetry parent must be a local directory")
        return value

    def _validated_initial_weights(
        self, value: Mapping[str, float] | None
    ) -> dict[str, float]:
        if value is None:
            total = sum(self.sample_counts.values())
            return {
                client_id: count / total for client_id, count in self.sample_counts.items()
            }
        weights = {client_id: float(weight) for client_id, weight in value.items()}
        if set(weights) != set(self.sample_counts) or not math.isclose(
            math.fsum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("previous_weights must match clients and sum to one")
        return weights

    def _artifact_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        if self.checkpoint_path is not None:
            paths.append(self.checkpoint_path)
        return tuple(paths)

    def _snapshot_telemetry(self) -> _TelemetrySnapshot | None:
        if self._telemetry_path is None:
            return None
        if Path(self.telemetry_sink.path).resolve(strict=False) != self._telemetry_path:
            raise ValueError("transactional telemetry path changed after validation")
        path = self._telemetry_path
        if not path.exists():
            return _TelemetrySnapshot(path=path, existed=False, offset=0)
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("telemetry path must remain a local regular file")
        return _TelemetrySnapshot(
            path=path,
            existed=True,
            offset=int(metadata.st_size),
        )

    def _snapshot_runtime(self) -> _RuntimeSnapshot:
        artifacts = {
            path: path.read_bytes() if path.is_file() else None
            for path in self._artifact_paths()
        }
        return _RuntimeSnapshot(
            global_state=_clone_tensor_state(self.global_state),
            model_state=_clone_tensor_state(self.model.state_dict()),
            optimizer_state=_clone(self.server_optimizer.get_optimizer_state()),
            previous_weights=dict(self.previous_weights),
            best_validation=_clone(self.best_validation),
            best_model_state=_clone_tensor_state(self.best_model_state),
            last_complete_round=self.last_complete_round,
            telemetry_records=_clone(self.telemetry_records),
            pending_telemetry=_clone(self.pending_telemetry),
            rng_state=capture_rng_state(),
            artifacts=artifacts,
            telemetry=self._snapshot_telemetry(),
        )

    @staticmethod
    def _restore_file(path: Path, content: bytes | None) -> None:
        if content is None:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".rollback", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _restore_runtime(self, snapshot: _RuntimeSnapshot) -> tuple[str, ...]:
        errors: list[str] = []
        restorers = (
            ("model", lambda: self.model.load_state_dict(_clone_tensor_state(snapshot.model_state))),
            (
                "optimizer",
                lambda: self.server_optimizer.load_optimizer_state(_clone(snapshot.optimizer_state)),
            ),
            ("RNG", lambda: restore_rng_state(snapshot.rng_state)),
        )
        for label, restore in restorers:
            try:
                restore()
            except Exception as error:  # keep the original round failure authoritative
                errors.append(f"{label}:{type(error).__name__}")
        self.global_state = _clone_tensor_state(snapshot.global_state)
        self.previous_weights = dict(snapshot.previous_weights)
        self.best_validation = _clone(snapshot.best_validation)
        self.best_model_state = _clone_tensor_state(snapshot.best_model_state)
        self.last_complete_round = snapshot.last_complete_round
        self.telemetry_records = _clone(snapshot.telemetry_records)
        self.pending_telemetry = _clone(snapshot.pending_telemetry)
        self._active_work = None
        for path, content in snapshot.artifacts.items():
            try:
                self._restore_file(path, content)
            except Exception as error:
                errors.append(f"artifact:{type(error).__name__}")
        if snapshot.telemetry is not None:
            try:
                self._restore_telemetry(snapshot.telemetry)
            except Exception as error:
                errors.append(f"telemetry:{type(error).__name__}")
        return tuple(errors)

    @staticmethod
    def _restore_telemetry(snapshot: _TelemetrySnapshot) -> None:
        path = snapshot.path
        if not snapshot.existed:
            if path.exists():
                metadata = path.lstat()
                if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("refusing to remove a non-regular telemetry path")
                path.unlink()
            return
        if not path.exists():
            raise FileNotFoundError("transactional telemetry file disappeared")
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("telemetry path ceased to be a regular file")
        if metadata.st_size < snapshot.offset:
            raise OSError("telemetry file shrank during the round transaction")
        with open(path, "r+b", buffering=0) as stream:
            os.ftruncate(stream.fileno(), snapshot.offset)
            stream.flush()
            os.fsync(stream.fileno())

    def _train_clients(self, round_index: int) -> dict[str, dict[str, torch.Tensor]]:
        result = self.train_clients(
            round_index=round_index,
            global_state=_clone_tensor_state(self.global_state),
            training_config=_clone(self.training_config),
            seed=self.training_seed + round_index,
        )
        if not isinstance(result, Mapping) or set(result) != set(self.sample_counts):
            raise ValueError("local updates must exactly match configured clients")
        return {
            client_id: _clone_tensor_state(result[client_id], callback_boundary=True)
            for client_id in self.sample_counts
        }

    def _collect_client_telemetry(
        self,
        round_index: int,
        local_updates: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> dict[str, ClientTelemetry]:
        result = self.collect_telemetry(
            round_index=round_index,
            global_state=_clone_tensor_state(self.global_state),
            local_updates={
                client_id: _clone_tensor_state(state)
                for client_id, state in local_updates.items()
            },
        )
        if not isinstance(result, Mapping) or set(result) != set(self.sample_counts):
            raise ValueError("client telemetry must exactly match configured clients")
        output: dict[str, ClientTelemetry] = {}
        for client_id in self.sample_counts:
            item = result[client_id]
            if type(item) is not ClientTelemetry:
                raise TypeError("telemetry callback must return exact ClientTelemetry records")
            if item.client_id != client_id or item.sample_count != self.sample_counts[client_id]:
                raise ValueError("telemetry identity/sample count mismatch")
            output[client_id] = item
        return output

    def _build_anchors(
        self, local_updates: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> list[CandidateAction]:
        del local_updates
        anchors = build_anchor_candidates(
            self.sample_counts,
            self.fedyogi_lr_scale,
            self.fedyogi_clip_norm,
        )
        if self.method == "FEDAVG_STRICT":
            return [anchors[0]]
        if self.method == "FEDYOGI_STRICT":
            return [anchors[1]]
        return anchors

    def _telemetry_payload(
        self, round_index: int, telemetry: Mapping[str, ClientTelemetry]
    ) -> dict[str, Any]:
        return {
            "round_index": round_index,
            "clients": [telemetry[client_id].to_prompt_dict() for client_id in self.sample_counts],
        }

    @staticmethod
    def _candidate_prompt_dict(candidate: CandidateAction) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "weights": dict(candidate.weights),
            "server_optimizer": candidate.server_optimizer,
            "server_lr_scale": candidate.server_lr_scale,
            "update_clip_norm": candidate.update_clip_norm,
            "source": candidate.source,
            "rationale": candidate.rationale,
        }

    @staticmethod
    def _vote_prompt_dict(vote: LocalCandidateVote) -> dict[str, Any]:
        return {
            "client_id": vote.client_id,
            "candidate_id": vote.candidate_id,
            "sample_count": vote.sample_count,
            "val_mape": vote.val_mape,
            "val_rmse": vote.val_rmse,
            "relative_mape": vote.relative_mape,
            "relative_rmse": vote.relative_rmse,
            "rank": vote.rank,
            "confidence": vote.confidence,
            "catastrophic_degradation": vote.catastrophic_degradation,
        }

    @staticmethod
    def _role_callable(dependency: Any, *, dependency_name: str) -> Callable[..., Any]:
        caller = getattr(dependency, "call", None)
        if callable(caller):
            return caller
        if callable(dependency):
            return dependency
        raise TypeError(f"{dependency_name} must be a staged callable or provide call()")

    def _call_role(self, dependency: Any, role: str, payload: Mapping[str, Any]) -> Any:
        if self._active_work is None:
            raise RuntimeError("agent call outside active round")
        caller = self._role_callable(dependency, dependency_name="agent dependency")
        result = caller(role=role, payload=_clone(payload))
        self._active_work.agent_calls.append(role)
        return result

    def _validated_proposals(
        self,
        value: Any,
        *,
        role: str,
    ) -> list[CandidateAction]:
        if isinstance(value, Mapping) and "proposals" in value:
            value = value["proposals"]
        if isinstance(value, CandidateAction) or isinstance(value, (str, bytes)):
            value = (value,)
        if not isinstance(value, Sequence):
            raise TypeError(f"{role} must return a candidate sequence")
        proposals = list(value)
        for candidate in proposals:
            if type(candidate) is not CandidateAction:
                raise TypeError("agent proposals must be exact CandidateAction records")
            candidate.validate(tuple(self.sample_counts))
            if candidate.source == "anchor" or candidate.candidate_id.startswith("anchor_"):
                raise ValueError("agent proposals cannot claim an anchor identity")
        return proposals

    def _build_method_proposals(
        self,
        round_index: int,
        client_telemetry: Mapping[str, ClientTelemetry],
        anchors: Sequence[CandidateAction],
    ) -> list[CandidateAction]:
        if self.method in {"FEDAVG_STRICT", "FEDYOGI_STRICT", "DPCV_FEDYOGI"}:
            return []
        payload = self._telemetry_payload(round_index, client_telemetry)
        self._active_work.telemetry_payload = _clone(payload)
        if self.method == "SA_PCV_FEDYOGI":
            if self.single_agent is None:
                raise TypeError("SA_PCV_FEDYOGI requires an explicit single_agent dependency")
            response = self._call_role(self.single_agent, "single_proposer", payload)
            if isinstance(response, Mapping) and "diagnostic" in response:
                self._active_work.diagnostic = _clone(response["diagnostic"])
            return self._validated_proposals(response, role="single_proposer")

        if self.agent_orchestrator is None:
            raise TypeError("FMAS_PCV_FEDYOGI requires a staged six-role agent_orchestrator")
        diagnostic = self._call_role(self.agent_orchestrator, "diagnostic", payload)
        if not isinstance(diagnostic, Mapping):
            raise TypeError("diagnostic role must return a mapping")
        self._active_work.diagnostic = _clone(diagnostic)
        proposals: list[CandidateAction] = []
        for role in _FMAS_PROPOSERS:
            response = self._call_role(
                self.agent_orchestrator,
                role,
                {**payload, "diagnostic": _clone(diagnostic)},
            )
            proposals.extend(self._validated_proposals(response, role=role))
        ids = [candidate.candidate_id for candidate in proposals]
        if len(ids) != len(set(ids)):
            raise ValueError("agent proposal candidate IDs must be unique")
        critique = self._call_role(
            self.agent_orchestrator,
            "critic",
            {
                **payload,
                "diagnostic": _clone(diagnostic),
                "candidates": [
                    self._candidate_prompt_dict(candidate)
                    for candidate in proposals
                ],
            },
        )
        if not isinstance(critique, Mapping):
            raise TypeError("critic role must return a mapping")
        self._active_work.critique = _clone(critique)
        accepted = critique.get("accepted_candidate_ids", ids)
        if type(accepted) not in (list, tuple) or any(type(item) is not str for item in accepted):
            raise TypeError("critic accepted_candidate_ids must be a string sequence")
        if not set(accepted).issubset(ids):
            raise ValueError("critic accepted an unknown candidate")
        rejected = critique.get("rejected", [])
        if type(rejected) not in (list, tuple):
            raise TypeError("critic rejected evidence must be a sequence")
        rejected_ids = []
        for item in rejected:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"candidate_id", "reason"}
                or type(item["candidate_id"]) is not str
                or type(item["reason"]) is not str
                or not item["reason"].strip()
            ):
                raise TypeError("critic rejected evidence is invalid")
            rejected_ids.append(item["candidate_id"])
        if (
            set(accepted) & set(rejected_ids)
            or set(accepted) | set(rejected_ids) != set(ids)
        ):
            raise ValueError("critic must classify every proposed candidate exactly once")
        self._active_work.critique = {
            "accepted_candidate_ids": list(accepted),
            "rejected": [dict(item) for item in rejected],
        }
        accepted_set = set(accepted)
        return [candidate for candidate in proposals if candidate.candidate_id in accepted_set]

    def _all_candidates(
        self,
        anchors: Sequence[CandidateAction],
        proposals: Sequence[CandidateAction],
        telemetry: Mapping[str, ClientTelemetry],
    ) -> list[CandidateAction]:
        if self.method == "DPCV_FEDYOGI":
            return build_deterministic_candidates(
                self.sample_counts,
                self.previous_weights,
                telemetry,
                self.fedyogi_lr_scale,
                self.fedyogi_clip_norm,
                budget=8,
            )
        combined = [*anchors, *proposals]
        ids = [candidate.candidate_id for candidate in combined]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique before deduplication")
        return deduplicate_candidates(combined, budget=8)

    def _clone_fedyogi_optimizer(self) -> Any:
        state = _clone(self.server_optimizer.get_optimizer_state())
        if isinstance(self.server_optimizer, FedYogiServerOptimizer):
            optimizer = FedYogiServerOptimizer()
            optimizer.load_optimizer_state(state)
            return optimizer
        try:
            optimizer = copy.deepcopy(self.server_optimizer)
            optimizer.load_optimizer_state(state)
            return optimizer
        except Exception as error:
            raise TypeError("FedYogi previews require a cloneable server optimizer") from error

    def _preview_candidates(
        self,
        candidates: Sequence[CandidateAction],
        local_updates: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> dict[str, _Preview]:
        previews: dict[str, _Preview] = {}
        base_optimizer_state = _clone(self.server_optimizer.get_optimizer_state())
        for candidate in candidates:
            averaged = weighted_average_state(local_updates, candidate.weights)
            if candidate.server_optimizer == "fedavg":
                optimizer = FedAvgServerOptimizer()
                model_state, telemetry = optimizer.step(
                    _clone_tensor_state(self.global_state),
                    averaged,
                    server_lr_scale=candidate.server_lr_scale,
                )
                optimizer_state = _clone(base_optimizer_state)
                optimizer_state["m"] = {}
                optimizer_state["v"] = {}
            elif candidate.server_optimizer == "fedyogi":
                optimizer = self._clone_fedyogi_optimizer()
                model_state, telemetry = optimizer.step(
                    _clone_tensor_state(self.global_state),
                    averaged,
                    server_lr_scale=candidate.server_lr_scale,
                    update_clip_norm_override=candidate.update_clip_norm,
                )
                optimizer_state = _clone(optimizer.get_optimizer_state())
            else:  # CandidateAction.validate normally makes this unreachable.
                raise ValueError("unsupported candidate server optimizer")
            previews[candidate.candidate_id] = _Preview(
                model_state=_clone_tensor_state(model_state),
                optimizer_state=optimizer_state,
                telemetry=_clone(telemetry),
            )
        if set(previews) != {candidate.candidate_id for candidate in candidates}:
            raise ValueError("candidate/state IDs are incomplete")
        return previews

    def _evaluate_on_clients(
        self,
        candidate_states: Mapping[str, _Preview],
        anchors: Sequence[CandidateAction],
    ) -> tuple[tuple[LocalCandidateVote, ...], dict[str, float], str]:
        state_payload = {
            candidate_id: _clone_tensor_state(preview.model_state)
            for candidate_id, preview in candidate_states.items()
        }
        anchor_ids = tuple(anchor.candidate_id for anchor in anchors)
        result = self.evaluate_candidates(
            candidate_states=state_payload,
            anchor_candidate_ids=anchor_ids,
        )
        supplied_scores = None
        supplied_stronger = None
        if isinstance(result, Mapping) and "votes" in result:
            supplied_scores = result.get("aggregate_mape")
            supplied_stronger = result.get("stronger_anchor_id")
            result = result["votes"]
        elif type(result) is tuple and len(result) == 3 and not all(
            type(item) is LocalCandidateVote for item in result
        ):
            result, supplied_scores, supplied_stronger = result
        votes = tuple(result)
        if any(type(vote) is not LocalCandidateVote for vote in votes):
            raise TypeError("candidate evaluator must return exact LocalCandidateVote records")
        summaries = aggregate_candidate_votes(votes)
        expected_ids = set(candidate_states)
        if set(summaries) != expected_ids:
            raise ValueError("vote candidate IDs must exactly match candidate state IDs")
        aggregate_mape = {
            candidate_id: float(summary.weighted_mape)
            for candidate_id, summary in summaries.items()
        }
        if supplied_scores is not None:
            normalized = {key: float(value) for key, value in supplied_scores.items()}
            if set(normalized) != expected_ids or any(
                not math.isclose(normalized[key], aggregate_mape[key], rel_tol=0.0, abs_tol=1e-12)
                for key in expected_ids
            ):
                raise ValueError("supplied aggregate MAPE does not match votes")
        computed_stronger = min(anchor_ids, key=lambda item: (aggregate_mape[item], item))
        stronger = computed_stronger if supplied_stronger is None else supplied_stronger
        if stronger != computed_stronger:
            raise ValueError("stronger anchor must be the best evaluated anchor")
        return votes, aggregate_mape, stronger

    def _coordinate(
        self,
        round_index: int,
        candidates: Sequence[CandidateAction],
        votes: Sequence[LocalCandidateVote],
    ) -> str:
        if self._active_work is None:
            raise RuntimeError("coordination outside active round")
        if self.method in {"FEDAVG_STRICT", "FEDYOGI_STRICT", "DPCV_FEDYOGI"}:
            return min(
                (candidate.candidate_id for candidate in candidates),
                key=lambda candidate_id: (
                    self._active_work.aggregate_mape[candidate_id],
                    candidate_id,
                ),
            )
        dependency = self.single_agent if self.method == "SA_PCV_FEDYOGI" else self.agent_orchestrator
        anchor_ids = [
            candidate.candidate_id
            for candidate in candidates
            if candidate.source == "anchor"
        ]
        non_anchor_ids = [
            candidate.candidate_id
            for candidate in candidates
            if candidate.source != "anchor"
        ]
        critique = (
            {
                "accepted_candidate_ids": non_anchor_ids,
                "rejected": [],
            }
            if self.method == "SA_PCV_FEDYOGI"
            else _clone(self._active_work.critique)
        )
        response = self._call_role(
            dependency,
            "coordinator",
            {
                **_clone(self._active_work.telemetry_payload),
                "diagnostic": _clone(self._active_work.diagnostic),
                "candidates": [
                    self._candidate_prompt_dict(candidate)
                    for candidate in candidates
                ],
                "critique": critique,
                "anchor_candidate_ids": anchor_ids,
                "client_votes": [self._vote_prompt_dict(vote) for vote in votes],
            },
        )
        if isinstance(response, Mapping):
            response = response.get("requested_candidate_id", response.get("selected_candidate_id"))
        if type(response) is not str or not response:
            raise TypeError("coordinator must return a requested candidate ID")
        return response

    def _gate(
        self,
        requested_id: str,
        candidates: Sequence[CandidateAction],
        votes: Sequence[LocalCandidateVote],
    ) -> CandidateDecision:
        if self._active_work is None:
            raise RuntimeError("gate outside active round")
        decision = self.gate_selector(
            requested_candidate_id=requested_id,
            candidates={candidate.candidate_id: candidate for candidate in candidates},
            votes=tuple(votes),
            aggregate_mape=dict(self._active_work.aggregate_mape),
            previous_weights=dict(self.previous_weights),
            stronger_anchor_id=self._active_work.stronger_anchor_id,
        )
        if type(decision) is not CandidateDecision:
            raise TypeError("gate must return an exact CandidateDecision")
        if decision.selected_candidate_id not in self._active_work.previews:
            raise ValueError("selected candidate state does not exist")
        return decision

    def _round_record(
        self,
        round_index: int,
        decision: CandidateDecision,
        selected: CandidateAction,
        preview: _Preview,
    ) -> dict[str, Any]:
        return {
            "event": "round_committed",
            "round_index": round_index,
            "method": self.method,
            "requested_candidate_id": decision.requested_candidate_id,
            "selected_candidate_id": decision.selected_candidate_id,
            "gate_status": decision.gate_status,
            "candidate_count": len(self._active_work.candidates),
            "agent_call_count": len(self._active_work.agent_calls),
            "stronger_anchor_id": self._active_work.stronger_anchor_id,
            "selected_mape": self._active_work.aggregate_mape[selected.candidate_id],
            "server_optimizer": selected.server_optimizer,
            "fedyogi_moments_reset": selected.server_optimizer == "fedavg",
            "optimizer": _clone(preview.telemetry),
        }

    def _append_telemetry(self, record: dict[str, Any]) -> None:
        if self.telemetry_sink is None:
            return
        if Path(self.telemetry_sink.path).resolve(strict=False) != self._telemetry_path:
            raise ValueError("transactional telemetry path changed during the round")
        self.telemetry_sink.append(_clone(record))

    def _checkpoint_payload(self, round_index: int) -> dict[str, Any]:
        return build_checkpoint_payload(
            round_index=round_index,
            freeze_id=self.freeze_id,
            method=self.method,
            training_seed=self.training_seed,
            llm_rep=self.llm_rep,
            global_model_state=self.global_state,
            server_optimizer_state=self.server_optimizer.get_optimizer_state(),
            previous_weights=self.previous_weights,
            best_validation=self.best_validation,
            best_model_state=self.best_model_state,
            partition_sha256=self.partition_sha256,
            config_sha256=self.config_sha256,
            prompt_hashes=self.prompt_hashes,
        )

    def _write_checkpoint(self, round_index: int) -> Path | None:
        if self.checkpoint_path is None and self.checkpoint_writer is None:
            return None
        payload = self._checkpoint_payload(round_index)
        if self.checkpoint_writer is not None:
            result = self.checkpoint_writer(self.checkpoint_path, payload)
            reported_path = self.checkpoint_path if result is None else Path(result)
            if reported_path.resolve(strict=False) != self.checkpoint_path.resolve(
                strict=False
            ):
                raise ValueError(
                    "checkpoint_writer must report the transactionally managed "
                    "checkpoint_path"
                )
            return self.checkpoint_path
        return save_checkpoint(self.checkpoint_path, payload)

    def _commit_round(
        self,
        round_index: int,
        decision: CandidateDecision,
        candidate_states: Mapping[str, _Preview],
    ) -> tuple[dict[str, Any], Path | None]:
        selected_candidates = {
            candidate.candidate_id: candidate for candidate in self._active_work.candidates
        }
        selected = selected_candidates[decision.selected_candidate_id]
        preview = candidate_states[decision.selected_candidate_id]
        # Runtime becomes visible before checkpointing; any later failure enters
        # the enclosing rollback and restores both runtime and durable artifacts.
        self.model.load_state_dict(_clone_tensor_state(preview.model_state))
        self.global_state = _clone_tensor_state(preview.model_state)
        self.server_optimizer.load_optimizer_state(_clone(preview.optimizer_state))
        self.previous_weights = {
            client_id: float(weight) for client_id, weight in selected.weights.items()
        }
        selected_mape = self._active_work.aggregate_mape[selected.candidate_id]
        old_best = self.best_validation.get("mape")
        if old_best is None or selected_mape < float(old_best):
            self.best_validation = {"mape": selected_mape, "round": round_index}
            self.best_model_state = _clone_tensor_state(preview.model_state)
        self.last_complete_round = round_index
        record = self._round_record(round_index, decision, selected, preview)
        self.pending_telemetry.append(_clone(record))
        self._append_telemetry(record)
        self.telemetry_records.append(_clone(record))
        self.pending_telemetry.clear()
        checkpoint_path = self._write_checkpoint(round_index)
        return record, checkpoint_path

    @staticmethod
    def _sanitized_failure(
        failure: Any,
    ) -> DeepSeekCallError | ExperimentRuntimeError:
        if type(failure) is DeepSeekCallError:
            category = getattr(failure, "category", None)
            role = getattr(failure, "role", None)
            if (
                type(category) is str
                and category in _SAFE_AGENT_FAILURE_CATEGORIES
                and type(role) is str
                and role in _SAFE_AGENT_ROLES
            ):
                return DeepSeekCallError(
                    category,
                    role,
                    "round stopped after a sanitized agent failure",
                )
        elif type(failure) is ExperimentRuntimeError:
            exception_type = getattr(failure, "exception_type", None)
            if (
                type(exception_type) is str
                and exception_type.isidentifier()
                and len(exception_type) <= 80
            ):
                return ExperimentRuntimeError(
                    exception_type,
                    "round stopped after a sanitized runtime failure",
                )
        return ExperimentRuntimeError(
            "InvalidExperimentPausedFailure",
            "nested ExperimentPaused.failure was invalid",
        )

    def _pause_report_record(
        self,
        round_index: int,
        failure: DeepSeekCallError | ExperimentRuntimeError,
        *,
        primary_report_error: Exception | None = None,
    ) -> dict[str, Any]:
        failure_record = {
            "category": failure.category,
            "role": getattr(failure, "role", "engine"),
            "exception_type": (
                failure.exception_type
                if isinstance(failure, ExperimentRuntimeError)
                else type(failure).__name__
            ),
        }
        report: dict[str, Any] = {
            "status": "paused",
            "failed_round": round_index,
            "last_complete_round": self.last_complete_round,
            "method": self.method,
            "failure": failure_record,
        }
        if primary_report_error is not None:
            report["primary_report_error"] = {
                "exception_type": type(primary_report_error).__name__,
            }
        return sanitize_telemetry_value(report)

    @staticmethod
    def _atomic_write_pause_report(target: Path, report: Mapping[str, Any]) -> Path:
        target = Path(target)
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise ValueError("pause report target must be a regular file")
        encoded = (json.dumps(report, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def _write_pause_report(
        self,
        round_index: int,
        failure: DeepSeekCallError | ExperimentRuntimeError,
    ) -> Path:
        report = self._pause_report_record(round_index, failure)
        if self.pause_report_writer is None:
            return self._atomic_write_pause_report(self.pause_report_path, report)
        result = self.pause_report_writer(self.pause_report_path, _clone(report))
        path = self.pause_report_path if result is None else Path(result)
        self._validate_pause_report(path, report)
        return path

    @staticmethod
    def _validate_pause_report(path: Path, expected: Mapping[str, Any]) -> None:
        if not path.exists() or path.is_symlink() or not path.is_file():
            raise OSError("primary pause writer did not persist a regular report")
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise OSError("primary pause writer persisted an invalid report") from error
        if actual != expected:
            raise OSError("primary pause writer persisted a stale or nonstandard report")

    def _fallback_pause_paths(self, round_index: int) -> tuple[Path, ...]:
        candidates: list[Path] = []
        if self.checkpoint_path is not None:
            candidates.append(self.checkpoint_path.parent / "PAUSED.json")
        candidates.append(self.pause_report_path.parent / "PAUSED.json")
        identity = sha256(
            (
                f"{self.method}|{round_index}|"
                f"{self.pause_report_path.resolve(strict=False)}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        candidates.append(Path(tempfile.gettempdir()) / f"pcv-{identity}-PAUSED.json")
        unique: list[Path] = []
        for candidate in candidates:
            if candidate not in unique and candidate != self.checkpoint_path:
                unique.append(candidate)
        return tuple(unique)

    def _write_fallback_pause_report(
        self,
        round_index: int,
        failure: DeepSeekCallError | ExperimentRuntimeError,
        primary_error: Exception,
    ) -> tuple[Path, bool, PauseReportPersistenceError | None]:
        report = self._pause_report_record(
            round_index,
            failure,
            primary_report_error=primary_error,
        )
        errors: list[str] = []
        attempted_path = self.pause_report_path
        for target in self._fallback_pause_paths(round_index):
            attempted_path = target
            try:
                path = self._atomic_write_pause_report(target, report)
                if path.is_file():
                    return path, True, None
                raise OSError("fallback pause report did not exist after write")
            except Exception as error:
                errors.append(type(error).__name__)
        return (
            attempted_path,
            False,
            PauseReportPersistenceError(attempted_path, errors),
        )

    def _pause_after_failure(
        self,
        *,
        round_index: int,
        failure: DeepSeekCallError | ExperimentRuntimeError,
        snapshot: _RuntimeSnapshot,
        cause: Exception,
    ) -> ExperimentPaused:
        rollback_errors = self._restore_runtime(snapshot)
        failure = self._sanitized_failure(failure)
        report_path = self.pause_report_path
        report_error: str | None = None
        report_persisted = False
        try:
            report_path = self._write_pause_report(round_index, failure)
            report_persisted = report_path.is_file()
        except Exception as primary_error:
            report_error = type(primary_error).__name__
            report_path, report_persisted, persistence_error = (
                self._write_fallback_pause_report(
                    round_index,
                    failure,
                    primary_error,
                )
            )
            if persistence_error is not None:
                rollback_errors = (
                    *rollback_errors,
                    f"pause_report:{type(persistence_error).__name__}",
                )
        paused = ExperimentPaused(
            failure,
            report_path,
            rollback_errors=rollback_errors,
            report_error=report_error,
            report_persisted=report_persisted,
        )
        paused.__cause__ = cause
        return paused

    def run_round(self, round_index: int) -> RoundResult:
        snapshot = self._snapshot_runtime()
        try:
            if type(round_index) is not int:
                raise TypeError("round_index must be an exact integer")
            expected = self.last_complete_round + 1
            if round_index != expected:
                raise ValueError(f"round_index must be contiguous: expected {expected}")
            self._active_work = _RoundWork(round_index=round_index)
            local_updates = self._train_clients(round_index)
            client_telemetry = self._collect_client_telemetry(round_index, local_updates)
            anchors = self._build_anchors(local_updates)
            proposals = self._build_method_proposals(
                round_index, client_telemetry, anchors
            )
            candidates = self._all_candidates(anchors, proposals, client_telemetry)
            self._active_work.candidates = tuple(candidates)
            previews = self._preview_candidates(candidates, local_updates)
            self._active_work.previews = previews
            votes, aggregate_mape, stronger_anchor = self._evaluate_on_clients(
                previews, anchors
            )
            self._active_work.votes = votes
            self._active_work.aggregate_mape = aggregate_mape
            self._active_work.stronger_anchor_id = stronger_anchor
            requested_id = self._coordinate(round_index, candidates, votes)
            decision = self._gate(requested_id, candidates, votes)
            record, checkpoint_path = self._commit_round(
                round_index, decision, previews
            )
            result = RoundResult(
                round_index=round_index,
                method=self.method,
                decision=decision,
                candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
                stronger_anchor_id=stronger_anchor,
                aggregate_mape=aggregate_mape,
                agent_call_count=len(self._active_work.agent_calls),
                optimizer_telemetry=record["optimizer"],
                checkpoint_path=checkpoint_path,
            )
            self._active_work = None
            return result
        except ExperimentPaused as paused_failure:
            nested_failure = self._sanitized_failure(
                getattr(paused_failure, "failure", None)
            )
            raise self._pause_after_failure(
                round_index=round_index,
                failure=nested_failure,
                snapshot=snapshot,
                cause=paused_failure,
            ) from paused_failure
        except DeepSeekCallError as failure:
            raise self._pause_after_failure(
                round_index=round_index,
                failure=failure,
                snapshot=snapshot,
                cause=failure,
            ) from failure
        except Exception as failure:
            wrapped = ExperimentRuntimeError(type(failure).__name__, str(failure))
            raise self._pause_after_failure(
                round_index=round_index,
                failure=wrapped,
                snapshot=snapshot,
                cause=failure,
            ) from failure


__all__ = [
    "FORMAL_METHODS",
    "ExperimentPaused",
    "ExperimentRuntimeError",
    "PauseReportPersistenceError",
    "PCVEngine",
    "RoundResult",
]
