"""Concrete client-local runtime used by the canonical strict runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data_preprocessing import load_strict_partition_frames
from src.models import CostEstimationMLP

from ..server_optimizers import FedYogiServerOptimizer
from .agents import (
    StrictDeepSeekClient,
    validate_coordinator_response,
    validate_critic_response,
    validate_diagnostic_response,
    validate_proposer_response,
)
from .client_evaluation import (
    MetricSums,
    aggregate_metric_sums,
    compute_metric_sums,
)
from .checkpoint import restore_training_checkpoint
from .engine import PCVEngine
from .protocol import ClientDataVault, LocalTrainingResult
from .provider_config import deepseek_client_settings_from_provenance
from .schemas import CandidateAction, ClientTelemetry
from .telemetry import AppendOnlyTelemetry
from .voting import aggregate_candidate_votes


_deepseek_settings_from_provenance = deepseek_client_settings_from_provenance


@dataclass(frozen=True, slots=True)
class VaultBundle:
    vaults: Mapping[str, ClientDataVault]
    train_sample_counts: Mapping[str, int]
    dataset_sha256: str
    partition_sha256: str
    install_diagnostics: Callable[..., None]


def _exact_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"configuration must be an exact mapping: {path}")
    return value


def _model_factory(config: Mapping[str, Any]) -> CostEstimationMLP:
    architecture = config["model"]["architecture"]
    return CostEstimationMLP(
        input_dim=int(architecture["input_dim"]),
        hidden_dims=list(architecture["hidden_dims"]),
        output_dim=int(architecture["output_dim"]),
        activation=str(architecture["activation"]),
        dropout=float(architecture["dropout"]),
    )


def _to_private_dataset(
    frame,
    *,
    preprocessor,
    feature_columns: Sequence[str],
    target_column: str,
) -> TensorDataset:
    features, target = preprocessor.transform(
        frame[list(feature_columns)], frame[target_column]
    )
    return TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(target, dtype=torch.float32),
    )


def _predict_metric_sums(
    dataset: TensorDataset,
    model_state: Mapping[str, torch.Tensor],
    *,
    config: Mapping[str, Any],
    inverse_transform,
) -> MetricSums:
    model = _model_factory(config)
    model.load_state_dict({key: value.detach().clone() for key, value in model_state.items()})
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for features, target in DataLoader(dataset, batch_size=128, shuffle=False):
            predictions.append(model(features).detach().cpu())
            targets.append(target.detach().cpu())
    return compute_metric_sums(
        torch.cat(targets).numpy(),
        torch.cat(predictions).numpy(),
        inverse_transform=inverse_transform,
    )


def _flatten_delta(
    local_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    values = [
        (local_state[key] - global_state[key]).detach().reshape(-1).float()
        for key in global_state
        if torch.is_floating_point(global_state[key])
    ]
    if not values:
        raise ValueError("model state has no floating-point parameters")
    return torch.cat(values)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator <= 0.0:
        return 0.0
    value = float(torch.dot(left, right) / denominator)
    return max(-1.0, min(1.0, value))


def build_client_vaults(
    *,
    project_root: Path,
    base_config: Mapping[str, Any],
    method_config: Mapping[str, Any],
    phase: str = "development",
) -> VaultBundle:
    """Create vaults locally, then expose only parameter and aggregate DTO methods."""

    if phase not in {"development", "formal_train", "formal_evaluate"}:
        raise ValueError("invalid strict runtime phase")
    loaded_partitions = (
        {"train", "locked_test"}
        if phase == "formal_evaluate"
        else {"train", "controller_validation"}
    )

    partition_path = project_root / "results/manifests/strict_partition_v1.csv"
    loaded = load_strict_partition_frames(
        dict(base_config),
        str(partition_path),
        allowed_partitions=loaded_partitions,
        sealed_data_directory=project_root / "Data/strict_partition_v1",
    )
    data_config = base_config["scene_c"]["data"]
    feature_columns = tuple(data_config["feature_columns"])
    target_column = str(data_config["target_column"])
    runtime_state: dict[str, dict[str, Any]] = {}
    vaults: dict[str, ClientDataVault] = {}
    train_counts: dict[str, int] = {}

    for client_id, partitions in sorted(loaded.client_frames.items()):
        datasets = {
            name: _to_private_dataset(
                partitions[name],
                preprocessor=loaded.preprocessor,
                feature_columns=feature_columns,
                target_column=target_column,
            )
            for name in loaded_partitions
        }
        train_counts[client_id] = len(datasets["train"])
        state: dict[str, Any] = {
            "train_loss": None,
            "global_state": None,
            "diagnostics": {"update_norm": 0.0, "cosine_to_mean": 0.0, "cosine_to_previous": 0.0},
        }
        runtime_state[client_id] = state

        def train_fn(dataset, global_state, training_config, seed, *, _state=state):
            if type(seed) is not int:
                raise TypeError("client training seed must be an exact integer")
            torch.manual_seed(seed)
            model = _model_factory(base_config)
            cloned_global = {
                key: value.detach().clone() for key, value in global_state.items()
            }
            model.load_state_dict(cloned_global)
            model.train()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(training_config["client_learning_rate"]),
            )
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                dataset,
                batch_size=int(training_config["batch_size"]),
                shuffle=True,
                generator=generator,
            )
            loss_sum = 0.0
            observations = 0
            for _ in range(int(training_config["local_epochs"])):
                for features, target in loader:
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(features)
                    loss = torch.nn.functional.mse_loss(prediction, target)
                    loss.backward()
                    optimizer.step()
                    batch_size = int(target.shape[0])
                    loss_sum += float(loss.detach()) * batch_size
                    observations += batch_size
            if observations <= 0:
                raise ValueError("client train partition is empty")
            train_loss = loss_sum / observations
            _state["train_loss"] = train_loss
            _state["global_state"] = cloned_global
            return LocalTrainingResult(
                model_state={
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                },
                sample_count=len(dataset),
                train_loss=train_loss,
            )

        def metric_sums_fn(dataset, model_state):
            return _predict_metric_sums(
                dataset,
                model_state,
                config=base_config,
                inverse_transform=loaded.preprocessor.inverse_transform_target,
            )

        def telemetry_fn(_client_id, dataset, model_state, *, _state=state, _count=len(datasets["train"])):
            if _state["train_loss"] is None:
                raise RuntimeError("telemetry requested before local training")
            metrics = aggregate_metric_sums([metric_sums_fn(dataset, model_state)])
            diagnostics = _state["diagnostics"]
            return ClientTelemetry(
                client_id=_client_id,
                sample_count=_count,
                train_loss=float(_state["train_loss"]),
                val_mape=float(metrics["mape"]),
                val_rmse=float(metrics["rmse"]),
                update_norm=float(diagnostics["update_norm"]),
                cosine_to_mean=float(diagnostics["cosine_to_mean"]),
                cosine_to_previous=float(diagnostics["cosine_to_previous"]),
            )

        vaults[client_id] = ClientDataVault(
            client_id=client_id,
            train_dataset=datasets["train"],
            controller_validation_dataset=datasets.get("controller_validation"),
            locked_test_dataset=datasets.get("locked_test"),
            train_fn=train_fn,
            telemetry_fn=telemetry_fn,
            metric_sums_fn=metric_sums_fn,
        )

    immutable_vaults = MappingProxyType(vaults)
    immutable_counts = MappingProxyType(train_counts)

    def install_diagnostics(
        global_state: Mapping[str, torch.Tensor],
        local_updates: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> None:
        deltas = {
            client_id: _flatten_delta(local_updates[client_id], global_state)
            for client_id in immutable_vaults
        }
        mean_delta = torch.stack(tuple(deltas.values())).mean(dim=0)
        for client_id, delta in deltas.items():
            runtime_state[client_id]["diagnostics"] = {
                "update_norm": float(torch.linalg.vector_norm(delta)),
                "cosine_to_mean": _cosine(delta, mean_delta),
                # This field is deliberately stateless. Keeping a previous-round
                # delta outside PCVEngine would escape its rollback snapshot.
                "cosine_to_previous": 0.0,
            }

    return VaultBundle(
        vaults=immutable_vaults,
        train_sample_counts=immutable_counts,
        dataset_sha256=loaded.dataset_sha256,
        partition_sha256=loaded.partition_sha256,
        install_diagnostics=install_diagnostics,
    )


class StagedDeepSeekAgent:
    """Adapt the strict JSON client to the engine's six ordered stage calls."""

    def __init__(self, client: StrictDeepSeekClient, prompt_dir: Path):
        self.client = client
        self.prompts = {
            role: (prompt_dir / f"{role}.md").read_text(encoding="utf-8")
            for role in (
                "diagnostic",
                "performance_proposer",
                "stability_proposer",
                "balance_proposer",
                "critic",
                "coordinator",
                "single_proposer",
            )
        }

    def call(self, *, role: str, payload: dict[str, Any]):
        clients = tuple(client["client_id"] for client in payload["clients"])
        if role == "diagnostic":
            validator = validate_diagnostic_response
        elif role in {
            "performance_proposer",
            "stability_proposer",
            "balance_proposer",
        }:
            validator = lambda value: validate_proposer_response(
                value,
                client_ids=clients,
                role=role,
            )
        elif role == "critic":
            candidate_ids = tuple(
                candidate["candidate_id"] for candidate in payload["candidates"]
            )
            validator = lambda value: validate_critic_response(
                value,
                candidate_ids=candidate_ids,
            )
        elif role == "coordinator":
            candidate_ids = tuple(
                candidate["candidate_id"] for candidate in payload["candidates"]
            )
            validator = lambda value: validate_coordinator_response(
                value,
                candidate_ids=candidate_ids,
            )
        elif role == "single_proposer":
            def validator(value):
                if type(value) is not dict or set(value) != {"diagnostic", "candidates"}:
                    raise ValueError("single-agent response must contain diagnostic/candidates")
                diagnostic = validate_diagnostic_response(value["diagnostic"])
                proposals = validate_proposer_response(
                    {"candidates": value["candidates"]},
                    client_ids=clients,
                    role="performance_proposer",
                )
                return {"diagnostic": diagnostic, "proposals": proposals}
        else:
            raise ValueError(f"unsupported staged role: {role}")
        return self.client.generate_json(role, self.prompts[role], payload, validator)


def _training_callbacks(bundle: VaultBundle):
    def train_clients(*, round_index, global_state, training_config, seed):
        del round_index
        return {
            client_id: dict(
                vault.train_local(
                    global_state,
                    training_config,
                    seed + client_index * 100_003,
                ).model_state
            )
            for client_index, (client_id, vault) in enumerate(
                bundle.vaults.items(),
                start=1,
            )
        }

    def collect_telemetry(*, round_index, global_state, local_updates):
        del round_index
        bundle.install_diagnostics(global_state, local_updates)
        return {
            client_id: vault.controller_telemetry(local_updates[client_id])
            for client_id, vault in bundle.vaults.items()
        }

    def evaluate_candidates(*, candidate_states, anchor_candidate_ids):
        first_anchor = anchor_candidate_ids[0]
        first_votes = tuple(
            vote
            for vault in bundle.vaults.values()
            for vote in vault.evaluate_candidates(candidate_states, first_anchor)
        )
        summaries = aggregate_candidate_votes(first_votes)
        stronger = min(
            anchor_candidate_ids,
            key=lambda candidate_id: (
                summaries[candidate_id].weighted_mape,
                candidate_id,
            ),
        )
        if stronger == first_anchor:
            return first_votes
        return tuple(
            vote
            for vault in bundle.vaults.values()
            for vote in vault.evaluate_candidates(candidate_states, stronger)
        )

    return train_clients, collect_telemetry, evaluate_candidates


def _write_json_no_replace(path: Path, record: Mapping[str, Any]) -> Path:
    content = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence path must be a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_numbered_pause_report(
    requested_path: Path, report: Mapping[str, Any]
) -> Path:
    """Persist every engine incident without replacing an earlier report."""

    requested_path = Path(requested_path)
    if requested_path.name != "PAUSED.json":
        raise ValueError("canonical pause report path must be PAUSED.json")
    for index in range(1000):
        candidate = requested_path.with_name(
            "PAUSED.json" if index == 0 else f"PAUSED.{index:03d}.json"
        )
        try:
            return _write_json_no_replace(candidate, report)
        except FileExistsError:
            continue
    raise RuntimeError("pause-report incident namespace is exhausted")


_AGGREGATE_METRIC_KEYS = {"sample_count", "mape", "rmse", "mae", "r2"}


def _validated_aggregate_metrics(value: Any) -> dict[str, float | int]:
    if type(value) is not dict or set(value) != _AGGREGATE_METRIC_KEYS:
        raise ValueError("aggregate metrics must use the exact metric schema")
    sample_count = value["sample_count"]
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("aggregate metric sample_count must be a positive exact integer")
    metrics: dict[str, float | int] = {"sample_count": sample_count}
    for name in ("mape", "rmse", "mae", "r2"):
        metric = value[name]
        if type(metric) not in (int, float) or not math.isfinite(float(metric)):
            raise ValueError("aggregate metrics must contain finite numeric values")
        metrics[name] = float(metric)
    if any(float(metrics[name]) < 0.0 for name in ("mape", "rmse", "mae")):
        raise ValueError("aggregate error metrics must be non-negative")
    if float(metrics["r2"]) > 1.0:
        raise ValueError("aggregate r2 cannot exceed one")
    return metrics


def _evaluation_binding(context) -> dict[str, str]:
    path = context.evaluation_provenance_path
    if path is None:
        raise ValueError("formal evaluation provenance is required")
    path = Path(path)
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("formal evaluation provenance is unreadable") from error
    checkpoint_sha256 = (
        audit.get("training_checkpoint_sha256") if type(audit) is dict else None
    )
    if (
        type(audit) is not dict
        or audit.get("phase") != "formal_evaluate"
        or audit.get("method") != context.args.method
        or audit.get("training_seed") != context.args.training_seed
        or audit.get("llm_rep") != context.args.llm_rep
        or type(checkpoint_sha256) is not str
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise ValueError("formal evaluation provenance identity mismatch")
    return {
        "training_checkpoint_sha256": checkpoint_sha256,
        "evaluation_provenance_sha256": _file_sha256(path),
    }


def _write_or_validate_locked_test_result(
    *, context, metrics: Mapping[str, Any] | None
) -> dict[str, float | int]:
    """Publish locked-test evidence once, or validate an exact same-checkpoint result."""

    path = Path(context.run_directory) / "locked_test_metrics.json"
    binding = _evaluation_binding(context)
    identity = {
        "schema_version": 1,
        "phase": "formal_evaluate",
        "method": context.args.method,
        "training_seed": context.args.training_seed,
        "llm_rep": context.args.llm_rep,
        **binding,
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("existing locked-test result is unreadable") from error
        if (
            type(existing) is not dict
            or set(existing) != {*identity, "locked_test"}
            or any(existing.get(key) != expected for key, expected in identity.items())
        ):
            raise ValueError("locked-test result identity mismatch")
        return _validated_aggregate_metrics(existing["locked_test"])
    if metrics is None:
        raise ValueError("locked-test metrics are required for initial publication")
    validated = _validated_aggregate_metrics(dict(metrics))
    _write_json_no_replace(path, {**identity, "locked_test": validated})
    return validated


def restore_engine_checkpoint(
    *,
    context,
    bundle: VaultBundle,
    model,
    server_optimizer,
    provenance: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Restore a run from one approved, identity-checked checkpoint load."""

    if context.args.resume_checkpoint is None:
        return 1, {}
    checkpoint_path = Path(context.args.resume_checkpoint)
    if not os.path.lexists(checkpoint_path) and hasattr(context, "run_directory"):
        if context.args.phase == "formal_evaluate":
            raise FileNotFoundError(
                "formal evaluation requires a completed training checkpoint"
            )
        expected = Path(context.run_directory) / "last_complete.pt"
        pause_reports = tuple(Path(context.run_directory).glob("PAUSED*.json"))
        round_telemetry = Path(context.run_directory) / "rounds.jsonl"
        if (
            checkpoint_path.resolve(strict=False) != expected.resolve(strict=False)
            or context.args.user_approved_resume is not True
            or not pause_reports
            or (round_telemetry.exists() and round_telemetry.stat().st_size > 0)
        ):
            raise FileNotFoundError("approved pre-checkpoint restart is not valid")
        return 1, {}
    prompt_hashes = provenance["prompt_hashes"] or {
        "engine": "no-agent-prompts"
    }
    return restore_training_checkpoint(
        model=model,
        server_optimizer=server_optimizer,
        user_approved_resume=context.args.user_approved_resume,
        resume_checkpoint=context.args.resume_checkpoint,
        requested_freeze_id=context.args.freeze_id or "development",
        requested_method=context.args.method,
        requested_training_seed=context.args.training_seed,
        requested_llm_rep=context.args.llm_rep,
        requested_partition_sha256=bundle.partition_sha256,
        requested_config_sha256=provenance["effective_config_sha256"],
        requested_prompt_hashes=prompt_hashes,
    )


def planned_round_indices(context, start_round: int, num_rounds: int) -> tuple[int, ...]:
    """Return the only legal rounds for training or locked-test evaluation."""

    if type(start_round) is not int or start_round < 1:
        raise ValueError("start_round must be a positive exact integer")
    if type(num_rounds) is not int or num_rounds < 1:
        raise ValueError("num_rounds must be a positive exact integer")
    if context.args.phase == "formal_evaluate":
        if start_round != num_rounds + 1:
            raise RuntimeError(
                "formal evaluation requires a completed training checkpoint"
            )
        return ()
    if start_round == num_rounds + 1:
        return ()
    if start_round > num_rounds + 1:
        raise RuntimeError("training checkpoint is already complete")
    return tuple(range(start_round, num_rounds + 1))


def _read_training_summary(path: Path, context, num_rounds: int) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("existing validation summary is unreadable") from error
    if (
        type(value) is not dict
        or set(value)
        != {
            "status",
            "phase",
            "method",
            "training_seed",
            "llm_rep",
            "completed_rounds",
            "best_validation",
        }
        or value["status"] != "complete"
        or value["phase"] != context.args.phase
        or value["method"] != context.args.method
        or value["training_seed"] != context.args.training_seed
        or value["llm_rep"] != context.args.llm_rep
        or value["completed_rounds"] != num_rounds
        or type(value["best_validation"]) is not dict
    ):
        raise ValueError("existing validation summary identity mismatch")
    return value


def _write_or_validate_completion(
    *, context, engine: PCVEngine, summary: Mapping[str, Any]
) -> Path:
    filename = (
        "EVALUATION_COMPLETE.json"
        if context.args.phase == "formal_evaluate"
        else "TRAINING_COMPLETE.json"
    )
    path = Path(context.run_directory) / filename
    result_file = (
        "locked_test_metrics.json"
        if context.args.phase == "formal_evaluate"
        else "validation_metrics.json"
    )
    result_path = Path(context.run_directory) / result_file
    record = {
        "status": "complete",
        "phase": context.args.phase,
        "method": context.args.method,
        "training_seed": context.args.training_seed,
        "llm_rep": context.args.llm_rep,
        "last_complete_round": engine.last_complete_round,
        "resolved_pause_reports": sorted(
            item.name for item in Path(context.run_directory).glob("PAUSED*.json")
        ),
        "resume_approved": context.args.user_approved_resume is True,
        "provenance": context.provenance_path.name,
        "evaluation_provenance": (
            None
            if context.evaluation_provenance_path is None
            else context.evaluation_provenance_path.name
        ),
        "result_status": summary["status"],
        "result_file": result_file,
        "result_sha256": _file_sha256(result_path),
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("completion status is unreadable") from error
        if type(existing) is not dict or existing != record:
            raise ValueError("completion status identity mismatch")
        return path
    return _write_json_no_replace(path, record)


def execute_strict_training(context) -> dict[str, Any]:
    """Run the frozen rounds; client rows remain inside ClientDataVault callbacks."""

    project_root = Path(__file__).resolve().parents[3]
    base_config = _exact_yaml(project_root / "configs/config.yaml")
    random.seed(context.args.training_seed)
    np.random.seed(context.args.training_seed)
    torch.manual_seed(context.args.training_seed)
    bundle = build_client_vaults(
        project_root=project_root,
        base_config=base_config,
        method_config=context.method_config,
        phase=context.args.phase,
    )
    model = _model_factory(base_config)
    server_optimizer = FedYogiServerOptimizer(
        server_lr=float(context.method_config["fedyogi_server_lr"]),
        beta1=float(context.method_config["fedyogi_beta1"]),
        beta2=float(context.method_config["fedyogi_beta2"]),
        tau=float(context.method_config["fedyogi_tau"]),
        max_coordinate_step_ratio=context.method_config[
            "fedyogi_max_coordinate_step_ratio"
        ],
    )
    train_clients, collect_telemetry, evaluate_candidates = _training_callbacks(bundle)

    provenance = json.loads(context.provenance_path.read_text(encoding="utf-8"))
    agent = None
    if context.api_key is not None:
        import requests

        call_telemetry = AppendOnlyTelemetry(
            context.run_directory / "agent_calls.jsonl",
            known_secrets=(context.api_key,),
        )
        deepseek_settings = _deepseek_settings_from_provenance(provenance)
        client = StrictDeepSeekClient(
            api_key=context.api_key,
            model_name=deepseek_settings["model"],
            base_url=deepseek_settings["base_url"],
            timeout_seconds=deepseek_settings["timeout_seconds"],
            session=requests.Session(),
            telemetry=call_telemetry,
        )
        agent = StagedDeepSeekAgent(client, project_root / "configs/prompts")

    start_round, resume_state = restore_engine_checkpoint(
        context=context,
        bundle=bundle,
        model=model,
        server_optimizer=server_optimizer,
        provenance=provenance,
    )
    engine = PCVEngine(
        method=context.args.method,
        model=model,
        server_optimizer=server_optimizer,
        sample_counts=bundle.train_sample_counts,
        train_clients=train_clients,
        collect_telemetry=collect_telemetry,
        evaluate_candidates=evaluate_candidates,
        training_config={
            "local_epochs": context.method_config["local_epochs"],
            "batch_size": context.method_config["batch_size"],
            "client_learning_rate": context.method_config["client_learning_rate"],
        },
        training_seed=context.args.training_seed,
        fedyogi_clip_norm=context.method_config["fedyogi_anchor_clip_norm"],
        agent_orchestrator=agent,
        single_agent=agent,
        telemetry_sink=AppendOnlyTelemetry(context.run_directory / "rounds.jsonl"),
        checkpoint_path=context.run_directory / "last_complete.pt",
        pause_report_path=context.run_directory / "PAUSED.json",
        pause_report_writer=_write_numbered_pause_report,
        freeze_id=context.args.freeze_id or "development",
        llm_rep=context.args.llm_rep,
        partition_sha256=bundle.partition_sha256,
        config_sha256=provenance["effective_config_sha256"],
        prompt_hashes=provenance["prompt_hashes"] or {"engine": "no-agent-prompts"},
        **resume_state,
    )
    num_rounds = int(context.method_config["num_rounds"])
    results = [
        engine.run_round(round_index)
        for round_index in planned_round_indices(context, start_round, num_rounds)
    ]
    if context.args.phase == "formal_evaluate":
        locked_path = context.run_directory / "locked_test_metrics.json"
        if locked_path.exists():
            test_metrics = _write_or_validate_locked_test_result(
                context=context,
                metrics=None,
            )
        else:
            test_metrics = aggregate_metric_sums(
                vault.final_test_sums(
                    engine.best_model_state,
                    {
                        "phase": context.args.phase,
                        "formal_frozen": context.manifest.formal_frozen,
                        "explicit_unlock": context.args.unlock_test,
                    },
                )
                for vault in bundle.vaults.values()
            )
            test_metrics = _write_or_validate_locked_test_result(
                context=context,
                metrics=test_metrics,
            )
        summary = {
            "status": "complete",
            "phase": context.args.phase,
            "method": context.args.method,
            "training_seed": context.args.training_seed,
            "llm_rep": context.args.llm_rep,
            "completed_rounds": engine.last_complete_round,
            "best_validation": dict(engine.best_validation),
            "locked_test": test_metrics,
        }
        _write_or_validate_completion(context=context, engine=engine, summary=summary)
        return summary

    validation_path = context.run_directory / "validation_metrics.json"
    if not results and validation_path.exists():
        summary = _read_training_summary(validation_path, context, num_rounds)
    else:
        validation = aggregate_metric_sums(
            vault.controller_metric_sums(engine.best_model_state)
            for vault in bundle.vaults.values()
        )
        summary = {
            "status": "complete",
            "phase": context.args.phase,
            "method": context.args.method,
            "training_seed": context.args.training_seed,
            "llm_rep": context.args.llm_rep,
            "completed_rounds": engine.last_complete_round,
            "best_validation": validation,
        }
        _write_json_no_replace(validation_path, summary)
    _write_or_validate_completion(context=context, engine=engine, summary=summary)
    return summary


__all__ = [
    "StagedDeepSeekAgent",
    "VaultBundle",
    "build_client_vaults",
    "execute_strict_training",
    "planned_round_indices",
    "restore_engine_checkpoint",
]
