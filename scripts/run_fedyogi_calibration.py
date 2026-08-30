"""Run the preregistered, controller-validation-only FedYogi calibration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_strict_federated import load_method_config
from src.federated_learning.pcv.engine import (
    ExperimentPaused,
    ExperimentRuntimeError,
)
from src.federated_learning.pcv.runtime import execute_strict_training


CONFIG_PATH = Path("configs/fedyogi_calibration_seed42_v2.yaml")
_APPROVED = {
    "schema_version": 2,
    "calibration_id": "fedyogi_lr_refinement_v2",
    "supersedes": {
        "summary": (
            "audits/fedyogi_calibration_seed42_v1_summary.json"
        ),
        "sha256": (
            "0d1ed24b032f13065b83ab9e3d00417a099bbc57c407578b3f71eeabd8137d30"
        ),
    },
    "approval": "user_approved_2026-08-30",
    "phase": "development",
    "method": "FEDYOGI_STRICT",
    "training_seed": 42,
    "num_rounds": 20,
    "partition_manifest": "results/manifests/strict_partition_v1.csv",
    "selection_partition": "controller_validation",
    "output_root": "results/development/seed42/baseline_calibration_v2",
    "eligibility_policy": {
        "required_completed_rounds": 20,
        "max_round_mape": 2.0,
        "max_final_to_best_mape_ratio": 1.5,
        "ratio_denominator_floor": 1.0e-12,
    },
    "readiness_policy": {
        "min_best_mape_improvement_from_first": 0.05,
    },
    "selection_rule": ["mape", "rmse", "mae", "server_lr"],
    "failure_policy": {
        "nonfinite_prediction": (
            "disqualify_without_retry_or_grid_replacement"
        ),
        "all_other_failures": "abort_calibration",
    },
    "grid": {
        "server_lr": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
        "beta1": 0.9,
        "beta2": 0.99,
        "tau": 0.001,
        "max_coordinate_step_ratio": None,
        "clip": None,
    },
}


@dataclass(frozen=True, slots=True)
class CalibrationUnit:
    server_lr: float
    run_id: str


@dataclass(frozen=True, slots=True)
class CalibrationEvidence:
    unit: CalibrationUnit
    metrics: Mapping[str, int | float] | None
    run_directory: Path
    status: str = "complete"
    failure: Mapping[str, Any] | None = None
    trajectory: tuple[float, ...] = ()
    stability: Mapping[str, Any] | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    content = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite: {path}")
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
    finally:
        temporary.unlink(missing_ok=True)


def load_calibration_config(path: Path, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    supplied = path if path.is_absolute() else project_root / path
    canonical = project_root / CONFIG_PATH
    if supplied.resolve(strict=True) != canonical.resolve(strict=True):
        raise ValueError("calibration must use the canonical preregistered config")
    loaded = yaml.safe_load(supplied.read_text(encoding="utf-8"))
    if type(loaded) is not dict or loaded != _APPROVED:
        raise ValueError("FedYogi calibration differs from the preregistered protocol")
    parent = project_root / str(loaded["supersedes"]["summary"])
    if _sha256(parent) != loaded["supersedes"]["sha256"]:
        raise ValueError("FedYogi calibration v1 evidence hash mismatch")
    return loaded


def _capture_snapshot(project_root: Path, config_path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=project_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=project_root, check=True, capture_output=True, text=True,
    ).stdout.strip())
    supplied = config_path if config_path.is_absolute() else project_root / config_path
    base_config = project_root / "configs/config.yaml"
    method_config = project_root / "configs/methods/fedyogi_strict.yaml"
    sealed_metadata = project_root / "Data/strict_partition_v1/metadata.json"
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "calibration_config_sha256": _sha256(supplied),
        "base_config_sha256": _sha256(base_config),
        "method_config_sha256": _sha256(method_config),
        "partition_sha256": _sha256(project_root / _APPROVED["partition_manifest"]),
        "sealed_partition_metadata_sha256": _sha256(sealed_metadata),
    }


def _effective_config_sha256(
    *, project_root: Path, method_config: Mapping[str, Any]
) -> str:
    digest = hashlib.sha256()
    for label, content in (
        (b"base-config", (project_root / "configs/config.yaml").read_bytes()),
        (
            b"calibrated-method-config",
            json.dumps(
                dict(method_config), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        ),
    ):
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _read_selected_mape_trajectory(
    path: Path, *, expected_rounds: int
) -> tuple[float, ...]:
    if type(expected_rounds) is not int or expected_rounds < 0:
        raise ValueError("expected_rounds must be a non-negative exact integer")
    if path.is_symlink() or not path.is_file():
        raise ValueError("round trajectory must be a regular file")
    trajectory: list[float] = []
    for expected_index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            raise ValueError("round trajectory contains an empty record")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("round trajectory contains invalid JSON") from error
        if (
            type(record) is not dict
            or record.get("event") != "round_committed"
            or type(record.get("round_index")) is not int
            or record.get("round_index") != expected_index
        ):
            raise ValueError("round trajectory must be contiguous committed rounds")
        mape = record.get("selected_mape")
        if (
            type(mape) not in (int, float)
            or not math.isfinite(float(mape))
            or float(mape) < 0.0
        ):
            raise ValueError("round trajectory MAPE must be finite and non-negative")
        trajectory.append(float(mape))
    if len(trajectory) != expected_rounds:
        raise ValueError("round trajectory length differs from expected rounds")
    return tuple(trajectory)


def _assess_stability(
    trajectory: tuple[float, ...], policy: Mapping[str, Any]
) -> dict[str, Any]:
    required_rounds = int(policy["required_completed_rounds"])
    if len(trajectory) != required_rounds or not trajectory:
        raise ValueError("stability assessment requires the complete trajectory")
    best_mape = min(trajectory)
    best_round_index = trajectory.index(best_mape) + 1
    first_mape = trajectory[0]
    final_mape = trajectory[-1]
    max_mape = max(trajectory)
    floor = float(policy["ratio_denominator_floor"])
    final_to_best = final_mape / max(best_mape, floor)
    improvement = (first_mape - best_mape) / max(first_mape, floor)
    improvement = min(1.0, max(0.0, improvement))
    checks = {
        "completed_rounds": len(trajectory) == required_rounds,
        "max_round_mape": max_mape <= float(policy["max_round_mape"]),
        "final_to_best_mape_ratio": (
            final_to_best
            <= float(policy["max_final_to_best_mape_ratio"])
        ),
    }
    return {
        "round_count": len(trajectory),
        "first_round_mape": first_mape,
        "best_round_index": best_round_index,
        "best_round_mape": best_mape,
        "final_round_mape": final_mape,
        "max_round_mape": max_mape,
        "final_to_best_mape_ratio": final_to_best,
        "best_improvement_from_first": improvement,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def _approved_numeric_divergence(
    paused: ExperimentPaused, run_directory: Path
) -> dict[str, int | str] | None:
    expected_message = "y_pred must contain only finite values"
    expected_sanitized_detail = "round stopped after a sanitized runtime failure"
    failure = paused.failure
    cause = paused.__cause__
    if (
        type(failure) is not ExperimentRuntimeError
        or failure.exception_type != "ValueError"
        or failure.detail != expected_sanitized_detail
        or type(cause) is not ValueError
        or str(cause) != expected_message
        or paused.report_persisted is not True
        or paused.report_error is not None
        or paused.rollback_errors
        or paused.report_path.parent.resolve(strict=False)
        != run_directory.resolve(strict=False)
        or not paused.report_path.is_file()
    ):
        return None
    report = json.loads(paused.report_path.read_text(encoding="utf-8"))
    report_failure = report.get("failure") if type(report) is dict else None
    if (
        type(report) is not dict
        or report.get("status") != "paused"
        or type(report.get("failed_round")) is not int
        or type(report.get("last_complete_round")) is not int
        or type(report_failure) is not dict
        or report_failure.get("category") != "runtime"
        or report_failure.get("exception_type") != "ValueError"
        or report_failure.get("role") != "engine"
    ):
        return None
    return {
        "failed_round": report["failed_round"],
        "last_complete_round": report["last_complete_round"],
        "reason": "nonfinite_prediction",
    }


def _run_unit(
    unit: CalibrationUnit, *, project_root: Path, output_root: Path,
    snapshot: Mapping[str, Any],
) -> CalibrationEvidence:
    run_directory = output_root / unit.run_id
    run_directory.mkdir(exist_ok=False)
    method_config = dict(load_method_config("FEDYOGI_STRICT", project_root=project_root))
    method_config.update({
        "fedyogi_server_lr": unit.server_lr,
        "fedyogi_beta1": 0.9,
        "fedyogi_beta2": 0.99,
        "fedyogi_tau": 0.001,
        "fedyogi_max_coordinate_step_ratio": None,
        "fedyogi_anchor_clip_norm": None,
    })
    effective = _effective_config_sha256(
        project_root=project_root, method_config=method_config
    )
    provenance = {
        "schema_version": 2,
        "audit": "baseline_fairness_fedyogi_calibration_v2",
        "calibration_id": _APPROVED["calibration_id"],
        "method": "FEDYOGI_STRICT",
        "phase": "development",
        "training_seed": 42,
        "llm_rep": 0,
        "run_id": unit.run_id,
        "git_commit": snapshot["git_commit"],
        "git_dirty": False,
        "calibration_config_sha256": snapshot["calibration_config_sha256"],
        "base_config_sha256": snapshot["base_config_sha256"],
        "method_config_sha256": snapshot["method_config_sha256"],
        "partition_sha256": snapshot["partition_sha256"],
        "sealed_partition_metadata_sha256": snapshot[
            "sealed_partition_metadata_sha256"
        ],
        "effective_config_sha256": effective,
        "prompt_hashes": {},
        "selected_partition": "controller_validation",
        "deepseek": {"enabled": False},
        "locked_test_unlocked": False,
        "server_lr": unit.server_lr,
    }
    provenance_path = run_directory / "provenance.json"
    _json_no_replace(provenance_path, provenance)
    args = SimpleNamespace(
        phase="development", method="FEDYOGI_STRICT", training_seed=42,
        llm_rep=0, resume_checkpoint=None, user_approved_resume=False,
        freeze_id=None, unlock_test=False,
    )
    context = SimpleNamespace(
        args=args, method_config=method_config, run_directory=run_directory,
        api_key=None, provenance_path=provenance_path,
        evaluation_provenance_path=None,
        manifest=SimpleNamespace(formal_frozen=False),
    )
    try:
        summary = execute_strict_training(context)
    except ExperimentPaused as paused:
        failure = _approved_numeric_divergence(paused, run_directory)
        if failure is None:
            raise
        required = {
            "provenance.json", "rounds.jsonl", "last_complete.pt",
            paused.report_path.name,
        }
        if not required.issubset({path.name for path in run_directory.iterdir()}):
            raise RuntimeError("divergent calibration unit lacks required evidence")
        if any(
            "locked_test" in path.name or "agent_call" in path.name
            for path in run_directory.iterdir()
        ):
            raise RuntimeError("forbidden calibration evidence was produced")
        trajectory = _read_selected_mape_trajectory(
            run_directory / "rounds.jsonl",
            expected_rounds=int(failure["last_complete_round"]),
        )
        return CalibrationEvidence(
            unit=unit,
            metrics=None,
            run_directory=run_directory,
            status="disqualified_numeric_divergence",
            failure=failure,
            trajectory=trajectory,
        )
    required = {
        "provenance.json", "rounds.jsonl", "last_complete.pt",
        "validation_metrics.json", "TRAINING_COMPLETE.json",
    }
    if not required.issubset({path.name for path in run_directory.iterdir()}):
        raise RuntimeError("calibration unit did not produce complete evidence")
    if any("locked_test" in path.name or "agent_call" in path.name for path in run_directory.iterdir()):
        raise RuntimeError("forbidden calibration evidence was produced")
    metrics = dict(summary["best_validation"])
    trajectory = _read_selected_mape_trajectory(
        run_directory / "rounds.jsonl",
        expected_rounds=int(_APPROVED["num_rounds"]),
    )
    stability = _assess_stability(
        trajectory, _APPROVED["eligibility_policy"]
    )
    if not math.isclose(
        float(metrics["mape"]),
        float(stability["best_round_mape"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("best checkpoint MAPE differs from round trajectory")
    return CalibrationEvidence(
        unit,
        metrics,
        run_directory,
        trajectory=trajectory,
        stability=stability,
    )


def run_calibration(
    *, config: Mapping[str, Any], project_root: Path = PROJECT_ROOT,
    output_root: Path | None = None, config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    if dict(config) != _APPROVED:
        raise ValueError("calibration configuration is not approved")
    snapshot = _capture_snapshot(project_root, config_path)
    if snapshot["git_dirty"]:
        raise ValueError("calibration requires a clean Git worktree")
    root = output_root or project_root / str(config["output_root"])
    if os.path.lexists(root):
        raise FileExistsError(f"refusing to overwrite: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    units = tuple(
        CalibrationUnit(
            float(lr), f"fedyogi-v2-lr-{str(lr).replace('.', 'p')}-seed42"
        )
        for lr in config["grid"]["server_lr"]
    )
    try:
        evidence = [
            _run_unit(
                unit, project_root=project_root, output_root=staging,
                snapshot=snapshot,
            )
            for unit in units
        ]
        eligible = tuple(
            item
            for item in evidence
            if (
                item.status == "complete"
                and item.metrics is not None
                and item.stability is not None
                and item.stability.get("eligible") is True
            )
        )
        if not eligible:
            raise RuntimeError(
                "calibration produced no stable eligible completed unit"
            )
        selected = min(
            eligible,
            key=lambda item: (
                float(item.metrics["mape"]), float(item.metrics["rmse"]),
                float(item.metrics["mae"]), item.unit.server_lr,
            ),
        )
        selected_improvement = float(
            selected.stability["best_improvement_from_first"]
        )
        readiness_threshold = float(
            config["readiness_policy"][
                "min_best_mape_improvement_from_first"
            ]
        )
        freeze_ready = selected_improvement >= readiness_threshold
        summary = {
            "schema_version": 2,
            "status": "complete",
            "calibration_id": config["calibration_id"],
            "calibration_outcome": (
                "selected_stable_improving_point"
                if freeze_ready
                else "selected_stable_but_stalled_point"
            ),
            "supersedes": dict(config["supersedes"]),
            "selection_partition": "controller_validation",
            "eligibility_policy": dict(config["eligibility_policy"]),
            "readiness_policy": dict(config["readiness_policy"]),
            "selection_rule": list(config["selection_rule"]),
            "failure_policy": dict(config["failure_policy"]),
            "snapshot": dict(snapshot),
            "runs": [
                {
                    "run_id": item.unit.run_id,
                    "server_lr": item.unit.server_lr,
                    "status": item.status,
                    "eligible_for_selection": (
                        item.status == "complete"
                        and item.stability is not None
                        and item.stability.get("eligible") is True
                    ),
                    "stability_eligible": (
                        None
                        if item.stability is None
                        else item.stability.get("eligible") is True
                    ),
                    "selected_mape_by_round": list(item.trajectory),
                    "stability": (
                        None
                        if item.stability is None
                        else dict(item.stability)
                    ),
                    "metrics": (
                        None if item.metrics is None else dict(item.metrics)
                    ),
                    "failure": (
                        None if item.failure is None else dict(item.failure)
                    ),
                }
                for item in evidence
            ],
            "selected_run_id": selected.unit.run_id,
            "selected_server_lr": selected.unit.server_lr,
            "selected_readiness": {
                "best_mape_improvement_from_first": selected_improvement,
                "minimum_required": readiness_threshold,
                "passed": freeze_ready,
            },
            "recommended_freeze_ready": freeze_ready,
            "locked_test_used": False,
            "deepseek_used": False,
        }
        _json_no_replace(staging / "calibration_summary.json", summary)
        os.rename(staging, root)
        return summary
    except BaseException:
        try:
            _json_no_replace(staging / "FAILED.json", {
                "schema_version": 2,
                "status": "failed",
                "audit": "baseline_fairness_fedyogi_calibration_v2",
                "snapshot": dict(snapshot),
            })
            failed = staging.with_name(f"{root.name}.failed-{staging.name.rsplit('.', 1)[-1]}")
            os.rename(staging, failed)
        except BaseException:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    config = load_calibration_config(args.config)
    run_calibration(config=config, config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
