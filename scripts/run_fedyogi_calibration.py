"""Run the preregistered, controller-validation-only FedYogi calibration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
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
from src.federated_learning.pcv.runtime import execute_strict_training


CONFIG_PATH = Path("configs/fedyogi_calibration_seed42.yaml")
_APPROVED = {
    "schema_version": 1,
    "phase": "development",
    "method": "FEDYOGI_STRICT",
    "training_seed": 42,
    "num_rounds": 20,
    "partition_manifest": "results/manifests/strict_partition_v1.csv",
    "selection_partition": "controller_validation",
    "output_root": "results/development/seed42/baseline_calibration",
    "selection_rule": ["mape", "rmse", "mae", "server_lr"],
    "grid": {
        "server_lr": [0.01, 0.1, 0.5],
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
    metrics: Mapping[str, int | float]
    run_directory: Path


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
        "schema_version": 1,
        "audit": "baseline_fairness_fedyogi_calibration",
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
    summary = execute_strict_training(context)
    required = {
        "provenance.json", "rounds.jsonl", "last_complete.pt",
        "validation_metrics.json", "TRAINING_COMPLETE.json",
    }
    if not required.issubset({path.name for path in run_directory.iterdir()}):
        raise RuntimeError("calibration unit did not produce complete evidence")
    if any("locked_test" in path.name or "agent_call" in path.name for path in run_directory.iterdir()):
        raise RuntimeError("forbidden calibration evidence was produced")
    return CalibrationEvidence(unit, dict(summary["best_validation"]), run_directory)


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
        CalibrationUnit(float(lr), f"fedyogi-lr-{str(lr).replace('.', 'p')}-seed42")
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
        selected = min(
            evidence,
            key=lambda item: (
                float(item.metrics["mape"]), float(item.metrics["rmse"]),
                float(item.metrics["mae"]), item.unit.server_lr,
            ),
        )
        summary = {
            "schema_version": 1,
            "status": "complete",
            "selection_partition": "controller_validation",
            "selection_rule": list(config["selection_rule"]),
            "snapshot": dict(snapshot),
            "runs": [
                {"run_id": item.unit.run_id, "server_lr": item.unit.server_lr,
                 "metrics": dict(item.metrics)} for item in evidence
            ],
            "selected_run_id": selected.unit.run_id,
            "selected_server_lr": selected.unit.server_lr,
            "locked_test_used": False,
            "deepseek_used": False,
        }
        _json_no_replace(staging / "calibration_summary.json", summary)
        os.rename(staging, root)
        return summary
    except BaseException:
        try:
            _json_no_replace(staging / "FAILED.json", {
                "schema_version": 1,
                "status": "failed",
                "audit": "baseline_fairness_fedyogi_calibration",
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
