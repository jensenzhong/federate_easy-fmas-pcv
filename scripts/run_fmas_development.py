"""Run the approved seed-42 FMAS-PCV development matrix sequentially.

This launcher has deliberately narrow authority: it can invoke only the nine
predeclared development runs.  It never enables a formal phase or unlocks the
test partition, and it stops at the first failed or non-terminal run.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAUSED_EXIT_CODE = 2

_APPROVED_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "phase": "development",
    "training_seed": 42,
    "partition_manifest": "results/manifests/strict_partition_v1.csv",
    "base_config": "configs/config.yaml",
    "output_root": "results/development/seed42",
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "timeout_seconds": 60,
    },
    "development_gate": {
        "baseline_selection": "lowest_mape_of_strict_baselines",
        "relative_mape_improvement_min": 0.0,
        "rmse_increase_ratio_max": 0.05,
        "r2_difference_min": -0.02,
        "required_passing_fmas_repetitions": 2,
    },
}

_APPROVED_METHOD_REPETITIONS = (
    ("FEDAVG_STRICT", 0),
    ("FEDYOGI_STRICT", 0),
    ("DPCV_FEDYOGI", 0),
    ("SA_PCV_FEDYOGI", 1),
    ("SA_PCV_FEDYOGI", 2),
    ("SA_PCV_FEDYOGI", 3),
    ("FMAS_PCV_FEDYOGI", 1),
    ("FMAS_PCV_FEDYOGI", 2),
    ("FMAS_PCV_FEDYOGI", 3),
)
_METHOD_ROLES = {
    "FEDAVG_STRICT": (),
    "FEDYOGI_STRICT": (),
    "DPCV_FEDYOGI": (),
    "SA_PCV_FEDYOGI": ("single_proposer", "coordinator"),
    "FMAS_PCV_FEDYOGI": (
        "diagnostic",
        "performance_proposer",
        "stability_proposer",
        "balance_proposer",
        "critic",
        "coordinator",
    ),
}
_METHOD_CONFIG_PATHS = {
    "FEDAVG_STRICT": "configs/methods/fedavg_strict.yaml",
    "FEDYOGI_STRICT": "configs/methods/fedyogi_strict.yaml",
    "DPCV_FEDYOGI": "configs/methods/dpcv_fedyogi.yaml",
    "SA_PCV_FEDYOGI": "configs/methods/sa_pcv_fedyogi.yaml",
    "FMAS_PCV_FEDYOGI": "configs/methods/fmas_pcv_fedyogi.yaml",
}


@dataclass(frozen=True, slots=True)
class DevelopmentRun:
    method: str
    training_seed: int
    llm_rep: int
    run_id: str


@dataclass(frozen=True, slots=True)
class RunEvidence:
    run: DevelopmentRun
    metrics: Mapping[str, int | float]
    partition_sha256: str
    git_commit: str


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    git_commit: str
    git_dirty: bool
    config_sha256: str
    partition_sha256: str
    sealed_partition_metadata_sha256: str
    base_config_sha256: str
    method_config_sha256: Mapping[str, str]
    effective_config_sha256: Mapping[str, str]
    prompt_hashes: Mapping[str, str]


def _require_development_seed(training_seed: int) -> None:
    if type(training_seed) is not int or training_seed != 42:
        raise ValueError("the approved development seed is exactly 42")


def build_run_matrix(*, training_seed: int) -> tuple[DevelopmentRun, ...]:
    """Return the immutable nine-run matrix in its required serial order."""

    _require_development_seed(training_seed)
    return tuple(
        DevelopmentRun(
            method=method,
            training_seed=training_seed,
            llm_rep=llm_rep,
            run_id=(
                f"{method.lower().replace('_', '-')}-seed{training_seed}-rep{llm_rep}"
            ),
        )
        for method, llm_rep in _APPROVED_METHOD_REPETITIONS
    )


def _exact_value_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _exact_value_equal(actual[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _exact_value_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _has_exact_fields(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(
        key in actual and _exact_value_equal(actual[key], value)
        for key, value in expected.items()
    )


def _read_approved_config(
    config_path: Path, *, project_root: Path
) -> Mapping[str, Any]:
    root = Path(project_root).resolve(strict=True)
    supplied = Path(config_path)
    if not supplied.is_absolute():
        supplied = root / supplied
    supplied = supplied.resolve(strict=True)
    expected_path = (root / "configs/development_seed42.yaml").resolve(strict=True)
    if supplied != expected_path:
        raise ValueError("only configs/development_seed42.yaml is approved")

    try:
        loaded = yaml.safe_load(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("development configuration is unreadable") from error
    if not _exact_value_equal(loaded, _APPROVED_CONFIG):
        raise ValueError("development configuration differs from the approved protocol")
    return loaded


def _read_json_regular(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact JSON object")
    return value


def _is_lower_hex(value: Any, length: int) -> bool:
    if type(value) is not str or len(value) != length or value.lower() != value:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evidence hash target must be a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _effective_config_sha256(base_path: Path, method_path: Path) -> str:
    digest = hashlib.sha256()
    for label, path in ((b"base-config", base_path), (b"method-config", method_path)):
        content = path.read_bytes()
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _read_git_state(project_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if revision.returncode != 0:
        raise ValueError("development matrix requires a valid Git HEAD")
    commit = revision.stdout.strip()
    if not _is_lower_hex(commit, 40):
        raise ValueError("development matrix Git commit is invalid")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if status.returncode != 0:
        raise ValueError("development matrix cannot inspect Git cleanliness")
    return commit, bool(status.stdout.strip())


def _capture_execution_snapshot(project_root: Path) -> ExecutionSnapshot:
    config_path = project_root / "configs/development_seed42.yaml"
    partition_path = project_root / "results/manifests/strict_partition_v1.csv"
    sealed_metadata_path = project_root / "Data/strict_partition_v1/metadata.json"
    base_path = project_root / "configs/config.yaml"
    method_paths = {
        method: project_root / relative
        for method, relative in _METHOD_CONFIG_PATHS.items()
    }
    prompt_roles = {role for roles in _METHOD_ROLES.values() for role in roles}
    git_commit, git_dirty = _read_git_state(project_root)
    if git_dirty:
        raise ValueError("development matrix requires a clean Git worktree")
    return ExecutionSnapshot(
        git_commit=git_commit,
        git_dirty=False,
        config_sha256=_file_sha256(config_path),
        partition_sha256=_file_sha256(partition_path),
        sealed_partition_metadata_sha256=_file_sha256(sealed_metadata_path),
        base_config_sha256=_file_sha256(base_path),
        method_config_sha256={
            method: _file_sha256(path) for method, path in method_paths.items()
        },
        effective_config_sha256={
            method: _effective_config_sha256(base_path, path)
            for method, path in method_paths.items()
        },
        prompt_hashes={
            role: _file_sha256(project_root / f"configs/prompts/{role}.md")
            for role in prompt_roles
        },
    )


def _is_linklike(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _validate_confined_directory(
    path: Path, *, project_root: Path, must_exist: bool
) -> None:
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(project_root)
    except ValueError as error:
        raise ValueError("output path escapes the project root") from error
    current = project_root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current):
            if _is_linklike(current) or not current.is_dir():
                raise ValueError("output path contains a link, junction, or non-directory")
            try:
                current.resolve(strict=True).relative_to(project_root)
            except ValueError as error:
                raise ValueError("output directory resolves outside the project root") from error
    if must_exist and (not path.is_dir() or _is_linklike(path)):
        raise ValueError("required output directory is missing or unsafe")


def _load_run_evidence(
    run: DevelopmentRun,
    run_directory: Path,
    *,
    snapshot: ExecutionSnapshot,
) -> RunEvidence:
    if os.path.lexists(run_directory / "locked_test_metrics.json"):
        raise ValueError("locked-test evidence is forbidden in development")

    validation_path = run_directory / "validation_metrics.json"
    validation = _read_json_regular(validation_path, label="validation metrics")
    validation_keys = {
        "status",
        "phase",
        "method",
        "training_seed",
        "llm_rep",
        "completed_rounds",
        "best_validation",
    }
    if set(validation) != validation_keys or not _has_exact_fields(
        validation,
        {
            "status": "complete",
            "phase": "development",
            "method": run.method,
            "training_seed": run.training_seed,
            "llm_rep": run.llm_rep,
            "completed_rounds": 20,
        },
    ):
        raise ValueError("validation metrics identity mismatch")
    metrics = validation["best_validation"]
    metric_keys = {"sample_count", "mape", "rmse", "mae", "r2"}
    if type(metrics) is not dict or set(metrics) != metric_keys:
        raise ValueError("validation metric fields mismatch")
    if type(metrics["sample_count"]) is not int or metrics["sample_count"] <= 0:
        raise ValueError("validation sample_count must be a positive integer")
    for name in ("mape", "rmse", "mae", "r2"):
        value = metrics[name]
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError(f"validation {name} must be finite")
    if any(metrics[name] < 0 for name in ("mape", "rmse", "mae")):
        raise ValueError("validation error metrics must be non-negative")

    provenance = _read_json_regular(
        run_directory / "provenance.json", label="run provenance"
    )
    provenance_keys = {
        "schema_version",
        "method",
        "phase",
        "training_seed",
        "llm_rep",
        "run_id",
        "freeze_id",
        "git_commit",
        "git_dirty",
        "partition_sha256",
        "sealed_partition_metadata_sha256",
        "method_config_sha256",
        "base_config_sha256",
        "effective_config_sha256",
        "prompt_hashes",
        "deepseek",
        "resume_requested",
        "locked_test_unlocked",
    }
    if set(provenance) != provenance_keys or not _has_exact_fields(
        provenance,
        {
            "schema_version": 1,
            "method": run.method,
            "phase": "development",
            "training_seed": run.training_seed,
            "llm_rep": run.llm_rep,
            "run_id": run.run_id,
            "freeze_id": None,
            "git_commit": snapshot.git_commit,
            "git_dirty": False,
            "locked_test_unlocked": False,
        },
    ):
        raise ValueError("run provenance identity mismatch")
    if type(provenance["resume_requested"]) is not bool:
        raise ValueError("resume provenance flag must be boolean")
    if not _is_lower_hex(provenance["git_commit"], 40):
        raise ValueError("run provenance Git commit is invalid")
    for name in (
        "partition_sha256",
        "sealed_partition_metadata_sha256",
        "method_config_sha256",
        "base_config_sha256",
        "effective_config_sha256",
    ):
        if not _is_lower_hex(provenance[name], 64):
            raise ValueError(f"run provenance {name} is invalid")
    expected_hashes = {
        "partition_sha256": snapshot.partition_sha256,
        "sealed_partition_metadata_sha256": snapshot.sealed_partition_metadata_sha256,
        "method_config_sha256": snapshot.method_config_sha256[run.method],
        "base_config_sha256": snapshot.base_config_sha256,
        "effective_config_sha256": snapshot.effective_config_sha256[run.method],
    }
    if any(provenance[name] != value for name, value in expected_hashes.items()):
        raise ValueError("run provenance does not match the execution snapshot")
    expected_roles = set(_METHOD_ROLES[run.method])
    prompt_hashes = provenance["prompt_hashes"]
    if (
        type(prompt_hashes) is not dict
        or set(prompt_hashes) != expected_roles
        or any(not _is_lower_hex(value, 64) for value in prompt_hashes.values())
        or any(prompt_hashes[role] != snapshot.prompt_hashes[role] for role in expected_roles)
    ):
        raise ValueError("run provenance prompt hashes mismatch")
    uses_llm = bool(expected_roles)
    expected_deepseek = {
        "enabled": uses_llm,
        "model": "deepseek-chat" if uses_llm else None,
        "base_url": "https://api.deepseek.com" if uses_llm else None,
        "temperature": 0.8 if uses_llm else None,
        "timeout_seconds": 60 if uses_llm else None,
    }
    if not _exact_value_equal(provenance["deepseek"], expected_deepseek):
        raise ValueError("run provenance DeepSeek settings mismatch")

    completion = _read_json_regular(
        run_directory / "TRAINING_COMPLETE.json", label="training completion"
    )
    completion_keys = {
        "status",
        "phase",
        "method",
        "training_seed",
        "llm_rep",
        "last_complete_round",
        "resolved_pause_reports",
        "resume_approved",
        "provenance",
        "evaluation_provenance",
        "result_status",
        "result_file",
        "result_sha256",
    }
    if set(completion) != completion_keys or not _has_exact_fields(
        completion,
        {
            "status": "complete",
            "phase": "development",
            "method": run.method,
            "training_seed": run.training_seed,
            "llm_rep": run.llm_rep,
            "last_complete_round": 20,
            "provenance": "provenance.json",
            "evaluation_provenance": None,
            "result_status": "complete",
            "result_file": "validation_metrics.json",
        },
    ):
        raise ValueError("training completion identity mismatch")
    if type(completion["resume_approved"]) is not bool:
        raise ValueError("completion resume flag must be boolean")
    resolved = completion["resolved_pause_reports"]
    pauses = sorted(path.name for path in run_directory.glob("PAUSED*.json"))
    if (
        type(resolved) is not list
        or any(type(name) is not str for name in resolved)
        or resolved != pauses
    ):
        raise ValueError("completion does not resolve every pause incident")
    expected_result_sha = _file_sha256(validation_path)
    if (
        not _is_lower_hex(completion["result_sha256"], 64)
        or completion["result_sha256"] != expected_result_sha
    ):
        raise ValueError("completion result SHA mismatch")

    return RunEvidence(
        run=run,
        metrics=dict(metrics),
        partition_sha256=provenance["partition_sha256"],
        git_commit=provenance["git_commit"],
    )


def _completion_is_terminal(
    run: DevelopmentRun, run_directory: Path, *, snapshot: ExecutionSnapshot
) -> bool:
    try:
        _load_run_evidence(run, run_directory, snapshot=snapshot)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _run_command(
    run: DevelopmentRun, *, project_root: Path, python_executable: str
) -> list[str]:
    return [
        python_executable,
        str(project_root / "experiments/run_strict_federated.py"),
        "--method",
        run.method,
        "--phase",
        "development",
        "--training-seed",
        str(run.training_seed),
        "--llm-rep",
        str(run.llm_rep),
        "--run-id",
        run.run_id,
    ]


def _inclusive_at_least(value: float, threshold: float) -> bool:
    return value >= threshold


def _inclusive_at_most(value: float, threshold: float) -> bool:
    return value <= threshold


def _build_development_gate(
    *,
    matrix: tuple[DevelopmentRun, ...],
    output_root: Path,
    snapshot: ExecutionSnapshot,
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = [
        _load_run_evidence(run, output_root / run.run_id, snapshot=snapshot)
        for run in matrix
    ]
    if {item.partition_sha256 for item in evidence} != {snapshot.partition_sha256}:
        raise ValueError("development runs do not share one partition")
    if len({item.git_commit for item in evidence}) != 1:
        raise ValueError("development runs do not share one Git commit")

    strict_baselines = [
        item
        for item in evidence
        if item.run.method in {"FEDAVG_STRICT", "FEDYOGI_STRICT"}
    ]
    if len(strict_baselines) != 2:
        raise ValueError("exactly two strict baselines are required")
    baseline = min(strict_baselines, key=lambda item: item.metrics["mape"])
    baseline_mape = Decimal(str(baseline.metrics["mape"]))
    baseline_rmse = Decimal(str(baseline.metrics["rmse"]))
    if baseline_mape <= 0 or baseline_rmse <= 0:
        raise ValueError("baseline MAPE and RMSE must be positive for relative gates")

    trajectory_records = []
    for item in evidence:
        if item.run.method not in {"SA_PCV_FEDYOGI", "FMAS_PCV_FEDYOGI"}:
            continue
        relative_mape = (
            baseline_mape - Decimal(str(item.metrics["mape"]))
        ) / baseline_mape
        rmse_increase = (
            Decimal(str(item.metrics["rmse"])) - baseline_rmse
        ) / baseline_rmse
        r2_difference = Decimal(str(item.metrics["r2"])) - Decimal(
            str(baseline.metrics["r2"])
        )
        checks = {
            "relative_mape_improvement": _inclusive_at_least(
                relative_mape,
                Decimal(str(gate_config["relative_mape_improvement_min"])),
            ),
            "rmse_increase_ratio": _inclusive_at_most(
                rmse_increase,
                Decimal(str(gate_config["rmse_increase_ratio_max"])),
            ),
            "r2_difference": _inclusive_at_least(
                r2_difference,
                Decimal(str(gate_config["r2_difference_min"])),
            ),
        }
        trajectory_records.append(
            {
                "method": item.run.method,
                "llm_rep": item.run.llm_rep,
                "run_id": item.run.run_id,
                "validation": dict(item.metrics),
                "relative_mape_improvement": float(relative_mape),
                "rmse_increase_ratio": float(rmse_increase),
                "r2_difference": float(r2_difference),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    if len(trajectory_records) != 6:
        raise ValueError("exactly six LLM trajectories are required")
    passing_fmas = sum(
        row["passed"]
        for row in trajectory_records
        if row["method"] == "FMAS_PCV_FEDYOGI"
    )
    required_fmas = gate_config["required_passing_fmas_repetitions"]
    return {
        "schema_version": 1,
        "status": "complete",
        "phase": "development",
        "training_seed": 42,
        "config_sha256": snapshot.config_sha256,
        "partition_sha256": snapshot.partition_sha256,
        "git_commit": snapshot.git_commit,
        "thresholds": dict(gate_config),
        "baseline": {
            "selection": gate_config["baseline_selection"],
            "method": baseline.run.method,
            "llm_rep": baseline.run.llm_rep,
            "run_id": baseline.run.run_id,
            "validation": dict(baseline.metrics),
        },
        "trajectories": trajectory_records,
        "passing_fmas_repetitions": passing_fmas,
        "required_passing_fmas_repetitions": required_fmas,
        "gate_passed": passing_fmas >= required_fmas,
        "evidence": {
            "partition": "controller_validation",
            "locked_test_used": False,
        },
    }


def _publish_json_no_replace(path: Path, record: Mapping[str, Any]) -> Path:
    encoded = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite immutable gate: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def run_development_matrix(
    *,
    training_seed: int,
    config_path: Path,
    project_root: Path = PROJECT_ROOT,
    command_runner: Callable[..., Any] = subprocess.run,
    python_executable: str = sys.executable,
) -> int:
    """Execute or safely continue the approved matrix, one process at a time."""

    _require_development_seed(training_seed)
    requested_root = Path(project_root).absolute()
    if _is_linklike(requested_root):
        raise ValueError("project root cannot be a link or junction")
    root = requested_root.resolve(strict=True)
    config = _read_approved_config(config_path, project_root=root)
    if config["training_seed"] != training_seed:
        raise ValueError("CLI seed and approved configuration seed must match")

    snapshot = _capture_execution_snapshot(root)
    matrix = build_run_matrix(training_seed=training_seed)
    output_root = root / config["output_root"]
    _validate_confined_directory(output_root, project_root=root, must_exist=False)
    gate_path = output_root / "development_gate.json"
    if os.path.lexists(gate_path):
        raise FileExistsError(f"refusing to overwrite immutable gate: {gate_path}")
    previous_run: DevelopmentRun | None = None
    previous_directory: Path | None = None

    for run in matrix:
        _validate_confined_directory(output_root, project_root=root, must_exist=False)
        if _capture_execution_snapshot(root) != snapshot:
            return PAUSED_EXIT_CODE
        if run.llm_rep > 0 and (
            previous_run is None
            or previous_directory is None
            or not _completion_is_terminal(
                previous_run, previous_directory, snapshot=snapshot
            )
        ):
            return PAUSED_EXIT_CODE

        run_directory = output_root / run.run_id
        _validate_confined_directory(run_directory, project_root=root, must_exist=False)
        if os.path.lexists(run_directory):
            if _completion_is_terminal(run, run_directory, snapshot=snapshot):
                previous_run, previous_directory = run, run_directory
                continue
            return PAUSED_EXIT_CODE

        completed = command_runner(
            _run_command(run, project_root=root, python_executable=python_executable),
            cwd=root,
            check=False,
        )
        returncode = completed.returncode
        if type(returncode) is not int:
            raise TypeError("subprocess returncode must be an integer")
        if returncode != 0:
            return returncode
        try:
            _validate_confined_directory(output_root, project_root=root, must_exist=True)
            _validate_confined_directory(run_directory, project_root=root, must_exist=True)
            current_snapshot = _capture_execution_snapshot(root)
        except (OSError, TypeError, ValueError):
            return PAUSED_EXIT_CODE
        if current_snapshot != snapshot:
            return PAUSED_EXIT_CODE
        if not _completion_is_terminal(run, run_directory, snapshot=snapshot):
            return PAUSED_EXIT_CODE
        previous_run, previous_directory = run, run_directory

    try:
        _validate_confined_directory(output_root, project_root=root, must_exist=True)
        if _capture_execution_snapshot(root) != snapshot:
            return PAUSED_EXIT_CODE
        gate = _build_development_gate(
            matrix=matrix,
            output_root=output_root,
            snapshot=snapshot,
            gate_config=config["development_gate"],
        )
    except (OSError, TypeError, ValueError):
        return PAUSED_EXIT_CODE
    _publish_json_no_replace(gate_path, gate)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the approved strict-federated seed-42 development matrix"
    )
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_development_matrix(
            training_seed=args.training_seed,
            config_path=args.config,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return PAUSED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DevelopmentRun",
    "PAUSED_EXIT_CODE",
    "build_parser",
    "build_run_matrix",
    "main",
    "run_development_matrix",
]
