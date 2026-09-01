"""Run the frozen five-seed FMAS-PCV matrix serially and fail closed."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_strict_federated import (
    effective_config_sha256,
    validate_formal_freeze,
)
from src.federated_learning.pcv.checkpoint import load_checkpoint
from src.federated_learning.pcv.provider_config import deepseek_provenance
from src.formal_protocol import (
    METHOD_CONFIG_PATHS,
    METHOD_PROMPT_ROLES,
    METHOD_REPETITIONS,
    PARTITION_MANIFEST,
    SEALED_PARTITION_METADATA,
    file_sha256,
)
from src.study_manifest import FORMAL_METHOD_ORDER, load_study_manifest


PAUSED_EXIT_CODE = 2


@dataclass(frozen=True, slots=True)
class FormalRun:
    method: str
    training_seed: int
    llm_rep: int


@dataclass(frozen=True, slots=True)
class FormalSnapshot:
    git_commit: str
    partition_sha256: str
    sealed_partition_metadata_sha256: str
    base_config_sha256: str
    method_config_sha256: Mapping[str, str]
    effective_config_sha256: Mapping[str, str]
    prompt_hashes: Mapping[str, str]


def build_run_matrix(formal_seeds: Sequence[int]) -> tuple[FormalRun, ...]:
    seeds = tuple(formal_seeds)
    if (
        len(seeds) != 5
        or len(set(seeds)) != 5
        or 42 in seeds
        or any(type(seed) is not int for seed in seeds)
    ):
        raise ValueError("formal matrix requires five unique exact seeds excluding 42")
    return tuple(
        FormalRun(method=method, training_seed=seed, llm_rep=rep)
        for seed in seeds
        for method in FORMAL_METHOD_ORDER
        for rep in METHOD_REPETITIONS[method]
    )


def _git_state(project_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    )
    return revision, dirty


def _capture_snapshot(project_root: Path) -> FormalSnapshot:
    commit, dirty = _git_state(project_root)
    if dirty:
        raise ValueError("formal matrix requires a clean Git worktree")
    base = project_root / "configs/config.yaml"
    method_paths = {
        method: project_root / METHOD_CONFIG_PATHS[method]
        for method in FORMAL_METHOD_ORDER
    }
    roles = sorted({role for method in FORMAL_METHOD_ORDER for role in METHOD_PROMPT_ROLES[method]})
    return FormalSnapshot(
        git_commit=commit,
        partition_sha256=file_sha256(project_root / PARTITION_MANIFEST),
        sealed_partition_metadata_sha256=file_sha256(
            project_root / SEALED_PARTITION_METADATA
        ),
        base_config_sha256=file_sha256(base),
        method_config_sha256={
            method: file_sha256(path) for method, path in method_paths.items()
        },
        effective_config_sha256={
            method: effective_config_sha256(base, path)
            for method, path in method_paths.items()
        },
        prompt_hashes={
            role: file_sha256(project_root / "configs/prompts" / f"{role}.md")
            for role in roles
        },
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _validate_metrics(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"sample_count", "mape", "rmse", "mae", "r2"}:
        raise ValueError("formal metrics use an invalid schema")
    if type(value["sample_count"]) is not int or value["sample_count"] <= 0:
        raise ValueError("formal metric sample_count is invalid")
    for name in ("mape", "rmse", "mae", "r2"):
        if type(value[name]) not in {int, float} or not math.isfinite(value[name]):
            raise ValueError(f"formal metric {name} is not finite")
    if any(value[name] < 0 for name in ("mape", "rmse", "mae")):
        raise ValueError("formal error metrics must be non-negative")
    return dict(value)


def _run_directory(root: Path, freeze_id: str, run: FormalRun) -> Path:
    return root / "results/formal" / freeze_id / run.method / str(run.training_seed) / str(run.llm_rep)


def _validate_training_run(
    run: FormalRun,
    run_directory: Path,
    *,
    freeze_id: str,
    snapshot: FormalSnapshot,
    allow_evaluation_artifacts: bool = False,
) -> dict[str, Any]:
    if (
        not allow_evaluation_artifacts
        and (run_directory / "locked_test_metrics.json").exists()
    ):
        raise ValueError("locked-test evidence is forbidden during formal training")
    validation_path = run_directory / "validation_metrics.json"
    validation = _read_json(validation_path, label="formal validation metrics")
    if (
        set(validation)
        != {"status", "phase", "method", "training_seed", "llm_rep", "completed_rounds", "best_validation"}
        or validation.get("status") != "complete"
        or validation.get("phase") != "formal_train"
        or validation.get("method") != run.method
        or validation.get("training_seed") != run.training_seed
        or validation.get("llm_rep") != run.llm_rep
        or validation.get("completed_rounds") != 20
    ):
        raise ValueError("formal validation identity mismatch")
    _validate_metrics(validation["best_validation"])

    provenance = _read_json(run_directory / "provenance.json", label="formal provenance")
    expected_provenance_fields = {
        "schema_version", "method", "phase", "training_seed", "llm_rep", "run_id",
        "freeze_id", "git_commit", "git_dirty", "partition_sha256",
        "sealed_partition_metadata_sha256", "method_config_sha256",
        "base_config_sha256", "effective_config_sha256", "prompt_hashes", "deepseek",
        "resume_requested", "locked_test_unlocked",
    }
    expected_roles = set(METHOD_PROMPT_ROLES[run.method])
    expected_prompts = {role: snapshot.prompt_hashes[role] for role in expected_roles}
    if (
        set(provenance) != expected_provenance_fields
        or provenance.get("schema_version") != 1
        or provenance.get("method") != run.method
        or provenance.get("phase") != "formal_train"
        or provenance.get("training_seed") != run.training_seed
        or provenance.get("llm_rep") != run.llm_rep
        or provenance.get("run_id") is not None
        or provenance.get("freeze_id") != freeze_id
        or provenance.get("git_commit") != snapshot.git_commit
        or provenance.get("git_dirty") is not False
        or provenance.get("partition_sha256") != snapshot.partition_sha256
        or provenance.get("sealed_partition_metadata_sha256")
        != snapshot.sealed_partition_metadata_sha256
        or provenance.get("method_config_sha256")
        != snapshot.method_config_sha256[run.method]
        or provenance.get("base_config_sha256") != snapshot.base_config_sha256
        or provenance.get("effective_config_sha256")
        != snapshot.effective_config_sha256[run.method]
        or provenance.get("prompt_hashes") != expected_prompts
        or provenance.get("deepseek") != deepseek_provenance(enabled=bool(expected_roles))
        or type(provenance.get("resume_requested")) is not bool
        or provenance.get("locked_test_unlocked") is not False
    ):
        raise ValueError("formal provenance identity mismatch")

    completion_path = run_directory / "TRAINING_COMPLETE.json"
    completion = _read_json(completion_path, label="formal training completion")
    pause_names = sorted(path.name for path in run_directory.glob("PAUSED*.json"))
    if (
        set(completion)
        != {"status", "phase", "method", "training_seed", "llm_rep", "last_complete_round",
            "resolved_pause_reports", "resume_approved", "provenance", "evaluation_provenance",
            "result_status", "result_file", "result_sha256"}
        or completion.get("status") != "complete"
        or completion.get("phase") != "formal_train"
        or completion.get("method") != run.method
        or completion.get("training_seed") != run.training_seed
        or completion.get("llm_rep") != run.llm_rep
        or completion.get("last_complete_round") != 20
        or completion.get("resolved_pause_reports") != pause_names
        or type(completion.get("resume_approved")) is not bool
        or completion.get("resume_approved") != bool(pause_names)
        or completion.get("provenance") != "provenance.json"
        or completion.get("evaluation_provenance") is not None
        or completion.get("result_status") != "complete"
        or completion.get("result_file") != "validation_metrics.json"
        or completion.get("result_sha256") != file_sha256(validation_path)
    ):
        raise ValueError("formal training completion identity mismatch")

    checkpoint_path = run_directory / "last_complete.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    if (
        checkpoint.get("last_complete_round") != 20
        or checkpoint.get("freeze_id") != freeze_id
        or checkpoint.get("method") != run.method
        or checkpoint.get("training_seed") != run.training_seed
        or checkpoint.get("llm_rep") != run.llm_rep
        or checkpoint.get("partition_sha256") != snapshot.partition_sha256
        or checkpoint.get("config_sha256") != snapshot.effective_config_sha256[run.method]
        or checkpoint.get("prompt_hashes")
        != (expected_prompts or {"engine": "no-agent-prompts"})
    ):
        raise ValueError("formal checkpoint identity mismatch")
    return {
        "method": run.method,
        "training_seed": run.training_seed,
        "llm_rep": run.llm_rep,
        "completion_sha256": file_sha256(completion_path),
        "validation_sha256": file_sha256(validation_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }


def _training_batch_record(
    *,
    freeze_id: str,
    git_commit: str,
    training_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = [dict(record) for record in training_records]
    expected_count = 5 * sum(
        len(METHOD_REPETITIONS[method]) for method in FORMAL_METHOD_ORDER
    )
    identities = []
    for record in records:
        if (
            set(record)
            != {
                "method",
                "training_seed",
                "llm_rep",
                "completion_sha256",
                "validation_sha256",
                "checkpoint_sha256",
            }
            or any(
                type(record.get(field)) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", record[field])
                for field in (
                    "completion_sha256",
                    "validation_sha256",
                    "checkpoint_sha256",
                )
            )
        ):
            raise ValueError("formal training batch contains an invalid run record")
        identities.append(
            (record["method"], record["training_seed"], record["llm_rep"])
        )
    if len(records) != expected_count or len(set(identities)) != expected_count:
        raise ValueError("formal training batch requires 45 unique completed runs")
    return {
        "schema_version": 1,
        "status": "complete",
        "phase": "formal_train",
        "freeze_id": freeze_id,
        "git_commit": git_commit,
        "run_count": len(records),
        "runs": records,
    }


def _validate_training_batch_record(
    value: Any,
    *,
    freeze_id: str,
    git_commit: str,
    training_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = _training_batch_record(
        freeze_id=freeze_id,
        git_commit=git_commit,
        training_records=training_records,
    )
    if type(value) is not dict or value != expected:
        raise ValueError("formal training batch does not match 45 validated runs")
    return dict(value)


def _validate_evaluation_run(
    run: FormalRun,
    run_directory: Path,
    *,
    training_record: Mapping[str, Any],
) -> dict[str, Any]:
    locked_path = run_directory / "locked_test_metrics.json"
    locked = _read_json(locked_path, label="locked-test metrics")
    completion_path = run_directory / "EVALUATION_COMPLETE.json"
    completion = _read_json(completion_path, label="formal evaluation completion")
    evaluation_name = completion.get("evaluation_provenance")
    if (
        type(evaluation_name) is not str
        or not re.fullmatch(r"evaluation_provenance(?:\.\d{3})?\.json", evaluation_name)
    ):
        raise ValueError("formal evaluation provenance filename is invalid")
    evaluation_path = run_directory / evaluation_name
    evaluation = _read_json(evaluation_path, label="formal evaluation provenance")
    training_provenance = _read_json(
        run_directory / "provenance.json", label="formal training provenance"
    )
    checkpoint_sha256 = file_sha256(run_directory / "last_complete.pt")
    stable_fields = {
        "schema_version", "method", "training_seed", "llm_rep", "run_id",
        "freeze_id", "git_commit", "git_dirty", "partition_sha256",
        "sealed_partition_metadata_sha256", "method_config_sha256",
        "base_config_sha256", "effective_config_sha256", "prompt_hashes", "deepseek",
    }
    pause_names = sorted(path.name for path in run_directory.glob("PAUSED*.json"))
    if (
        set(locked)
        != {"schema_version", "phase", "method", "training_seed", "llm_rep",
            "training_checkpoint_sha256", "evaluation_provenance_sha256", "locked_test"}
        or locked.get("schema_version") != 1
        or locked.get("phase") != "formal_evaluate"
        or locked.get("method") != run.method
        or locked.get("training_seed") != run.training_seed
        or locked.get("llm_rep") != run.llm_rep
        or locked.get("training_checkpoint_sha256") != checkpoint_sha256
        or checkpoint_sha256 != training_record["checkpoint_sha256"]
        or locked.get("evaluation_provenance_sha256") != file_sha256(evaluation_path)
        or set(evaluation) != {
            *set(training_provenance), "training_checkpoint_sha256"
        }
        or any(evaluation.get(field) != training_provenance.get(field) for field in stable_fields)
        or evaluation.get("phase") != "formal_evaluate"
        or evaluation.get("method") != run.method
        or evaluation.get("training_seed") != run.training_seed
        or evaluation.get("llm_rep") != run.llm_rep
        or evaluation.get("freeze_id") != training_provenance.get("freeze_id")
        or evaluation.get("resume_requested") is not True
        or evaluation.get("locked_test_unlocked") is not True
        or evaluation.get("training_checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("formal evaluation provenance or locked-test identity mismatch")
    _validate_metrics(locked["locked_test"])
    if (
        set(completion)
        != {"status", "phase", "method", "training_seed", "llm_rep", "last_complete_round",
            "resolved_pause_reports", "resume_approved", "provenance", "evaluation_provenance",
            "result_status", "result_file", "result_sha256"}
        or completion.get("status") != "complete"
        or completion.get("phase") != "formal_evaluate"
        or completion.get("method") != run.method
        or completion.get("training_seed") != run.training_seed
        or completion.get("llm_rep") != run.llm_rep
        or completion.get("last_complete_round") != 20
        or completion.get("resolved_pause_reports") != pause_names
        or completion.get("resume_approved") is not True
        or completion.get("provenance") != "provenance.json"
        or completion.get("evaluation_provenance") != evaluation_name
        or completion.get("result_status") != "complete"
        or completion.get("result_file") != "locked_test_metrics.json"
        or completion.get("result_sha256") != file_sha256(locked_path)
    ):
        raise ValueError("formal evaluation completion identity mismatch")
    return {
        **dict(training_record),
        "evaluation_completion_sha256": file_sha256(completion_path),
        "locked_test_sha256": file_sha256(locked_path),
    }


def _command(
    run: FormalRun,
    *,
    phase: str,
    freeze_id: str,
    project_root: Path,
    python_executable: str,
) -> list[str]:
    command = [
        python_executable,
        str(project_root / "experiments/run_strict_federated.py"),
        "--method", run.method,
        "--phase", phase,
        "--training-seed", str(run.training_seed),
        "--llm-rep", str(run.llm_rep),
        "--freeze-id", freeze_id,
    ]
    if phase == "formal_evaluate":
        command.extend(
            [
                "--resume-checkpoint",
                str(_run_directory(project_root, freeze_id, run) / "last_complete.pt"),
                "--user-approved-resume",
                "--unlock-test",
            ]
        )
    return command


def _publish_json_no_replace(path: Path, record: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite immutable batch manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
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
    finally:
        temporary.unlink(missing_ok=True)


def _publish_or_validate_json(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if path.exists():
        existing = _read_json(path, label=label)
        if existing != dict(record):
            raise ValueError(f"existing {label} does not match validated evidence")
        return
    _publish_json_no_replace(path, record)


def run_formal_matrix(
    *,
    phase: str,
    freeze_id: str,
    project_root: Path = PROJECT_ROOT,
    command_runner: Callable[..., Any] = subprocess.run,
    python_executable: str = sys.executable,
) -> int:
    if phase not in {"formal_train", "formal_evaluate"}:
        raise ValueError("formal matrix phase is invalid")
    root = Path(project_root).resolve(strict=True)
    manifest = load_study_manifest(root / "study_manifest.yaml")
    if manifest.formal_frozen is not True or freeze_id not in manifest.paper_eligible_freeze_ids:
        raise ValueError("requested freeze is not active and paper eligible")
    matrix = build_run_matrix(manifest.formal_seeds)
    snapshot = _capture_snapshot(root)
    validation_args = argparse.Namespace(freeze_id=freeze_id)
    validate_formal_freeze(
        project_root=root,
        manifest=manifest,
        args=validation_args,
        git_commit=snapshot.git_commit,
        git_dirty=False,
    )

    training_records = []
    for run in matrix:
        directory = _run_directory(root, freeze_id, run)
        if not directory.exists():
            if phase == "formal_evaluate":
                return PAUSED_EXIT_CODE
            break
        try:
            training_records.append(
                _validate_training_run(
                    run,
                    directory,
                    freeze_id=freeze_id,
                    snapshot=snapshot,
                    allow_evaluation_artifacts=phase == "formal_evaluate",
                )
            )
        except (OSError, TypeError, ValueError):
            return PAUSED_EXIT_CODE
    training_batch_path = root / "results/formal" / freeze_id / "TRAINING_BATCH_COMPLETE.json"

    if phase == "formal_train" and len(training_records) == len(matrix):
        expected_batch = _training_batch_record(
            freeze_id=freeze_id,
            git_commit=snapshot.git_commit,
            training_records=training_records,
        )
        try:
            _publish_or_validate_json(
                training_batch_path,
                expected_batch,
                label="formal training batch",
            )
        except (OSError, TypeError, ValueError):
            return PAUSED_EXIT_CODE
        return 0

    if phase == "formal_evaluate":
        if len(training_records) != len(matrix) or not training_batch_path.is_file():
            return PAUSED_EXIT_CODE
        batch = _read_json(training_batch_path, label="formal training batch")
        try:
            _validate_training_batch_record(
                batch,
                freeze_id=freeze_id,
                git_commit=snapshot.git_commit,
                training_records=training_records,
            )
        except (TypeError, ValueError):
            return PAUSED_EXIT_CODE

    for index, run in enumerate(matrix):
        directory = _run_directory(root, freeze_id, run)
        if phase == "formal_train" and index < len(training_records):
            continue
        if phase == "formal_evaluate" and (directory / "EVALUATION_COMPLETE.json").exists():
            try:
                _validate_evaluation_run(run, directory, training_record=training_records[index])
            except (OSError, TypeError, ValueError):
                return PAUSED_EXIT_CODE
            continue
        if _capture_snapshot(root) != snapshot:
            return PAUSED_EXIT_CODE
        completed = command_runner(
            _command(
                run,
                phase=phase,
                freeze_id=freeze_id,
                project_root=root,
                python_executable=python_executable,
            ),
            cwd=root,
            check=False,
        )
        if type(completed.returncode) is not int:
            raise TypeError("subprocess returncode must be an exact integer")
        if completed.returncode != 0:
            return completed.returncode
        if _capture_snapshot(root) != snapshot:
            return PAUSED_EXIT_CODE
        try:
            if phase == "formal_train":
                training_records.append(
                    _validate_training_run(
                        run,
                        directory,
                        freeze_id=freeze_id,
                        snapshot=snapshot,
                    )
                )
            else:
                _validate_evaluation_run(
                    run, directory, training_record=training_records[index]
                )
        except (OSError, TypeError, ValueError):
            return PAUSED_EXIT_CODE

    if phase == "formal_train":
        _publish_or_validate_json(
            training_batch_path,
            _training_batch_record(
                freeze_id=freeze_id,
                git_commit=snapshot.git_commit,
                training_records=training_records,
            ),
            label="formal training batch",
        )
    else:
        evaluation_records = [
            _validate_evaluation_run(
                run,
                _run_directory(root, freeze_id, run),
                training_record=training_records[index],
            )
            for index, run in enumerate(matrix)
        ]
        evaluation_batch_path = (
            root / "results/formal" / freeze_id / "EVALUATION_BATCH_COMPLETE.json"
        )
        _publish_or_validate_json(
            evaluation_batch_path,
            {
                "schema_version": 1,
                "status": "complete",
                "phase": "formal_evaluate",
                "freeze_id": freeze_id,
                "run_count": len(matrix),
                "runs": evaluation_records,
            },
            label="formal evaluation batch",
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen FMAS-PCV formal matrix")
    parser.add_argument("--phase", choices=("formal_train", "formal_evaluate"), required=True)
    parser.add_argument("--freeze-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_formal_matrix(phase=args.phase, freeze_id=args.freeze_id)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return PAUSED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FormalRun",
    "PAUSED_EXIT_CODE",
    "build_parser",
    "build_run_matrix",
    "main",
    "run_formal_matrix",
]
