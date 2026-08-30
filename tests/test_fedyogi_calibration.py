import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts import run_fedyogi_calibration as calibration
from src.federated_learning.pcv.engine import (
    ExperimentPaused,
    ExperimentRuntimeError,
)


def test_calibration_script_is_directly_executable():
    completed = subprocess.run(
        [sys.executable, "scripts/run_fedyogi_calibration.py", "--help"],
        cwd=Path.cwd(), capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_calibration_config_is_exactly_preregistered():
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42.yaml")
    )
    assert config == {
        "schema_version": 1,
        "phase": "development",
        "method": "FEDYOGI_STRICT",
        "training_seed": 42,
        "num_rounds": 20,
        "partition_manifest": "results/manifests/strict_partition_v1.csv",
        "selection_partition": "controller_validation",
        "output_root": "results/development/seed42/baseline_calibration",
        "selection_rule": ["mape", "rmse", "mae", "server_lr"],
        "failure_policy": {
            "approval": "user_approved_2026-08-30",
            "nonfinite_prediction": (
                "disqualify_without_retry_or_grid_replacement"
            ),
            "all_other_failures": "abort_calibration",
        },
        "grid": {
            "server_lr": [0.01, 0.1, 0.5],
            "beta1": 0.9,
            "beta2": 0.99,
            "tau": 0.001,
            "max_coordinate_step_ratio": None,
            "clip": None,
        },
    }


def test_matrix_is_fixed_serial_and_no_api_or_locked_test(tmp_path, monkeypatch):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42.yaml")
    )
    calls = []

    def fake_run(unit, *, project_root, output_root, snapshot):
        calls.append(unit.server_lr)
        run_dir = output_root / unit.run_id
        run_dir.mkdir()
        for name, value in {
            "provenance.json": {"locked_test_unlocked": False, "deepseek": {"enabled": False}},
            "rounds.jsonl": {},
            "last_complete.pt": {},
            "validation_metrics.json": {
                "status": "complete",
                "best_validation": {"sample_count": 3, "mape": unit.server_lr, "rmse": 2.0, "mae": 1.0, "r2": 0.0},
            },
        }.items():
            path = run_dir / name
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return calibration.CalibrationEvidence(
            unit=unit,
            metrics={"sample_count": 3, "mape": unit.server_lr, "rmse": 2.0, "mae": 1.0, "r2": 0.0},
            run_directory=run_dir,
        )

    monkeypatch.setattr(calibration, "_run_unit", fake_run)
    monkeypatch.setattr(calibration, "_capture_snapshot", lambda *_: {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "calibration_config_sha256": "b" * 64,
        "base_config_sha256": "c" * 64,
        "method_config_sha256": "d" * 64,
        "partition_sha256": "e" * 64,
        "sealed_partition_metadata_sha256": "f" * 64,
    })
    output = tmp_path / "baseline_calibration"
    result = calibration.run_calibration(config=config, project_root=Path.cwd(), output_root=output)
    assert calls == [0.01, 0.1, 0.5]
    assert result["selected_server_lr"] == 0.01
    assert result["selection_rule"] == ["mape", "rmse", "mae", "server_lr"]
    assert not list(output.rglob("*locked_test*"))
    assert not list(output.rglob("*agent_call*"))


def test_calibration_refuses_dirty_git_and_existing_output(tmp_path, monkeypatch):
    config = calibration.load_calibration_config(Path("configs/fedyogi_calibration_seed42.yaml"))
    monkeypatch.setattr(calibration, "_capture_snapshot", lambda *_: {
        "git_commit": "a" * 40, "git_dirty": True,
        "calibration_config_sha256": "b" * 64,
        "base_config_sha256": "c" * 64,
        "method_config_sha256": "d" * 64,
        "partition_sha256": "e" * 64,
        "sealed_partition_metadata_sha256": "f" * 64,
    })
    with pytest.raises(ValueError, match="clean Git"):
        calibration.run_calibration(config=config, project_root=Path.cwd(), output_root=tmp_path / "new")
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(calibration, "_capture_snapshot", lambda *_: {
        "git_commit": "a" * 40, "git_dirty": False,
        "calibration_config_sha256": "b" * 64,
        "base_config_sha256": "c" * 64,
        "method_config_sha256": "d" * 64,
        "partition_sha256": "e" * 64,
        "sealed_partition_metadata_sha256": "f" * 64,
    })
    with pytest.raises(FileExistsError):
        calibration.run_calibration(config=config, project_root=Path.cwd(), output_root=existing)


def test_noncanonical_config_copy_is_rejected(tmp_path):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42.yaml")
    )
    supplied = tmp_path / "approved-copy.yaml"
    supplied.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical preregistered"):
        calibration.load_calibration_config(supplied)


def test_failed_unit_does_not_occupy_final_output(tmp_path, monkeypatch):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42.yaml")
    )
    snapshot = {
        "git_commit": "a" * 40, "git_dirty": False,
        "calibration_config_sha256": "b" * 64,
        "base_config_sha256": "c" * 64,
        "method_config_sha256": "d" * 64,
        "partition_sha256": "e" * 64,
        "sealed_partition_metadata_sha256": "f" * 64,
    }
    monkeypatch.setattr(calibration, "_capture_snapshot", lambda *_: snapshot)
    monkeypatch.setattr(
        calibration, "_run_unit",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unit failed")),
    )
    output = tmp_path / "baseline_calibration"
    with pytest.raises(RuntimeError, match="unit failed"):
        calibration.run_calibration(
            config=config, project_root=Path.cwd(), output_root=output
        )
    assert not output.exists()
    failed = list(tmp_path.glob("baseline_calibration.failed-*"))
    assert len(failed) == 1
    assert json.loads((failed[0] / "FAILED.json").read_text(encoding="utf-8"))[
        "status"
    ] == "failed"


def test_exact_nonfinite_prediction_is_disqualified_with_evidence(
    tmp_path, monkeypatch
):
    output = tmp_path / "staging"
    output.mkdir()
    unit = calibration.CalibrationUnit(0.5, "fedyogi-lr-0p5-seed42")
    snapshot = {
        "git_commit": "a" * 40, "git_dirty": False,
        "calibration_config_sha256": "b" * 64,
        "base_config_sha256": "c" * 64,
        "method_config_sha256": "d" * 64,
        "partition_sha256": "e" * 64,
        "sealed_partition_metadata_sha256": "f" * 64,
    }

    def paused_training(context):
        (context.run_directory / "rounds.jsonl").write_text(
            '{"round_index":1}\n', encoding="utf-8"
        )
        (context.run_directory / "last_complete.pt").write_bytes(b"checkpoint")
        report = {
            "status": "paused", "failed_round": 2,
            "last_complete_round": 1,
            "failure": {
                "category": "runtime", "exception_type": "ValueError",
                "role": "engine",
            },
        }
        report_path = context.run_directory / "PAUSED.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        cause = ValueError("y_pred must contain only finite values")
        try:
            raise cause
        except ValueError as error:
            raise ExperimentPaused(
                ExperimentRuntimeError(
                    type(error).__name__,
                    "round stopped after a sanitized runtime failure",
                ),
                report_path,
            ) from error

    monkeypatch.setattr(calibration, "execute_strict_training", paused_training)
    evidence = calibration._run_unit(
        unit, project_root=Path.cwd(), output_root=output, snapshot=snapshot
    )
    assert evidence.status == "disqualified_numeric_divergence"
    assert evidence.metrics is None
    assert evidence.failure == {
        "failed_round": 2,
        "last_complete_round": 1,
        "reason": "nonfinite_prediction",
    }


def test_sanitized_nonfinite_disqualification_requires_exact_original_cause(
    tmp_path,
):
    run_directory = tmp_path / "fedyogi-lr-0p5-seed42"
    run_directory.mkdir()
    report = {
        "status": "paused", "failed_round": 2,
        "last_complete_round": 1,
        "failure": {
            "category": "runtime", "exception_type": "ValueError",
            "role": "engine",
        },
    }
    report_path = run_directory / "PAUSED.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    paused = ExperimentPaused(
        ExperimentRuntimeError(
            "ValueError", "round stopped after a sanitized runtime failure"
        ),
        report_path,
    )
    paused.__cause__ = ValueError("y_pred must contain only finite values")

    assert calibration._approved_numeric_divergence(paused, run_directory) == {
        "failed_round": 2,
        "last_complete_round": 1,
        "reason": "nonfinite_prediction",
    }


def test_other_value_error_is_not_disqualified_as_numeric_divergence(tmp_path):
    run_directory = tmp_path / "fedyogi-lr-0p5-seed42"
    run_directory.mkdir()
    report_path = run_directory / "PAUSED.json"
    report_path.write_text(json.dumps({
        "status": "paused", "failed_round": 2,
        "last_complete_round": 1,
        "failure": {
            "category": "runtime", "exception_type": "ValueError",
            "role": "engine",
        },
    }), encoding="utf-8")
    paused = ExperimentPaused(
        ExperimentRuntimeError(
            "ValueError", "round stopped after a sanitized runtime failure"
        ),
        report_path,
    )
    paused.__cause__ = ValueError("aggregate metrics must contain finite values")

    assert calibration._approved_numeric_divergence(paused, run_directory) is None


def test_disqualified_grid_point_is_recorded_but_not_selected(
    tmp_path, monkeypatch
):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42.yaml")
    )
    monkeypatch.setattr(calibration, "_capture_snapshot", lambda *_: {
        "git_commit": "a" * 40, "git_dirty": False,
        "calibration_config_sha256": "b" * 64,
        "base_config_sha256": "c" * 64,
        "method_config_sha256": "d" * 64,
        "partition_sha256": "e" * 64,
        "sealed_partition_metadata_sha256": "f" * 64,
    })

    def fake_run(unit, *, output_root, **kwargs):
        run_dir = output_root / unit.run_id
        run_dir.mkdir()
        if unit.server_lr == 0.5:
            return calibration.CalibrationEvidence(
                unit=unit, metrics=None, run_directory=run_dir,
                status="disqualified_numeric_divergence",
                failure={
                    "failed_round": 2, "last_complete_round": 1,
                    "reason": "nonfinite_prediction",
                },
            )
        mape = {0.01: 0.998714, 0.1: 0.902689}[unit.server_lr]
        return calibration.CalibrationEvidence(
            unit=unit,
            metrics={
                "sample_count": 103, "mape": mape, "rmse": 1.0,
                "mae": 1.0, "r2": 0.0,
            },
            run_directory=run_dir,
        )

    monkeypatch.setattr(calibration, "_run_unit", fake_run)
    output = tmp_path / "baseline_calibration"
    result = calibration.run_calibration(
        config=config, project_root=Path.cwd(), output_root=output
    )
    assert result["selected_server_lr"] == 0.1
    assert [item["status"] for item in result["runs"]] == [
        "complete", "complete", "disqualified_numeric_divergence"
    ]
    assert result["failure_policy"] == config["failure_policy"]
