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


def _complete_evidence(unit, run_directory, *, mape, eligible=True):
    trajectory = tuple([1.0] * 19 + [mape])
    return calibration.CalibrationEvidence(
        unit=unit,
        metrics={
            "sample_count": 103, "mape": mape, "rmse": 2.0,
            "mae": 1.0, "r2": 0.0,
        },
        run_directory=run_directory,
        trajectory=trajectory,
        stability={
            "round_count": 20,
            "first_round_mape": 1.0,
            "best_round_index": 20,
            "best_round_mape": mape,
            "final_round_mape": mape,
            "max_round_mape": 1.0,
            "final_to_best_mape_ratio": 1.0,
            "best_improvement_from_first": 1.0 - mape,
            "checks": {
                "completed_rounds": True,
                "max_round_mape": eligible,
                "final_to_best_mape_ratio": eligible,
            },
            "eligible": eligible,
        },
    )


def test_calibration_script_is_directly_executable():
    completed = subprocess.run(
        [sys.executable, "scripts/run_fedyogi_calibration.py", "--help"],
        cwd=Path.cwd(), capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_calibration_config_is_exactly_preregistered():
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42_v2.yaml")
    )
    assert config == {
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


def test_matrix_is_fixed_serial_and_no_api_or_locked_test(tmp_path, monkeypatch):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42_v2.yaml")
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
        return _complete_evidence(
            unit, run_dir, mape=unit.server_lr, eligible=True
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
    output = tmp_path / "baseline_calibration_v2"
    result = calibration.run_calibration(config=config, project_root=Path.cwd(), output_root=output)
    assert calls == [0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
    assert result["selected_server_lr"] == 0.01
    assert result["selection_rule"] == ["mape", "rmse", "mae", "server_lr"]
    assert not list(output.rglob("*locked_test*"))
    assert not list(output.rglob("*agent_call*"))


def test_calibration_refuses_dirty_git_and_existing_output(tmp_path, monkeypatch):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42_v2.yaml")
    )
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
        Path("configs/fedyogi_calibration_seed42_v2.yaml")
    )
    supplied = tmp_path / "approved-copy.yaml"
    supplied.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical preregistered"):
        calibration.load_calibration_config(supplied)


def test_failed_unit_does_not_occupy_final_output(tmp_path, monkeypatch):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42_v2.yaml")
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
    output = tmp_path / "baseline_calibration_v2"
    with pytest.raises(RuntimeError, match="unit failed"):
        calibration.run_calibration(
            config=config, project_root=Path.cwd(), output_root=output
        )
    assert not output.exists()
    failed = list(tmp_path.glob("baseline_calibration_v2.failed-*"))
    assert len(failed) == 1
    assert json.loads((failed[0] / "FAILED.json").read_text(encoding="utf-8"))[
        "status"
    ] == "failed"


def test_exact_nonfinite_prediction_is_disqualified_with_evidence(
    tmp_path, monkeypatch
):
    output = tmp_path / "staging"
    output.mkdir()
    unit = calibration.CalibrationUnit(0.1, "fedyogi-v2-lr-0p1-seed42")
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
            '{"event":"round_committed","round_index":1,'
            '"selected_mape":1.0}\n', encoding="utf-8"
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
    assert evidence.trajectory == (1.0,)


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


def test_run_unit_rethrows_nonapproved_pause(tmp_path, monkeypatch):
    output = tmp_path / "staging"
    output.mkdir()
    unit = calibration.CalibrationUnit(0.03, "fedyogi-v2-lr-0p03-seed42")
    snapshot = {
        "git_commit": "a" * 40, "git_dirty": False,
        "calibration_config_sha256": "b" * 64,
        "base_config_sha256": "c" * 64,
        "method_config_sha256": "d" * 64,
        "partition_sha256": "e" * 64,
        "sealed_partition_metadata_sha256": "f" * 64,
    }

    def paused_training(context):
        report_path = context.run_directory / "PAUSED.json"
        cause = ValueError("different runtime failure")
        paused = ExperimentPaused(
            ExperimentRuntimeError(
                "ValueError", "round stopped after a sanitized runtime failure"
            ),
            report_path,
        )
        paused.__cause__ = cause
        raise paused

    monkeypatch.setattr(calibration, "execute_strict_training", paused_training)
    with pytest.raises(ExperimentPaused):
        calibration._run_unit(
            unit, project_root=Path.cwd(), output_root=output, snapshot=snapshot
        )


def test_unstable_better_point_is_not_selected(
    tmp_path, monkeypatch
):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42_v2.yaml")
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
        mape = {
            0.01: 0.99, 0.02: 0.80, 0.03: 0.70,
            0.05: 0.60, 0.075: 0.50, 0.1: 0.40,
        }[unit.server_lr]
        return _complete_evidence(
            unit, run_dir, mape=mape, eligible=unit.server_lr != 0.1
        )

    monkeypatch.setattr(calibration, "_run_unit", fake_run)
    output = tmp_path / "baseline_calibration_v2"
    result = calibration.run_calibration(
        config=config, project_root=Path.cwd(), output_root=output
    )
    assert result["selected_server_lr"] == 0.075
    assert result["recommended_freeze_ready"] is True
    assert result["runs"][-1]["status"] == "complete"
    assert result["runs"][-1]["stability_eligible"] is False
    assert result["failure_policy"] == config["failure_policy"]


def test_round_trajectory_and_stability_are_strict(tmp_path):
    rounds = tmp_path / "rounds.jsonl"
    records = [
        {
            "event": "round_committed",
            "round_index": index,
            "selected_mape": 1.0 - index * 0.01,
        }
        for index in range(1, 21)
    ]
    rounds.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    trajectory = calibration._read_selected_mape_trajectory(
        rounds, expected_rounds=20
    )
    stability = calibration._assess_stability(
        trajectory,
        {
            "required_completed_rounds": 20,
            "max_round_mape": 2.0,
            "max_final_to_best_mape_ratio": 1.5,
            "ratio_denominator_floor": 1.0e-12,
        },
    )
    assert len(trajectory) == 20
    assert stability["eligible"] is True
    assert stability["best_round_index"] == 20

    records[-1]["round_index"] = 19
    rounds.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contiguous"):
        calibration._read_selected_mape_trajectory(rounds, expected_rounds=20)

    records[-1]["round_index"] = 20
    records[0]["round_index"] = True
    rounds.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contiguous"):
        calibration._read_selected_mape_trajectory(rounds, expected_rounds=20)


@pytest.mark.parametrize("invalid_mape", [True, -0.1, float("nan"), float("inf")])
def test_round_trajectory_rejects_invalid_mape(tmp_path, invalid_mape):
    rounds = tmp_path / "rounds.jsonl"
    rounds.write_text(json.dumps({
        "event": "round_committed",
        "round_index": 1,
        "selected_mape": invalid_mape,
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite and non-negative"):
        calibration._read_selected_mape_trajectory(rounds, expected_rounds=1)


def test_exploding_trajectory_is_stability_ineligible():
    trajectory = (
        1.0, 0.99, 0.98, 0.90, 8.0, 100.0, 1000.0,
        2000.0, 4000.0, 12000.0, 7000.0, 5000.0, 4000.0,
        3000.0, 2000.0, 1500.0, 1200.0, 1100.0, 1050.0, 1288.0,
    )
    stability = calibration._assess_stability(
        trajectory,
        {
            "required_completed_rounds": 20,
            "max_round_mape": 2.0,
            "max_final_to_best_mape_ratio": 1.5,
            "ratio_denominator_floor": 1.0e-12,
        },
    )
    assert stability["eligible"] is False
    assert stability["checks"] == {
        "completed_rounds": True,
        "max_round_mape": False,
        "final_to_best_mape_ratio": False,
    }


def test_zero_error_trajectory_has_zero_improvement():
    stability = calibration._assess_stability(
        (0.0,) * 20,
        {
            "required_completed_rounds": 20,
            "max_round_mape": 2.0,
            "max_final_to_best_mape_ratio": 1.5,
            "ratio_denominator_floor": 1.0e-12,
        },
    )
    assert stability["eligible"] is True
    assert stability["best_improvement_from_first"] == 0.0


def test_stable_but_stalled_selection_is_not_freeze_ready(tmp_path, monkeypatch):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42_v2.yaml")
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
        return _complete_evidence(unit, run_dir, mape=0.99, eligible=True)

    monkeypatch.setattr(calibration, "_run_unit", fake_run)
    result = calibration.run_calibration(
        config=config,
        project_root=Path.cwd(),
        output_root=tmp_path / "baseline_calibration_v2",
    )
    assert result["selected_server_lr"] == 0.01
    assert result["recommended_freeze_ready"] is False
    assert result["calibration_outcome"] == "selected_stable_but_stalled_point"


def test_all_unstable_points_abort_without_final_output(tmp_path, monkeypatch):
    config = calibration.load_calibration_config(
        Path("configs/fedyogi_calibration_seed42_v2.yaml")
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
        return _complete_evidence(unit, run_dir, mape=0.4, eligible=False)

    monkeypatch.setattr(calibration, "_run_unit", fake_run)
    output = tmp_path / "baseline_calibration_v2"
    with pytest.raises(RuntimeError, match="no stable eligible"):
        calibration.run_calibration(
            config=config, project_root=Path.cwd(), output_root=output
        )
    assert not output.exists()
    assert len(list(tmp_path.glob("baseline_calibration_v2.failed-*"))) == 1
