import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts import run_fedyogi_calibration as calibration


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
