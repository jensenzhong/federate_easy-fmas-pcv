import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import scripts.run_fmas_development as development_module
from scripts.run_fmas_development import (
    PAUSED_EXIT_CODE,
    build_run_matrix,
    main,
    run_development_matrix,
)


EXPECTED = [
    ("FEDAVG_STRICT", 0),
    ("FEDYOGI_STRICT", 0),
    ("DPCV_FEDYOGI", 0),
    ("SA_PCV_FEDYOGI", 1),
    ("SA_PCV_FEDYOGI", 2),
    ("SA_PCV_FEDYOGI", 3),
    ("FMAS_PCV_FEDYOGI", 1),
    ("FMAS_PCV_FEDYOGI", 2),
    ("FMAS_PCV_FEDYOGI", 3),
]


def test_launcher_help_runs_through_the_real_script_entrypoint():
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/run_fmas_development.py", "--help"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run the approved strict-federated seed-42 development matrix" in (
        completed.stdout
    )


@pytest.fixture(autouse=True)
def _stable_git_state(monkeypatch):
    state = {"commit": "b" * 40, "dirty": False}
    monkeypatch.setattr(
        development_module,
        "_read_git_state",
        lambda _root: (state["commit"], state["dirty"]),
        raising=False,
    )
    return state


def _write_config(root: Path, *, seed: int = 42, phase: str = "development") -> Path:
    path = root / "configs" / "development_seed42.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"phase: {phase}",
                f"training_seed: {seed}",
                "partition_manifest: results/manifests/strict_partition_v1.csv",
                "base_config: configs/config.yaml",
                "output_root: results/development/seed42",
                "deepseek:",
                "  model: deepseek-v4-flash",
                "  base_url: https://api.deepseek.com",
                "  timeout_seconds: 120",
                "development_gate:",
                "  baseline_selection: lowest_mape_of_strict_baselines",
                "  relative_mape_improvement_min: 0.0",
                "  rmse_increase_ratio_max: 0.05",
                "  r2_difference_min: -0.02",
                "  required_passing_fmas_repetitions: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "experiments").mkdir(exist_ok=True)
    (root / "experiments" / "run_strict_federated.py").write_text(
        "# test sentinel\n", encoding="utf-8"
    )
    related_files = {
        "results/manifests/strict_partition_v1.csv": "partition-sentinel\n",
        "Data/strict_partition_v1/metadata.json": '{"sealed":true}\n',
        "configs/config.yaml": "base: strict\n",
        "configs/methods/fedavg_strict.yaml": "method: FEDAVG_STRICT\n",
        "configs/methods/fedyogi_strict.yaml": "method: FEDYOGI_STRICT\n",
        "configs/methods/dpcv_fedyogi.yaml": "method: DPCV_FEDYOGI\n",
        "configs/methods/sa_pcv_fedyogi.yaml": "method: SA_PCV_FEDYOGI\n",
        "configs/methods/fmas_pcv_fedyogi.yaml": "method: FMAS_PCV_FEDYOGI\n",
    }
    for role in {role for roles in ROLE_NAMES.values() for role in roles}:
        related_files[f"configs/prompts/{role}.md"] = f"{role} prompt\n"
    for relative, content in related_files.items():
        related = root / relative
        related.parent.mkdir(parents=True, exist_ok=True)
        related.write_text(content, encoding="utf-8")
    return path


ROLE_NAMES = {
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


def _default_metrics(run):
    values = {
        ("FEDAVG_STRICT", 0): (0.12, 1.2, 0.70),
        ("FEDYOGI_STRICT", 0): (0.10, 1.0, 0.75),
        ("DPCV_FEDYOGI", 0): (0.095, 1.01, 0.76),
        ("SA_PCV_FEDYOGI", 1): (0.09, 1.04, 0.74),
        ("SA_PCV_FEDYOGI", 2): (0.11, 1.00, 0.76),
        ("SA_PCV_FEDYOGI", 3): (0.10, 1.049, 0.73),
        ("FMAS_PCV_FEDYOGI", 1): (0.08, 1.02, 0.77),
        # Exactly on all three inclusive gate boundaries relative to FedYogi.
        ("FMAS_PCV_FEDYOGI", 2): (0.10, 1.05, 0.73),
        ("FMAS_PCV_FEDYOGI", 3): (0.11, 1.00, 0.78),
    }
    mape, rmse, r2 = values[(run.method, run.llm_rep)]
    return {"sample_count": 103, "mape": mape, "rmse": rmse, "mae": 0.5, "r2": r2}


def _completion(
    run,
    run_directory: Path,
    *,
    resolved=(),
    metrics=None,
    partition_sha=None,
    git_dirty=False,
    git_commit="b" * 40,
):
    run_directory.mkdir(parents=True, exist_ok=True)
    validation = {
        "status": "complete",
        "phase": "development",
        "method": run.method,
        "training_seed": run.training_seed,
        "llm_rep": run.llm_rep,
        "completed_rounds": 20,
        "best_validation": metrics or _default_metrics(run),
    }
    validation_path = run_directory / "validation_metrics.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    root = run_directory.parents[3]
    method_paths = {
        "FEDAVG_STRICT": "configs/methods/fedavg_strict.yaml",
        "FEDYOGI_STRICT": "configs/methods/fedyogi_strict.yaml",
        "DPCV_FEDYOGI": "configs/methods/dpcv_fedyogi.yaml",
        "SA_PCV_FEDYOGI": "configs/methods/sa_pcv_fedyogi.yaml",
        "FMAS_PCV_FEDYOGI": "configs/methods/fmas_pcv_fedyogi.yaml",
    }
    base_bytes = (root / "configs/config.yaml").read_bytes()
    method_bytes = (root / method_paths[run.method]).read_bytes()
    effective = hashlib.sha256()
    for label, content in ((b"base-config", base_bytes), (b"method-config", method_bytes)):
        effective.update(len(label).to_bytes(4, "big"))
        effective.update(label)
        effective.update(len(content).to_bytes(8, "big"))
        effective.update(content)
    provenance = {
        "schema_version": 1,
        "method": run.method,
        "phase": "development",
        "training_seed": run.training_seed,
        "llm_rep": run.llm_rep,
        "run_id": run.run_id,
        "freeze_id": None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "partition_sha256": partition_sha or hashlib.sha256(
            (root / "results/manifests/strict_partition_v1.csv").read_bytes()
        ).hexdigest(),
        "sealed_partition_metadata_sha256": hashlib.sha256(
            (root / "Data/strict_partition_v1/metadata.json").read_bytes()
        ).hexdigest(),
        "method_config_sha256": hashlib.sha256(method_bytes).hexdigest(),
        "base_config_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "effective_config_sha256": effective.hexdigest(),
        "prompt_hashes": {
            role: hashlib.sha256(
                (root / f"configs/prompts/{role}.md").read_bytes()
            ).hexdigest()
            for role in ROLE_NAMES[run.method]
        },
        "deepseek": {
            "enabled": bool(ROLE_NAMES[run.method]),
            "model": "deepseek-v4-flash" if ROLE_NAMES[run.method] else None,
            "base_url": "https://api.deepseek.com" if ROLE_NAMES[run.method] else None,
            "temperature": 0.8 if ROLE_NAMES[run.method] else None,
            "timeout_seconds": 120 if ROLE_NAMES[run.method] else None,
        },
        "resume_requested": False,
        "locked_test_unlocked": False,
    }
    (run_directory / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    (run_directory / "TRAINING_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "phase": "development",
                "method": run.method,
                "training_seed": run.training_seed,
                "llm_rep": run.llm_rep,
                "last_complete_round": 20,
                "resolved_pause_reports": list(resolved),
                "resume_approved": False,
                "provenance": "provenance.json",
                "evaluation_provenance": None,
                "result_status": "complete",
                "result_file": "validation_metrics.json",
                "result_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_development_matrix_is_exact_and_predeclared():
    matrix = build_run_matrix(training_seed=42)
    assert [(row.method, row.llm_rep) for row in matrix] == EXPECTED
    assert all(row.training_seed == 42 for row in matrix)
    assert len({row.run_id for row in matrix}) == 9


@pytest.mark.parametrize("seed", [0, 41, 43, 314])
def test_development_matrix_rejects_every_nondevelopment_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        build_run_matrix(training_seed=seed)


def test_launcher_runs_exact_commands_serially_and_never_requests_formal_or_test(tmp_path):
    config = _write_config(tmp_path)
    observed = []
    matrix = build_run_matrix(training_seed=42)

    def fake_run(command, *, cwd, check):
        assert cwd == tmp_path
        assert check is False
        observed.append(tuple(command))
        run = matrix[len(observed) - 1]
        # A completion appears only when this blocking subprocess returns. This
        # makes any accidental parallel scheduling fail the next precondition.
        _completion(run, tmp_path / "results/development/seed42" / run.run_id)
        return SimpleNamespace(returncode=0)

    assert (
        run_development_matrix(
            training_seed=42,
            config_path=config,
            project_root=tmp_path,
            command_runner=fake_run,
            python_executable="python-test",
        )
        == 0
    )
    assert len(observed) == 9
    for command, run in zip(observed, matrix, strict=True):
        assert command[:2] == (
            "python-test",
            str(tmp_path / "experiments/run_strict_federated.py"),
        )
        assert command[2:] == (
            "--method",
            run.method,
            "--phase",
            "development",
            "--training-seed",
            "42",
            "--llm-rep",
            str(run.llm_rep),
            "--run-id",
            run.run_id,
        )
        assert "formal_train" not in command
        assert "formal_evaluate" not in command
        assert "--unlock-test" not in command
    assert [command[-1] for command in observed] == [run.run_id for run in matrix]
    gate = json.loads(
        (tmp_path / "results/development/seed42/development_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["baseline"]["method"] == "FEDYOGI_STRICT"
    assert [(row["method"], row["llm_rep"]) for row in gate["trajectories"]] == EXPECTED[3:]
    assert gate["passing_fmas_repetitions"] == 2
    assert gate["gate_passed"] is True
    boundary = next(
        row
        for row in gate["trajectories"]
        if row["method"] == "FMAS_PCV_FEDYOGI" and row["llm_rep"] == 2
    )
    assert math.isclose(boundary["relative_mape_improvement"], 0.0, abs_tol=1e-12)
    assert math.isclose(boundary["rmse_increase_ratio"], 0.05, abs_tol=1e-12)
    assert math.isclose(boundary["r2_difference"], -0.02, abs_tol=1e-12)
    assert boundary["passed"] is True
    assert gate["evidence"] == {
        "partition": "controller_validation",
        "locked_test_used": False,
    }
    assert gate["partition_sha256"] == hashlib.sha256(
        (tmp_path / "results/manifests/strict_partition_v1.csv").read_bytes()
    ).hexdigest()
    assert len(gate["config_sha256"]) == 64


def test_nonzero_exit_stops_immediately_and_returns_same_status(tmp_path):
    config = _write_config(tmp_path)
    matrix = build_run_matrix(training_seed=42)
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append(command)
        run = matrix[len(calls) - 1]
        if len(calls) == 2:
            return SimpleNamespace(returncode=17)
        _completion(run, tmp_path / "results/development/seed42" / run.run_id)
        return SimpleNamespace(returncode=0)

    assert run_development_matrix(
        training_seed=42,
        config_path=config,
        project_root=tmp_path,
        command_runner=fake_run,
    ) == 17
    assert len(calls) == 2


def test_zero_exit_with_unresolved_pause_stops_before_next_run(tmp_path):
    config = _write_config(tmp_path)
    matrix = build_run_matrix(training_seed=42)
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append(command)
        run = matrix[0]
        run_dir = tmp_path / "results/development/seed42" / run.run_id
        _completion(run, run_dir)
        (run_dir / "PAUSED.json").write_text('{"status":"paused"}', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    assert run_development_matrix(
        training_seed=42,
        config_path=config,
        project_root=tmp_path,
        command_runner=fake_run,
    ) == PAUSED_EXIT_CODE
    assert len(calls) == 1


def test_historical_pause_named_by_completion_is_terminal_success(tmp_path):
    config = _write_config(tmp_path)
    matrix = build_run_matrix(training_seed=42)
    calls = []

    def fake_run(command, *, cwd, check):
        run = matrix[len(calls)]
        calls.append(command)
        run_dir = tmp_path / "results/development/seed42" / run.run_id
        (run_dir / "PAUSED.json").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "PAUSED.json").write_text('{"status":"paused"}', encoding="utf-8")
        _completion(run, run_dir, resolved=("PAUSED.json",))
        return SimpleNamespace(returncode=0)

    assert run_development_matrix(
        training_seed=42,
        config_path=config,
        project_root=tmp_path,
        command_runner=fake_run,
    ) == 0
    assert len(calls) == 9


def test_minimal_fake_completion_cannot_skip_a_run(tmp_path):
    config = _write_config(tmp_path)
    first = build_run_matrix(training_seed=42)[0]

    def fake_run(command, *, cwd, check):
        run_dir = tmp_path / "results/development/seed42" / first.run_id
        run_dir.mkdir(parents=True)
        (run_dir / "TRAINING_COMPLETE.json").write_text(
            '{"status":"complete"}', encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    assert run_development_matrix(
        training_seed=42,
        config_path=config,
        project_root=tmp_path,
        command_runner=fake_run,
    ) == PAUSED_EXIT_CODE


@pytest.mark.parametrize(
    "corruption", ["sha", "provenance", "rounds", "nan", "dirty", "git_commit"]
)
def test_gate_rejects_inconsistent_or_nonfinite_run_evidence(tmp_path, corruption):
    config = _write_config(tmp_path)
    matrix = build_run_matrix(training_seed=42)
    calls = []

    def fake_run(command, *, cwd, check):
        run = matrix[len(calls)]
        calls.append(command)
        run_dir = tmp_path / "results/development/seed42" / run.run_id
        metrics = _default_metrics(run)
        if corruption == "nan" and len(calls) == 9:
            metrics = dict(metrics, mape=math.nan)
        _completion(
            run,
            run_dir,
            metrics=metrics,
            partition_sha=(
                "9" * 64 if corruption == "provenance" and len(calls) == 9 else None
            ),
            git_dirty=corruption == "dirty" and len(calls) == 9,
            git_commit=("8" * 40 if corruption == "git_commit" and len(calls) == 9 else "b" * 40),
        )
        if corruption == "sha" and len(calls) == 9:
            completion = json.loads((run_dir / "TRAINING_COMPLETE.json").read_text())
            completion["result_sha256"] = "0" * 64
            (run_dir / "TRAINING_COMPLETE.json").write_text(json.dumps(completion))
        if corruption == "rounds" and len(calls) == 9:
            completion = json.loads((run_dir / "TRAINING_COMPLETE.json").read_text())
            completion["last_complete_round"] = 19
            (run_dir / "TRAINING_COMPLETE.json").write_text(json.dumps(completion))
        return SimpleNamespace(returncode=0)

    assert run_development_matrix(
        training_seed=42,
        config_path=config,
        project_root=tmp_path,
        command_runner=fake_run,
    ) == PAUSED_EXIT_CODE
    assert not (tmp_path / "results/development/seed42/development_gate.json").exists()


def test_related_file_hash_drift_stops_after_current_process(tmp_path):
    config = _write_config(tmp_path)
    first = build_run_matrix(training_seed=42)[0]
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append(command)
        _completion(first, tmp_path / "results/development/seed42" / first.run_id)
        (tmp_path / "configs/config.yaml").write_text("drifted: true\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    assert run_development_matrix(
        training_seed=42,
        config_path=config,
        project_root=tmp_path,
        command_runner=fake_run,
    ) == PAUSED_EXIT_CODE
    assert len(calls) == 1


def test_clean_git_commit_drift_stops_after_current_process(
    tmp_path, _stable_git_state
):
    config = _write_config(tmp_path)
    matrix = build_run_matrix(training_seed=42)
    calls = []

    def fake_run(command, *, cwd, check):
        run = matrix[len(calls)]
        calls.append(command)
        _completion(
            run,
            tmp_path / "results/development/seed42" / run.run_id,
            git_commit=_stable_git_state["commit"],
        )
        if len(calls) == 1:
            _stable_git_state["commit"] = "c" * 40
        return SimpleNamespace(returncode=0)

    assert run_development_matrix(
        training_seed=42,
        config_path=config,
        project_root=tmp_path,
        command_runner=fake_run,
    ) == PAUSED_EXIT_CODE
    assert len(calls) == 1


def test_final_gate_rereads_earlier_evidence(tmp_path):
    config = _write_config(tmp_path)
    matrix = build_run_matrix(training_seed=42)

    def fake_run(command, *, cwd, check):
        run = matrix[fake_run.calls]
        fake_run.calls += 1
        _completion(run, tmp_path / "results/development/seed42" / run.run_id)
        if fake_run.calls == 9:
            first_path = (
                tmp_path
                / "results/development/seed42"
                / matrix[0].run_id
                / "validation_metrics.json"
            )
            first_path.write_text('{"tampered":true}', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    fake_run.calls = 0
    assert run_development_matrix(
        training_seed=42,
        config_path=config,
        project_root=tmp_path,
        command_runner=fake_run,
    ) == PAUSED_EXIT_CODE
    assert not (tmp_path / "results/development/seed42/development_gate.json").exists()


def test_output_root_symlink_is_refused_before_launch(tmp_path):
    config = _write_config(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    output_root = tmp_path / "results/development/seed42"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_root.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this Windows host")

    with pytest.raises(ValueError, match="link, junction"):
        run_development_matrix(
            training_seed=42,
            config_path=config,
            project_root=tmp_path,
            command_runner=lambda *args, **kwargs: pytest.fail("must not launch"),
        )


def test_locked_test_artifact_is_rejected_without_opening_it(tmp_path):
    config = _write_config(tmp_path)
    matrix = build_run_matrix(training_seed=42)

    def fake_run(command, *, cwd, check):
        run = matrix[fake_run.calls]
        fake_run.calls += 1
        run_dir = tmp_path / "results/development/seed42" / run.run_id
        _completion(run, run_dir)
        if fake_run.calls == 9:
            (run_dir / "locked_test_metrics.json").write_text("do-not-read", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    fake_run.calls = 0
    assert run_development_matrix(
        training_seed=42,
        config_path=config,
        project_root=tmp_path,
        command_runner=fake_run,
    ) == PAUSED_EXIT_CODE
    assert not (tmp_path / "results/development/seed42/development_gate.json").exists()


def test_existing_gate_is_never_overwritten_or_reexecuted(tmp_path):
    config = _write_config(tmp_path)
    gate = tmp_path / "results/development/seed42/development_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(b"original")

    with pytest.raises(FileExistsError, match="overwrite"):
        run_development_matrix(
            training_seed=42,
            config_path=config,
            project_root=tmp_path,
            command_runner=lambda *args, **kwargs: pytest.fail("must not launch"),
        )
    assert gate.read_bytes() == b"original"


@pytest.mark.parametrize(
    ("seed", "phase"),
    [(41, "development"), (42, "formal_train"), (42, "formal_evaluate")],
)
def test_config_cannot_change_seed_or_enter_formal_phase(tmp_path, seed, phase):
    config = _write_config(tmp_path, seed=seed, phase=phase)

    with pytest.raises(ValueError):
        run_development_matrix(
            training_seed=42,
            config_path=config,
            project_root=tmp_path,
            command_runner=lambda *args, **kwargs: pytest.fail("must not launch"),
        )


def test_gate_threshold_types_are_exact_not_python_truthy_equivalents(tmp_path):
    config = _write_config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "relative_mape_improvement_min: 0.0",
            "relative_mape_improvement_min: false",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved protocol"):
        run_development_matrix(
            training_seed=42,
            config_path=config,
            project_root=tmp_path,
            command_runner=lambda *args, **kwargs: pytest.fail("must not launch"),
        )


def test_gate_thresholds_are_exactly_inclusive_without_hidden_tolerance():
    below_zero = math.nextafter(0.0, -math.inf)
    above_five_percent = math.nextafter(0.05, math.inf)

    assert not development_module._inclusive_at_least(below_zero, 0.0)
    assert not development_module._inclusive_at_most(above_five_percent, 0.05)


def test_cli_delegates_without_reading_environment_or_running_network(monkeypatch):
    captured = {}

    def fake_matrix(**kwargs):
        captured.update(kwargs)
        return 23

    monkeypatch.setattr("scripts.run_fmas_development.run_development_matrix", fake_matrix)
    assert main(["--training-seed", "42", "--config", "configs/development_seed42.yaml"]) == 23
    assert captured["training_seed"] == 42
    assert captured["config_path"] == Path("configs/development_seed42.yaml")
