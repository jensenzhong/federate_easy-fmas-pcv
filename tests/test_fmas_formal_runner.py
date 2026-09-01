import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_fmas_formal as formal_module
from scripts.run_fmas_formal import (
    FormalRun,
    FormalSnapshot,
    PAUSED_EXIT_CODE,
    build_run_matrix,
    run_formal_matrix,
)


SEEDS = (314, 2718, 2025, 3407, 9001)
FREEZE_ID = "abc123freeze"


def test_formal_matrix_is_exact_45_run_protocol():
    matrix = build_run_matrix(SEEDS)
    assert len(matrix) == 45
    assert [(run.method, run.llm_rep) for run in matrix[:9]] == [
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
    assert {run.training_seed for run in matrix} == set(SEEDS)


@pytest.mark.parametrize("seeds", [(42, 1, 2, 3, 4), (1, 1, 2, 3, 4), (1, 2, 3)])
def test_formal_matrix_rejects_invalid_seed_sets(seeds):
    with pytest.raises(ValueError, match="five unique"):
        build_run_matrix(seeds)


def test_formal_train_command_never_unlocks_test(tmp_path):
    run = FormalRun("FMAS_PCV_FEDYOGI", 314, 1)
    command = formal_module._command(
        run,
        phase="formal_train",
        freeze_id=FREEZE_ID,
        project_root=tmp_path,
        python_executable="python-test",
    )
    assert "formal_train" in command
    assert "--unlock-test" not in command
    assert "--resume-checkpoint" not in command


def test_formal_evaluation_command_is_explicit_and_checkpoint_bound(tmp_path):
    run = FormalRun("FEDAVG_STRICT", 314, 0)
    command = formal_module._command(
        run,
        phase="formal_evaluate",
        freeze_id=FREEZE_ID,
        project_root=tmp_path,
        python_executable="python-test",
    )
    assert "--unlock-test" in command
    assert "--user-approved-resume" in command
    checkpoint = command[command.index("--resume-checkpoint") + 1]
    assert checkpoint.endswith("FEDAVG_STRICT\\314\\0\\last_complete.pt") or checkpoint.endswith(
        "FEDAVG_STRICT/314/0/last_complete.pt"
    )


def _snapshot():
    return FormalSnapshot(
        git_commit="b" * 40,
        partition_sha256="c" * 64,
        sealed_partition_metadata_sha256="d" * 64,
        base_config_sha256="e" * 64,
        method_config_sha256={method: "f" * 64 for method in formal_module.FORMAL_METHOD_ORDER},
        effective_config_sha256={method: "1" * 64 for method in formal_module.FORMAL_METHOD_ORDER},
        prompt_hashes={},
    )


def _patch_protocol(monkeypatch):
    manifest = SimpleNamespace(
        formal_frozen=True,
        paper_eligible_freeze_ids=(FREEZE_ID,),
        formal_seeds=SEEDS,
    )
    snapshot = _snapshot()
    monkeypatch.setattr(formal_module, "load_study_manifest", lambda path: manifest)
    monkeypatch.setattr(formal_module, "_capture_snapshot", lambda root: snapshot)
    monkeypatch.setattr(formal_module, "validate_formal_freeze", lambda **kwargs: None)
    return snapshot


def test_first_nonzero_exit_stops_without_retry(tmp_path, monkeypatch):
    _patch_protocol(monkeypatch)
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append(command)
        return SimpleNamespace(returncode=17)

    assert run_formal_matrix(
        phase="formal_train",
        freeze_id=FREEZE_ID,
        project_root=tmp_path,
        command_runner=fake_run,
        python_executable="python-test",
    ) == 17
    assert len(calls) == 1


def test_existing_incomplete_run_stops_before_any_new_call(tmp_path, monkeypatch):
    _patch_protocol(monkeypatch)
    first = build_run_matrix(SEEDS)[0]
    directory = formal_module._run_directory(tmp_path, FREEZE_ID, first)
    directory.mkdir(parents=True)
    calls = []
    assert run_formal_matrix(
        phase="formal_train",
        freeze_id=FREEZE_ID,
        project_root=tmp_path,
        command_runner=lambda *args, **kwargs: calls.append(args),
    ) == PAUSED_EXIT_CODE
    assert calls == []


def test_formal_evaluation_can_continue_after_an_earlier_run_completed(
    tmp_path, monkeypatch
):
    _patch_protocol(monkeypatch)
    matrix = build_run_matrix(SEEDS)
    training_records = []
    observed_allow_flags = []
    for run in matrix:
        formal_module._run_directory(tmp_path, FREEZE_ID, run).mkdir(parents=True)
        training_records.append(
            {
                "method": run.method,
                "training_seed": run.training_seed,
                "llm_rep": run.llm_rep,
                "completion_sha256": "a" * 64,
                "validation_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
            }
        )
    batch_path = tmp_path / "results/formal" / FREEZE_ID / "TRAINING_BATCH_COMPLETE.json"
    batch_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "phase": "formal_train",
                "freeze_id": FREEZE_ID,
                "git_commit": "b" * 40,
                "run_count": 45,
                "runs": training_records,
            }
        ),
        encoding="utf-8",
    )
    first_dir = formal_module._run_directory(tmp_path, FREEZE_ID, matrix[0])
    (first_dir / "EVALUATION_COMPLETE.json").write_text("{}", encoding="utf-8")

    def fake_training(run, directory, **kwargs):
        observed_allow_flags.append(kwargs["allow_evaluation_artifacts"])
        index = matrix.index(run)
        return training_records[index]

    monkeypatch.setattr(formal_module, "_validate_training_run", fake_training)
    monkeypatch.setattr(
        formal_module,
        "_validate_evaluation_run",
        lambda run, directory, training_record: dict(training_record),
    )
    calls = []

    def stop_on_second(command, *, cwd, check):
        calls.append(command)
        return SimpleNamespace(returncode=17)

    assert run_formal_matrix(
        phase="formal_evaluate",
        freeze_id=FREEZE_ID,
        project_root=tmp_path,
        command_runner=stop_on_second,
        python_executable="python-test",
    ) == 17
    assert len(calls) == 1
    assert observed_allow_flags == [True] * 45
