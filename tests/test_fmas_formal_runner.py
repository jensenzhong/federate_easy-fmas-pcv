import hashlib
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
    roles = {
        role
        for method in formal_module.FORMAL_METHOD_ORDER
        for role in formal_module.METHOD_PROMPT_ROLES[method]
    }
    return FormalSnapshot(
        git_commit="b" * 40,
        partition_sha256="c" * 64,
        sealed_partition_metadata_sha256="d" * 64,
        base_config_sha256="e" * 64,
        method_config_sha256={method: "f" * 64 for method in formal_module.FORMAL_METHOD_ORDER},
        effective_config_sha256={method: "1" * 64 for method in formal_module.FORMAL_METHOD_ORDER},
        prompt_hashes={role: "2" * 64 for role in roles},
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


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _training_fixture(tmp_path, monkeypatch, *, run=None):
    run = run or FormalRun("FEDAVG_STRICT", 314, 0)
    snapshot = _snapshot()
    directory = formal_module._run_directory(tmp_path, FREEZE_ID, run)
    directory.mkdir(parents=True, exist_ok=True)
    validation = {
        "status": "complete",
        "phase": "formal_train",
        "method": run.method,
        "training_seed": run.training_seed,
        "llm_rep": run.llm_rep,
        "completed_rounds": 20,
        "best_validation": {
            "sample_count": 10,
            "mape": 0.4,
            "rmse": 2.0,
            "mae": 1.0,
            "r2": 0.5,
        },
    }
    _write_json(directory / "validation_metrics.json", validation)
    roles = set(formal_module.METHOD_PROMPT_ROLES[run.method])
    prompts = {role: snapshot.prompt_hashes[role] for role in roles}
    provenance = {
        "schema_version": 1,
        "method": run.method,
        "phase": "formal_train",
        "training_seed": run.training_seed,
        "llm_rep": run.llm_rep,
        "run_id": None,
        "freeze_id": FREEZE_ID,
        "git_commit": snapshot.git_commit,
        "git_dirty": False,
        "partition_sha256": snapshot.partition_sha256,
        "sealed_partition_metadata_sha256": snapshot.sealed_partition_metadata_sha256,
        "method_config_sha256": snapshot.method_config_sha256[run.method],
        "base_config_sha256": snapshot.base_config_sha256,
        "effective_config_sha256": snapshot.effective_config_sha256[run.method],
        "prompt_hashes": prompts,
        "deepseek": formal_module.deepseek_provenance(enabled=bool(roles)),
        "resume_requested": False,
        "locked_test_unlocked": False,
    }
    _write_json(directory / "provenance.json", provenance)
    checkpoint_path = directory / "last_complete.pt"
    checkpoint_path.write_bytes(b"formal-checkpoint")
    completion = {
        "status": "complete",
        "phase": "formal_train",
        "method": run.method,
        "training_seed": run.training_seed,
        "llm_rep": run.llm_rep,
        "last_complete_round": 20,
        "resolved_pause_reports": [],
        "resume_approved": False,
        "provenance": "provenance.json",
        "evaluation_provenance": None,
        "result_status": "complete",
        "result_file": "validation_metrics.json",
        "result_sha256": hashlib.sha256(
            (directory / "validation_metrics.json").read_bytes()
        ).hexdigest(),
    }
    _write_json(directory / "TRAINING_COMPLETE.json", completion)
    checkpoint = {
        "last_complete_round": 20,
        "freeze_id": FREEZE_ID,
        "method": run.method,
        "training_seed": run.training_seed,
        "llm_rep": run.llm_rep,
        "partition_sha256": snapshot.partition_sha256,
        "config_sha256": snapshot.effective_config_sha256[run.method],
        "prompt_hashes": prompts or {"engine": "no-agent-prompts"},
    }
    monkeypatch.setattr(formal_module, "load_checkpoint", lambda path: checkpoint)
    record = formal_module._validate_training_run(
        run,
        directory,
        freeze_id=FREEZE_ID,
        snapshot=snapshot,
    )
    return run, snapshot, directory, provenance, record


def test_evaluation_prescan_allows_prior_locked_result(tmp_path, monkeypatch):
    run, snapshot, directory, _, _ = _training_fixture(tmp_path, monkeypatch)
    _write_json(directory / "locked_test_metrics.json", {})

    record = formal_module._validate_training_run(
        run,
        directory,
        freeze_id=FREEZE_ID,
        snapshot=snapshot,
        allow_evaluation_artifacts=True,
    )

    assert record["method"] == run.method


@pytest.mark.parametrize(
    ("pause_report", "resume_approved"),
    [(False, 0), (True, 1)],
)
def test_training_completion_rejects_integer_resume_approval(
    tmp_path,
    monkeypatch,
    pause_report,
    resume_approved,
):
    run, snapshot, directory, _, _ = _training_fixture(tmp_path, monkeypatch)
    completion_path = directory / "TRAINING_COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if pause_report:
        _write_json(directory / "PAUSED.json", {"status": "resolved"})
        completion["resolved_pause_reports"] = ["PAUSED.json"]
    completion["resume_approved"] = resume_approved
    _write_json(completion_path, completion)

    with pytest.raises(ValueError, match="training completion identity"):
        formal_module._validate_training_run(
            run,
            directory,
            freeze_id=FREEZE_ID,
            snapshot=snapshot,
        )


def test_training_batch_requires_exact_real_run_records():
    records = [
        {
            "method": run.method,
            "training_seed": run.training_seed,
            "llm_rep": run.llm_rep,
            "completion_sha256": "a" * 64,
            "validation_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
        }
        for run in build_run_matrix(SEEDS)
    ]
    batch = {
        "schema_version": 1,
        "status": "complete",
        "phase": "formal_train",
        "freeze_id": FREEZE_ID,
        "git_commit": "b" * 40,
        "run_count": 45,
        "runs": [{} for _ in range(45)],
    }
    with pytest.raises(ValueError, match="training batch"):
        formal_module._validate_training_batch_record(
            batch,
            freeze_id=FREEZE_ID,
            git_commit="b" * 40,
            training_records=records,
        )


def _evaluation_fixture(tmp_path, monkeypatch, *, evaluation_name="evaluation_provenance.json"):
    run, _, directory, provenance, training_record = _training_fixture(
        tmp_path, monkeypatch
    )
    checkpoint_sha = hashlib.sha256((directory / "last_complete.pt").read_bytes()).hexdigest()
    evaluation = dict(provenance)
    evaluation.update(
        {
            "phase": "formal_evaluate",
            "resume_requested": True,
            "locked_test_unlocked": True,
            "training_checkpoint_sha256": checkpoint_sha,
        }
    )
    evaluation_path = directory / evaluation_name
    _write_json(evaluation_path, evaluation)
    locked = {
        "schema_version": 1,
        "phase": "formal_evaluate",
        "method": run.method,
        "training_seed": run.training_seed,
        "llm_rep": run.llm_rep,
        "training_checkpoint_sha256": checkpoint_sha,
        "evaluation_provenance_sha256": hashlib.sha256(
            evaluation_path.read_bytes()
        ).hexdigest(),
        "locked_test": {
            "sample_count": 10,
            "mape": 0.4,
            "rmse": 2.0,
            "mae": 1.0,
            "r2": 0.5,
        },
    }
    _write_json(directory / "locked_test_metrics.json", locked)
    completion = {
        "status": "complete",
        "phase": "formal_evaluate",
        "method": run.method,
        "training_seed": run.training_seed,
        "llm_rep": run.llm_rep,
        "last_complete_round": 20,
        "resolved_pause_reports": [],
        "resume_approved": True,
        "provenance": "provenance.json",
        "evaluation_provenance": evaluation_name,
        "result_status": "complete",
        "result_file": "locked_test_metrics.json",
        "result_sha256": hashlib.sha256(
            (directory / "locked_test_metrics.json").read_bytes()
        ).hexdigest(),
    }
    _write_json(directory / "EVALUATION_COMPLETE.json", completion)
    return run, directory, evaluation_path, training_record


def test_evaluation_rejects_unconfined_provenance_path(tmp_path, monkeypatch):
    run, directory, _, training_record = _evaluation_fixture(tmp_path, monkeypatch)
    outside = directory.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    completion_path = directory / "EVALUATION_COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["evaluation_provenance"] = "../outside.json"
    _write_json(completion_path, completion)

    with pytest.raises(ValueError, match="provenance filename"):
        formal_module._validate_evaluation_run(
            run, directory, training_record=training_record
        )


def test_evaluation_rejects_wrong_provenance_identity(tmp_path, monkeypatch):
    run, directory, evaluation_path, training_record = _evaluation_fixture(
        tmp_path, monkeypatch
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["freeze_id"] = "wrong-freeze"
    _write_json(evaluation_path, evaluation)
    locked_path = directory / "locked_test_metrics.json"
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    locked["evaluation_provenance_sha256"] = hashlib.sha256(
        evaluation_path.read_bytes()
    ).hexdigest()
    _write_json(locked_path, locked)
    completion_path = directory / "EVALUATION_COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["result_sha256"] = hashlib.sha256(locked_path.read_bytes()).hexdigest()
    _write_json(completion_path, completion)

    with pytest.raises(ValueError, match="evaluation provenance"):
        formal_module._validate_evaluation_run(
            run, directory, training_record=training_record
        )


def test_batch_publication_is_idempotent_only_for_exact_evidence(tmp_path):
    path = tmp_path / "EVALUATION_BATCH_COMPLETE.json"
    record = {"schema_version": 1, "status": "complete", "runs": []}

    formal_module._publish_or_validate_json(path, record, label="formal evaluation batch")
    formal_module._publish_or_validate_json(path, record, label="formal evaluation batch")

    with pytest.raises(ValueError, match="does not match"):
        formal_module._publish_or_validate_json(
            path,
            {"schema_version": 1, "status": "complete", "runs": [{}]},
            label="formal evaluation batch",
        )


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
