from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import experiments.run_strict_federated as runner_module

from experiments.run_strict_federated import (
    FORMAL_METHODS,
    LLM_METHODS,
    build_parser,
    create_run_directory,
    effective_config_sha256,
    execute,
    load_method_config,
    resolve_api_key,
    resolve_run_directory,
    validate_invocation,
    write_provenance,
)
from src.federated_learning.pcv.engine import ExperimentPaused
from src.study_manifest import load_study_manifest
from scripts.run_multi_seed import DEFAULT_SCENARIOS, reject_new_formal_methods
from src.federated_learning.pcv import runtime as runtime_module
from src.models import CostEstimationMLP


def test_runner_accepts_only_formal_methods():
    parser = build_parser()
    args = parser.parse_args(
        ["--method", "FMAS_PCV_FEDYOGI", "--phase", "development"]
    )
    assert args.method == "FMAS_PCV_FEDYOGI"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--method", "LLM_GCA_FEDYOGI_TR", "--phase", "development"]
        )


def test_run_directory_never_overwrites(tmp_path: Path):
    first = resolve_run_directory(tmp_path, "dev-a")
    first.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        resolve_run_directory(tmp_path, "dev-a")


def test_run_id_is_one_safe_path_component(tmp_path: Path):
    for unsafe in ("", ".", "..", "../escape", "a/b", "a\\b", "C:escape"):
        with pytest.raises(ValueError):
            resolve_run_directory(tmp_path, unsafe)


COMMON = {
    "num_rounds": 20,
    "local_epochs": 20,
    "batch_size": 32,
    "client_learning_rate": 0.0005,
    "checkpoint_metric": "aggregated_client_val_mape",
    "candidate_budget": 8,
    "min_client_weight": 0.05,
    "max_client_weight": 0.80,
    "weight_l1_limit": 0.35,
    "best_candidate_tolerance": 0.002,
    "anchor_mape_tolerance": 0.001,
    "catastrophic_client_relative_mape": 0.05,
    "fedyogi_server_lr": 0.0175,
    "fedyogi_beta1": 0.9,
    "fedyogi_beta2": 0.99,
    "fedyogi_tau": 0.001,
    "fedyogi_max_coordinate_step_ratio": None,
    "fedyogi_anchor_clip_norm": None,
}


@pytest.mark.parametrize(
    ("method", "optimizer", "mode", "roles"),
    [
        ("FEDAVG_STRICT", "fedavg", "anchor_only", []),
        ("FEDYOGI_STRICT", "fedyogi", "anchor_only", []),
        ("DPCV_FEDYOGI", "fedyogi", "deterministic", []),
        (
            "SA_PCV_FEDYOGI",
            "fedyogi",
            "single_agent",
            ["single_proposer", "coordinator"],
        ),
        (
            "FMAS_PCV_FEDYOGI",
            "fedyogi",
            "multi_agent",
            [
                "diagnostic",
                "performance_proposer",
                "stability_proposer",
                "balance_proposer",
                "critic",
                "coordinator",
            ],
        ),
    ],
)
def test_method_configs_have_only_the_predeclared_differences(
    method, optimizer, mode, roles
):
    config = load_method_config(method)
    assert {key: config[key] for key in COMMON} == COMMON
    assert config["method"] == method
    assert config["server_optimizer"] == optimizer
    assert config["proposal_mode"] == mode
    assert config["deepseek_roles"] == roles
    assert set(config) == {
        *COMMON,
        "method",
        "server_optimizer",
        "proposal_mode",
        "deepseek_roles",
    }


def test_all_formal_methods_explicitly_register_the_same_fedyogi_base_parameters():
    from experiments.run_strict_federated import FORMAL_METHOD_ORDER

    expected = {
        "fedyogi_server_lr": 0.0175,
        "fedyogi_beta1": 0.9,
        "fedyogi_beta2": 0.99,
        "fedyogi_tau": 0.001,
        "fedyogi_max_coordinate_step_ratio": None,
        "fedyogi_anchor_clip_norm": None,
    }
    for method in FORMAL_METHOD_ORDER:
        config = load_method_config(method)
        assert {key: config[key] for key in expected} == expected


class _EnvironmentProbe(dict):
    def __init__(self):
        super().__init__({"DEEPSEEK_API_KEY": "fake-test-key"})
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        return super().get(key, default)


@pytest.mark.parametrize("method", sorted(FORMAL_METHODS - LLM_METHODS))
def test_non_llm_methods_never_read_deepseek_key(method):
    environment = _EnvironmentProbe()
    assert resolve_api_key(method, environment) is None
    assert environment.reads == []


@pytest.mark.parametrize("method", sorted(LLM_METHODS))
def test_llm_methods_read_only_the_named_environment_key(method):
    environment = _EnvironmentProbe()
    assert resolve_api_key(method, environment) == "fake-test-key"
    assert environment.reads == ["DEEPSEEK_API_KEY"]


def _args(*extra):
    return build_parser().parse_args(
        [
            "--method",
            "FEDAVG_STRICT",
            "--phase",
            "development",
            "--training-seed",
            "42",
            "--run-id",
            "dev-a",
            *extra,
        ]
    )


def test_unfrozen_manifest_refuses_every_formal_phase_before_output(tmp_path: Path):
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    for phase in ("formal_train", "formal_evaluate"):
        args = build_parser().parse_args(
            [
                "--method",
                "FEDAVG_STRICT",
                "--phase",
                phase,
                "--training-seed",
                "314",
                "--freeze-id",
                "not-frozen",
                "--unlock-test",
            ]
        )
        with pytest.raises(RuntimeError, match="frozen"):
            validate_invocation(args, manifest)
    assert list(tmp_path.iterdir()) == []


def test_formal_runner_refuses_an_unbound_or_dirty_freeze_before_output(
    tmp_path, monkeypatch
):
    manifest = replace(
        load_study_manifest(Path("study_manifest.yaml")),
        formal_frozen=True,
        paper_eligible_freeze_ids=("freeze-a",),
    )
    monkeypatch.setattr(runner_module, "load_study_manifest", lambda path: manifest)
    monkeypatch.setattr(runner_module, "_git_metadata", lambda root: ("commit", False))
    args = build_parser().parse_args(
        [
            "--method", "FEDAVG_STRICT",
            "--phase", "formal_train",
            "--training-seed", str(manifest.formal_seeds[0]),
            "--freeze-id", "freeze-a",
        ]
    )

    with pytest.raises(RuntimeError, match="freeze"):
        runner_module.execute(
            args,
            results_root=tmp_path / "unbound",
            environment=_EnvironmentProbe(),
            training_executor=lambda context: "must-not-run",
        )
    assert not (tmp_path / "unbound").exists()

    monkeypatch.setattr(runner_module, "_git_metadata", lambda root: ("commit", True))
    with pytest.raises(RuntimeError, match="clean"):
        runner_module.execute(
            args,
            results_root=tmp_path / "dirty",
            environment=_EnvironmentProbe(),
            training_executor=lambda context: "must-not-run",
        )
    assert not (tmp_path / "dirty").exists()


def test_development_cannot_unlock_test_or_use_a_formal_seed():
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    with pytest.raises(RuntimeError, match="locked test"):
        validate_invocation(_args("--unlock-test"), manifest)
    with pytest.raises(ValueError, match="development seed"):
        validate_invocation(
            build_parser().parse_args(
                [
                    "--method",
                    "FEDAVG_STRICT",
                    "--phase",
                    "development",
                    "--training-seed",
                    "314",
                    "--run-id",
                    "bad-seed",
                ]
            ),
            manifest,
        )


def test_resume_requires_checkpoint_and_explicit_approval_together():
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    with pytest.raises(RuntimeError, match="resume"):
        validate_invocation(_args("--user-approved-resume"), manifest)
    with pytest.raises(RuntimeError, match="approval"):
        validate_invocation(
            _args("--resume-checkpoint", "missing.pt"), manifest
        )


def test_create_run_directory_is_atomic_and_provenance_is_no_overwrite(tmp_path: Path):
    run_dir = create_run_directory(tmp_path, "dev-a")
    assert run_dir.is_dir()
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "dev-a")

    record = {"method": "FEDAVG_STRICT", "llm_enabled": False}
    path = write_provenance(run_dir, record)
    assert path.is_file()
    assert "fake-test-key" not in path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_provenance(run_dir, record)


def test_legacy_runner_rejects_new_formal_keys_but_keeps_historical_keys():
    reject_new_formal_methods(["C", "MAS_ADAPTIVE", "LLM_GCA_FEDYOGI_TR"])
    reject_new_formal_methods(DEFAULT_SCENARIOS)
    for method in FORMAL_METHODS:
        with pytest.raises(RuntimeError, match="run_strict_federated"):
            reject_new_formal_methods([method])


def test_formal_evaluation_requires_an_approved_training_checkpoint(tmp_path):
    manifest = replace(
        load_study_manifest(Path("study_manifest.yaml")),
        formal_frozen=True,
        paper_eligible_freeze_ids=("freeze-a",),
    )
    args = build_parser().parse_args(
        [
            "--method",
            "FEDAVG_STRICT",
            "--phase",
            "formal_evaluate",
            "--training-seed",
            str(manifest.formal_seeds[0]),
            "--freeze-id",
            "freeze-a",
            "--unlock-test",
        ]
    )
    with pytest.raises(RuntimeError, match="checkpoint"):
        validate_invocation(args, manifest)


def test_approved_resume_reopens_only_its_existing_immutable_run(tmp_path):
    initial_context, _ = execute(
        _args("--run-id", "resume-a"),
        results_root=tmp_path,
        environment=_EnvironmentProbe(),
        training_executor=lambda context: "initial",
    )
    checkpoint = initial_context.run_directory / "last_complete.pt"
    provenance_before = initial_context.provenance_path.read_bytes()

    resumed_context, result = execute(
        _args(
            "--run-id",
            "resume-a",
            "--resume-checkpoint",
            str(checkpoint),
            "--user-approved-resume",
        ),
        results_root=tmp_path,
        environment=_EnvironmentProbe(),
        training_executor=lambda context: "resumed",
    )

    assert result == "resumed"
    assert resumed_context.run_directory == initial_context.run_directory
    assert resumed_context.provenance_path == initial_context.provenance_path
    assert resumed_context.provenance_path.read_bytes() == provenance_before


def test_effective_config_hash_changes_with_base_or_method_config(tmp_path):
    base = tmp_path / "base.yaml"
    method = tmp_path / "method.yaml"
    base.write_text("model: one\n", encoding="utf-8")
    method.write_text("method: one\n", encoding="utf-8")
    initial = effective_config_sha256(base, method)

    base.write_text("model: two\n", encoding="utf-8")
    assert effective_config_sha256(base, method) != initial
    base.write_text("model: one\n", encoding="utf-8")
    method.write_text("method: two\n", encoding="utf-8")
    assert effective_config_sha256(base, method) != initial


def test_repeated_setup_failure_writes_a_new_pause_report(tmp_path):
    failure = runner_module.ExperimentRuntimeError("RuntimeError", "sanitized")
    first = runner_module._write_preflight_pause(tmp_path, failure)
    first_bytes = first.read_bytes()
    second = runner_module._write_preflight_pause(tmp_path, failure)

    assert first.name == "PAUSED.json"
    assert second.name == "PAUSED.001.json"
    assert first.read_bytes() == first_bytes
    assert second.is_file()


def test_failed_evaluation_attempts_get_distinct_immutable_audits(tmp_path):
    record = {"phase": "formal_evaluate", "locked_test_unlocked": True}
    first = runner_module._write_evaluation_provenance(tmp_path, record)
    second = runner_module._write_evaluation_provenance(tmp_path, record)

    assert first.name == "evaluation_provenance.json"
    assert second.name == "evaluation_provenance.001.json"
    assert first.read_bytes() == second.read_bytes()


def test_formal_evaluation_adds_separate_audit_without_overwriting_training(
    tmp_path, monkeypatch
):
    manifest = replace(
        load_study_manifest(Path("study_manifest.yaml")),
        formal_frozen=True,
        paper_eligible_freeze_ids=("freeze-a",),
    )
    monkeypatch.setattr(runner_module, "load_study_manifest", lambda path: manifest)
    monkeypatch.setattr(runner_module, "validate_formal_freeze", lambda **kwargs: None)
    train_args = build_parser().parse_args(
        [
            "--method", "FEDAVG_STRICT",
            "--phase", "formal_train",
            "--training-seed", str(manifest.formal_seeds[0]),
            "--freeze-id", "freeze-a",
        ]
    )
    train_context, _ = runner_module.execute(
        train_args,
        results_root=tmp_path,
        environment=_EnvironmentProbe(),
        training_executor=lambda context: "trained",
    )
    checkpoint = train_context.run_directory / "last_complete.pt"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    training_provenance = train_context.provenance_path.read_bytes()
    evaluate_args = build_parser().parse_args(
        [
            "--method", "FEDAVG_STRICT",
            "--phase", "formal_evaluate",
            "--training-seed", str(manifest.formal_seeds[0]),
            "--freeze-id", "freeze-a",
            "--resume-checkpoint", str(checkpoint),
            "--user-approved-resume",
            "--unlock-test",
        ]
    )

    evaluate_context, _ = runner_module.execute(
        evaluate_args,
        results_root=tmp_path,
        environment=_EnvironmentProbe(),
        training_executor=lambda context: "evaluated",
    )

    assert evaluate_context.provenance_path.read_bytes() == training_provenance
    evaluation_audit = json.loads(
        (evaluate_context.run_directory / "evaluation_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert evaluation_audit["phase"] == "formal_evaluate"
    assert evaluation_audit["locked_test_unlocked"] is True


class _PreflightResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"status":"ready","model":"deepseek-v4-flash"}'
                    }
                }
            ]
        }


class _PreflightSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _PreflightResponse()


def test_preflight_is_exactly_one_aggregate_free_call_and_never_records_key(tmp_path):
    session = _PreflightSession()
    args = build_parser().parse_args(
        [
            "--method",
            "FMAS_PCV_FEDYOGI",
            "--phase",
            "development",
            "--training-seed",
            "42",
            "--run-id",
            "preflight-a",
            "--preflight-only",
        ]
    )

    context, result = execute(
        args,
        results_root=tmp_path,
        environment={"DEEPSEEK_API_KEY": "fake-test-key"},
        preflight_session=session,
    )

    assert result == {"status": "ready", "model": "deepseek-v4-flash"}
    assert len(session.calls) == 1
    _, request = session.calls[0]
    assert request["json"]["model"] == "deepseek-v4-flash"
    assert "Do not echo" in request["json"]["messages"][0]["content"]
    user_payload = json.loads(request["json"]["messages"][1]["content"])
    assert user_payload == {"round_index": 0, "clients": []}
    all_output = "".join(
        path.read_text(encoding="utf-8")
        for path in context.run_directory.iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )
    assert "fake-test-key" not in all_output


def test_training_runtime_uses_exact_deepseek_settings_from_provenance():
    settings = runtime_module._deepseek_settings_from_provenance(
        {
            "deepseek": {
                "enabled": True,
                "model": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "temperature": 0.8,
                "timeout_seconds": 60,
            }
        }
    )

    assert settings == {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "timeout_seconds": 60,
    }


def test_training_runtime_rejects_old_deepseek_model_provenance():
    with pytest.raises(ValueError, match="frozen protocol"):
        runtime_module._deepseek_settings_from_provenance(
            {
                "deepseek": {
                    "enabled": True,
                    "model": "deepseek-chat",
                    "base_url": "https://api.deepseek.com",
                    "temperature": 0.8,
                    "timeout_seconds": 60,
                }
            }
        )


def test_missing_llm_key_stops_with_a_persisted_pause_report_and_no_call(tmp_path):
    args = build_parser().parse_args(
        [
            "--method",
            "FMAS_PCV_FEDYOGI",
            "--phase",
            "development",
            "--training-seed",
            "42",
            "--run-id",
            "missing-key",
            "--preflight-only",
        ]
    )

    with pytest.raises(ExperimentPaused) as paused:
        execute(args, results_root=tmp_path, environment={})

    assert paused.value.failure.category == "authentication"
    assert paused.value.report_path.is_file()
    report = json.loads(paused.value.report_path.read_text(encoding="utf-8"))
    assert report["category"] == "authentication"


def test_provenance_exists_before_any_training_executor_runs(tmp_path):
    observations = []

    def executor(context):
        observations.append(context.provenance_path.is_file())
        return "done"

    context, result = execute(
        _args(),
        results_root=tmp_path,
        environment=_EnvironmentProbe(),
        training_executor=executor,
    )

    assert result == "done"
    assert observations == [True]
    assert context.run_directory.is_dir()


def test_runtime_constructs_private_client_vaults_from_the_fixed_loader(
    tmp_path, monkeypatch
):
    class Preprocessor:
        def transform(self, features, target):
            return features.to_numpy(dtype=float), target.to_numpy(dtype=float).reshape(-1, 1)

        def inverse_transform_target(self, values):
            return np.asarray(values, dtype=float)

    feature_columns = [f"f{index}" for index in range(10)]
    rows = pd.DataFrame(
        {
            **{column: [1.0, 2.0] for column in feature_columns},
            "target": [2.0, 3.0],
        }
    )
    loaded = SimpleNamespace(
        client_frames={
            client_id: {
                "train": rows.copy(),
                "controller_validation": rows.copy(),
                "locked_test": rows.copy(),
            }
            for client_id in ("Client 1", "Client 2", "Client 3")
        },
        preprocessor=Preprocessor(),
        dataset_sha256="d" * 64,
        partition_sha256="p" * 64,
    )
    loader_calls = []

    def fake_loader(config, path, **kwargs):
        loader_calls.append((Path(path), kwargs))
        return loaded

    monkeypatch.setattr(
        runtime_module,
        "load_strict_partition_frames",
        fake_loader,
    )
    base_config = {
        "scene_c": {
            "data": {
                "feature_columns": feature_columns,
                "target_column": "target",
            }
        },
        "model": {
            "architecture": {
                "input_dim": 10,
                "hidden_dims": [4],
                "output_dim": 1,
                "activation": "relu",
                "dropout": 0.0,
            }
        },
    }

    bundle = runtime_module.build_client_vaults(
        project_root=tmp_path,
        base_config=base_config,
        method_config=COMMON,
    )

    assert loader_calls == [
        (
            tmp_path / "results/manifests/strict_partition_v1.csv",
            {
                "allowed_partitions": {"train", "controller_validation"},
                "sealed_data_directory": tmp_path / "Data/strict_partition_v1",
            },
        )
    ]

    assert dict(bundle.train_sample_counts) == {
        "Client 1": 2,
        "Client 2": 2,
        "Client 3": 2,
    }
    model = CostEstimationMLP(
        input_dim=10,
        hidden_dims=[4],
        output_dim=1,
        activation="relu",
        dropout=0.0,
    )
    sums = bundle.vaults["Client 1"].controller_metric_sums(model.state_dict())
    assert sums.n == 2
    for vault in bundle.vaults.values():
        assert not hasattr(vault, "train_dataset")
        assert not hasattr(vault, "locked_test_dataset")


def test_runtime_resume_restores_once_and_passes_all_engine_state(monkeypatch):
    observed = []
    restored = {
        "previous_weights": {"Client 1": 1.0},
        "best_validation": {"mape": 0.2, "round": 2},
        "best_model_state": {"weight": "saved"},
        "last_complete_round": 2,
    }

    def fake_restore(**kwargs):
        observed.append(kwargs)
        return 3, restored

    monkeypatch.setattr(
        runtime_module,
        "restore_training_checkpoint",
        fake_restore,
        raising=False,
    )
    context = SimpleNamespace(
        args=SimpleNamespace(
            resume_checkpoint=Path("run/last_complete.pt"),
            user_approved_resume=True,
            freeze_id=None,
            method="FEDAVG_STRICT",
            training_seed=42,
            llm_rep=0,
        )
    )
    bundle = SimpleNamespace(partition_sha256="partition-hash")
    model = object()
    optimizer = object()

    start_round, engine_state = runtime_module.restore_engine_checkpoint(
        context=context,
        bundle=bundle,
        model=model,
        server_optimizer=optimizer,
        provenance={
            "effective_config_sha256": "config-hash",
            "prompt_hashes": {},
        },
    )

    assert start_round == 3
    assert engine_state == restored
    assert observed == [
        {
            "model": model,
            "server_optimizer": optimizer,
            "user_approved_resume": True,
            "resume_checkpoint": Path("run/last_complete.pt"),
            "requested_freeze_id": "development",
            "requested_method": "FEDAVG_STRICT",
            "requested_training_seed": 42,
            "requested_llm_rep": 0,
            "requested_partition_sha256": "partition-hash",
            "requested_config_sha256": "config-hash",
            "requested_prompt_hashes": {"engine": "no-agent-prompts"},
        }
    ]


def test_runtime_can_restart_from_round_zero_when_pause_precedes_checkpoint(tmp_path):
    checkpoint = tmp_path / "last_complete.pt"
    (tmp_path / "PAUSED.json").write_text("{}", encoding="utf-8")
    context = SimpleNamespace(
        run_directory=tmp_path,
        args=SimpleNamespace(
            resume_checkpoint=checkpoint,
            user_approved_resume=True,
            freeze_id=None,
            method="FEDAVG_STRICT",
            training_seed=42,
            llm_rep=0,
            phase="development",
        ),
    )

    assert runtime_module.restore_engine_checkpoint(
        context=context,
        bundle=SimpleNamespace(partition_sha256="partition-hash"),
        model=object(),
        server_optimizer=object(),
        provenance={"effective_config_sha256": "config-hash", "prompt_hashes": {}},
    ) == (1, {})


def test_runtime_round_plan_separates_training_resume_and_locked_evaluation():
    development = SimpleNamespace(args=SimpleNamespace(phase="development"))
    formal_train = SimpleNamespace(args=SimpleNamespace(phase="formal_train"))
    formal_evaluate = SimpleNamespace(args=SimpleNamespace(phase="formal_evaluate"))

    assert runtime_module.planned_round_indices(development, 3, 5) == (3, 4, 5)
    assert runtime_module.planned_round_indices(formal_train, 4, 5) == (4, 5)
    assert runtime_module.planned_round_indices(formal_evaluate, 6, 5) == ()
    assert runtime_module.planned_round_indices(development, 6, 5) == ()
    with pytest.raises(RuntimeError, match="completed training checkpoint"):
        runtime_module.planned_round_indices(formal_evaluate, 5, 5)
    with pytest.raises(RuntimeError, match="already complete"):
        runtime_module.planned_round_indices(development, 7, 5)


def test_terminal_completion_resolves_preserved_pause_incidents(tmp_path):
    (tmp_path / "PAUSED.json").write_text("{}", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}", encoding="utf-8")
    context = SimpleNamespace(
        run_directory=tmp_path,
        provenance_path=provenance,
        evaluation_provenance_path=None,
        args=SimpleNamespace(
            phase="development",
            method="FEDAVG_STRICT",
            training_seed=42,
            llm_rep=0,
            user_approved_resume=True,
        ),
    )
    engine = SimpleNamespace(last_complete_round=20)
    summary = {
        "status": "complete",
        "phase": "development",
        "method": "FEDAVG_STRICT",
        "training_seed": 42,
        "llm_rep": 0,
        "completed_rounds": 20,
        "best_validation": {
            "sample_count": 3,
            "mape": 0.1,
            "rmse": 1.0,
            "mae": 0.5,
            "r2": 0.8,
        },
    }
    result_path = tmp_path / "validation_metrics.json"
    result_path.write_text(json.dumps(summary), encoding="utf-8")

    path = runtime_module._write_or_validate_completion(
        context=context,
        engine=engine,
        summary=summary,
    )

    status = json.loads(path.read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["resolved_pause_reports"] == ["PAUSED.json"]
    assert status["resume_approved"] is True
    assert status["result_file"] == "validation_metrics.json"
    assert status["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()

    path.write_text('{"status":"complete"}', encoding="utf-8")
    with pytest.raises(ValueError, match="completion status identity mismatch"):
        runtime_module._write_or_validate_completion(
            context=context,
            engine=engine,
            summary=summary,
        )


def test_runtime_pause_writer_preserves_every_engine_incident(tmp_path):
    requested = tmp_path / "PAUSED.json"
    first_record = {"status": "paused", "round": 1}
    second_record = {"status": "paused", "round": 2}

    first = runtime_module._write_numbered_pause_report(requested, first_record)
    second = runtime_module._write_numbered_pause_report(requested, second_record)

    assert first.name == "PAUSED.json"
    assert second.name == "PAUSED.001.json"
    assert json.loads(first.read_text(encoding="utf-8")) == first_record
    assert json.loads(second.read_text(encoding="utf-8")) == second_record


def _formal_result_context(tmp_path, audit_name="evaluation_provenance.json"):
    audit = tmp_path / audit_name
    audit.write_text(
        json.dumps(
            {
                "phase": "formal_evaluate",
                "method": "FEDAVG_STRICT",
                "training_seed": 101,
                "llm_rep": 0,
                "training_checkpoint_sha256": "c" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        run_directory=tmp_path,
        evaluation_provenance_path=audit,
        args=SimpleNamespace(
            phase="formal_evaluate",
            method="FEDAVG_STRICT",
            training_seed=101,
            llm_rep=0,
        ),
    )


def test_locked_test_result_has_exact_finite_schema_and_checkpoint_binding(tmp_path):
    context = _formal_result_context(tmp_path)
    metrics = {
        "sample_count": 105,
        "mape": 0.12,
        "rmse": 2.5,
        "mae": 1.25,
        "r2": 0.75,
    }

    returned = runtime_module._write_or_validate_locked_test_result(
        context=context,
        metrics=metrics,
    )
    record_path = tmp_path / "locked_test_metrics.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert returned == metrics
    assert record["training_checkpoint_sha256"] == "c" * 64
    assert record["evaluation_provenance_sha256"] == hashlib.sha256(
        context.evaluation_provenance_path.read_bytes()
    ).hexdigest()
    assert record["locked_test"] == metrics

    equivalent_retry = _formal_result_context(
        tmp_path, "evaluation_provenance.001.json"
    )
    assert runtime_module._write_or_validate_locked_test_result(
        context=equivalent_retry,
        metrics=None,
    ) == metrics


def test_locked_test_result_rejects_stale_audit_and_nonfinite_metrics(tmp_path):
    context = _formal_result_context(tmp_path)
    metrics = {
        "sample_count": 105,
        "mape": 0.12,
        "rmse": 2.5,
        "mae": 1.25,
        "r2": 0.75,
    }
    runtime_module._write_or_validate_locked_test_result(
        context=context,
        metrics=metrics,
    )

    stale = _formal_result_context(tmp_path, "evaluation_provenance.001.json")
    stale_record = json.loads(stale.evaluation_provenance_path.read_text(encoding="utf-8"))
    stale_record["training_checkpoint_sha256"] = "d" * 64
    stale.evaluation_provenance_path.write_text(
        json.dumps(stale_record, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="locked-test result identity mismatch"):
        runtime_module._write_or_validate_locked_test_result(
            context=stale,
            metrics=None,
        )

    record_path = tmp_path / "locked_test_metrics.json"
    corrupt = json.loads(record_path.read_text(encoding="utf-8"))
    corrupt["locked_test"]["mape"] = float("nan")
    record_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        runtime_module._write_or_validate_locked_test_result(
            context=context,
            metrics=None,
        )
