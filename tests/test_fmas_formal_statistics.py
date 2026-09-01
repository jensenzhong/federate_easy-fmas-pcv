import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import scripts.statistical_analysis as statistics_module
from scripts.statistical_analysis import (
    aggregate_formal_repetitions,
    analyze_fmas_formal_results,
    load_frozen_formal_results,
    main,
)
from src.formal_protocol import METHOD_PROMPT_ROLES, METHOD_REPETITIONS, file_sha256
from src.federated_learning.pcv.provider_config import deepseek_provenance
from src.study_manifest import load_study_manifest


SEEDS = (314, 2718, 2025, 3407, 9001)


def _raw_results():
    rows = []
    offsets = {
        "FEDAVG_STRICT": 0.50,
        "FEDYOGI_STRICT": 0.62,
        "DPCV_FEDYOGI": 0.49,
        "SA_PCV_FEDYOGI": 0.475,
        "FMAS_PCV_FEDYOGI": 0.45,
    }
    for method, base in offsets.items():
        for seed_index, seed in enumerate(SEEDS):
            for rep in METHOD_REPETITIONS[method]:
                jitter = 0.002 * rep if rep else 0.0
                seed_slope = 0.0025 if method == "FMAS_PCV_FEDYOGI" else 0.003
                value = base + seed_slope * seed_index + jitter
                rows.append(
                    {
                        "method": method,
                        "training_seed": seed,
                        "llm_rep": rep,
                        "test_mape": value,
                        "test_rmse": value * 1_000_000,
                        "test_mae": value * 800_000,
                        "test_r2": 1.0 - value,
                    }
                )
    return pd.DataFrame(rows)


def test_llm_repetitions_are_averaged_within_seed():
    raw = _raw_results()
    assert len(raw) == 45
    per_seed = aggregate_formal_repetitions(raw)
    assert len(per_seed) == 25
    fmas = per_seed[per_seed["method"] == "FMAS_PCV_FEDYOGI"]
    assert len(fmas) == 5
    assert set(fmas["n_repetitions"]) == {3}
    baselines = per_seed[per_seed["method"] == "FEDAVG_STRICT"]
    assert set(baselines["n_repetitions"]) == {1}


def test_paired_statistics_use_five_observations_and_report_holm():
    report = analyze_fmas_formal_results(aggregate_formal_repetitions(_raw_results()))
    assert {row["paired_n"] for row in report["comparisons"]} == {5}
    assert all("holm_adjusted_p_value" in row for row in report["comparisons"])
    primary = next(row for row in report["comparisons"] if row["comparator"] == "FEDAVG_STRICT")
    assert primary["seed_wins"] == 5
    assert set(primary["metrics"]) == {"test_mape", "test_rmse", "test_mae", "test_r2"}
    assert len(primary["mean_mape_improvement_ci95"]) == 2
    assert report["stable_improvement"] is True


def test_legacy_method_is_rejected_from_hierarchical_aggregation():
    raw = _raw_results()
    raw.loc[0, "method"] = "LLM_GCA_FEDYOGI_TR"
    with pytest.raises(ValueError, match="repetition coverage"):
        aggregate_formal_repetitions(raw)


def test_duplicate_llm_repetition_is_rejected():
    raw = _raw_results()
    duplicate = raw[
        (raw["method"] == "FMAS_PCV_FEDYOGI")
        & (raw["training_seed"] == SEEDS[0])
        & (raw["llm_rep"] == 1)
    ]
    raw = pd.concat([raw, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="repetition coverage"):
        aggregate_formal_repetitions(raw)


def test_loader_rejects_noneligible_freeze_before_reading_results(tmp_path):
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    with pytest.raises(ValueError, match="not paper eligible"):
        load_frozen_formal_results(tmp_path, manifest, "not-approved")


def test_loader_rejects_seed42_even_for_an_eligible_freeze(tmp_path):
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    manifest = replace(
        manifest,
        formal_seeds=(42, 2718, 2025, 3407, 9001),
        paper_eligible_freeze_ids=("freeze-a",),
        formal_frozen=True,
        stage="formal_ready",
    )
    with pytest.raises(ValueError, match="seed 42"):
        load_frozen_formal_results(tmp_path, manifest, "freeze-a")


def test_duplicate_llm_repetition_is_rejected():
    raw = _raw_results()
    duplicate = raw[
        (raw["method"] == "FMAS_PCV_FEDYOGI")
        & (raw["training_seed"] == SEEDS[0])
        & (raw["llm_rep"] == 1)
    ]
    raw = pd.concat([raw, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate repetition"):
        aggregate_formal_repetitions(raw)


def test_negative_error_metric_is_rejected_before_aggregation():
    raw = _raw_results()
    raw.loc[0, "test_mape"] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_formal_repetitions(raw)


def test_four_of_five_wins_has_stable_claim_category():
    per_seed = aggregate_formal_repetitions(_raw_results())
    fmas_index = per_seed[per_seed["method"] == "FMAS_PCV_FEDYOGI"].index
    fedavg = per_seed[per_seed["method"] == "FEDAVG_STRICT"].set_index(
        "training_seed"
    )["test_mape"]
    values = [
        fedavg.loc[seed] - 0.005 if index < 4 else fedavg.loc[seed] + 0.005
        for index, seed in enumerate(SEEDS)
    ]
    per_seed.loc[fmas_index, "test_mape"] = values

    report = analyze_fmas_formal_results(per_seed)

    assert report["stable_improvement"] is True
    assert report["significant_improvement"] is False
    assert report["claim_status"] == "stable_improvement"


def test_formal_cli_uses_frozen_loader_and_writes_hierarchical_outputs(
    tmp_path, monkeypatch
):
    raw = _raw_results()
    calls = []
    manifest = replace(
        load_study_manifest(Path("study_manifest.yaml")),
        formal_frozen=True,
        stage="formal_ready",
        paper_eligible_freeze_ids=("freeze-a",),
    )
    monkeypatch.setattr(
        "scripts.statistical_analysis.load_study_manifest", lambda path: manifest
    )

    def fake_loader(results_root, loaded_manifest, freeze_id):
        calls.append((Path(results_root), loaded_manifest, freeze_id))
        return raw

    monkeypatch.setattr(
        "scripts.statistical_analysis.load_frozen_formal_results", fake_loader
    )
    results_root = tmp_path / "results"

    assert main(
        [
            "--freeze-id",
            "freeze-a",
            "--results-root",
            str(results_root),
        ]
    ) == 0
    assert calls and calls[0][2] == "freeze-a"
    output = results_root / "paper" / "freeze-a" / "statistics"
    assert (output / "formal_raw_runs.csv").is_file()
    assert (output / "formal_per_seed.csv").is_file()
    assert (output / "formal_analysis.json").is_file()


def test_formal_cli_routes_to_frozen_batch_analysis(monkeypatch, tmp_path):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"claim_status": "mean_improvement_trend"}

    monkeypatch.setattr(statistics_module, "run_frozen_formal_statistics", fake_run)
    assert statistics_module.main(
        [
            "--freeze-id", "freeze-a",
            "--results-root", str(tmp_path / "results"),
            "--study-manifest", str(tmp_path / "study_manifest.yaml"),
        ]
    ) == 0
    assert captured == {
        "results_root": tmp_path / "results",
        "freeze_id": "freeze-a",
        "manifest_path": tmp_path / "study_manifest.yaml",
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_frozen_loader_accepts_one_complete_45_run_evidence_chain(
    tmp_path, monkeypatch
):
    freeze_id = "freeze-a"
    manifest = replace(
        load_study_manifest(Path("study_manifest.yaml")),
        stage="formal_ready",
        formal_frozen=True,
        paper_eligible_freeze_ids=(freeze_id,),
        formal_seeds=SEEDS,
    )
    results_root = tmp_path / "results"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/formal_frozen.yaml").write_text("{}\n", encoding="utf-8")
    prompt_hashes = {
        role: f"hash-{role}"
        for roles in METHOD_PROMPT_ROLES.values()
        for role in roles
    }
    freeze_payload = {
        "partition_sha256": "p" * 64,
        "sealed_partition_metadata_sha256": "s" * 64,
        "method_config_sha256s": {
            method: f"method-{method}" for method in manifest.formal_methods
        },
        "prompt_sha256s": prompt_hashes,
    }
    monkeypatch.setattr(
        statistics_module,
        "validate_freeze_record",
        lambda *args, **kwargs: freeze_payload,
    )
    batch_records = []
    for method in manifest.formal_methods:
        for seed in SEEDS:
            for rep in METHOD_REPETITIONS[method]:
                run_root = results_root / "formal" / freeze_id / method / str(seed) / str(rep)
                run_root.mkdir(parents=True)
                checkpoint = run_root / "last_complete.pt"
                checkpoint.write_bytes(f"{method}-{seed}-{rep}".encode())
                prompts = {role: prompt_hashes[role] for role in METHOD_PROMPT_ROLES[method]}
                provenance = {
                    "schema_version": 1,
                    "method": method,
                    "phase": "formal_train",
                    "training_seed": seed,
                    "llm_rep": rep,
                    "run_id": None,
                    "freeze_id": freeze_id,
                    "git_commit": "g" * 40,
                    "git_dirty": False,
                    "partition_sha256": freeze_payload["partition_sha256"],
                    "sealed_partition_metadata_sha256": freeze_payload[
                        "sealed_partition_metadata_sha256"
                    ],
                    "method_config_sha256": freeze_payload["method_config_sha256s"][method],
                    "base_config_sha256": "b" * 64,
                    "effective_config_sha256": "e" * 64,
                    "prompt_hashes": prompts,
                    "deepseek": deepseek_provenance(enabled=bool(prompts)),
                    "resume_requested": False,
                    "locked_test_unlocked": False,
                }
                _write_json(run_root / "provenance.json", provenance)
                metrics = {
                    "sample_count": 105,
                    "mape": 0.4 + rep * 0.001,
                    "rmse": 1_000_000.0,
                    "mae": 800_000.0,
                    "r2": 0.4,
                }
                validation = {
                    "status": "complete",
                    "phase": "formal_train",
                    "method": method,
                    "training_seed": seed,
                    "llm_rep": rep,
                    "completed_rounds": 20,
                    "best_validation": metrics,
                }
                _write_json(run_root / "validation_metrics.json", validation)
                training_completion = {
                    "status": "complete",
                    "phase": "formal_train",
                    "method": method,
                    "training_seed": seed,
                    "llm_rep": rep,
                    "last_complete_round": 20,
                    "resolved_pause_reports": [],
                    "resume_approved": False,
                    "provenance": "provenance.json",
                    "evaluation_provenance": None,
                    "result_status": "complete",
                    "result_file": "validation_metrics.json",
                    "result_sha256": file_sha256(run_root / "validation_metrics.json"),
                }
                _write_json(run_root / "TRAINING_COMPLETE.json", training_completion)
                evaluation = dict(provenance)
                evaluation.update(
                    {
                        "phase": "formal_evaluate",
                        "resume_requested": True,
                        "locked_test_unlocked": True,
                        "training_checkpoint_sha256": file_sha256(checkpoint),
                    }
                )
                _write_json(run_root / "evaluation_provenance.json", evaluation)
                locked = {
                    "schema_version": 1,
                    "phase": "formal_evaluate",
                    "method": method,
                    "training_seed": seed,
                    "llm_rep": rep,
                    "training_checkpoint_sha256": file_sha256(checkpoint),
                    "evaluation_provenance_sha256": file_sha256(
                        run_root / "evaluation_provenance.json"
                    ),
                    "locked_test": metrics,
                }
                _write_json(run_root / "locked_test_metrics.json", locked)
                evaluation_completion = {
                    **training_completion,
                    "phase": "formal_evaluate",
                    "resume_approved": True,
                    "evaluation_provenance": "evaluation_provenance.json",
                    "result_file": "locked_test_metrics.json",
                    "result_sha256": file_sha256(run_root / "locked_test_metrics.json"),
                }
                _write_json(run_root / "EVALUATION_COMPLETE.json", evaluation_completion)
                batch_records.append(
                    {
                        "method": method,
                        "training_seed": seed,
                        "llm_rep": rep,
                        "completion_sha256": file_sha256(run_root / "TRAINING_COMPLETE.json"),
                        "validation_sha256": file_sha256(run_root / "validation_metrics.json"),
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "evaluation_completion_sha256": file_sha256(
                            run_root / "EVALUATION_COMPLETE.json"
                        ),
                        "locked_test_sha256": file_sha256(
                            run_root / "locked_test_metrics.json"
                        ),
                    }
                )
    _write_json(
        results_root / "formal" / freeze_id / "EVALUATION_BATCH_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "phase": "formal_evaluate",
            "freeze_id": freeze_id,
            "run_count": 45,
            "runs": batch_records,
        },
    )
    frame = load_frozen_formal_results(results_root, manifest, freeze_id)
    assert len(frame) == 45
    assert set(frame["training_seed"]) == set(SEEDS)

    first = batch_records[0]
    first_checkpoint = (
        results_root
        / "formal"
        / freeze_id
        / first["method"]
        / str(first["training_seed"])
        / str(first["llm_rep"])
        / "last_complete.pt"
    )
    first_checkpoint.write_bytes(b"tampered-checkpoint")
    with pytest.raises(ValueError, match="batch evidence"):
        load_frozen_formal_results(results_root, manifest, freeze_id)
