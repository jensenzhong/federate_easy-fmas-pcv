"""Deterministic formal-protocol and freeze identities for FMAS-PCV."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.federated_learning.pcv.provider_config import deepseek_protocol_config
from src.study_manifest import FORMAL_METHOD_ORDER, load_study_manifest


PARTITION_MANIFEST = "results/manifests/strict_partition_v1.csv"
SEALED_PARTITION_METADATA = "Data/strict_partition_v1/metadata.json"
DEVELOPMENT_CONFIG = "configs/development_seed42.yaml"
BASE_CONFIG = "configs/config.yaml"
DEVELOPMENT_GATE = "results/development/seed42/development_gate.json"
BASELINE_FAIRNESS_AUDIT = "audits/baseline_fairness_audit.json"

METHOD_CONFIG_PATHS = {
    "FEDAVG_STRICT": "configs/methods/fedavg_strict.yaml",
    "FEDYOGI_STRICT": "configs/methods/fedyogi_strict.yaml",
    "DPCV_FEDYOGI": "configs/methods/dpcv_fedyogi.yaml",
    "SA_PCV_FEDYOGI": "configs/methods/sa_pcv_fedyogi.yaml",
    "FMAS_PCV_FEDYOGI": "configs/methods/fmas_pcv_fedyogi.yaml",
}

METHOD_PROMPT_ROLES = {
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

METHOD_REPETITIONS = {
    method: ([1, 2, 3] if METHOD_PROMPT_ROLES[method] else [0])
    for method in FORMAL_METHOD_ORDER
}

ANALYSIS_PROTOCOL = {
    "primary_metric": "mape",
    "secondary_metrics": ["rmse", "mae", "r2"],
    "primary_strict_baseline": "FEDAVG_STRICT",
    "llm_repetition_aggregation": "mean_within_training_seed",
    "paired_observation_unit": "training_seed",
    "stable_improvement_min_seed_wins": 4,
    "formal_seed_count": 5,
    "multiple_comparison_correction": "holm",
    "alpha": 0.05,
    "comparison_family": [
        "FMAS_PCV_FEDYOGI_vs_FEDAVG_STRICT",
        "FMAS_PCV_FEDYOGI_vs_FEDYOGI_STRICT",
        "FMAS_PCV_FEDYOGI_vs_DPCV_FEDYOGI",
        "FMAS_PCV_FEDYOGI_vs_SA_PCV_FEDYOGI",
    ],
}

TRAINING_PROTOCOL = {
    "rounds": 20,
    "local_epochs": 20,
    "checkpoint_selection": "lowest_aggregated_client_val_mape",
    "metric_schema": ["sample_count", "mape", "rmse", "mae", "r2"],
    "client_participation": "all_clients_every_round",
}

FAILURE_PROTOCOL = {
    "automatic_transport_retry": False,
    "max_json_parse_regeneration_retries": 1,
    "json_parse_regeneration_requires_identical_request": True,
    "non_parse_failures_retry": False,
    "fallback_model": None,
    "resume_requires_explicit_user_approval": True,
    "locked_test_policy": "separate_formal_evaluate_after_complete_training_batch",
}

_MUTABLE_STUDY_FIELDS = {"stage", "formal_frozen", "paper_eligible_freeze_ids"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"freeze input must be a regular file: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _read_yaml_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"YAML input must be a regular file: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"YAML input is unreadable: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"YAML input must contain one object: {path}")
    return value


def normalized_study_protocol(project_root: Path) -> dict[str, Any]:
    raw = _read_yaml_object(Path(project_root) / "study_manifest.yaml")
    missing = _MUTABLE_STUDY_FIELDS - set(raw)
    if missing:
        raise ValueError(f"study manifest is missing state fields: {sorted(missing)}")
    return {key: value for key, value in raw.items() if key not in _MUTABLE_STUDY_FIELDS}


def frozen_execution_config_sha256(project_root: Path) -> str:
    """Hash execution semantics while excluding generated freeze state."""

    root = Path(project_root)
    record = {
        "study_protocol": normalized_study_protocol(root),
        "files": {
            path: file_sha256(root / path)
            for path in (
                BASE_CONFIG,
                DEVELOPMENT_CONFIG,
                *(METHOD_CONFIG_PATHS[method] for method in FORMAL_METHOD_ORDER),
            )
        },
    }
    return sha256(canonical_json_bytes(record)).hexdigest()


def formal_protocol_semantics(project_root: Path) -> dict[str, Any]:
    manifest = load_study_manifest(Path(project_root) / "study_manifest.yaml")
    return {
        "schema_version": 1,
        "formal_seeds": list(manifest.formal_seeds),
        "formal_methods": list(manifest.formal_methods),
        "method_repetitions": METHOD_REPETITIONS,
        "partition_manifest": PARTITION_MANIFEST,
        "deepseek": deepseek_protocol_config(),
        "training": TRAINING_PROTOCOL,
        "analysis": ANALYSIS_PROTOCOL,
        "failure_policy": FAILURE_PROTOCOL,
    }


def formal_config_sha256(project_root: Path) -> str:
    return sha256(canonical_json_bytes(formal_protocol_semantics(project_root))).hexdigest()


def prompt_sha256s(project_root: Path) -> dict[str, str]:
    roles = sorted({role for method in FORMAL_METHOD_ORDER for role in METHOD_PROMPT_ROLES[method]})
    return {
        role: file_sha256(Path(project_root) / "configs/prompts" / f"{role}.md")
        for role in roles
    }


def method_config_sha256s(project_root: Path) -> dict[str, str]:
    return {
        method: file_sha256(Path(project_root) / METHOD_CONFIG_PATHS[method])
        for method in FORMAL_METHOD_ORDER
    }


def build_freeze_payload(
    project_root: Path,
    *,
    source_commit: str,
    development_gate_path: Path | None = None,
    baseline_audit_path: Path | None = None,
    supersedes_freeze_id: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    gate_path = development_gate_path or root / DEVELOPMENT_GATE
    audit_path = baseline_audit_path or root / BASELINE_FAIRNESS_AUDIT
    manifest = load_study_manifest(root / "study_manifest.yaml")
    return {
        "git_commit": source_commit,
        "supersedes_freeze_id": supersedes_freeze_id,
        "partition_sha256": file_sha256(root / PARTITION_MANIFEST),
        "sealed_partition_metadata_sha256": file_sha256(root / SEALED_PARTITION_METADATA),
        "development_config_sha256": file_sha256(root / DEVELOPMENT_CONFIG),
        "formal_config_sha256": formal_config_sha256(root),
        "execution_config_sha256": frozen_execution_config_sha256(root),
        "method_config_sha256s": method_config_sha256s(root),
        "prompt_sha256s": prompt_sha256s(root),
        "deepseek": deepseek_protocol_config(),
        "formal_seeds": list(manifest.formal_seeds),
        "formal_methods": list(manifest.formal_methods),
        "method_repetitions": METHOD_REPETITIONS,
        "development_gate_sha256": file_sha256(gate_path),
        "baseline_fairness_audit_sha256": file_sha256(audit_path),
        "analysis": ANALYSIS_PROTOCOL,
        "training": TRAINING_PROTOCOL,
        "failure_policy": FAILURE_PROTOCOL,
    }


def freeze_id_from_payload(payload: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(dict(payload))).hexdigest()[:16]


def formal_frozen_document(payload: Mapping[str, Any], freeze_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "formal_frozen": True,
        "freeze_id": freeze_id,
        "supersedes_freeze_id": payload["supersedes_freeze_id"],
        "formal_seeds": list(payload["formal_seeds"]),
        "partition_manifest": PARTITION_MANIFEST,
        "partition_sha256": payload["partition_sha256"],
        "sealed_partition_metadata_sha256": payload["sealed_partition_metadata_sha256"],
        "config_sha256": payload["execution_config_sha256"],
        "prompt_hashes": dict(payload["prompt_sha256s"]),
        "git_commit": payload["git_commit"],
        "deepseek": dict(payload["deepseek"]),
        "method_repetitions": dict(payload["method_repetitions"]),
        "development_gate_sha256": payload["development_gate_sha256"],
        "baseline_fairness_audit_sha256": payload["baseline_fairness_audit_sha256"],
        "analysis": dict(payload["analysis"]),
    }


def validate_freeze_record(
    project_root: Path,
    *,
    freeze_id: str,
    frozen_document: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root)
    record_path = root / "results/manifests" / f"{freeze_id}.json"
    if record_path.is_symlink() or not record_path.is_file():
        raise RuntimeError("formal freeze manifest is missing")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("formal freeze manifest is unreadable") from error
    if type(record) is not dict or set(record) != {
        "schema_version",
        "freeze_id",
        "payload",
        "formal_protocol",
    }:
        raise RuntimeError("formal freeze manifest schema mismatch")
    if record.get("schema_version") != 1 or record.get("freeze_id") != freeze_id:
        raise RuntimeError("formal freeze manifest identity mismatch")
    payload = record.get("payload")
    if type(payload) is not dict or freeze_id_from_payload(payload) != freeze_id:
        raise RuntimeError("formal freeze id does not match its payload")
    current_protected = {
        "partition_sha256": file_sha256(root / PARTITION_MANIFEST),
        "sealed_partition_metadata_sha256": file_sha256(
            root / SEALED_PARTITION_METADATA
        ),
        "development_config_sha256": file_sha256(root / DEVELOPMENT_CONFIG),
        "formal_config_sha256": formal_config_sha256(root),
        "execution_config_sha256": frozen_execution_config_sha256(root),
        "method_config_sha256s": method_config_sha256s(root),
        "prompt_sha256s": prompt_sha256s(root),
        "deepseek": deepseek_protocol_config(),
        "formal_seeds": list(load_study_manifest(root / "study_manifest.yaml").formal_seeds),
        "formal_methods": list(FORMAL_METHOD_ORDER),
        "method_repetitions": METHOD_REPETITIONS,
        "analysis": ANALYSIS_PROTOCOL,
        "training": TRAINING_PROTOCOL,
        "failure_policy": FAILURE_PROTOCOL,
    }
    if (
        type(payload.get("git_commit")) is not str
        or len(payload["git_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in payload["git_commit"])
        or any(payload.get(key) != value for key, value in current_protected.items())
        or type(payload.get("development_gate_sha256")) is not str
        or type(payload.get("baseline_fairness_audit_sha256")) is not str
        or (
            payload.get("supersedes_freeze_id") is not None
            and (
                type(payload["supersedes_freeze_id"]) is not str
                or len(payload["supersedes_freeze_id"]) != 16
                or any(
                    character not in "0123456789abcdef"
                    for character in payload["supersedes_freeze_id"]
                )
            )
        )
    ):
        raise RuntimeError("formal freeze payload does not match current protected inputs")
    if record.get("formal_protocol") != formal_protocol_semantics(root):
        raise RuntimeError("formal protocol semantics mismatch")
    if dict(frozen_document) != formal_frozen_document(payload, freeze_id):
        raise RuntimeError("formal frozen YAML does not match the freeze manifest")
    return payload


__all__ = [
    "ANALYSIS_PROTOCOL",
    "BASELINE_FAIRNESS_AUDIT",
    "DEVELOPMENT_GATE",
    "FAILURE_PROTOCOL",
    "METHOD_CONFIG_PATHS",
    "METHOD_PROMPT_ROLES",
    "METHOD_REPETITIONS",
    "PARTITION_MANIFEST",
    "SEALED_PARTITION_METADATA",
    "TRAINING_PROTOCOL",
    "build_freeze_payload",
    "canonical_json_bytes",
    "file_sha256",
    "formal_config_sha256",
    "formal_frozen_document",
    "formal_protocol_semantics",
    "freeze_id_from_payload",
    "frozen_execution_config_sha256",
    "method_config_sha256s",
    "normalized_study_protocol",
    "prompt_sha256s",
    "validate_freeze_record",
]
