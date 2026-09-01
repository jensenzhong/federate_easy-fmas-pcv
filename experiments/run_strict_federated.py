"""Canonical entry point for the strict FMAS-PCV study.

This module owns phase, provenance, immutable-output, and secret-reading gates.
Training is supplied as an explicit executor so preflight and runner policy can
be tested without loading client data or making network calls.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
import unicodedata

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated_learning.pcv.agents import (  # noqa: E402
    DeepSeekCallError,
    StrictDeepSeekClient,
)
from src.federated_learning.pcv.engine import (  # noqa: E402
    ExperimentPaused,
    ExperimentRuntimeError,
)
from src.federated_learning.pcv.checkpoint import load_checkpoint  # noqa: E402
from src.federated_learning.pcv.protocol import (  # noqa: E402
    TestPartitionLocked,
    require_test_unlock,
)
from src.federated_learning.pcv.provider_config import (  # noqa: E402
    DEEPSEEK_MODEL,
    deepseek_client_settings_from_provenance,
    deepseek_protocol_config,
    deepseek_provenance,
)
from src.federated_learning.pcv.telemetry import AppendOnlyTelemetry  # noqa: E402
from src.formal_protocol import (  # noqa: E402
    ANALYSIS_PROTOCOL,
    METHOD_CONFIG_PATHS,
    METHOD_PROMPT_ROLES,
    METHOD_REPETITIONS,
    PARTITION_MANIFEST,
    SEALED_PARTITION_METADATA,
    frozen_execution_config_sha256 as _stable_execution_config_sha256,
    validate_freeze_record,
)
from src.study_manifest import (  # noqa: E402
    FORMAL_METHOD_ORDER,
    StudyManifest,
    load_study_manifest,
)


FORMAL_METHODS = frozenset(FORMAL_METHOD_ORDER)
LLM_METHODS = frozenset({"SA_PCV_FEDYOGI", "FMAS_PCV_FEDYOGI"})
PHASES = ("development", "formal_train", "formal_evaluate")
COMMON_METHOD_CONFIG = {
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
METHOD_DIFFERENCES = {
    "FEDAVG_STRICT": ("fedavg", "anchor_only"),
    "FEDYOGI_STRICT": ("fedyogi", "anchor_only"),
    "DPCV_FEDYOGI": ("fedyogi", "deterministic"),
    "SA_PCV_FEDYOGI": ("fedyogi", "single_agent"),
    "FMAS_PCV_FEDYOGI": ("fedyogi", "multi_agent"),
}
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SENSITIVE_KEY_FRAGMENTS = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "privatekey",
    "secret",
)


@dataclass(frozen=True, slots=True)
class RunContext:
    args: argparse.Namespace
    manifest: StudyManifest
    method_config: Mapping[str, Any]
    run_directory: Path
    api_key: str | None
    provenance_path: Path
    evaluation_provenance_path: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical strict-federated FMAS-PCV runner"
    )
    parser.add_argument("--method", required=True, choices=tuple(FORMAL_METHOD_ORDER))
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--llm-rep", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--freeze-id")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--user-approved-resume", action="store_true")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _safe_component(value: str, *, name: str) -> str:
    if type(value) is not str or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{name} must be one safe path component")
    if value in {".", ".."} or ":" in value:
        raise ValueError(f"{name} must be one safe path component")
    return value


def resolve_run_directory(base_directory: Path, run_id: str) -> Path:
    """Resolve a development run path and reject every existing filesystem node."""

    root = Path(base_directory).resolve(strict=False)
    component = _safe_component(run_id, name="run_id")
    target = root / component
    if os.path.lexists(target):
        raise FileExistsError(f"run directory already exists: {target}")
    return target


def create_run_directory(base_directory: Path, run_id: str) -> Path:
    target = resolve_run_directory(base_directory, run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"configuration must be an exact mapping: {path}")
    return value


def load_method_config(
    method: str, *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    if method not in FORMAL_METHODS:
        raise ValueError(f"unknown formal method: {method!r}")
    config = _read_yaml(project_root / METHOD_CONFIG_PATHS[method])
    expected_keys = {
        *COMMON_METHOD_CONFIG,
        "method",
        "server_optimizer",
        "proposal_mode",
        "deepseek_roles",
    }
    if set(config) != expected_keys:
        raise ValueError(f"method config fields mismatch for {method}")
    if config["method"] != method:
        raise ValueError(f"method config identity mismatch for {method}")
    for key, expected in COMMON_METHOD_CONFIG.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f"method config mismatch for {method}.{key}")
    optimizer, proposal_mode = METHOD_DIFFERENCES[method]
    roles = list(METHOD_PROMPT_ROLES[method])
    if (
        config["server_optimizer"] != optimizer
        or config["proposal_mode"] != proposal_mode
        or config["deepseek_roles"] != roles
    ):
        raise ValueError(f"method-specific config mismatch for {method}")
    return config


def resolve_api_key(method: str, environment: Mapping[str, str]) -> str | None:
    if method not in LLM_METHODS:
        return None
    value = environment.get("DEEPSEEK_API_KEY")
    if type(value) is not str or not value.strip():
        raise DeepSeekCallError(
            "authentication", "preflight", "DEEPSEEK_API_KEY is missing"
        )
    return value


def validate_invocation(args: argparse.Namespace, manifest: StudyManifest) -> None:
    if args.method not in manifest.formal_methods:
        raise ValueError("method is not enabled by study_manifest.yaml")
    if type(args.training_seed) is not int or type(args.llm_rep) is not int:
        raise TypeError("seed and llm_rep must be exact integers")
    if args.llm_rep < 0:
        raise ValueError("llm_rep must be non-negative")

    is_formal = args.phase in {"formal_train", "formal_evaluate"}
    if is_formal and manifest.formal_frozen is not True:
        raise RuntimeError("formal phases require a frozen study manifest")
    if args.phase == "development":
        if args.training_seed != manifest.development_seed:
            raise ValueError("development seed must equal the manifest development seed")
        if args.run_id is None:
            raise ValueError("development runs require --run-id")
        _safe_component(args.run_id, name="run_id")
        if args.freeze_id is not None:
            raise ValueError("development runs cannot claim a freeze_id")
        if args.unlock_test:
            raise RuntimeError("locked test cannot be unlocked during development")
    else:
        if args.training_seed not in manifest.formal_seeds:
            raise ValueError("formal training seed is not predeclared")
        if args.freeze_id is None:
            raise ValueError("formal phases require --freeze-id")
        _safe_component(args.freeze_id, name="freeze_id")
        if args.freeze_id not in manifest.paper_eligible_freeze_ids:
            raise RuntimeError("freeze_id is not paper eligible in the manifest")
        if args.run_id is not None:
            raise ValueError("formal paths are derived; --run-id is not accepted")

    if args.phase == "formal_evaluate":
        try:
            require_test_unlock(
                phase=args.phase,
                formal_frozen=manifest.formal_frozen,
                explicit_unlock=args.unlock_test,
            )
        except TestPartitionLocked as error:
            raise RuntimeError("locked test requires explicit frozen formal unlock") from error
        if args.resume_checkpoint is None:
            raise RuntimeError(
                "formal evaluation requires the completed training checkpoint"
            )
    elif args.unlock_test:
        raise RuntimeError("locked test is unavailable outside formal_evaluate")

    if (args.resume_checkpoint is None) != (not args.user_approved_resume):
        if args.resume_checkpoint is None:
            raise RuntimeError("resume approval requires --resume-checkpoint")
        raise RuntimeError("resume checkpoint requires explicit user approval")
    if args.method not in LLM_METHODS and args.llm_rep != 0:
        raise ValueError("non-LLM methods require llm_rep=0")
    if args.method in LLM_METHODS and not args.preflight_only and args.llm_rep <= 0:
        raise ValueError("LLM training runs require a positive llm_rep")
    if args.preflight_only:
        if args.phase != "development" or args.method not in LLM_METHODS:
            raise ValueError("preflight-only is development-only and requires an LLM method")
        if args.resume_checkpoint is not None:
            raise ValueError("preflight cannot resume training")


def _assert_provenance_safe(value: Any) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("provenance keys must be exact strings")
            normalized = "".join(
                character
                for character in unicodedata.normalize("NFKC", key).casefold()
                if character.isalnum()
            )
            if any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS):
                raise ValueError("sensitive provenance fields are forbidden")
            _assert_provenance_safe(item)
    elif type(value) is list:
        for item in value:
            _assert_provenance_safe(item)
    elif type(value) not in (type(None), bool, int, float, str):
        raise TypeError("provenance values must be exact JSON values")


def _publish_bytes_no_replace(path: Path, content: bytes) -> Path:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        linked = True
        return path
    finally:
        temporary.unlink(missing_ok=True)
        if not linked and os.path.lexists(path):
            # A concurrent publisher owns this path; never remove it.
            pass


def write_provenance(
    run_directory: Path,
    record: dict[str, Any],
    *,
    filename: str = "provenance.json",
) -> Path:
    if type(record) is not dict:
        raise TypeError("provenance record must be an exact dictionary")
    _assert_provenance_safe(record)
    encoded = (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if filename != "provenance.json" and not re.fullmatch(
        r"evaluation_provenance(?:\.\d{3})?\.json", filename
    ):
        raise ValueError("invalid provenance filename")
    return _publish_bytes_no_replace(Path(run_directory) / filename, encoded)


def _write_evaluation_provenance(
    run_directory: Path, record: dict[str, Any]
) -> Path:
    for index in range(1000):
        filename = (
            "evaluation_provenance.json"
            if index == 0
            else f"evaluation_provenance.{index:03d}.json"
        )
        try:
            return write_provenance(
                run_directory,
                record,
                filename=filename,
            )
        except FileExistsError:
            continue
    raise RuntimeError("evaluation attempt namespace is exhausted")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def effective_config_sha256(base_config_path: Path, method_config_path: Path) -> str:
    """Bind the exact base and method bytes into one unambiguous run identity."""

    digest = sha256()
    for label, path in (
        (b"base-config", Path(base_config_path)),
        (b"method-config", Path(method_config_path)),
    ):
        content = path.read_bytes()
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def frozen_execution_config_sha256(project_root: Path) -> str:
    """Hash execution semantics while ignoring generated freeze-state fields."""

    return _stable_execution_config_sha256(project_root)


def _matches_frozen_source_commit(
    *, project_root: Path, frozen_commit: Any, current_commit: str, freeze_id: str
) -> bool:
    """Accept HEAD or one metadata-only freeze commit based on the frozen source."""

    if type(frozen_commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", frozen_commit):
        return False
    if frozen_commit == current_commit:
        return True
    try:
        count = subprocess.run(
            ["git", "rev-list", "--count", f"{frozen_commit}..{current_commit}"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "diff", "--name-only", frozen_commit, current_commit, "--"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    allowed = {
        "study_manifest.yaml",
        "configs/formal_frozen.yaml",
        f"results/manifests/{freeze_id}.json",
    }
    return count == "1" and bool(changed) and set(changed) <= allowed


def validate_formal_freeze(
    *,
    project_root: Path,
    manifest: StudyManifest,
    args: argparse.Namespace,
    git_commit: str,
    git_dirty: bool,
) -> None:
    """Fail closed unless the complete formal execution matches its freeze."""

    if git_dirty is not False:
        raise RuntimeError("formal execution requires a clean Git worktree")
    payload = _read_yaml(project_root / "configs/formal_frozen.yaml")
    expected_fields = {
        "schema_version", "formal_frozen", "freeze_id", "formal_seeds",
        "partition_manifest", "partition_sha256",
        "sealed_partition_metadata_sha256", "config_sha256", "prompt_hashes",
        "git_commit", "deepseek", "method_repetitions",
        "development_gate_sha256", "baseline_fairness_audit_sha256", "analysis",
    }
    if set(payload) != expected_fields or payload.get("formal_frozen") is not True:
        raise RuntimeError("formal freeze payload is incomplete or inactive")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise RuntimeError("formal freeze schema mismatch")
    if payload.get("freeze_id") != args.freeze_id:
        raise RuntimeError("formal freeze id mismatch")
    if payload.get("formal_seeds") != list(manifest.formal_seeds):
        raise RuntimeError("formal freeze seed list mismatch")
    if not _matches_frozen_source_commit(
        project_root=project_root,
        frozen_commit=payload.get("git_commit"),
        current_commit=git_commit,
        freeze_id=args.freeze_id,
    ):
        raise RuntimeError("formal freeze Git commit mismatch")
    partition_name = PARTITION_MANIFEST
    if payload.get("partition_manifest") != partition_name or payload.get(
        "partition_sha256"
    ) != _file_sha256(project_root / partition_name):
        raise RuntimeError("formal freeze partition mismatch")
    if payload.get("sealed_partition_metadata_sha256") != _file_sha256(
        project_root / SEALED_PARTITION_METADATA
    ):
        raise RuntimeError("formal freeze sealed partition mismatch")
    if payload.get("config_sha256") != frozen_execution_config_sha256(project_root):
        raise RuntimeError("formal freeze configuration mismatch")
    prompt_roles = sorted(
        {
            role
            for method in FORMAL_METHOD_ORDER
            for role in METHOD_PROMPT_ROLES[method]
        }
    )
    expected_prompts = {
        role: _file_sha256(project_root / "configs/prompts" / f"{role}.md")
        for role in prompt_roles
    }
    if payload.get("prompt_hashes") != expected_prompts:
        raise RuntimeError("formal freeze prompt hashes mismatch")
    if payload.get("deepseek") != deepseek_protocol_config():
        raise RuntimeError("formal freeze DeepSeek parameters mismatch")
    if payload.get("method_repetitions") != METHOD_REPETITIONS:
        raise RuntimeError("formal freeze repetition protocol mismatch")
    if payload.get("analysis") != ANALYSIS_PROTOCOL:
        raise RuntimeError("formal freeze analysis protocol mismatch")
    validated = validate_freeze_record(
        project_root,
        freeze_id=args.freeze_id,
        frozen_document=payload,
    )
    if validated.get("git_commit") != payload.get("git_commit"):
        raise RuntimeError("formal freeze source commit mismatch")


def _git_metadata(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return commit, dirty


def _require_complete_training_batch(
    results_root: Path,
    freeze_id: str,
    *,
    project_root: Path,
    manifest: StudyManifest,
    git_commit: str,
) -> None:
    path = Path(results_root) / "formal" / freeze_id / "TRAINING_BATCH_COMPLETE.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("formal evaluation requires the complete training batch")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("formal training batch manifest is unreadable") from error
    expected_runs = [
        (method, seed, rep)
        for seed in manifest.formal_seeds
        for method in FORMAL_METHOD_ORDER
        for rep in METHOD_REPETITIONS[method]
    ]
    if (
        type(record) is not dict
        or set(record)
        != {"schema_version", "status", "phase", "freeze_id", "git_commit", "run_count", "runs"}
        or record.get("schema_version") != 1
        or record.get("status") != "complete"
        or record.get("phase") != "formal_train"
        or record.get("freeze_id") != freeze_id
        or record.get("git_commit") != git_commit
        or record.get("run_count") != len(expected_runs)
        or type(record.get("runs")) is not list
        or len(record["runs"]) != len(expected_runs)
    ):
        raise RuntimeError("formal training batch manifest is incomplete")
    for run_record, (method, seed, rep) in zip(
        record["runs"], expected_runs, strict=True
    ):
        if (
            type(run_record) is not dict
            or set(run_record)
            != {
                "method", "training_seed", "llm_rep", "completion_sha256",
                "validation_sha256", "checkpoint_sha256",
            }
            or run_record.get("method") != method
            or run_record.get("training_seed") != seed
            or run_record.get("llm_rep") != rep
        ):
            raise RuntimeError("formal training batch run identity mismatch")
        run_root = Path(results_root) / "formal" / freeze_id / method / str(seed) / str(rep)
        for field, filename in (
            ("completion_sha256", "TRAINING_COMPLETE.json"),
            ("validation_sha256", "validation_metrics.json"),
            ("checkpoint_sha256", "last_complete.pt"),
        ):
            expected_sha = run_record.get(field)
            if (
                type(expected_sha) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
                or _file_sha256(run_root / filename) != expected_sha
            ):
                raise RuntimeError("formal training batch evidence hash mismatch")
        try:
            completion = json.loads(
                (run_root / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
            )
            validation = json.loads(
                (run_root / "validation_metrics.json").read_text(encoding="utf-8")
            )
            provenance = json.loads(
                (run_root / "provenance.json").read_text(encoding="utf-8")
            )
            checkpoint = load_checkpoint(run_root / "last_complete.pt")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError("formal training batch evidence is unreadable") from error
        method_path = project_root / METHOD_CONFIG_PATHS[method]
        base_path = project_root / "configs/config.yaml"
        method_config = load_method_config(method, project_root=project_root)
        expected_prompts = _prompt_hashes(method_config, project_root)
        checkpoint_prompts = expected_prompts or {"engine": "no-agent-prompts"}
        pause_names = sorted(item.name for item in run_root.glob("PAUSED*.json"))
        if (
            type(completion) is not dict
            or set(completion)
            != {
                "status", "phase", "method", "training_seed", "llm_rep",
                "last_complete_round", "resolved_pause_reports", "resume_approved",
                "provenance", "evaluation_provenance", "result_status",
                "result_file", "result_sha256",
            }
            or completion.get("status") != "complete"
            or completion.get("phase") != "formal_train"
            or completion.get("method") != method
            or completion.get("training_seed") != seed
            or completion.get("llm_rep") != rep
            or completion.get("last_complete_round") != 20
            or completion.get("resolved_pause_reports") != pause_names
            or type(completion.get("resume_approved")) is not bool
            or completion.get("provenance") != "provenance.json"
            or completion.get("evaluation_provenance") is not None
            or completion.get("result_status") != "complete"
            or completion.get("result_file") != "validation_metrics.json"
            or type(validation) is not dict
            or set(validation)
            != {
                "status", "phase", "method", "training_seed", "llm_rep",
                "completed_rounds", "best_validation",
            }
            or validation.get("status") != "complete"
            or validation.get("phase") != "formal_train"
            or validation.get("method") != method
            or validation.get("training_seed") != seed
            or validation.get("llm_rep") != rep
            or validation.get("completed_rounds") != 20
            or type(provenance) is not dict
            or provenance.get("schema_version") != 1
            or provenance.get("method") != method
            or provenance.get("phase") != "formal_train"
            or provenance.get("training_seed") != seed
            or provenance.get("llm_rep") != rep
            or provenance.get("run_id") is not None
            or provenance.get("freeze_id") != freeze_id
            or provenance.get("git_commit") != git_commit
            or provenance.get("git_dirty") is not False
            or provenance.get("partition_sha256")
            != _file_sha256(project_root / PARTITION_MANIFEST)
            or provenance.get("sealed_partition_metadata_sha256")
            != _file_sha256(project_root / SEALED_PARTITION_METADATA)
            or provenance.get("method_config_sha256") != _file_sha256(method_path)
            or provenance.get("base_config_sha256") != _file_sha256(base_path)
            or provenance.get("effective_config_sha256")
            != effective_config_sha256(base_path, method_path)
            or provenance.get("prompt_hashes") != expected_prompts
            or provenance.get("deepseek")
            != deepseek_provenance(enabled=method in LLM_METHODS)
            or provenance.get("locked_test_unlocked") is not False
            or checkpoint.get("last_complete_round") != 20
            or checkpoint.get("freeze_id") != freeze_id
            or checkpoint.get("method") != method
            or checkpoint.get("training_seed") != seed
            or checkpoint.get("llm_rep") != rep
            or checkpoint.get("partition_sha256")
            != provenance.get("partition_sha256")
            or checkpoint.get("config_sha256")
            != provenance.get("effective_config_sha256")
            or checkpoint.get("prompt_hashes") != checkpoint_prompts
        ):
            raise RuntimeError("formal training batch evidence identity mismatch")


def _prompt_hashes(method_config: Mapping[str, Any], project_root: Path) -> dict[str, str]:
    hashes = {}
    for role in method_config["deepseek_roles"]:
        path = project_root / "configs" / "prompts" / f"{role}.md"
        hashes[role] = _file_sha256(path)
    return hashes


def _create_context_run_directory(
    args: argparse.Namespace, *, results_root: Path
) -> Path:
    if args.phase == "development":
        leaf = (
            Path(results_root).resolve(strict=False)
            / "development"
            / "seed42"
            / _safe_component(args.run_id, name="run_id")
        )
    else:
        leaf = (
            Path(results_root).resolve(strict=False)
            / "formal"
            / _safe_component(args.freeze_id, name="freeze_id")
            / args.method
            / str(args.training_seed)
            / str(args.llm_rep)
        )

    if args.resume_checkpoint is not None:
        checkpoint = Path(args.resume_checkpoint).resolve(strict=False)
        if checkpoint.name != "last_complete.pt" or checkpoint.parent != leaf:
            raise ValueError(
                "resume checkpoint must be the managed last_complete.pt in the run directory"
            )
        if not leaf.exists() or not leaf.is_dir() or leaf.is_symlink():
            raise FileNotFoundError("resume run directory is missing or unsafe")
        if args.phase == "formal_evaluate" and (
            leaf / "EVALUATION_COMPLETE.json"
        ).exists():
            raise FileExistsError("formal evaluation is already complete")
        return leaf

    if os.path.lexists(leaf):
        raise FileExistsError(f"run directory already exists: {leaf}")
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.mkdir(exist_ok=False)
    return leaf


def _reuse_resume_provenance(path: Path, expected: Mapping[str, Any]) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("resume provenance is missing or unsafe")
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("resume provenance is unreadable") from error
    if type(existing) is not dict:
        raise ValueError("resume provenance must be an exact dictionary")
    _assert_provenance_safe(existing)
    stable_fields = {
        "schema_version",
        "method",
        "training_seed",
        "llm_rep",
        "run_id",
        "freeze_id",
        "git_commit",
        "git_dirty",
        "partition_sha256",
        "sealed_partition_metadata_sha256",
        "method_config_sha256",
        "base_config_sha256",
        "effective_config_sha256",
        "prompt_hashes",
        "deepseek",
    }
    if any(existing.get(field) != expected.get(field) for field in stable_fields):
        raise ValueError("resume provenance does not match the requested run")
    allowed_phases = {expected["phase"]}
    if expected["phase"] == "formal_evaluate":
        allowed_phases.add("formal_train")
    if existing.get("phase") not in allowed_phases:
        raise ValueError("resume provenance phase is incompatible")
    return path


def prepare_run(
    args: argparse.Namespace,
    *,
    project_root: Path = PROJECT_ROOT,
    results_root: Path | None = None,
    environment: Mapping[str, str] = os.environ,
) -> RunContext:
    manifest = load_study_manifest(project_root / "study_manifest.yaml")
    validate_invocation(args, manifest)
    git_commit, git_dirty = _git_metadata(project_root)
    if args.phase in {"formal_train", "formal_evaluate"}:
        validate_formal_freeze(
            project_root=project_root,
            manifest=manifest,
            args=args,
            git_commit=git_commit,
            git_dirty=git_dirty,
        )
    method_config = load_method_config(args.method, project_root=project_root)
    if results_root is None:
        results_root = project_root / "results"
    if args.phase == "formal_evaluate":
        _require_complete_training_batch(
            results_root,
            args.freeze_id,
            project_root=project_root,
            manifest=manifest,
            git_commit=git_commit,
        )
    run_directory = _create_context_run_directory(args, results_root=results_root)

    partition_path = project_root / "results/manifests/strict_partition_v1.csv"
    method_path = project_root / METHOD_CONFIG_PATHS[args.method]
    base_config_path = project_root / "configs/config.yaml"
    provenance = {
        "schema_version": 1,
        "method": args.method,
        "phase": args.phase,
        "training_seed": args.training_seed,
        "llm_rep": args.llm_rep,
        "run_id": args.run_id,
        "freeze_id": args.freeze_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "partition_sha256": _file_sha256(partition_path),
        "sealed_partition_metadata_sha256": _file_sha256(
            project_root / "Data/strict_partition_v1/metadata.json"
        ),
        "method_config_sha256": _file_sha256(method_path),
        "base_config_sha256": _file_sha256(base_config_path),
        "effective_config_sha256": effective_config_sha256(
            base_config_path,
            method_path,
        ),
        "prompt_hashes": _prompt_hashes(method_config, project_root),
        "deepseek": deepseek_provenance(enabled=args.method in LLM_METHODS),
        "resume_requested": args.resume_checkpoint is not None,
        "locked_test_unlocked": args.phase == "formal_evaluate" and args.unlock_test,
    }
    evaluation_provenance_path = None
    if args.resume_checkpoint is None:
        provenance_path = write_provenance(run_directory, provenance)
    else:
        provenance_path = _reuse_resume_provenance(
            run_directory / "provenance.json",
            provenance,
        )
        if args.phase == "formal_evaluate":
            evaluation_record = dict(provenance)
            evaluation_record["training_checkpoint_sha256"] = _file_sha256(
                Path(args.resume_checkpoint)
            )
            evaluation_provenance_path = _write_evaluation_provenance(
                run_directory,
                evaluation_record,
            )
    try:
        api_key = (
            None
            if args.phase == "formal_evaluate"
            else resolve_api_key(args.method, environment)
        )
    except DeepSeekCallError as failure:
        report_path = _write_preflight_pause(run_directory, failure)
        raise ExperimentPaused(failure, report_path) from failure
    return RunContext(
        args=args,
        manifest=manifest,
        method_config=method_config,
        run_directory=run_directory,
        api_key=api_key,
        provenance_path=provenance_path,
        evaluation_provenance_path=evaluation_provenance_path,
    )


def _validate_preflight_response(value: Any) -> dict[str, str]:
    if type(value) is not dict or set(value) != {"status", "model"}:
        raise ValueError("preflight response must contain exact status/model fields")
    if value != {"status": "ready", "model": DEEPSEEK_MODEL}:
        raise ValueError("preflight response values do not match the contract")
    return dict(value)


def run_preflight(context: RunContext, *, session: Any = None) -> dict[str, str]:
    if context.args.preflight_only is not True or context.api_key is None:
        raise ValueError("run_preflight requires an approved LLM preflight context")
    if session is None:
        import requests

        session = requests.Session()
    telemetry = AppendOnlyTelemetry(
        context.run_directory / "deepseek_calls.jsonl",
        known_secrets=(context.api_key,),
    )
    provenance = json.loads(context.provenance_path.read_text(encoding="utf-8"))
    settings = deepseek_client_settings_from_provenance(provenance)
    client = StrictDeepSeekClient(
        api_key=context.api_key,
        model_name=settings["model"],
        base_url=settings["base_url"],
        timeout_seconds=settings["timeout_seconds"],
        session=session,
        telemetry=telemetry,
    )
    return client.generate_json(
        "preflight",
        (
            "The user message is a protocol-safe placeholder, not an answer "
            "template. Do not echo or transform the user message. Return exactly "
            "one JSON object and no other text: "
            f'{{"status":"ready","model":"{DEEPSEEK_MODEL}"}}'
        ),
        {"round_index": 0, "clients": []},
        _validate_preflight_response,
    )


def _write_preflight_pause(run_directory: Path, failure: Exception) -> Path:
    category = getattr(failure, "category", "runtime")
    role = getattr(failure, "role", "preflight")
    record = {
        "status": "paused",
        "category": category if type(category) is str else "runtime",
        "role": role if type(role) is str else "preflight",
        "exception_type": type(failure).__name__,
    }
    encoded = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
    for index in range(1000):
        filename = "PAUSED.json" if index == 0 else f"PAUSED.{index:03d}.json"
        try:
            return _publish_bytes_no_replace(run_directory / filename, encoded)
        except FileExistsError:
            continue
    raise RuntimeError("pause-report incident namespace is exhausted")


def execute(
    args: argparse.Namespace,
    *,
    training_executor: Callable[[RunContext], Any] | None = None,
    project_root: Path = PROJECT_ROOT,
    results_root: Path | None = None,
    environment: Mapping[str, str] = os.environ,
    preflight_session: Any = None,
) -> tuple[RunContext, Any]:
    context = prepare_run(
        args,
        project_root=project_root,
        results_root=results_root,
        environment=environment,
    )
    if args.preflight_only:
        try:
            return context, run_preflight(context, session=preflight_session)
        except Exception as failure:
            sanitized = (
                failure
                if type(failure) is DeepSeekCallError
                else ExperimentRuntimeError(
                    type(failure).__name__, "preflight stopped before training"
                )
            )
            report_path = _write_preflight_pause(context.run_directory, sanitized)
            raise ExperimentPaused(sanitized, report_path) from failure
    if training_executor is None:
        from src.federated_learning.pcv.runtime import execute_strict_training

        training_executor = execute_strict_training
    try:
        return context, training_executor(context)
    except ExperimentPaused:
        raise
    except Exception as failure:
        sanitized = ExperimentRuntimeError(
            type(failure).__name__,
            "strict runtime stopped before completing the requested run",
        )
        report_path = _write_preflight_pause(context.run_directory, sanitized)
        raise ExperimentPaused(sanitized, report_path) from failure


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context, result = execute(args)
    except ExperimentPaused as paused:
        print(f"PAUSED: {paused.report_path}", file=sys.stderr)
        return 2
    print(f"COMPLETE: {context.run_directory}")
    if args.preflight_only:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_METHODS",
    "LLM_METHODS",
    "RunContext",
    "build_parser",
    "create_run_directory",
    "execute",
    "effective_config_sha256",
    "frozen_execution_config_sha256",
    "load_method_config",
    "main",
    "prepare_run",
    "resolve_api_key",
    "resolve_run_directory",
    "run_preflight",
    "validate_invocation",
    "validate_formal_freeze",
    "write_provenance",
]
