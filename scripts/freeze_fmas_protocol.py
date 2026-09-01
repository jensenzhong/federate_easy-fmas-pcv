"""Freeze the approved FMAS-PCV protocol without running any experiment."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formal_protocol import (
    BASELINE_FAIRNESS_AUDIT,
    DEVELOPMENT_GATE,
    METHOD_CONFIG_PATHS,
    METHOD_PROMPT_ROLES,
    PARTITION_MANIFEST,
    SEALED_PARTITION_METADATA,
    build_freeze_payload,
    canonical_json_bytes,
    file_sha256,
    formal_frozen_document,
    formal_protocol_semantics,
    freeze_id_from_payload,
)
from src.federated_learning.pcv.provider_config import deepseek_provenance
from src.study_manifest import FORMAL_METHOD_ORDER
from src.study_manifest import load_study_manifest


_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_FREEZE_ID = re.compile(r"[0-9a-f]{16}\Z")
_LOCK_NAME = ".freeze.lock"
_DEVELOPMENT_RUNS = (
    ("FEDAVG_STRICT", 0, "fedavg-strict-seed42-rep0"),
    ("FEDYOGI_STRICT", 0, "fedyogi-strict-seed42-rep0"),
    ("DPCV_FEDYOGI", 0, "dpcv-fedyogi-seed42-rep0"),
    ("SA_PCV_FEDYOGI", 1, "sa-pcv-fedyogi-seed42-rep1"),
    ("SA_PCV_FEDYOGI", 2, "sa-pcv-fedyogi-seed42-rep2"),
    ("SA_PCV_FEDYOGI", 3, "sa-pcv-fedyogi-seed42-rep3"),
    ("FMAS_PCV_FEDYOGI", 1, "fmas-pcv-fedyogi-seed42-rep1"),
    ("FMAS_PCV_FEDYOGI", 2, "fmas-pcv-fedyogi-seed42-rep2"),
    ("FMAS_PCV_FEDYOGI", 3, "fmas-pcv-fedyogi-seed42-rep3"),
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _transaction_journal(
    *,
    freeze_id: str,
    source_commit: str,
    manifest_path: Path,
    frozen_path: Path,
    original_manifest: bytes,
    original_frozen: bytes,
    final_manifest: bytes,
    final_frozen: bytes,
    freeze_manifest: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pid": os.getpid(),
        "freeze_id": freeze_id,
        "source_commit": source_commit,
        "manifest_path": manifest_path.name,
        "frozen_path": frozen_path.relative_to(manifest_path.parent).as_posix(),
        "original_manifest_base64": base64.b64encode(original_manifest).decode("ascii"),
        "original_frozen_base64": base64.b64encode(original_frozen).decode("ascii"),
        "original_manifest_sha256": _sha256_bytes(original_manifest),
        "original_frozen_sha256": _sha256_bytes(original_frozen),
        "final_manifest_sha256": _sha256_bytes(final_manifest),
        "final_frozen_sha256": _sha256_bytes(final_frozen),
        "freeze_manifest_sha256": _sha256_bytes(freeze_manifest),
    }


def _acquire_freeze_lock(path: Path, journal: Mapping[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("another freeze transaction is active or requires recovery") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(dict(journal)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_transaction_journal(path: Path) -> dict[str, Any]:
    journal = _read_json(path, label="freeze transaction journal")
    expected = {
        "schema_version", "pid", "freeze_id", "source_commit", "manifest_path",
        "frozen_path", "original_manifest_base64", "original_frozen_base64",
        "original_manifest_sha256", "original_frozen_sha256",
        "final_manifest_sha256", "final_frozen_sha256", "freeze_manifest_sha256",
    }
    if (
        set(journal) != expected
        or journal.get("schema_version") != 1
        or not _FREEZE_ID.fullmatch(str(journal.get("freeze_id", "")))
        or not _COMMIT.fullmatch(str(journal.get("source_commit", "")))
        or journal.get("manifest_path") != "study_manifest.yaml"
        or journal.get("frozen_path") != "configs/formal_frozen.yaml"
    ):
        raise RuntimeError("freeze transaction journal is invalid")
    return journal


def _decode_original(journal: Mapping[str, Any], field: str, sha_field: str) -> bytes:
    try:
        content = base64.b64decode(journal[field], validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("freeze transaction backup is invalid") from error
    if _sha256_bytes(content) != journal.get(sha_field):
        raise RuntimeError("freeze transaction backup hash mismatch")
    return content


def recover_freeze_transaction(
    *, project_root: Path = PROJECT_ROOT
) -> tuple[str, str]:
    """Explicitly finish or roll back a preserved interrupted freeze transaction."""

    root = Path(project_root).resolve(strict=True)
    lock_path = root / "results/manifests" / _LOCK_NAME
    journal = _read_transaction_journal(lock_path)
    freeze_id = journal["freeze_id"]
    manifest_path = root / journal["manifest_path"]
    frozen_path = root / journal["frozen_path"]
    freeze_manifest_path = root / "results/manifests" / f"{freeze_id}.json"
    if os.path.lexists(freeze_manifest_path):
        if (
            file_sha256(freeze_manifest_path) != journal["freeze_manifest_sha256"]
            or file_sha256(manifest_path) != journal["final_manifest_sha256"]
            or file_sha256(frozen_path) != journal["final_frozen_sha256"]
        ):
            raise RuntimeError("published freeze transaction is inconsistent")
        lock_path.unlink()
        return "committed", freeze_id

    original_manifest = _decode_original(
        journal, "original_manifest_base64", "original_manifest_sha256"
    )
    original_frozen = _decode_original(
        journal, "original_frozen_base64", "original_frozen_sha256"
    )
    _replace_bytes(manifest_path, original_manifest)
    _replace_bytes(frozen_path, original_frozen)
    if (
        file_sha256(manifest_path) != journal["original_manifest_sha256"]
        or file_sha256(frozen_path) != journal["original_frozen_sha256"]
    ):
        raise RuntimeError("freeze transaction rollback verification failed")
    lock_path.unlink()
    return "rolled_back", freeze_id


def _git_state(project_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    commit = revision.stdout.strip()
    if revision.returncode != 0 or status.returncode != 0 or not _COMMIT.fullmatch(commit):
        raise ValueError("freeze requires a valid Git HEAD")
    return commit, bool(status.stdout.strip())


def _validate_git_lineage(project_root: Path, *, gate_commit: str, source_commit: str) -> None:
    if not _COMMIT.fullmatch(gate_commit):
        raise ValueError("development gate Git commit is invalid")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", gate_commit, source_commit],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if ancestor.returncode != 0:
        raise ValueError("development gate commit is not an ancestor of the freeze source")
    changed = subprocess.run(
        ["git", "diff", "--name-only", gate_commit, source_commit, "--"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.splitlines()
    allowed_exact = {
        ".gitignore",
        "PROJECT_STATUS.md",
        "study_manifest.yaml",
        "configs/formal_frozen.yaml",
        "experiments/run_strict_federated.py",
        "scripts/freeze_fmas_protocol.py",
        "scripts/run_fmas_formal.py",
        "scripts/statistical_analysis.py",
        "src/federated_learning/pcv/agents.py",
        "src/formal_protocol.py",
        "src/study_manifest.py",
    }
    forbidden = [
        path
        for path in changed
        if path not in allowed_exact
        and not path.startswith("audits/")
        and not path.startswith("docs/")
        and not path.startswith("tests/")
        and not re.fullmatch(r"results/manifests/[0-9a-f]{16}\.json", path)
    ]
    if forbidden:
        raise ValueError(
            f"training-relevant files changed after the development gate: {forbidden}"
        )


def _effective_config_sha256(base_path: Path, method_path: Path) -> str:
    digest = hashlib.sha256()
    for label, path in ((b"base-config", base_path), (b"method-config", method_path)):
        content = path.read_bytes()
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_development_provenance(
    project_root: Path,
    *,
    gate_commit: str,
) -> None:
    base_path = project_root / "configs/config.yaml"
    base_sha = file_sha256(base_path)
    partition_sha = file_sha256(project_root / PARTITION_MANIFEST)
    sealed_sha = file_sha256(project_root / SEALED_PARTITION_METADATA)
    for method, rep, run_id in _DEVELOPMENT_RUNS:
        provenance = _read_json(
            project_root / "results/development/seed42" / run_id / "provenance.json",
            label=f"development provenance {run_id}",
        )
        method_path = project_root / METHOD_CONFIG_PATHS[method]
        expected_prompts = {
            role: file_sha256(project_root / "configs/prompts" / f"{role}.md")
            for role in METHOD_PROMPT_ROLES[method]
        }
        if (
            provenance.get("schema_version") != 1
            or provenance.get("method") != method
            or provenance.get("phase") != "development"
            or provenance.get("training_seed") != 42
            or provenance.get("llm_rep") != rep
            or provenance.get("run_id") != run_id
            or provenance.get("freeze_id") is not None
            or provenance.get("git_commit") != gate_commit
            or provenance.get("git_dirty") is not False
            or provenance.get("partition_sha256") != partition_sha
            or provenance.get("sealed_partition_metadata_sha256") != sealed_sha
            or provenance.get("base_config_sha256") != base_sha
            or provenance.get("method_config_sha256") != file_sha256(method_path)
            or provenance.get("effective_config_sha256")
            != _effective_config_sha256(base_path, method_path)
            or provenance.get("prompt_hashes") != expected_prompts
            or provenance.get("deepseek")
            != deepseek_provenance(enabled=bool(expected_prompts))
            or provenance.get("locked_test_unlocked") is not False
        ):
            raise ValueError(f"development provenance drift detected for {run_id}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _validate_gate(
    gate: Mapping[str, Any], *, project_root: Path, development_gate_path: Path
) -> None:
    manifest = load_study_manifest(project_root / "study_manifest.yaml")
    if (
        gate.get("schema_version") != 1
        or gate.get("status") != "complete"
        or gate.get("phase") != "development"
        or gate.get("training_seed") != manifest.development_seed
        or gate.get("gate_passed") is not True
        or type(gate.get("passing_fmas_repetitions")) is not int
        or gate["passing_fmas_repetitions"]
        < gate.get("required_passing_fmas_repetitions", 10**9)
    ):
        raise ValueError("development gate did not pass the approved criteria")
    evidence = gate.get("evidence")
    if type(evidence) is not dict or evidence != {
        "partition": "controller_validation",
        "locked_test_used": False,
    }:
        raise ValueError("development gate contains forbidden or ambiguous evidence")
    if gate.get("partition_sha256") != file_sha256(project_root / PARTITION_MANIFEST):
        raise ValueError("development gate partition hash mismatch")
    if gate.get("config_sha256") != file_sha256(
        project_root / "configs/development_seed42.yaml"
    ):
        raise ValueError("development gate configuration hash mismatch")
    if development_gate_path != project_root / DEVELOPMENT_GATE:
        raise ValueError("only the approved seed-42 development gate may be frozen")


def _validate_baseline_audit(
    audit: Mapping[str, Any], *, project_root: Path, gate_sha256: str
) -> None:
    scope = audit.get("scope")
    contamination = audit.get("locked_test_contamination")
    comparability = audit.get("comparability_gate")
    if (
        audit.get("schema_version") != 1
        or audit.get("overall_status") != "PASS"
        or audit.get("recommended_freeze_ready") is not True
        or audit.get("rerun_required") is not False
        or type(scope) is not dict
        or scope.get("development_gate_sha256") != gate_sha256
        or scope.get("git_commit") is None
        or scope.get("partition_sha256") != file_sha256(project_root / PARTITION_MANIFEST)
        or scope.get("sealed_partition_metadata_sha256")
        != file_sha256(project_root / SEALED_PARTITION_METADATA)
        or scope.get("base_config_sha256")
        != file_sha256(project_root / "configs/config.yaml")
        or type(contamination) is not dict
        or contamination.get("status") != "PASS"
        or contamination.get("found") is not False
        or contamination.get("locked_test_unlocked") is not False
        or type(comparability) is not dict
        or comparability.get("status") != "PASS"
    ):
        raise ValueError("baseline fairness audit is not freeze-ready")


def _yaml_bytes(value: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).encode("utf-8")


def _replace_bytes(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_no_replace(path: Path, content: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing freeze manifest: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _validate_supersede_request(
    *,
    project_root: Path,
    manifest: Any,
    supersede_freeze_id: str | None,
) -> None:
    if manifest.stage == "development":
        if (
            manifest.formal_frozen is not False
            or manifest.paper_eligible_freeze_ids
            or supersede_freeze_id is not None
        ):
            raise ValueError("study manifest is not in the unfrozen development state")
        return
    if (
        manifest.stage != "formal_ready"
        or manifest.formal_frozen is not True
        or len(manifest.paper_eligible_freeze_ids) != 1
        or supersede_freeze_id != manifest.paper_eligible_freeze_ids[0]
        or not _FREEZE_ID.fullmatch(str(supersede_freeze_id or ""))
    ):
        raise ValueError("frozen studies require the exact active freeze id to supersede")
    old_record_path = project_root / "results/manifests" / f"{supersede_freeze_id}.json"
    old_record = _read_json(old_record_path, label="superseded freeze manifest")
    old_payload = old_record.get("payload")
    old_frozen = yaml.safe_load(
        (project_root / "configs/formal_frozen.yaml").read_text(encoding="utf-8")
    )
    if (
        old_record.get("freeze_id") != supersede_freeze_id
        or type(old_payload) is not dict
        or freeze_id_from_payload(old_payload) != supersede_freeze_id
        or type(old_frozen) is not dict
        or old_frozen.get("freeze_id") != supersede_freeze_id
    ):
        raise ValueError("superseded freeze evidence is incomplete or inconsistent")


def _freeze_protocol_locked(
    *,
    development_gate_path: Path,
    project_root: Path = PROJECT_ROOT,
    supersede_freeze_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    root = Path(project_root).resolve(strict=True)
    if os.path.lexists(root / "results/manifests" / _LOCK_NAME):
        raise RuntimeError("another freeze transaction is active or requires recovery")
    gate_path = Path(development_gate_path)
    if not gate_path.is_absolute():
        gate_path = root / gate_path
    gate_path = gate_path.resolve(strict=True)

    commit, dirty = _git_state(root)
    if dirty:
        raise ValueError("freeze requires a clean Git worktree")

    manifest_path = root / "study_manifest.yaml"
    frozen_path = root / "configs/formal_frozen.yaml"
    manifest = load_study_manifest(manifest_path)
    raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    _validate_supersede_request(
        project_root=root,
        manifest=manifest,
        supersede_freeze_id=supersede_freeze_id,
    )
    if (
        len(manifest.formal_seeds) != 5
        or len(set(manifest.formal_seeds)) != 5
        or manifest.development_seed in manifest.formal_seeds
    ):
        raise ValueError("formal seeds must be five unique non-development seeds")

    gate = _read_json(gate_path, label="development gate")
    _validate_gate(gate, project_root=root, development_gate_path=gate_path)
    gate_sha256 = file_sha256(gate_path)
    audit_path = root / BASELINE_FAIRNESS_AUDIT
    audit = _read_json(audit_path, label="baseline fairness audit")
    _validate_baseline_audit(audit, project_root=root, gate_sha256=gate_sha256)
    gate_commit = gate.get("git_commit")
    if audit["scope"].get("git_commit") != gate_commit:
        raise ValueError("development gate and fairness audit Git commits differ")
    _validate_git_lineage(root, gate_commit=gate_commit, source_commit=commit)
    _validate_development_provenance(root, gate_commit=gate_commit)

    payload = build_freeze_payload(
        root,
        source_commit=commit,
        development_gate_path=gate_path,
        baseline_audit_path=audit_path,
        supersedes_freeze_id=supersede_freeze_id,
    )
    freeze_id = freeze_id_from_payload(payload)
    freeze_manifest_path = root / "results/manifests" / f"{freeze_id}.json"
    if os.path.lexists(freeze_manifest_path):
        raise FileExistsError(f"freeze id already exists: {freeze_id}")

    final_manifest = dict(raw_manifest)
    final_manifest["stage"] = "formal_ready"
    final_manifest["formal_frozen"] = True
    final_manifest["paper_eligible_freeze_ids"] = [freeze_id]
    final_frozen = formal_frozen_document(payload, freeze_id)
    record = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "payload": payload,
        "formal_protocol": formal_protocol_semantics(root),
    }
    original_manifest = manifest_path.read_bytes()
    original_frozen = frozen_path.read_bytes()
    final_manifest_bytes = _yaml_bytes(final_manifest)
    final_frozen_bytes = _yaml_bytes(final_frozen)
    freeze_manifest_bytes = (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )

    if _git_state(root) != (commit, False):
        raise ValueError("freeze inputs changed during validation")
    lock_path = root / "results/manifests" / _LOCK_NAME
    journal = _transaction_journal(
        freeze_id=freeze_id,
        source_commit=commit,
        manifest_path=manifest_path,
        frozen_path=frozen_path,
        original_manifest=original_manifest,
        original_frozen=original_frozen,
        final_manifest=final_manifest_bytes,
        final_frozen=final_frozen_bytes,
        freeze_manifest=freeze_manifest_bytes,
    )
    _acquire_freeze_lock(lock_path, journal)
    try:
        _replace_bytes(manifest_path, final_manifest_bytes)
        _replace_bytes(frozen_path, final_frozen_bytes)
        rebuilt_payload = build_freeze_payload(
            root,
            source_commit=commit,
            development_gate_path=gate_path,
            baseline_audit_path=audit_path,
            supersedes_freeze_id=supersede_freeze_id,
        )
        if rebuilt_payload != payload:
            raise ValueError("freeze state changed protected execution semantics")
        _publish_no_replace(freeze_manifest_path, freeze_manifest_bytes)
    except BaseException as failure:
        try:
            _replace_bytes(manifest_path, original_manifest)
            _replace_bytes(frozen_path, original_frozen)
            if (
                manifest_path.read_bytes() != original_manifest
                or frozen_path.read_bytes() != original_frozen
            ):
                raise RuntimeError("rollback verification failed")
            lock_path.unlink()
        except BaseException as recovery_failure:
            raise RuntimeError(
                f"freeze transaction requires explicit recovery: {lock_path}"
            ) from recovery_failure
        raise
    lock_path.unlink()
    return freeze_id, record


def freeze_protocol(
    *,
    development_gate_path: Path,
    project_root: Path = PROJECT_ROOT,
    supersede_freeze_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    root = Path(project_root).resolve(strict=True)
    return _freeze_protocol_locked(
        development_gate_path=development_gate_path,
        project_root=root,
        supersede_freeze_id=supersede_freeze_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the approved FMAS-PCV protocol")
    parser.add_argument(
        "--development-gate",
        type=Path,
    )
    parser.add_argument("--supersede-freeze-id")
    parser.add_argument("--recover-transaction", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.recover_transaction:
            if args.development_gate is not None or args.supersede_freeze_id is not None:
                raise ValueError("transaction recovery cannot be combined with freeze arguments")
            status, freeze_id = recover_freeze_transaction()
            print(f"RECOVERY_STATUS={status}")
            print(f"FREEZE_ID={freeze_id}")
            return 0
        if args.development_gate is None:
            raise ValueError("--development-gate is required unless recovering")
        freeze_id, record = freeze_protocol(
            development_gate_path=args.development_gate,
            supersede_freeze_id=args.supersede_freeze_id,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"FREEZE_ID={freeze_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "freeze_protocol",
    "main",
    "recover_freeze_transaction",
]
