import json
from pathlib import Path
import shutil

import pytest
import yaml

import scripts.freeze_fmas_protocol as freeze_module
from scripts.freeze_fmas_protocol import freeze_protocol, recover_freeze_transaction
from src.federated_learning.pcv.agents import MAX_JSON_PARSE_REGENERATION_RETRIES
from src.formal_protocol import (
    DEVELOPMENT_GATE,
    FAILURE_PROTOCOL,
    build_freeze_payload,
    freeze_id_from_payload,
    frozen_execution_config_sha256,
    validate_freeze_record,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _protocol_tree(tmp_path: Path) -> Path:
    paths = [
        "configs/config.yaml",
        "configs/development_seed42.yaml",
        "configs/formal_frozen.yaml",
        "configs/methods",
        "configs/prompts",
        "results/manifests/strict_partition_v1.csv",
        "Data/strict_partition_v1/metadata.json",
        DEVELOPMENT_GATE,
        "audits/baseline_fairness_audit.json",
    ]
    for relative in paths:
        source = SOURCE_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    manifest = yaml.safe_load((SOURCE_ROOT / "study_manifest.yaml").read_text(encoding="utf-8"))
    manifest["stage"] = "development"
    manifest["formal_frozen"] = False
    manifest["paper_eligible_freeze_ids"] = []
    (tmp_path / "study_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "configs/formal_frozen.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "formal_frozen": False,
                "freeze_id": None,
                "formal_seeds": manifest["formal_seeds"],
                "partition_manifest": "results/manifests/strict_partition_v1.csv",
                "partition_sha256": None,
                "sealed_partition_metadata_sha256": None,
                "config_sha256": None,
                "prompt_hashes": {},
                "git_commit": None,
                "deepseek": {
                    "model": "deepseek-v4-flash",
                    "base_url": "https://api.deepseek.com",
                    "temperature": 0.8,
                    "timeout_seconds": 120,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def clean_protocol(tmp_path, monkeypatch):
    root = _protocol_tree(tmp_path)
    monkeypatch.setattr(freeze_module, "_git_state", lambda project_root: (COMMIT, False))
    monkeypatch.setattr(freeze_module, "_validate_git_lineage", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        freeze_module, "_validate_development_provenance", lambda *args, **kwargs: None
    )
    return root


def test_freeze_is_deterministic_and_preserves_execution_hash(clean_protocol):
    before = frozen_execution_config_sha256(clean_protocol)
    freeze_id, record = freeze_protocol(
        development_gate_path=clean_protocol / DEVELOPMENT_GATE,
        project_root=clean_protocol,
    )
    after = frozen_execution_config_sha256(clean_protocol)
    assert before == after == record["payload"]["execution_config_sha256"]
    assert freeze_id == freeze_id_from_payload(record["payload"])
    manifest = yaml.safe_load((clean_protocol / "study_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["stage"] == "formal_ready"
    assert manifest["formal_frozen"] is True
    assert manifest["paper_eligible_freeze_ids"] == [freeze_id]
    assert (clean_protocol / "results/manifests" / f"{freeze_id}.json").is_file()


def test_freeze_hash_covers_the_approved_json_parse_regeneration_policy(clean_protocol):
    assert FAILURE_PROTOCOL["automatic_transport_retry"] is False
    assert FAILURE_PROTOCOL["max_json_parse_regeneration_retries"] == (
        MAX_JSON_PARSE_REGENERATION_RETRIES
    )
    assert FAILURE_PROTOCOL["json_parse_regeneration_requires_identical_request"] is True
    assert FAILURE_PROTOCOL["non_parse_failures_retry"] is False
    payload = build_freeze_payload(clean_protocol, source_commit=COMMIT)
    assert payload["failure_policy"] == FAILURE_PROTOCOL
    changed = dict(payload)
    changed["failure_policy"] = {
        **payload["failure_policy"],
        "max_json_parse_regeneration_retries": 0,
    }
    assert freeze_id_from_payload(changed) != freeze_id_from_payload(payload)


def test_runtime_validation_does_not_require_ignored_development_outputs(clean_protocol):
    freeze_id, _ = freeze_protocol(
        development_gate_path=clean_protocol / DEVELOPMENT_GATE,
        project_root=clean_protocol,
    )
    shutil.rmtree(clean_protocol / "results/development")
    (clean_protocol / "audits/baseline_fairness_audit.json").unlink()
    frozen = yaml.safe_load(
        (clean_protocol / "configs/formal_frozen.yaml").read_text(encoding="utf-8")
    )
    payload = validate_freeze_record(
        clean_protocol,
        freeze_id=freeze_id,
        frozen_document=frozen,
    )
    assert payload["git_commit"] == COMMIT


def test_dirty_tree_refuses_without_writes(tmp_path, monkeypatch):
    root = _protocol_tree(tmp_path)
    before_manifest = (root / "study_manifest.yaml").read_bytes()
    before_frozen = (root / "configs/formal_frozen.yaml").read_bytes()
    monkeypatch.setattr(freeze_module, "_git_state", lambda project_root: (COMMIT, True))
    with pytest.raises(ValueError, match="clean"):
        freeze_protocol(
            development_gate_path=root / DEVELOPMENT_GATE,
            project_root=root,
        )
    assert (root / "study_manifest.yaml").read_bytes() == before_manifest
    assert (root / "configs/formal_frozen.yaml").read_bytes() == before_frozen


def test_existing_transaction_lock_refuses_without_writes(tmp_path, monkeypatch):
    root = _protocol_tree(tmp_path)
    lock = root / "results/manifests/.freeze.lock"
    lock.write_text("active", encoding="utf-8")
    before = (root / "study_manifest.yaml").read_bytes()
    with pytest.raises(RuntimeError, match="transaction"):
        freeze_protocol(
            development_gate_path=root / DEVELOPMENT_GATE,
            project_root=root,
        )
    assert (root / "study_manifest.yaml").read_bytes() == before
    assert lock.read_text(encoding="utf-8") == "active"


def test_interrupted_transaction_can_be_explicitly_rolled_back(clean_protocol):
    manifest_path = clean_protocol / "study_manifest.yaml"
    frozen_path = clean_protocol / "configs/formal_frozen.yaml"
    original_manifest = manifest_path.read_bytes()
    original_frozen = frozen_path.read_bytes()
    lock_path = clean_protocol / "results/manifests/.freeze.lock"
    journal = freeze_module._transaction_journal(
        freeze_id="f" * 16,
        source_commit=COMMIT,
        manifest_path=manifest_path,
        frozen_path=frozen_path,
        original_manifest=original_manifest,
        original_frozen=original_frozen,
        final_manifest=b"changed-manifest",
        final_frozen=b"changed-frozen",
        freeze_manifest=b"{}\n",
    )
    freeze_module._acquire_freeze_lock(lock_path, journal)
    manifest_path.write_bytes(b"changed-manifest")
    frozen_path.write_bytes(b"changed-frozen")

    status, freeze_id = recover_freeze_transaction(project_root=clean_protocol)

    assert (status, freeze_id) == ("rolled_back", "f" * 16)
    assert manifest_path.read_bytes() == original_manifest
    assert frozen_path.read_bytes() == original_frozen
    assert not lock_path.exists()


def test_failed_manifest_publication_never_deletes_a_foreign_manifest(
    clean_protocol, monkeypatch
):
    foreign_paths = []

    def publish_foreign_then_fail(path, content):
        path.write_bytes(b"foreign")
        foreign_paths.append(path)
        raise FileExistsError("concurrent publisher")

    monkeypatch.setattr(freeze_module, "_publish_no_replace", publish_foreign_then_fail)
    before_manifest = (clean_protocol / "study_manifest.yaml").read_bytes()
    before_frozen = (clean_protocol / "configs/formal_frozen.yaml").read_bytes()
    with pytest.raises(FileExistsError, match="concurrent"):
        freeze_protocol(
            development_gate_path=clean_protocol / DEVELOPMENT_GATE,
            project_root=clean_protocol,
        )
    assert foreign_paths and foreign_paths[0].read_bytes() == b"foreign"
    assert (clean_protocol / "study_manifest.yaml").read_bytes() == before_manifest
    assert (clean_protocol / "configs/formal_frozen.yaml").read_bytes() == before_frozen


def test_explicit_supersede_preserves_old_record_and_activates_only_new_freeze(
    clean_protocol, monkeypatch
):
    old_id, _ = freeze_protocol(
        development_gate_path=clean_protocol / DEVELOPMENT_GATE,
        project_root=clean_protocol,
    )
    old_path = clean_protocol / "results/manifests" / f"{old_id}.json"
    old_bytes = old_path.read_bytes()
    monkeypatch.setattr(freeze_module, "_git_state", lambda project_root: ("b" * 40, False))

    new_id, record = freeze_protocol(
        development_gate_path=clean_protocol / DEVELOPMENT_GATE,
        project_root=clean_protocol,
        supersede_freeze_id=old_id,
    )

    assert new_id != old_id
    assert record["payload"]["supersedes_freeze_id"] == old_id
    assert old_path.read_bytes() == old_bytes
    manifest = yaml.safe_load((clean_protocol / "study_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["paper_eligible_freeze_ids"] == [new_id]


def test_failed_development_gate_refuses_without_freeze(clean_protocol):
    gate_path = clean_protocol / DEVELOPMENT_GATE
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["gate_passed"] = False
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        freeze_protocol(
            development_gate_path=gate_path,
            project_root=clean_protocol,
        )
    manifest = yaml.safe_load((clean_protocol / "study_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["formal_frozen"] is False


def test_development_gate_must_bind_the_current_development_config(clean_protocol):
    gate_path = clean_protocol / DEVELOPMENT_GATE
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["config_sha256"] = "0" * 64
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration hash"):
        freeze_protocol(
            development_gate_path=gate_path,
            project_root=clean_protocol,
        )


def test_gate_and_fairness_audit_must_name_the_same_development_commit(
    clean_protocol,
):
    audit_path = clean_protocol / "audits/baseline_fairness_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["scope"]["git_commit"] = "c" * 40
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="Git commits differ"):
        freeze_protocol(
            development_gate_path=clean_protocol / DEVELOPMENT_GATE,
            project_root=clean_protocol,
        )


def test_development_seed_in_formal_seeds_is_rejected(clean_protocol):
    path = clean_protocol / "study_manifest.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest["formal_seeds"][0] = 42
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="development seed"):
        freeze_protocol(
            development_gate_path=clean_protocol / DEVELOPMENT_GATE,
            project_root=clean_protocol,
        )


def test_existing_freeze_id_is_never_overwritten(clean_protocol):
    payload = build_freeze_payload(clean_protocol, source_commit=COMMIT)
    freeze_id = freeze_id_from_payload(payload)
    path = clean_protocol / "results/manifests" / f"{freeze_id}.json"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_protocol(
            development_gate_path=clean_protocol / DEVELOPMENT_GATE,
            project_root=clean_protocol,
        )
    assert path.read_bytes() == b"original"
