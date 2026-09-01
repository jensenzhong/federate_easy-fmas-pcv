from pathlib import Path

import pytest
import yaml

from src.study_manifest import load_study_manifest


def test_manifest_exposes_exact_formal_method_order():
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    assert manifest.formal_methods == (
        "FEDAVG_STRICT",
        "FEDYOGI_STRICT",
        "DPCV_FEDYOGI",
        "SA_PCV_FEDYOGI",
        "FMAS_PCV_FEDYOGI",
    )


def test_seed_42_is_development_only():
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    assert manifest.development_seed == 42
    assert 42 not in manifest.formal_seeds


def test_manifest_freeze_state_is_internally_consistent():
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    if manifest.formal_frozen:
        assert manifest.stage == "formal_ready"
        assert manifest.paper_eligible_freeze_ids
    else:
        assert manifest.stage == "development"
        assert manifest.paper_eligible_freeze_ids == ()


def test_manifest_rejects_inconsistent_freeze_state(tmp_path):
    source = yaml.safe_load(Path("study_manifest.yaml").read_text(encoding="utf-8"))
    source["stage"] = "formal_ready"
    source["formal_frozen"] = False
    source["paper_eligible_freeze_ids"] = []
    path = tmp_path / "study_manifest.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unfrozen study"):
        load_study_manifest(path)
