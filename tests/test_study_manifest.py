from pathlib import Path

import pytest

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


def test_unfrozen_manifest_cannot_name_paper_eligible_batch():
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    assert manifest.formal_frozen is False
    assert manifest.paper_eligible_freeze_ids == ()
