import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import load_strict_partition_frames
from src.federated_learning.pcv.protocol import (
    PartitionRatios,
    build_partition_manifest,
)


def _frame():
    rows = []
    for client in ("Client 1", "Client 2", "Client 3"):
        for index in range(40):
            rows.append(
                {
                    "Client": client,
                    "ContAmnt": float(index + 1),
                    "source_index": len(rows),
                }
            )
    return pd.DataFrame(rows)


def test_partition_is_client_local_disjoint_and_complete():
    frame = _frame()
    manifest = build_partition_manifest(
        frame,
        client_column="Client",
        target_column="ContAmnt",
        source_index_column="source_index",
        dataset_sha256="0" * 64,
        split_seed=20260730,
        ratios=PartitionRatios(0.70, 0.15, 0.15),
        quantile_bins=5,
    )
    assert len(manifest) == len(frame)
    assert manifest["row_id"].is_unique
    assert set(manifest["partition"]) == {
        "train",
        "controller_validation",
        "locked_test",
    }
    for _, rows in manifest.groupby("client_id"):
        assert abs((rows["partition"] == "train").mean() - 0.70) <= 0.05


def test_partition_does_not_change_with_training_seed():
    frame = _frame()
    kwargs = dict(
        client_column="Client",
        target_column="ContAmnt",
        source_index_column="source_index",
        dataset_sha256="1" * 64,
        split_seed=20260730,
        ratios=PartitionRatios(0.70, 0.15, 0.15),
        quantile_bins=5,
    )
    left = build_partition_manifest(frame, **kwargs)
    right = build_partition_manifest(frame, **kwargs)
    pd.testing.assert_frame_equal(left, right)


def test_strict_loader_fits_feature_stats_on_train_partition_only(tmp_path):
    frame = pd.DataFrame(
        {
            "Feature": [1.0, 3.0, 1000.0, 5.0, 2.0, 4.0, 6.0, 8.0],
            "ContAmnt": [10.0, 30.0, 100.0, 50.0, 20.0, 40.0, 60.0, 80.0],
            "Client": [
                "Client 1",
                "Client 1",
                "Client 1",
                "Client 1",
                "Client 2",
                "Client 2",
                "Client 2",
                "Client 2",
            ],
        }
    )
    raw_path = tmp_path / "canonical.csv"
    frame.to_csv(raw_path, index=False)
    dataset_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest = pd.DataFrame(
        {
            "source_index": range(len(frame)),
            "client_id": frame["Client"],
            "partition": [
                "train",
                "train",
                "controller_validation",
                "locked_test",
                "train",
                "train",
                "controller_validation",
                "locked_test",
            ],
            "dataset_sha256": dataset_sha256,
        }
    )
    manifest_path = tmp_path / "partition.csv"
    manifest.to_csv(manifest_path, index=False)
    config = {
        "scene_c": {
            "data": {
                "raw_csv": str(raw_path),
                "rename_map": {},
                "feature_columns": ["Feature"],
                "target_column": "ContAmnt",
                "client_column": "Client",
            }
        },
        "preprocessing": {
            "scaler": "StandardScaler",
            "target_transform": "power_0.25",
            "random_seed": 42,
        },
    }

    result = load_strict_partition_frames(config, str(manifest_path))

    train_values = np.array([1.0, 3.0, 2.0, 4.0])
    np.testing.assert_allclose(
        result.preprocessor.feature_scaler.mean_,
        [train_values.mean()],
    )
    np.testing.assert_allclose(
        result.preprocessor.feature_scaler.scale_,
        [train_values.std(ddof=0)],
    )
    assert (
        result.client_frames["Client 1"]["controller_validation"]["Feature"]
        .iloc[0]
        == 1000.0
    )


def test_manifest_generator_is_directly_executable_from_repository_root():
    repository_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/create_strict_partition_manifest.py",
            "--help",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_strict_loader_rejects_noncanonical_source_indices(tmp_path):
    frame = pd.DataFrame(
        {
            "Feature": [1.0, 2.0, 3.0, 4.0],
            "ContAmnt": [10.0, 20.0, 30.0, 40.0],
            "Client": ["Client 1"] * 4,
        }
    )
    raw_path = tmp_path / "canonical.csv"
    frame.to_csv(raw_path, index=False)
    dataset_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest = pd.DataFrame(
        {
            "source_index": [0, 1, 2, 3, 999],
            "client_id": ["Client 1"] * 5,
            "partition": [
                "train",
                "train",
                "controller_validation",
                "locked_test",
                "train",
            ],
            "dataset_sha256": [dataset_sha256] * 5,
        }
    )
    manifest_path = tmp_path / "partition.csv"
    manifest.to_csv(manifest_path, index=False)
    config = {
        "scene_c": {
            "data": {
                "raw_csv": str(raw_path),
                "rename_map": {},
                "feature_columns": ["Feature"],
                "target_column": "ContAmnt",
                "client_column": "Client",
            }
        },
        "preprocessing": {
            "scaler": "StandardScaler",
            "target_transform": "power_0.25",
            "random_seed": 42,
        },
    }

    with pytest.raises(
        ValueError,
        match="partition manifest does not cover the canonical dataset exactly",
    ):
        load_strict_partition_frames(config, str(manifest_path))
