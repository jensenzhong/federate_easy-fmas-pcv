import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import scripts.create_strict_partition_manifest as manifest_script
from scripts.create_strict_partition_data import seal_partition_data
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


def _expected_row_id(dataset_sha256: str, source_index: int) -> str:
    return hashlib.sha256(
        f"{dataset_sha256}:{source_index}".encode("utf-8")
    ).hexdigest()[:24]


def _write_manifest_pair(
    manifest_path: Path,
    manifest: pd.DataFrame,
    dataset_sha256: str,
) -> None:
    manifest.to_csv(manifest_path, index=False)
    partition_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "publication_protocol": "strict_partition_csv_commit_v1",
                "publication_schema": 1,
                "csv_name": manifest_path.name,
                "dataset_sha256": dataset_sha256,
                "partition_sha256": partition_sha256,
                "rows": len(manifest),
                "split_seed": 20260730,
            }
        ),
        encoding="utf-8",
    )


def _strict_loader_fixture(tmp_path: Path):
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
    manifest.insert(
        0,
        "row_id",
        [
            _expected_row_id(dataset_sha256, source_index)
            for source_index in manifest["source_index"]
        ],
    )
    manifest_path = tmp_path / "partition.csv"
    _write_manifest_pair(manifest_path, manifest, dataset_sha256)
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
    return config, manifest_path, manifest, dataset_sha256


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


@pytest.mark.parametrize(
    "ratios",
    [
        PartitionRatios(0.0, 0.5, 0.5),
        PartitionRatios(0.5, 0.0, 0.5),
        PartitionRatios(0.5, 0.5, 0.0),
    ],
)
def test_partition_rejects_nonpositive_ratios(ratios):
    with pytest.raises(ValueError, match="partition ratios must be strictly positive"):
        build_partition_manifest(
            _frame(),
            client_column="Client",
            target_column="ContAmnt",
            source_index_column="source_index",
            dataset_sha256="0" * 64,
            split_seed=20260730,
            ratios=ratios,
            quantile_bins=5,
        )


def test_partition_rejects_nonpositive_quantile_bins():
    with pytest.raises(ValueError, match="quantile_bins must be positive"):
        build_partition_manifest(
            _frame(),
            client_column="Client",
            target_column="ContAmnt",
            source_index_column="source_index",
            dataset_sha256="0" * 64,
            split_seed=20260730,
            ratios=PartitionRatios(0.70, 0.15, 0.15),
            quantile_bins=0,
        )


def test_partition_rejects_client_too_small_for_first_stratified_split():
    frame = pd.DataFrame(
        {
            "Client": ["Tiny"] * 10,
            "ContAmnt": range(1, 11),
            "source_index": range(10),
        }
    )

    with pytest.raises(
        ValueError,
        match="client_id='Tiny'.*holdout size.*5 target strata",
    ):
        build_partition_manifest(
            frame,
            client_column="Client",
            target_column="ContAmnt",
            source_index_column="source_index",
            dataset_sha256="0" * 64,
            split_seed=20260730,
            ratios=PartitionRatios(0.70, 0.15, 0.15),
            quantile_bins=5,
        )


def test_partition_rejects_holdout_strata_too_small_for_second_split():
    frame = pd.DataFrame(
        {
            "Client": ["Layered"] * 20,
            "ContAmnt": range(1, 21),
            "source_index": range(20),
        }
    )

    with pytest.raises(
        ValueError,
        match="client_id='Layered'.*holdout target strata.*at least 2",
    ):
        build_partition_manifest(
            frame,
            client_column="Client",
            target_column="ContAmnt",
            source_index_column="source_index",
            dataset_sha256="0" * 64,
            split_seed=20260730,
            ratios=PartitionRatios(0.70, 0.15, 0.15),
            quantile_bins=5,
        )


def test_strict_loader_fits_feature_stats_on_train_partition_only(tmp_path):
    config, manifest_path, _, _ = _strict_loader_fixture(tmp_path)

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


def test_training_loader_does_not_materialize_locked_test_rows(tmp_path):
    config, manifest_path, _, _ = _strict_loader_fixture(tmp_path)

    result = load_strict_partition_frames(
        config,
        str(manifest_path),
        allowed_partitions={"train", "controller_validation"},
    )

    assert result.client_frames
    assert all(
        set(partitions) == {"train", "controller_validation"}
        for partitions in result.client_frames.values()
    )


def test_sealed_training_loader_never_opens_locked_test_file(tmp_path, monkeypatch):
    config, manifest_path, _, _ = _strict_loader_fixture(tmp_path)
    sealed = seal_partition_data(
        config=config,
        manifest_path=manifest_path,
        output_directory=tmp_path / "sealed",
    )
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        if path.name == "locked_test.csv":
            raise AssertionError("training opened the physically sealed locked test")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    result = load_strict_partition_frames(
        config,
        str(manifest_path),
        allowed_partitions={"train", "controller_validation"},
        sealed_data_directory=sealed,
    )

    assert all("locked_test" not in parts for parts in result.client_frames.values())


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


def _publishable_manifest(dataset_sha256: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [_expected_row_id(dataset_sha256, 0)],
            "source_index": [0],
            "client_id": ["Client 1"],
            "partition": ["train"],
            "dataset_sha256": [dataset_sha256],
        }
    )


def test_manifest_publisher_writes_verified_csv_json_pair(tmp_path):
    output = tmp_path / "strict_partition_v1.csv"
    dataset_sha256 = "0" * 64
    manifest = _publishable_manifest(dataset_sha256)

    manifest_script.publish_partition_artifacts(
        manifest,
        output,
        dataset_sha256=dataset_sha256,
        split_seed=20260730,
    )

    metadata_path = output.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert output.exists()
    assert (
        metadata["publication_protocol"]
        == "strict_partition_csv_commit_v1"
    )
    assert metadata["publication_schema"] == 1
    assert metadata["csv_name"] == output.name
    assert metadata["dataset_sha256"] == dataset_sha256
    assert metadata["partition_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert metadata["rows"] == len(manifest)
    assert not list(tmp_path.glob(".strict_partition_v1.*.tmp"))


def test_manifest_publisher_leaves_no_final_when_metadata_publish_fails(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "strict_partition_v1.csv"
    metadata_path = output.with_suffix(".json")
    dataset_sha256 = "0" * 64
    real_link = manifest_script.os.link

    def fail_json_link(source, destination):
        if Path(destination).suffix == ".json":
            raise OSError("simulated JSON publish failure")
        return real_link(source, destination)

    monkeypatch.setattr(manifest_script.os, "link", fail_json_link)

    with pytest.raises(OSError, match="simulated JSON publish failure"):
        manifest_script.publish_partition_artifacts(
            _publishable_manifest(dataset_sha256),
            output,
            dataset_sha256=dataset_sha256,
            split_seed=20260730,
        )

    assert not output.exists()
    assert not metadata_path.exists()
    assert not list(tmp_path.glob(".strict_partition_v1.*.tmp"))


def test_manifest_publisher_rolls_back_sidecar_when_csv_commit_fails(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "strict_partition_v1.csv"
    metadata_path = output.with_suffix(".json")
    dataset_sha256 = "0" * 64
    real_link = manifest_script.os.link
    link_destinations = []

    def fail_csv_link(source, destination):
        destination = Path(destination)
        link_destinations.append(destination)
        if destination == output:
            raise OSError("simulated CSV commit failure")
        return real_link(source, destination)

    monkeypatch.setattr(manifest_script.os, "link", fail_csv_link)

    with pytest.raises(OSError, match="simulated CSV commit failure"):
        manifest_script.publish_partition_artifacts(
            _publishable_manifest(dataset_sha256),
            output,
            dataset_sha256=dataset_sha256,
            split_seed=20260730,
        )

    assert link_destinations == [metadata_path, output]
    assert not output.exists()
    assert not metadata_path.exists()
    assert not list(tmp_path.glob(".strict_partition_v1.*.tmp"))


def _orphan_metadata(
    output: Path,
    manifest: pd.DataFrame,
    dataset_sha256: str,
) -> dict:
    csv_bytes = manifest.to_csv(index=False).encode("utf-8")
    return {
        "publication_protocol": "strict_partition_csv_commit_v1",
        "publication_schema": 1,
        "csv_name": output.name,
        "dataset_sha256": dataset_sha256,
        "partition_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "split_seed": 20260730,
        "rows": len(manifest),
    }


def test_manifest_publisher_recovers_valid_uncommitted_sidecar(tmp_path):
    output = tmp_path / "strict_partition_v1.csv"
    metadata_path = output.with_suffix(".json")
    dataset_sha256 = "0" * 64
    manifest = _publishable_manifest(dataset_sha256)
    metadata_path.write_text(
        json.dumps(_orphan_metadata(output, manifest, dataset_sha256)),
        encoding="utf-8",
    )

    manifest_script.publish_partition_artifacts(
        manifest,
        output,
        dataset_sha256=dataset_sha256,
        split_seed=20260730,
    )

    assert output.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["partition_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert metadata["csv_name"] == output.name


def test_manifest_publisher_refuses_unknown_sidecar_without_deleting_it(
    tmp_path,
):
    output = tmp_path / "strict_partition_v1.csv"
    metadata_path = output.with_suffix(".json")
    unknown = {
        "publication_protocol": "strict_partition_csv_commit_v1",
        "publication_schema": 1,
        "csv_name": "different.csv",
    }
    metadata_path.write_text(json.dumps(unknown), encoding="utf-8")
    original_bytes = metadata_path.read_bytes()

    with pytest.raises(FileExistsError, match="unrecognized partition metadata"):
        manifest_script.publish_partition_artifacts(
            _publishable_manifest("0" * 64),
            output,
            dataset_sha256="0" * 64,
            split_seed=20260730,
        )

    assert not output.exists()
    assert metadata_path.read_bytes() == original_bytes


def test_csv_commit_point_observes_complete_valid_sidecar(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "strict_partition_v1.csv"
    metadata_path = output.with_suffix(".json")
    real_link = manifest_script.os.link
    observed_csv_commit = False

    def observe_link(source, destination):
        nonlocal observed_csv_commit
        destination = Path(destination)
        if destination == output:
            observed_csv_commit = True
            assert metadata_path.exists()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert metadata["csv_name"] == output.name
            assert metadata["partition_sha256"] == hashlib.sha256(
                Path(source).read_bytes()
            ).hexdigest()
        return real_link(source, destination)

    monkeypatch.setattr(manifest_script.os, "link", observe_link)

    manifest_script.publish_partition_artifacts(
        _publishable_manifest("0" * 64),
        output,
        dataset_sha256="0" * 64,
        split_seed=20260730,
    )

    assert observed_csv_commit


def test_manifest_publisher_cleans_temp_when_second_temp_creation_fails(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "strict_partition_v1.csv"
    real_temporary_path = manifest_script._temporary_artifact_path

    def fail_json_temporary_path(path, artifact_suffix):
        if artifact_suffix == "json":
            raise OSError("simulated JSON temp creation failure")
        return real_temporary_path(path, artifact_suffix)

    monkeypatch.setattr(
        manifest_script,
        "_temporary_artifact_path",
        fail_json_temporary_path,
    )

    with pytest.raises(OSError, match="simulated JSON temp creation failure"):
        manifest_script.publish_partition_artifacts(
            _publishable_manifest("0" * 64),
            output,
            dataset_sha256="0" * 64,
            split_seed=20260730,
        )

    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert not list(tmp_path.glob(".strict_partition_v1.*.tmp"))


@pytest.mark.parametrize(
    ("existing_suffix", "error_match"),
    [
        (".csv", "partition output already exists"),
        (".json", "unrecognized partition metadata"),
    ],
)
def test_manifest_publisher_refuses_any_existing_final(
    tmp_path,
    existing_suffix,
    error_match,
):
    output = tmp_path / "strict_partition_v1.csv"
    existing_path = output.with_suffix(existing_suffix)
    existing_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match=error_match):
        manifest_script.publish_partition_artifacts(
            _publishable_manifest("0" * 64),
            output,
            dataset_sha256="0" * 64,
            split_seed=20260730,
        )

    assert existing_path.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".strict_partition_v1.*.tmp"))


def test_manifest_publisher_does_not_overwrite_concurrent_final(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "strict_partition_v1.csv"

    metadata_path = output.with_suffix(".json")

    def concurrent_link(_source, destination):
        Path(destination).write_bytes(b"concurrent")
        raise FileExistsError("simulated concurrent publisher")

    monkeypatch.setattr(manifest_script.os, "link", concurrent_link)

    with pytest.raises(FileExistsError, match="simulated concurrent publisher"):
        manifest_script.publish_partition_artifacts(
            _publishable_manifest("0" * 64),
            output,
            dataset_sha256="0" * 64,
            split_seed=20260730,
        )

    assert not output.exists()
    assert metadata_path.read_bytes() == b"concurrent"
    assert not list(tmp_path.glob(".strict_partition_v1.*.tmp"))


def test_strict_loader_rejects_noncanonical_source_indices(tmp_path):
    config, manifest_path, manifest, dataset_sha256 = _strict_loader_fixture(
        tmp_path
    )
    manifest.loc[len(manifest)] = {
        "row_id": _expected_row_id(dataset_sha256, 999),
        "source_index": 999,
        "client_id": "Client 1",
        "partition": "train",
        "dataset_sha256": dataset_sha256,
    }
    _write_manifest_pair(manifest_path, manifest, dataset_sha256)

    with pytest.raises(
        ValueError,
        match="partition manifest does not cover the canonical dataset exactly",
    ):
        load_strict_partition_frames(config, str(manifest_path))


def test_strict_loader_rejects_manifest_schema_missing_row_id(tmp_path):
    config, manifest_path, manifest, dataset_sha256 = _strict_loader_fixture(
        tmp_path
    )
    _write_manifest_pair(
        manifest_path,
        manifest.drop(columns=["row_id"]),
        dataset_sha256,
    )

    with pytest.raises(ValueError, match="manifest missing columns.*row_id"):
        load_strict_partition_frames(config, str(manifest_path))


def test_strict_loader_rejects_tampered_row_id(tmp_path):
    config, manifest_path, manifest, dataset_sha256 = _strict_loader_fixture(
        tmp_path
    )
    manifest.loc[0, "row_id"] = "f" * 24
    _write_manifest_pair(manifest_path, manifest, dataset_sha256)

    with pytest.raises(ValueError, match="row_id mismatch"):
        load_strict_partition_frames(config, str(manifest_path))


def test_strict_loader_rejects_cross_client_assignment(tmp_path):
    config, manifest_path, manifest, dataset_sha256 = _strict_loader_fixture(
        tmp_path
    )
    manifest.loc[0, "client_id"] = "Client 2"
    _write_manifest_pair(manifest_path, manifest, dataset_sha256)

    with pytest.raises(ValueError, match="client_id ownership mismatch"):
        load_strict_partition_frames(config, str(manifest_path))


def test_strict_loader_rejects_unknown_partition(tmp_path):
    config, manifest_path, manifest, dataset_sha256 = _strict_loader_fixture(
        tmp_path
    )
    manifest.loc[0, "partition"] = "validation"
    _write_manifest_pair(manifest_path, manifest, dataset_sha256)

    with pytest.raises(ValueError, match="invalid partition"):
        load_strict_partition_frames(config, str(manifest_path))


def test_strict_loader_requires_same_stem_metadata(tmp_path):
    config, manifest_path, _, _ = _strict_loader_fixture(tmp_path)
    manifest_path.with_suffix(".json").unlink()

    with pytest.raises(ValueError, match="partition metadata"):
        load_strict_partition_frames(config, str(manifest_path))


@pytest.mark.parametrize(
    "metadata_key",
    [
        "publication_protocol",
        "publication_schema",
        "csv_name",
        "dataset_sha256",
        "partition_sha256",
    ],
)
def test_strict_loader_rejects_tampered_metadata(tmp_path, metadata_key):
    config, manifest_path, _, _ = _strict_loader_fixture(tmp_path)
    metadata_path = manifest_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[metadata_key] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=f"metadata {metadata_key} mismatch"):
        load_strict_partition_frames(config, str(manifest_path))
