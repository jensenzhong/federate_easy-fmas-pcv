import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.federated_learning.pcv.protocol import (
    PARTITION_PUBLICATION_PROTOCOL,
    PARTITION_PUBLICATION_SCHEMA,
    PartitionRatios,
    build_partition_manifest,
)
from src.study_manifest import load_study_manifest
from src.utils import load_config


def _temporary_artifact_path(output: Path, artifact_suffix: str) -> Path:
    descriptor, path = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.stem}.",
        suffix=f".{artifact_suffix}.tmp",
    )
    os.close(descriptor)
    return Path(path)


def _fsync_manifest_csv(path: Path, manifest: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        manifest.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_metadata_json(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_candidate_orphan_metadata(
    metadata_path: Path,
    output: Path,
) -> dict:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileExistsError(
            f"unrecognized partition metadata: {metadata_path}"
        ) from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("publication_protocol")
        != PARTITION_PUBLICATION_PROTOCOL
        or metadata.get("publication_schema")
        != PARTITION_PUBLICATION_SCHEMA
        or metadata.get("csv_name") != output.name
    ):
        raise FileExistsError(
            f"unrecognized partition metadata: {metadata_path}"
        )
    return metadata


def publish_partition_artifacts(
    manifest: pd.DataFrame,
    output: Path,
    *,
    dataset_sha256: str,
    split_seed: int,
) -> dict:
    """Publish a verified CSV/JSON pair without replacing existing files."""

    output = Path(output)
    metadata_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"partition output already exists: {output}")
    orphan_metadata = None
    if os.path.lexists(metadata_path):
        orphan_metadata = _read_candidate_orphan_metadata(
            metadata_path,
            output,
        )

    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        csv_temp = _temporary_artifact_path(output, "csv")
        temporary_paths.append(csv_temp)
        json_temp = _temporary_artifact_path(output, "json")
        temporary_paths.append(json_temp)
        _fsync_manifest_csv(csv_temp, manifest)
        verified_manifest = pd.read_csv(
            csv_temp,
            dtype={
                "row_id": "string",
                "client_id": "string",
                "partition": "string",
                "dataset_sha256": "string",
            },
        )
        pd.testing.assert_frame_equal(
            verified_manifest,
            manifest.reset_index(drop=True),
            check_dtype=False,
        )
        partition_sha256 = hashlib.sha256(csv_temp.read_bytes()).hexdigest()
        metadata = {
            "publication_protocol": PARTITION_PUBLICATION_PROTOCOL,
            "publication_schema": PARTITION_PUBLICATION_SCHEMA,
            "csv_name": output.name,
            "dataset_sha256": dataset_sha256,
            "partition_sha256": partition_sha256,
            "split_seed": split_seed,
            "rows": len(manifest),
        }
        _fsync_metadata_json(json_temp, metadata)
        verified_metadata = json.loads(json_temp.read_text(encoding="utf-8"))
        if verified_metadata != metadata:
            raise RuntimeError("partition metadata temporary verification failed")

        if orphan_metadata is not None:
            if orphan_metadata != metadata:
                raise FileExistsError(
                    f"unrecognized partition metadata: {metadata_path}"
                )
            metadata_path.unlink()

        os.link(json_temp, metadata_path)
        published_paths.append(metadata_path)
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError("published partition metadata verification failed")

        os.link(csv_temp, output)
        published_paths.append(output)
        if hashlib.sha256(output.read_bytes()).hexdigest() != partition_sha256:
            raise RuntimeError("published partition CSV hash verification failed")
        return metadata
    except BaseException:
        for published_path in reversed(published_paths):
            try:
                published_path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", default="study_manifest.yaml")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--output",
        default="results/manifests/strict_partition_v1.csv",
    )
    args = parser.parse_args()
    output = Path(args.output)

    study = load_study_manifest(Path(args.study_manifest))
    config = load_config(args.config)
    data_cfg = config["scene_c"]["data"]
    raw_path = Path(data_cfg["raw_csv"])
    dataset_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    frame = pd.read_csv(raw_path).rename(
        columns=data_cfg.get("rename_map", {})
    )
    frame["source_index"] = frame.index.astype(int)
    protocol = study.data_protocol
    manifest = build_partition_manifest(
        frame,
        client_column=data_cfg["client_column"],
        target_column=data_cfg["target_column"],
        source_index_column="source_index",
        dataset_sha256=dataset_sha256,
        split_seed=study.split_seed,
        ratios=PartitionRatios(
            protocol["train_ratio"],
            protocol["controller_validation_ratio"],
            protocol["locked_test_ratio"],
        ),
        quantile_bins=int(protocol["target_quantile_bins"]),
    )
    publish_partition_artifacts(
        manifest,
        output,
        dataset_sha256=dataset_sha256,
        split_seed=study.split_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
