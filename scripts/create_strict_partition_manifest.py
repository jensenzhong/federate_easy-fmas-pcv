import argparse
import hashlib
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.federated_learning.pcv.protocol import (
    PartitionRatios,
    build_partition_manifest,
)
from src.study_manifest import load_study_manifest
from src.utils import load_config


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
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"partition output already exists: {output}")

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
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    partition_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_sha256": dataset_sha256,
                "partition_sha256": partition_sha256,
                "split_seed": study.split_seed,
                "rows": len(manifest),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
