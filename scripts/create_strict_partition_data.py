"""Seal physical strict-partition CSVs so training never opens locked-test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import load_strict_partition_frames  # noqa: E402
from src.utils import load_config  # noqa: E402


PARTITIONS = ("train", "controller_validation", "locked_test")
PUBLICATION_PROTOCOL = "strict_partition_physical_seal_v1"
PUBLICATION_SCHEMA = 1


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    with path.open("w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_partition_data(
    *,
    config: dict,
    manifest_path: Path,
    output_directory: Path,
) -> Path:
    """Publish all three physical partitions as one no-overwrite directory."""

    output_directory = Path(output_directory)
    if os.path.lexists(output_directory):
        raise FileExistsError(f"sealed partition directory exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    loaded = load_strict_partition_frames(config, str(manifest_path))
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            suffix=".tmp",
            dir=output_directory.parent,
        )
    )
    try:
        files = {}
        for partition in PARTITIONS:
            frame = pd.concat(
                [
                    partitions[partition]
                    for _, partitions in sorted(loaded.client_frames.items())
                ],
                ignore_index=True,
            ).sort_values("source_index", kind="stable").reset_index(drop=True)
            filename = f"{partition}.csv"
            digest = _write_csv(temporary / filename, frame)
            files[partition] = {
                "filename": filename,
                "rows": len(frame),
                "sha256": digest,
            }
        metadata = {
            "publication_protocol": PUBLICATION_PROTOCOL,
            "publication_schema": PUBLICATION_SCHEMA,
            "manifest_name": Path(manifest_path).name,
            "dataset_sha256": loaded.dataset_sha256,
            "partition_sha256": loaded.partition_sha256,
            "files": files,
        }
        metadata_path = temporary / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError("sealed partition metadata verification failed")
        os.rename(temporary, output_directory)
        return output_directory
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--manifest", default="results/manifests/strict_partition_v1.csv"
    )
    parser.add_argument("--output", default="Data/strict_partition_v1")
    args = parser.parse_args(argv)
    seal_partition_data(
        config=load_config(args.config),
        manifest_path=Path(args.manifest),
        output_directory=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
