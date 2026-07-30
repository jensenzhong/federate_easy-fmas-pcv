"""Deterministic, client-local data partition construction."""

from dataclasses import dataclass
from hashlib import sha256

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PartitionRatios:
    train: float
    controller_validation: float
    locked_test: float

    def validate(self) -> None:
        if (
            abs(
                self.train
                + self.controller_validation
                + self.locked_test
                - 1.0
            )
            > 1e-9
        ):
            raise ValueError("partition ratios must sum to one")


def _row_id(dataset_sha256: str, source_index: int) -> str:
    return sha256(
        f"{dataset_sha256}:{source_index}".encode("utf-8")
    ).hexdigest()[:24]


def build_partition_manifest(
    frame: pd.DataFrame,
    *,
    client_column: str,
    target_column: str,
    source_index_column: str,
    dataset_sha256: str,
    split_seed: int,
    ratios: PartitionRatios,
    quantile_bins: int,
) -> pd.DataFrame:
    ratios.validate()
    output = []
    for client_id, client_frame in frame.groupby(client_column, sort=True):
        local = client_frame.copy()
        ranks = local[target_column].rank(method="first")
        local["_target_bin"] = pd.qcut(
            ranks,
            q=min(quantile_bins, len(local)),
            labels=False,
        )
        train, holdout = train_test_split(
            local,
            test_size=1.0 - ratios.train,
            random_state=split_seed,
            stratify=local["_target_bin"],
        )
        test_fraction = ratios.locked_test / (
            ratios.controller_validation + ratios.locked_test
        )
        validation, test = train_test_split(
            holdout,
            test_size=test_fraction,
            random_state=split_seed,
            stratify=holdout["_target_bin"],
        )
        for partition, rows in (
            ("train", train),
            ("controller_validation", validation),
            ("locked_test", test),
        ):
            for source_index in rows[source_index_column].astype(int):
                output.append(
                    {
                        "row_id": _row_id(dataset_sha256, source_index),
                        "source_index": source_index,
                        "client_id": str(client_id),
                        "partition": partition,
                        "dataset_sha256": dataset_sha256,
                    }
                )
    return (
        pd.DataFrame(output)
        .sort_values(["client_id", "source_index"])
        .reset_index(drop=True)
    )
