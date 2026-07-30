"""Deterministic, client-local data partition construction."""

from dataclasses import dataclass
from hashlib import sha256
import math
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split


PARTITION_PUBLICATION_PROTOCOL = "strict_partition_csv_commit_v1"
PARTITION_PUBLICATION_SCHEMA = 1


class PrivacyViolation(RuntimeError):
    """Raised when a prompt payload crosses the aggregate-only boundary."""


class TestPartitionLocked(RuntimeError):
    """Raised when locked-test access is requested before formal unlock."""


ROOT_PROMPT_KEYS = frozenset({"round_index", "clients"})
REQUIRED_CLIENT_PROMPT_KEYS = frozenset(
    {
        "client_id",
        "sample_count",
        "train_loss",
        "val_mape",
        "val_rmse",
        "update_norm",
    }
)
OPTIONAL_CLIENT_PROMPT_KEYS = frozenset(
    {"cosine_to_mean", "cosine_to_previous"}
)
CLIENT_PROMPT_KEYS = REQUIRED_CLIENT_PROMPT_KEYS | OPTIONAL_CLIENT_PROMPT_KEYS
CLIENT_METRIC_KEYS = frozenset(
    {
        "train_loss",
        "val_mape",
        "val_rmse",
        "update_norm",
        "cosine_to_mean",
        "cosine_to_previous",
    }
)
APPROVED_PROMPT_KEYS = ROOT_PROMPT_KEYS | CLIENT_PROMPT_KEYS


def _normalized_prompt_mapping(
    value,
    *,
    allowed_keys: frozenset[str],
    context: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise PrivacyViolation(f"{context} must be an exact mapping")

    normalized = {}
    for key, item in value.items():
        if type(key) is not str:
            raise PrivacyViolation(f"{context} field names must be exact strings")
        normalized_key = unicodedata.normalize("NFKC", key).casefold()
        if normalized_key in normalized:
            raise PrivacyViolation(
                f"{context} contains colliding normalized field: {key}"
            )
        if normalized_key not in allowed_keys:
            raise PrivacyViolation(f"unapproved {context} field: {key}")
        normalized[normalized_key] = item
    return normalized


def _require_fields(
    value: dict[str, object],
    required_fields: frozenset[str],
    *,
    context: str,
) -> None:
    missing = required_fields - value.keys()
    if missing:
        raise PrivacyViolation(
            f"{context} missing required fields: {sorted(missing)}"
        )


def assert_prompt_payload_safe(payload) -> None:
    """Validate the complete aggregate client-telemetry prompt schema."""
    root = _normalized_prompt_mapping(
        payload,
        allowed_keys=ROOT_PROMPT_KEYS,
        context="prompt root",
    )
    _require_fields(root, ROOT_PROMPT_KEYS, context="prompt root")

    round_index = root["round_index"]
    if type(round_index) is not int or round_index < 0:
        raise PrivacyViolation("round_index must be a non-negative exact int")

    clients = root["clients"]
    if type(clients) not in (list, tuple):
        raise PrivacyViolation("clients must be an exact list or tuple")

    for index, client_payload in enumerate(clients):
        context = f"clients[{index}]"
        client = _normalized_prompt_mapping(
            client_payload,
            allowed_keys=CLIENT_PROMPT_KEYS,
            context=context,
        )
        _require_fields(
            client,
            REQUIRED_CLIENT_PROMPT_KEYS,
            context=context,
        )

        client_id = client["client_id"]
        if type(client_id) is not str or not client_id.strip():
            raise PrivacyViolation(f"{context}.client_id must be non-empty")

        sample_count = client["sample_count"]
        if type(sample_count) is not int or sample_count <= 0:
            raise PrivacyViolation(
                f"{context}.sample_count must be a positive exact int"
            )

        for metric_key in CLIENT_METRIC_KEYS & client.keys():
            metric = client[metric_key]
            if type(metric) not in (int, float) or (
                type(metric) is float and not math.isfinite(metric)
            ):
                raise PrivacyViolation(
                    f"{context}.{metric_key} must be a finite exact number"
                )


def require_test_unlock(
    *,
    phase: str,
    formal_frozen: bool,
    explicit_unlock: bool,
) -> None:
    """Require the exact frozen formal-evaluation unlock combination."""
    if (
        phase != "formal_evaluate"
        or formal_frozen is not True
        or explicit_unlock is not True
    ):
        raise TestPartitionLocked(
            "locked test is unavailable before frozen formal evaluation"
        )


@dataclass(frozen=True)
class PartitionRatios:
    train: float
    controller_validation: float
    locked_test: float

    def validate(self) -> None:
        values = (
            self.train,
            self.controller_validation,
            self.locked_test,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("partition ratios must be strictly positive")
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
    if (
        isinstance(quantile_bins, bool)
        or not isinstance(quantile_bins, int)
        or quantile_bins <= 0
    ):
        raise ValueError("quantile_bins must be positive")

    output = []
    for client_id, client_frame in frame.groupby(client_column, sort=True):
        local = client_frame.copy()
        ranks = local[target_column].rank(method="first")
        local["_target_bin"] = pd.qcut(
            ranks,
            q=min(quantile_bins, len(local)),
            labels=False,
        )
        target_bin_counts = local["_target_bin"].value_counts()
        target_strata = len(target_bin_counts)
        holdout_fraction = 1.0 - ratios.train
        holdout_size = math.ceil(holdout_fraction * len(local))
        train_size = len(local) - holdout_size
        client_context = f"client_id={client_id!r}"
        if target_bin_counts.min() < 2:
            raise ValueError(
                f"{client_context}: target strata need at least 2 rows each "
                "for the train/holdout split"
            )
        if train_size < target_strata:
            raise ValueError(
                f"{client_context}: train size {train_size} cannot cover "
                f"{target_strata} target strata"
            )
        if holdout_size < target_strata:
            raise ValueError(
                f"{client_context}: holdout size {holdout_size} cannot cover "
                f"{target_strata} target strata"
            )
        try:
            train, holdout = train_test_split(
                local,
                test_size=holdout_fraction,
                random_state=split_seed,
                stratify=local["_target_bin"],
            )
        except ValueError as exc:
            raise ValueError(
                f"{client_context}: train/holdout stratified split "
                f"is infeasible: {exc}"
            ) from exc

        test_fraction = ratios.locked_test / (
            ratios.controller_validation + ratios.locked_test
        )
        holdout_bin_counts = holdout["_target_bin"].value_counts()
        holdout_strata = len(holdout_bin_counts)
        locked_test_size = math.ceil(test_fraction * len(holdout))
        validation_size = len(holdout) - locked_test_size
        if holdout_bin_counts.min() < 2:
            raise ValueError(
                f"{client_context}: holdout target strata need at least 2 "
                "rows each for the controller split"
            )
        if validation_size < holdout_strata:
            raise ValueError(
                f"{client_context}: controller-validation size "
                f"{validation_size} cannot cover {holdout_strata} "
                "holdout target strata"
            )
        if locked_test_size < holdout_strata:
            raise ValueError(
                f"{client_context}: locked-test size {locked_test_size} "
                f"cannot cover {holdout_strata} holdout target strata"
            )
        try:
            validation, test = train_test_split(
                holdout,
                test_size=test_fraction,
                random_state=split_seed,
                stratify=holdout["_target_bin"],
            )
        except ValueError as exc:
            raise ValueError(
                f"{client_context}: controller-validation/locked-test "
                f"stratified split is infeasible: {exc}"
            ) from exc
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
