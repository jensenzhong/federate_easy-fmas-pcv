"""Deterministic, client-local data partition construction."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import math
from numbers import Real
from types import MappingProxyType
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from .client_evaluation import MetricSums, aggregate_metric_sums, build_vote
from .schemas import ClientTelemetry, LocalCandidateVote


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


@dataclass(frozen=True, slots=True)
class LocalTrainingResult:
    """Sanitized client training result with cloned parameter tensors."""

    model_state: Mapping[str, torch.Tensor]
    sample_count: int
    train_loss: float

    def __post_init__(self) -> None:
        if type(self.model_state) is not dict:
            raise TypeError("model_state must be an exact dictionary")
        cloned_state: dict[str, torch.Tensor] = {}
        for name, tensor in self.model_state.items():
            if type(name) is not str or not name:
                raise TypeError("model_state keys must be non-empty exact strings")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("model_state values must be torch.Tensor instances")
            cloned_state[name] = tensor.detach().clone()
        if type(self.sample_count) is not int:
            raise TypeError("sample_count must be an exact integer")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if isinstance(self.train_loss, bool) or not isinstance(self.train_loss, Real):
            raise TypeError("train_loss must be a real scalar")
        train_loss = float(self.train_loss)
        if not math.isfinite(train_loss):
            raise ValueError("train_loss must be finite")
        object.__setattr__(self, "model_state", MappingProxyType(cloned_state))
        object.__setattr__(self, "train_loss", train_loss)


class ClientDataVault:
    """Own client-private partitions and expose sanitized aggregate operations."""

    __slots__ = (
        "client_id",
        "__train_dataset",
        "__controller_validation_dataset",
        "__locked_test_dataset",
        "__train_fn",
        "__telemetry_fn",
        "__metric_sums_fn",
    )

    def __init__(
        self,
        *,
        client_id: str,
        train_dataset,
        controller_validation_dataset,
        locked_test_dataset,
        train_fn: Callable,
        telemetry_fn: Callable,
        metric_sums_fn: Callable,
    ) -> None:
        if type(client_id) is not str or not client_id.strip():
            raise ValueError("client_id must be a non-empty exact string")
        for name, callback in (
            ("train_fn", train_fn),
            ("telemetry_fn", telemetry_fn),
            ("metric_sums_fn", metric_sums_fn),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        self.client_id = client_id
        self.__train_dataset = train_dataset
        self.__controller_validation_dataset = controller_validation_dataset
        self.__locked_test_dataset = locked_test_dataset
        self.__train_fn = train_fn
        self.__telemetry_fn = telemetry_fn
        self.__metric_sums_fn = metric_sums_fn

    @staticmethod
    def _positive_exact_int(value, *, name: str) -> int:
        if type(value) is not int:
            raise TypeError(f"{name} must be an exact integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _finite_scalar(value, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    def train_local(
        self,
        global_state,
        training_config,
        seed,
    ) -> LocalTrainingResult:
        result = self.__train_fn(
            self.__train_dataset,
            global_state,
            training_config,
            seed,
        )
        if type(result) is not LocalTrainingResult:
            raise TypeError("train_fn must return exact LocalTrainingResult")
        return LocalTrainingResult(
            model_state=dict(result.model_state),
            sample_count=result.sample_count,
            train_loss=result.train_loss,
        )

    def controller_telemetry(self, model_state) -> ClientTelemetry:
        telemetry = self.__telemetry_fn(
            self.client_id,
            self.__controller_validation_dataset,
            model_state,
        )
        if type(telemetry) is not ClientTelemetry:
            raise TypeError("telemetry_fn must return exact ClientTelemetry")
        if type(telemetry.client_id) is not str or telemetry.client_id != self.client_id:
            raise ValueError("telemetry client_id must match the vault client")
        return ClientTelemetry(
            client_id=self.client_id,
            sample_count=self._positive_exact_int(
                telemetry.sample_count,
                name="telemetry sample_count",
            ),
            train_loss=self._finite_scalar(
                telemetry.train_loss,
                name="telemetry train_loss",
            ),
            val_mape=self._finite_scalar(
                telemetry.val_mape,
                name="telemetry val_mape",
            ),
            val_rmse=self._finite_scalar(
                telemetry.val_rmse,
                name="telemetry val_rmse",
            ),
            update_norm=self._finite_scalar(
                telemetry.update_norm,
                name="telemetry update_norm",
            ),
            cosine_to_mean=self._finite_scalar(
                telemetry.cosine_to_mean,
                name="telemetry cosine_to_mean",
            ),
            cosine_to_previous=self._finite_scalar(
                telemetry.cosine_to_previous,
                name="telemetry cosine_to_previous",
            ),
        )

    def _validated_metric_sums(self, dataset, model_state) -> MetricSums:
        sums = self.__metric_sums_fn(dataset, model_state)
        if type(sums) is not MetricSums:
            raise TypeError("metric_sums_fn must return exact MetricSums")
        return MetricSums(
            n=int(sums.n),
            ape_sum=float(sums.ape_sum),
            se_sum=float(sums.se_sum),
            ae_sum=float(sums.ae_sum),
            y_sum=float(sums.y_sum),
            y_sq_sum=float(sums.y_sq_sum),
        )

    def evaluate_candidates(
        self,
        candidate_states: dict[str, dict],
        stronger_anchor_id: str,
    ) -> list[LocalCandidateVote]:
        if not isinstance(candidate_states, dict):
            raise TypeError("candidate_states must be a dictionary")
        candidate_items = list(candidate_states.items())
        if not candidate_items:
            raise ValueError("candidate evaluation requires at least one candidate")

        candidate_ids = [candidate_id for candidate_id, _ in candidate_items]
        if any(
            not isinstance(candidate_id, str) or not candidate_id.strip()
            for candidate_id in candidate_ids
        ):
            raise ValueError("candidate IDs must be non-empty strings")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("duplicate candidate IDs are not allowed")
        if stronger_anchor_id not in set(candidate_ids):
            raise ValueError("stronger anchor must exist among candidate states")

        candidate_sums = [
            (
                candidate_id,
                self._validated_metric_sums(
                    self.__controller_validation_dataset,
                    model_state,
                ),
            )
            for candidate_id, model_state in candidate_items
        ]
        sample_counts = {sums.n for _, sums in candidate_sums}
        if len(sample_counts) != 1:
            raise ValueError("candidate sample_count values must be consistent")

        metrics = {
            candidate_id: aggregate_metric_sums([sums])
            for candidate_id, sums in candidate_sums
        }
        ranked_ids = sorted(
            metrics,
            key=lambda candidate_id: (
                float(metrics[candidate_id]["mape"]),
                candidate_id,
            ),
        )
        anchor = metrics[stronger_anchor_id]
        confidence = 1.0 / len(ranked_ids)
        return [
            build_vote(
                client_id=self.client_id,
                candidate_id=candidate_id,
                sample_count=int(metrics[candidate_id]["sample_count"]),
                candidate_mape=float(metrics[candidate_id]["mape"]),
                candidate_rmse=float(metrics[candidate_id]["rmse"]),
                anchor_mape=float(anchor["mape"]),
                anchor_rmse=float(anchor["rmse"]),
                rank=rank,
                confidence=confidence,
            )
            for rank, candidate_id in enumerate(ranked_ids, start=1)
        ]

    def final_test_sums(self, model_state, unlock_context) -> MetricSums:
        if not isinstance(unlock_context, dict):
            raise TypeError("unlock_context must be a dictionary")
        require_test_unlock(**unlock_context)
        return self._validated_metric_sums(self.__locked_test_dataset, model_state)


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
