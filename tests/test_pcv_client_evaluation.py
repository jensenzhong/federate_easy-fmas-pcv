import math

import numpy as np
import pytest

from src.federated_learning.pcv.client_evaluation import (
    MetricSums,
    aggregate_metric_sums,
    build_vote,
    compute_metric_sums,
)
from src.federated_learning.pcv.schemas import LocalCandidateVote


def test_client_evaluation_module_exposes_metric_sums():
    assert MetricSums


def test_compute_metric_sums_inverse_transforms_to_original_currency_scale():
    transformed = []

    def inverse_transform(values):
        transformed.append(np.asarray(values).copy())
        return np.asarray(values) * 100.0

    sums = compute_metric_sums(
        np.array([0.1, 0.2]),
        np.array([0.11, 0.18]),
        inverse_transform=inverse_transform,
    )

    assert len(transformed) == 2
    assert sums.n == 2
    assert sums.ape_sum == pytest.approx(0.2)
    assert sums.se_sum == pytest.approx(5.0)
    assert sums.ae_sum == pytest.approx(3.0)
    assert sums.y_sum == pytest.approx(30.0)
    assert sums.y_sq_sum == pytest.approx(500.0)


def test_metric_sums_aggregate_without_predictions_labels_or_tensors():
    combined = aggregate_metric_sums(
        [
            MetricSums(2, 0.4, 5.0, 3.0, 6.0, 20.0),
            MetricSums(1, 0.1, 4.0, 2.0, 4.0, 16.0),
        ]
    )

    assert combined == {
        "sample_count": 3,
        "mape": pytest.approx(0.5 / 3),
        "rmse": pytest.approx(math.sqrt(3.0)),
        "mae": pytest.approx(5.0 / 3),
        "r2": pytest.approx(-2.375),
    }
    assert not any(
        forbidden in combined
        for forbidden in ("labels", "predictions", "y_true", "y_pred", "tensor")
    )


@pytest.mark.parametrize(
    ("truth", "prediction", "epsilon"),
    [
        ([], [], 1e-8),
        ([1.0], [1.0, 2.0], 1e-8),
        (np.ones((2, 1)), np.ones(2), 1e-8),
        ([np.nan], [1.0], 1e-8),
        ([1.0], [np.inf], 1e-8),
        ([1.0], [1.0], 0.0),
        ([1.0], [1.0], -1.0),
        ([1.0], [1.0], np.inf),
    ],
)
def test_compute_metric_sums_rejects_invalid_inputs(truth, prediction, epsilon):
    with pytest.raises((TypeError, ValueError)):
        compute_metric_sums(
            truth,
            prediction,
            inverse_transform=lambda values: values,
            epsilon=epsilon,
        )


def test_compute_metric_sums_rejects_non_finite_inverse_transform_output():
    with pytest.raises(ValueError, match="inverse-transform"):
        compute_metric_sums(
            [1.0],
            [1.0],
            inverse_transform=lambda values: np.array([np.nan]),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"n": True},
        {"n": 1.5},
        {"ape_sum": -1.0},
        {"se_sum": np.inf},
        {"ae_sum": np.nan},
        {"y_sum": np.inf},
        {"y_sq_sum": -1.0},
    ],
)
def test_metric_sums_reject_invalid_counts_and_non_finite_sums(kwargs):
    values = {
        "n": 1,
        "ape_sum": 0.0,
        "se_sum": 0.0,
        "ae_sum": 0.0,
        "y_sum": 0.0,
        "y_sq_sum": 0.0,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        MetricSums(**values)


def test_aggregate_rejects_empty_or_invalid_items():
    with pytest.raises(ValueError, match="observations"):
        aggregate_metric_sums([])
    with pytest.raises(TypeError, match="MetricSums"):
        aggregate_metric_sums([object()])


def test_zero_and_constant_targets_have_finite_metrics_and_defined_r2():
    zero = compute_metric_sums(
        [0.0, 0.0],
        [0.0, 1.0],
        inverse_transform=lambda values: values,
    )
    constant = compute_metric_sums(
        [4.0, 4.0],
        [3.0, 5.0],
        inverse_transform=lambda values: values,
    )

    for metrics in (aggregate_metric_sums([zero]), aggregate_metric_sums([constant])):
        assert metrics["r2"] == 0.0
        assert all(math.isfinite(value) for value in metrics.values())


def test_aggregate_r2_avoids_overflow_before_dividing_by_sample_count():
    metrics = aggregate_metric_sums(
        [
            MetricSums(
                n=4,
                ape_sum=0.0,
                se_sum=1.25e307,
                ae_sum=0.0,
                y_sum=2.0e154,
                y_sq_sum=1.25e308,
            )
        ]
    )

    assert metrics["r2"] == pytest.approx(0.5)


def test_aggregate_r2_clamps_rounding_scale_negative_target_variance_to_zero():
    metrics = aggregate_metric_sums(
        [
            MetricSums(
                n=3,
                ape_sum=0.0,
                se_sum=1.0,
                ae_sum=0.0,
                y_sum=3.0,
                y_sq_sum=np.nextafter(3.0, 0.0),
            )
        ]
    )

    assert metrics["r2"] == 0.0


def test_aggregate_rejects_inconsistent_target_sufficient_statistics():
    with pytest.raises(ValueError, match="inconsistent"):
        aggregate_metric_sums(
            [
                MetricSums(
                    n=3,
                    ape_sum=0.0,
                    se_sum=1.0,
                    ae_sum=0.0,
                    y_sum=3.0,
                    y_sq_sum=2.0,
                )
            ]
        )


def _vote(**overrides):
    values = {
        "client_id": "client_01",
        "candidate_id": "candidate_01",
        "sample_count": 30,
        "candidate_mape": 0.42,
        "candidate_rmse": 10.0,
        "anchor_mape": 0.40,
        "anchor_rmse": 9.0,
        "rank": 1,
        "confidence": 0.8,
    }
    values.update(overrides)
    return build_vote(**values)


def test_vote_catastrophic_threshold_is_strictly_more_than_five_percent_relative():
    assert _vote(candidate_mape=0.42).catastrophic_degradation is False
    above_threshold = np.nextafter(0.40 * 1.05, math.inf)
    assert _vote(candidate_mape=above_threshold).catastrophic_degradation is True


def test_vote_with_zero_anchor_remains_finite_and_detects_degradation():
    vote = _vote(
        candidate_mape=0.1,
        candidate_rmse=0.2,
        anchor_mape=0.0,
        anchor_rmse=0.0,
    )

    assert vote.catastrophic_degradation is True
    assert math.isfinite(vote.relative_mape)
    assert math.isfinite(vote.relative_rmse)


def test_vote_contains_only_aggregate_fields():
    vote = _vote()

    assert isinstance(vote, LocalCandidateVote)
    assert set(vars(vote)) == {
        "client_id",
        "candidate_id",
        "sample_count",
        "val_mape",
        "val_rmse",
        "relative_mape",
        "relative_rmse",
        "rank",
        "confidence",
        "catastrophic_degradation",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"client_id": ""},
        {"candidate_id": ""},
        {"sample_count": 0},
        {"sample_count": True},
        {"candidate_mape": np.nan},
        {"candidate_rmse": -1.0},
        {"anchor_mape": np.inf},
        {"anchor_rmse": -1.0},
        {"rank": 0},
        {"rank": True},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": np.nan},
    ],
)
def test_vote_rejects_invalid_identity_metrics_rank_and_confidence(overrides):
    with pytest.raises((TypeError, ValueError)):
        _vote(**overrides)
