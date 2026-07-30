import pytest

from src.federated_learning.pcv.protocol import (
    PrivacyViolation,
    TestPartitionLocked as LockedTestError,
    assert_prompt_payload_safe,
    require_test_unlock,
)


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "raw_features",
        "raw_labels",
        "labels",
        "row_predictions",
        "predictions",
        "residuals",
        "test_mape",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_metrics",
        "locked_test",
    ),
)
def test_prompt_rejects_raw_label_prediction_and_test_fields(forbidden_key):
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe({forbidden_key: [1.0]})


def test_prompt_rejects_forbidden_keys_case_insensitively():
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe({"TeSt_MaPe": 0.4})


@pytest.mark.parametrize(
    "unknown_key",
    (
        "features",
        "ground_truth",
        "teſt_mape",
    ),
)
def test_prompt_rejects_unknown_or_unicode_normalized_keys(unknown_key):
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe({unknown_key: 0.4})


def test_prompt_rejects_non_exact_string_keys():
    class K(str):
        pass

    for key in (1, K("round_index")):
        with pytest.raises(PrivacyViolation):
            assert_prompt_payload_safe({key: 2})


@pytest.mark.parametrize(
    "payload",
    (
        {"clients": [{"client_id": "client_01", "raw_features": [1.0]}]},
        {"clients": [{"client_id": "client_01", "labels": [2.0]}]},
        {"clients": ({"client_id": "client_01", "row_predictions": [3.0]},)},
    ),
)
def test_prompt_rejects_forbidden_keys_in_nested_dict_list_or_tuple(payload):
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe(payload)


@pytest.mark.parametrize(
    "unsafe_value",
    (
        {1, 2},
        iter((1, 2)),
        object(),
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_prompt_rejects_non_json_or_non_finite_values(unsafe_value):
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe({"round_index": unsafe_value})


def test_prompt_accepts_approved_aggregate_telemetry():
    assert_prompt_payload_safe(
        {
            "round_index": 2,
            "clients": [
                {
                    "client_id": "client_01",
                    "sample_count": 100,
                    "train_loss": 0.3,
                    "val_mape": 0.4,
                    "val_rmse": 1.2,
                    "update_norm": 0.8,
                    "cosine_to_mean": 0.9,
                    "cosine_to_previous": 0.7,
                }
            ],
        }
    )


def test_prompt_normalizes_approved_key_spelling():
    assert_prompt_payload_safe({"ＲＯＵＮＤ_ＩＮＤＥＸ": 2})


@pytest.mark.parametrize(
    ("phase", "formal_frozen", "explicit_unlock"),
    (
        ("development", False, False),
        ("development", True, True),
        ("formal_evaluate", False, True),
        ("formal_evaluate", True, False),
        ("formal_evaluate", 1, True),
        ("formal_evaluate", True, 1),
    ),
)
def test_locked_test_rejects_until_all_three_unlock_conditions_are_exact(
    phase,
    formal_frozen,
    explicit_unlock,
):
    with pytest.raises(LockedTestError):
        require_test_unlock(
            phase=phase,
            formal_frozen=formal_frozen,
            explicit_unlock=explicit_unlock,
        )


def test_locked_test_unlocks_only_for_frozen_formal_evaluation():
    require_test_unlock(
        phase="formal_evaluate",
        formal_frozen=True,
        explicit_unlock=True,
    )
