import pytest

from src.federated_learning.pcv.client_evaluation import MetricSums
from src.federated_learning.pcv.protocol import (
    ClientDataVault,
    PrivacyViolation,
    TestPartitionLocked as LockedTestError,
    assert_prompt_payload_safe,
    require_test_unlock,
)
from src.federated_learning.pcv.schemas import ClientTelemetry, LocalCandidateVote


def _approved_payload():
    return {
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
    payload = _approved_payload()
    payload["round_index"] = unsafe_value
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe(payload)


def test_prompt_accepts_approved_aggregate_telemetry():
    assert_prompt_payload_safe(_approved_payload())


def test_prompt_normalizes_approved_key_spelling():
    assert_prompt_payload_safe(
        {
            "ＲＯＵＮＤ_ＩＮＤＥＸ": 2,
            "ＣＬＩＥＮＴＳ": [],
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("round_index", True),
        ("round_index", -1),
        ("round_index", 2.0),
        ("clients", ["client_01"]),
    ),
)
def test_prompt_rejects_wrong_root_field_types(field, value):
    payload = _approved_payload()
    payload[field] = value
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe(payload)


def test_prompt_rejects_normalized_root_key_collisions():
    payload = _approved_payload()
    payload["ＣＬＩＥＮＴＳ"] = []
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe(payload)


@pytest.mark.parametrize("missing_field", ("round_index", "clients"))
def test_prompt_rejects_missing_required_root_fields(missing_field):
    payload = _approved_payload()
    del payload[missing_field]
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe(payload)


@pytest.mark.parametrize(
    "missing_field",
    (
        "client_id",
        "sample_count",
        "train_loss",
        "val_mape",
        "val_rmse",
        "update_norm",
    ),
)
def test_prompt_rejects_missing_required_client_fields(missing_field):
    payload = _approved_payload()
    del payload["clients"][0][missing_field]
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("client_id", ""),
        ("client_id", 1),
        ("sample_count", True),
        ("sample_count", 0),
        ("sample_count", 100.0),
        ("sample_count", {"value": 100}),
        ("train_loss", [0.3]),
        ("val_mape", True),
        ("val_rmse", float("nan")),
        ("update_norm", float("inf")),
    ),
)
def test_prompt_rejects_wrong_client_field_types(field, value):
    payload = _approved_payload()
    payload["clients"][0][field] = value
    with pytest.raises(PrivacyViolation):
        assert_prompt_payload_safe(payload)


def test_prompt_allows_optional_cosine_fields_to_be_absent():
    payload = _approved_payload()
    del payload["clients"][0]["cosine_to_mean"]
    del payload["clients"][0]["cosine_to_previous"]
    assert_prompt_payload_safe(payload)


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


def _metric_sums(*, n=10, mape=0.2, rmse=2.0):
    return MetricSums(
        n=n,
        ape_sum=mape * n,
        se_sum=rmse * rmse * n,
        ae_sum=rmse * n,
        y_sum=0.0,
        y_sq_sum=0.0,
    )


def _telemetry(client_id="client_01"):
    return ClientTelemetry(
        client_id=client_id,
        sample_count=10,
        train_loss=0.2,
        val_mape=0.3,
        val_rmse=2.0,
        update_norm=1.0,
        cosine_to_mean=0.9,
        cosine_to_previous=0.8,
    )


def _vault(*, telemetry_fn=None, metric_sums_fn=None, calls=None):
    calls = [] if calls is None else calls
    train_dataset = object()
    controller_validation_dataset = object()
    locked_test_dataset = object()

    def train_fn(dataset, global_state, training_config, seed):
        calls.append(("train", dataset))
        return {"model_update": global_state, "seed": seed}

    def default_telemetry_fn(client_id, dataset, model_state):
        calls.append(("telemetry", dataset))
        return _telemetry(client_id)

    def default_metric_sums_fn(dataset, model_state):
        calls.append(("metric", dataset))
        return _metric_sums()

    vault = ClientDataVault(
        client_id="client_01",
        train_dataset=train_dataset,
        controller_validation_dataset=controller_validation_dataset,
        locked_test_dataset=locked_test_dataset,
        train_fn=train_fn,
        telemetry_fn=telemetry_fn or default_telemetry_fn,
        metric_sums_fn=metric_sums_fn or default_metric_sums_fn,
    )
    return vault, calls, train_dataset, controller_validation_dataset, locked_test_dataset


def test_client_data_vault_has_no_public_partition_or_tensor_accessors():
    vault, _, _, _, _ = _vault()

    public_names = {name for name in dir(vault) if not name.startswith("_")}
    assert public_names == {
        "client_id",
        "controller_telemetry",
        "evaluate_candidates",
        "final_test_sums",
        "train_local",
    }
    assert not hasattr(vault, "__dict__")
    for name in (
        "train_dataset",
        "controller_validation_dataset",
        "locked_test_dataset",
        "labels",
        "predictions",
        "tensors",
    ):
        with pytest.raises(AttributeError):
            getattr(vault, name)


def test_vault_routes_each_operation_to_only_its_private_partition():
    vault, calls, train_data, controller_data, test_data = _vault()

    vault.train_local({"weight": 1}, {"epochs": 1}, 7)
    telemetry = vault.controller_telemetry({"weight": 2})
    votes = vault.evaluate_candidates(
        {"anchor": {"mape": 0.2}, "candidate": {"mape": 0.2}},
        stronger_anchor_id="anchor",
    )
    sums = vault.final_test_sums(
        {"weight": 3},
        {
            "phase": "formal_evaluate",
            "formal_frozen": True,
            "explicit_unlock": True,
        },
    )

    assert calls == [
        ("train", train_data),
        ("telemetry", controller_data),
        ("metric", controller_data),
        ("metric", controller_data),
        ("metric", test_data),
    ]
    assert isinstance(telemetry, ClientTelemetry)
    assert all(isinstance(vote, LocalCandidateVote) for vote in votes)
    assert isinstance(sums, MetricSums)


def test_vault_candidate_ranking_is_stable_on_mape_ties():
    def metric_sums_fn(dataset, model_state):
        return _metric_sums(
            n=12,
            mape=model_state["mape"],
            rmse=model_state["rmse"],
        )

    vault, _, _, _, _ = _vault(metric_sums_fn=metric_sums_fn)
    votes = vault.evaluate_candidates(
        {
            "candidate_b": {"mape": 0.2, "rmse": 3.0},
            "anchor": {"mape": 0.3, "rmse": 2.0},
            "candidate_a": {"mape": 0.2, "rmse": 4.0},
        },
        stronger_anchor_id="anchor",
    )

    assert [(vote.candidate_id, vote.rank) for vote in votes] == [
        ("candidate_a", 1),
        ("candidate_b", 2),
        ("anchor", 3),
    ]
    assert all(vote.sample_count == 12 for vote in votes)
    assert all(vote.confidence == pytest.approx(1.0 / 3.0) for vote in votes)


class _DuplicateCandidateMapping(dict):
    def items(self):
        state = {"mape": 0.2, "rmse": 1.0}
        return [("duplicate", state), ("duplicate", state)]


@pytest.mark.parametrize(
    ("candidate_states", "anchor_id", "message"),
    [
        ({}, "anchor", "at least one"),
        ({"candidate": {}}, "anchor", "stronger anchor"),
        ({"": {}}, "", "candidate IDs"),
        (_DuplicateCandidateMapping(), "duplicate", "duplicate"),
    ],
)
def test_vault_rejects_empty_missing_anchor_or_invalid_candidate_ids(
    candidate_states,
    anchor_id,
    message,
):
    vault, _, _, _, _ = _vault()

    with pytest.raises((TypeError, ValueError), match=message):
        vault.evaluate_candidates(candidate_states, stronger_anchor_id=anchor_id)


def test_vault_rejects_candidate_sample_count_mismatch():
    def metric_sums_fn(dataset, model_state):
        return _metric_sums(n=model_state["n"])

    vault, _, _, _, _ = _vault(metric_sums_fn=metric_sums_fn)

    with pytest.raises(ValueError, match="sample_count"):
        vault.evaluate_candidates(
            {"anchor": {"n": 10}, "candidate": {"n": 11}},
            stronger_anchor_id="anchor",
        )


@pytest.mark.parametrize("callback_name", ["telemetry", "metric"])
def test_vault_rejects_malicious_callback_result_types(callback_name):
    kwargs = {}
    if callback_name == "telemetry":
        kwargs["telemetry_fn"] = lambda client_id, dataset, state: {
            "labels": [1.0],
            "predictions": [1.0],
        }
    else:
        kwargs["metric_sums_fn"] = lambda dataset, state: {
            "labels": [1.0],
            "predictions": [1.0],
        }
    vault, _, _, _, _ = _vault(**kwargs)

    with pytest.raises(TypeError):
        if callback_name == "telemetry":
            vault.controller_telemetry({})
        else:
            vault.evaluate_candidates({"anchor": {}}, stronger_anchor_id="anchor")


@pytest.mark.parametrize(
    "unlock_context",
    [
        {"phase": "development", "formal_frozen": True, "explicit_unlock": True},
        {"phase": "formal_evaluate", "formal_frozen": False, "explicit_unlock": True},
        {"phase": "formal_evaluate", "formal_frozen": True, "explicit_unlock": False},
    ],
)
def test_vault_final_test_requires_all_three_unlock_conditions(unlock_context):
    metric_calls = []

    def metric_sums_fn(dataset, state):
        metric_calls.append(dataset)
        return _metric_sums()

    vault, _, _, _, _ = _vault(metric_sums_fn=metric_sums_fn)

    with pytest.raises(LockedTestError):
        vault.final_test_sums({}, unlock_context)
    assert metric_calls == []


def test_vault_final_test_rejects_non_metric_sums_callback_result():
    vault, _, _, _, _ = _vault(
        metric_sums_fn=lambda dataset, state: {"tensor": object()}
    )

    with pytest.raises(TypeError, match="MetricSums"):
        vault.final_test_sums(
            {},
            {
                "phase": "formal_evaluate",
                "formal_frozen": True,
                "explicit_unlock": True,
            },
        )
