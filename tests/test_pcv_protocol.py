from dataclasses import FrozenInstanceError

import pytest
import torch

from src.federated_learning.pcv.client_evaluation import MetricSums
from src.federated_learning.pcv.protocol import (
    ClientDataVault,
    LocalTrainingResult,
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


def _vault(*, train_fn=None, telemetry_fn=None, metric_sums_fn=None, calls=None):
    calls = [] if calls is None else calls
    train_dataset = object()
    controller_validation_dataset = object()
    locked_test_dataset = object()

    def default_train_fn(dataset, global_state, training_config, seed):
        calls.append(("train", dataset))
        return LocalTrainingResult(
            model_state=global_state,
            sample_count=10,
            train_loss=0.2,
        )

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
        train_fn=train_fn or default_train_fn,
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

    training = vault.train_local(
        {"weight": torch.tensor([1.0])},
        {"epochs": 1},
        7,
    )
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
    assert type(training) is LocalTrainingResult
    assert type(telemetry) is ClientTelemetry
    assert all(type(vote) is LocalCandidateVote for vote in votes)
    assert type(sums) is MetricSums


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


def test_local_training_result_deep_clones_tensors_and_freezes_mapping():
    source_tensor = torch.tensor([1.0])
    source_state = {"weight": source_tensor}

    result = LocalTrainingResult(
        model_state=source_state,
        sample_count=5,
        train_loss=0.25,
    )
    source_tensor.add_(10.0)
    source_state["extra"] = torch.tensor([2.0])

    assert list(result.model_state) == ["weight"]
    assert torch.equal(result.model_state["weight"], torch.tensor([1.0]))
    assert result.model_state["weight"] is not source_tensor
    with pytest.raises(TypeError):
        result.model_state["extra"] = torch.tensor([2.0])
    with pytest.raises(FrozenInstanceError):
        result.sample_count = 6


class _CustomStateDict(dict):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_state": []},
        {"model_state": _CustomStateDict(weight=torch.tensor([1.0]))},
        {"model_state": {"weight": 1.0}},
        {"model_state": {1: torch.tensor([1.0])}},
        {"sample_count": 0},
        {"sample_count": True},
        {"train_loss": float("inf")},
        {"train_loss": torch.tensor(0.2)},
    ],
)
def test_local_training_result_rejects_custom_or_invalid_payload(kwargs):
    values = {
        "model_state": {"weight": torch.tensor([1.0])},
        "sample_count": 5,
        "train_loss": 0.25,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        LocalTrainingResult(**values)


def test_local_training_result_rejects_extra_payload_fields():
    with pytest.raises(TypeError):
        LocalTrainingResult(
            model_state={"weight": torch.tensor([1.0])},
            sample_count=5,
            train_loss=0.25,
            labels=[1.0],
        )


def test_train_local_requires_exact_dto_and_reclones_callback_tensors():
    callback_result = LocalTrainingResult(
        model_state={"weight": torch.tensor([1.0])},
        sample_count=5,
        train_loss=0.25,
    )
    vault, _, _, _, _ = _vault(train_fn=lambda *args: callback_result)

    returned = vault.train_local(
        {"weight": torch.tensor([0.0])},
        {"epochs": 1},
        3,
    )
    callback_result.model_state["weight"].add_(10.0)

    assert type(returned) is LocalTrainingResult
    assert returned is not callback_result
    assert returned.model_state is not callback_result.model_state
    assert torch.equal(returned.model_state["weight"], torch.tensor([1.0]))


class _AliasedTrainingResult(LocalTrainingResult):
    leaked_dataset = object()


class _AliasedTelemetry(ClientTelemetry):
    leaked_dataset = object()


class _AliasedMetricSums(MetricSums):
    leaked_test_dataset = object()


def test_vault_rejects_training_result_subclass_with_hidden_alias():
    result = _AliasedTrainingResult(
        model_state={"weight": torch.tensor([1.0])},
        sample_count=5,
        train_loss=0.25,
    )
    vault, _, _, _, _ = _vault(train_fn=lambda *args: result)

    with pytest.raises(TypeError, match="exact LocalTrainingResult"):
        vault.train_local({"weight": torch.tensor([0.0])}, {}, 1)


def test_vault_rejects_telemetry_subclass_with_hidden_alias():
    telemetry = _AliasedTelemetry(**vars(_telemetry()))
    vault, _, _, _, _ = _vault(telemetry_fn=lambda *args: telemetry)

    with pytest.raises(TypeError, match="exact ClientTelemetry"):
        vault.controller_telemetry({})


def test_vault_rejects_metric_sums_subclass_with_hidden_test_dataset():
    sums = _AliasedMetricSums(
        n=10,
        ape_sum=2.0,
        se_sum=40.0,
        ae_sum=20.0,
        y_sum=0.0,
        y_sq_sum=0.0,
    )
    vault, _, _, _, _ = _vault(metric_sums_fn=lambda *args: sums)

    with pytest.raises(TypeError, match="exact MetricSums"):
        vault.final_test_sums(
            {},
            {
                "phase": "formal_evaluate",
                "formal_frozen": True,
                "explicit_unlock": True,
            },
        )


@pytest.mark.parametrize("callback_name", ["train", "telemetry", "metric"])
def test_vault_rejects_callback_returning_private_dataset_directly(callback_name):
    kwargs = {}
    if callback_name == "train":
        kwargs["train_fn"] = lambda dataset, *args: dataset
    elif callback_name == "telemetry":
        kwargs["telemetry_fn"] = lambda client_id, dataset, state: dataset
    else:
        kwargs["metric_sums_fn"] = lambda dataset, state: dataset
    vault, _, _, _, _ = _vault(**kwargs)

    with pytest.raises(TypeError):
        if callback_name == "train":
            vault.train_local({"weight": torch.tensor([0.0])}, {}, 1)
        elif callback_name == "telemetry":
            vault.controller_telemetry({})
        else:
            vault.evaluate_candidates({"anchor": {}}, stronger_anchor_id="anchor")


def test_controller_telemetry_rebuilds_exact_validated_value():
    callback_result = _telemetry()
    vault, _, _, _, _ = _vault(telemetry_fn=lambda *args: callback_result)

    returned = vault.controller_telemetry({})

    assert type(returned) is ClientTelemetry
    assert returned == callback_result
    assert returned is not callback_result


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_count": 0},
        {"sample_count": True},
        {"train_loss": float("nan")},
        {"val_mape": float("inf")},
        {"val_rmse": float("nan")},
        {"update_norm": float("inf")},
        {"cosine_to_mean": float("nan")},
        {"cosine_to_previous": float("inf")},
    ],
)
def test_controller_telemetry_rejects_invalid_scalar_fields(overrides):
    values = vars(_telemetry()).copy()
    values.update(overrides)
    callback_result = ClientTelemetry(**values)
    vault, _, _, _, _ = _vault(telemetry_fn=lambda *args: callback_result)

    with pytest.raises((TypeError, ValueError)):
        vault.controller_telemetry({})


def test_final_test_sums_rebuilds_exact_metric_sums_value():
    callback_result = _metric_sums()
    vault, _, _, _, _ = _vault(metric_sums_fn=lambda *args: callback_result)

    returned = vault.final_test_sums(
        {},
        {
            "phase": "formal_evaluate",
            "formal_frozen": True,
            "explicit_unlock": True,
        },
    )

    assert type(returned) is MetricSums
    assert returned == callback_result
    assert returned is not callback_result


class _AliasedFloat(float):
    leaked_dataset = object()


def test_final_test_sums_normalizes_custom_numeric_fields_to_plain_float():
    callback_result = MetricSums(
        n=10,
        ape_sum=_AliasedFloat(2.0),
        se_sum=40.0,
        ae_sum=20.0,
        y_sum=0.0,
        y_sq_sum=0.0,
    )
    vault, _, _, _, _ = _vault(metric_sums_fn=lambda *args: callback_result)

    returned = vault.final_test_sums(
        {},
        {
            "phase": "formal_evaluate",
            "formal_frozen": True,
            "explicit_unlock": True,
        },
    )

    assert type(returned.ape_sum) is float
