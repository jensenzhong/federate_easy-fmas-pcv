from dataclasses import FrozenInstanceError

import pytest

from src.federated_learning.pcv.schemas import (
    ALLOWED_CLIP_NORMS,
    ALLOWED_LR_SCALES,
    CandidateAction,
    CandidateDecision,
    ClientTelemetry,
    LocalCandidateVote,
)


CLIENT_IDS = ("client_01", "client_02", "client_03")


def _candidate(**overrides) -> CandidateAction:
    values = {
        "candidate_id": "performance_01",
        "weights": {
            "client_01": 0.3,
            "client_02": 0.4,
            "client_03": 0.3,
        },
        "server_optimizer": "fedyogi",
        "server_lr_scale": 1.0,
        "update_clip_norm": 1.0,
        "source": "performance_proposer",
        "rationale": "lower aggregate validation error",
    }
    values.update(overrides)
    return CandidateAction(**values)


def test_schema_records_are_frozen():
    records = (
        ClientTelemetry("client_01", 10, 0.2, 0.3, 1.0, 0.5, 0.9, 0.8),
        _candidate(),
        LocalCandidateVote(
            "client_01",
            "performance_01",
            10,
            0.3,
            1.0,
            1.0,
            1.0,
            1,
            0.8,
            False,
        ),
        CandidateDecision("performance_01", "performance_01", "accepted", "safe"),
    )

    for record in records:
        with pytest.raises(FrozenInstanceError):
            record.rationale = "mutated"


def test_client_telemetry_converts_to_prompt_dictionary():
    telemetry = ClientTelemetry(
        client_id="client_01",
        sample_count=10,
        train_loss=0.2,
        val_mape=0.3,
        val_rmse=1.0,
        update_norm=0.5,
        cosine_to_mean=0.9,
        cosine_to_previous=0.8,
    )

    assert telemetry.to_prompt_dict() == {
        "client_id": "client_01",
        "sample_count": 10,
        "train_loss": 0.2,
        "val_mape": 0.3,
        "val_rmse": 1.0,
        "update_norm": 0.5,
        "cosine_to_mean": 0.9,
        "cosine_to_previous": 0.8,
    }


def test_candidate_action_accepts_each_legal_action_value():
    for lr_scale in ALLOWED_LR_SCALES:
        for clip_norm in ALLOWED_CLIP_NORMS:
            _candidate(
                server_lr_scale=lr_scale,
                update_clip_norm=clip_norm,
            ).validate(CLIENT_IDS)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("server_optimizer", "unknown"),
        ("server_optimizer", "FedYogi"),
        ("server_lr_scale", 3.0),
        ("server_lr_scale", float("nan")),
        ("server_lr_scale", float("inf")),
        ("server_lr_scale", True),
        ("update_clip_norm", 7.0),
        ("update_clip_norm", float("nan")),
        ("update_clip_norm", float("inf")),
        ("update_clip_norm", True),
    ),
)
def test_candidate_action_rejects_illegal_or_non_finite_actions(field, value):
    with pytest.raises(ValueError):
        _candidate(**{field: value}).validate(CLIENT_IDS)


@pytest.mark.parametrize(
    "weights",
    (
        {"client_01": 0.5, "client_02": 0.5},
        {"client_01": 0.3, "client_02": 0.4, "other": 0.3},
        {"client_01": 0.95, "client_02": 0.03, "client_03": 0.02},
        {"client_01": 0.3, "client_02": 0.3, "client_03": 0.3},
        {"client_01": float("nan"), "client_02": 0.4, "client_03": 0.3},
        {"client_01": float("inf"), "client_02": 0.4, "client_03": 0.3},
        {"client_01": True, "client_02": 0.4, "client_03": -0.4},
        {"client_01": "0.3", "client_02": 0.4, "client_03": 0.3},
    ),
)
def test_candidate_action_rejects_illegal_weights(weights):
    with pytest.raises(ValueError):
        _candidate(weights=weights).validate(CLIENT_IDS)


@pytest.mark.parametrize(
    "client_ids",
    (
        (),
        ("", "client_02", "client_03"),
        ("client_01", "client_01", "client_03"),
    ),
)
def test_candidate_action_rejects_empty_or_duplicate_expected_client_ids(client_ids):
    with pytest.raises(ValueError):
        _candidate().validate(client_ids)


@pytest.mark.parametrize("field", ("candidate_id", "source", "rationale"))
def test_candidate_action_rejects_blank_identity_fields(field):
    with pytest.raises(ValueError):
        _candidate(**{field: "  "}).validate(CLIENT_IDS)


def test_candidate_weights_are_defensively_copied_and_immutable():
    source_weights = {
        "client_01": 0.3,
        "client_02": 0.4,
        "client_03": 0.3,
    }
    candidate = _candidate(weights=source_weights)
    candidate.validate(CLIENT_IDS)

    source_weights["client_01"] = 0.8
    assert candidate.weights["client_01"] == 0.3
    with pytest.raises(TypeError):
        candidate.weights["client_01"] = 0.8


def test_candidate_decision_diagnostics_are_defensively_copied_and_immutable():
    source_diagnostics = {"vote_margin": 0.2}
    decision = CandidateDecision(
        "performance_01",
        "performance_01",
        "accepted",
        "safe",
        diagnostics=source_diagnostics,
    )

    source_diagnostics["vote_margin"] = 0.9
    assert decision.diagnostics["vote_margin"] == 0.2
    with pytest.raises(TypeError):
        decision.diagnostics["vote_margin"] = 0.9
