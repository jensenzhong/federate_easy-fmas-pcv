from dataclasses import FrozenInstanceError
import math

import pytest

from src.federated_learning.pcv.gate import select_with_gate
from src.federated_learning.pcv.schemas import (
    CandidateAction,
    LocalCandidateVote,
)


CLIENT_IDS = ("c1", "c2", "c3")
PREVIOUS_WEIGHTS = {"c1": 0.3, "c2": 0.4, "c3": 0.3}


def _candidate(candidate_id, weights=None, **overrides):
    values = {
        "weights": PREVIOUS_WEIGHTS if weights is None else weights,
        "server_optimizer": "fedyogi",
        "server_lr_scale": 1.0,
        "update_clip_norm": 1.0,
        "source": "test",
        "rationale": "test",
    }
    values.update(overrides)
    return CandidateAction(candidate_id=candidate_id, **values)


def _vote(client_id, candidate_id, **overrides):
    values = {
        "sample_count": 10,
        "val_mape": 0.4,
        "val_rmse": 10.0,
        "relative_mape": 0.0,
        "relative_rmse": 0.0,
        "rank": 1,
        "confidence": 0.9,
        "catastrophic_degradation": False,
    }
    values.update(overrides)
    return LocalCandidateVote(
        client_id=client_id,
        candidate_id=candidate_id,
        **values,
    )


def _valid_inputs():
    candidates = {
        "anchor": _candidate("anchor"),
        "best": _candidate("best"),
        "proposal": _candidate("proposal"),
    }
    votes = [
        _vote(client_id, candidate_id)
        for candidate_id in candidates
        for client_id in CLIENT_IDS
    ]
    return {
        "requested_candidate_id": "proposal",
        "candidates": candidates,
        "votes": votes,
        "aggregate_mape": {
            "anchor": 0.400,
            "best": 0.399,
            "proposal": 0.400,
        },
        "previous_weights": PREVIOUS_WEIGHTS,
        "stronger_anchor_id": "anchor",
    }


def test_gate_accepts_candidate_that_passes_every_check():
    decision = select_with_gate(**_valid_inputs())

    assert decision.requested_candidate_id == "proposal"
    assert decision.selected_candidate_id == "proposal"
    assert decision.gate_status == "accepted"


def test_gate_rejects_missing_requested_candidate_first():
    inputs = _valid_inputs()
    inputs["requested_candidate_id"] = "missing"

    decision = select_with_gate(**inputs)

    assert decision.selected_candidate_id == "anchor"
    assert decision.gate_status == "rejected_missing_candidate"


def test_gate_rejects_illegal_requested_action_before_metric_checks():
    inputs = _valid_inputs()
    inputs["candidates"]["proposal"] = _candidate(
        "proposal",
        {"c1": 0.4, "c2": 0.4, "c3": 0.4},
    )
    inputs["votes"] = [
        _vote(
            vote.client_id,
            vote.candidate_id,
            catastrophic_degradation=vote.candidate_id == "proposal",
        )
        for vote in inputs["votes"]
    ]

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "rejected_illegal_action"


def test_gate_rejects_client_catastrophe_before_trust_and_score_checks():
    inputs = _valid_inputs()
    inputs["candidates"]["proposal"] = _candidate(
        "proposal",
        {"c1": 0.5, "c2": 0.2, "c3": 0.3},
    )
    inputs["votes"] = [
        _vote(
            vote.client_id,
            vote.candidate_id,
            catastrophic_degradation=(
                vote.candidate_id == "proposal" and vote.client_id == "c2"
            ),
        )
        for vote in inputs["votes"]
    ]
    inputs["aggregate_mape"]["proposal"] = 0.5

    decision = select_with_gate(**inputs)

    assert decision.selected_candidate_id == "anchor"
    assert decision.gate_status == "rejected_client_degradation"


def test_gate_rejects_l1_distance_above_trust_region_before_scores():
    inputs = _valid_inputs()
    inputs["candidates"]["proposal"] = _candidate(
        "proposal",
        {"c1": 0.5, "c2": 0.2, "c3": 0.3},
    )
    inputs["aggregate_mape"]["proposal"] = 0.5

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "rejected_trust_region"


def test_gate_rejects_candidate_not_within_best_legal_mape():
    inputs = _valid_inputs()
    inputs["aggregate_mape"].update(best=0.390, proposal=0.393)

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "rejected_not_near_best"


def test_gate_rejects_candidate_worse_than_stronger_anchor():
    inputs = _valid_inputs()
    inputs["aggregate_mape"].update(
        anchor=0.400,
        best=0.400,
        proposal=0.4011,
    )

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "rejected_anchor_degradation"


def test_each_rejection_reason_has_a_unique_status():
    statuses = {
        "rejected_missing_candidate",
        "rejected_illegal_action",
        "rejected_client_degradation",
        "rejected_trust_region",
        "rejected_not_near_best",
        "rejected_anchor_degradation",
    }

    assert len(statuses) == 6


def test_gate_accepts_exact_threshold_equalities():
    inputs = _valid_inputs()
    inputs["candidates"]["proposal"] = _candidate(
        "proposal",
        {"c1": 0.475, "c2": 0.225, "c3": 0.3},
    )
    inputs["aggregate_mape"].update(
        anchor=0.401,
        best=0.400,
        proposal=0.402,
    )

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "accepted"
    assert decision.diagnostics["l1_distance"] == pytest.approx(0.35)


def test_gate_rejects_smallest_representable_l1_excess():
    inputs = _valid_inputs()
    inputs["candidates"]["proposal"] = _candidate(
        "proposal",
        {
            "c1": 0.475,
            "c2": math.nextafter(0.225, -math.inf),
            "c3": 0.3,
        },
    )
    actual_l1 = math.fsum(
        abs(
            inputs["candidates"]["proposal"].weights[client_id]
            - PREVIOUS_WEIGHTS[client_id]
        )
        for client_id in CLIENT_IDS
    )
    assert actual_l1 == math.nextafter(0.35, math.inf)

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "rejected_trust_region"


def test_gate_rejects_smallest_representable_best_mape_excess():
    inputs = _valid_inputs()
    just_over_limit = math.nextafter(0.002, math.inf)
    inputs["aggregate_mape"].update(
        anchor=0.4,
        best=0.0,
        proposal=just_over_limit,
    )
    assert just_over_limit > inputs["aggregate_mape"]["best"] + 0.002

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "rejected_not_near_best"


def test_gate_rejects_smallest_representable_anchor_mape_excess():
    inputs = _valid_inputs()
    just_over_limit = math.nextafter(0.001, math.inf)
    inputs["aggregate_mape"].update(
        anchor=0.0,
        best=just_over_limit,
        proposal=just_over_limit,
    )
    assert just_over_limit > inputs["aggregate_mape"]["anchor"] + 0.001

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "rejected_anchor_degradation"


def test_gate_best_legal_tie_is_deterministic():
    inputs = _valid_inputs()
    inputs["candidates"] = {
        "proposal": _candidate("proposal"),
        "beta": _candidate("beta"),
        "anchor": _candidate("anchor"),
        "alpha": _candidate("alpha"),
    }
    inputs["votes"] = [
        _vote(client_id, candidate_id)
        for candidate_id in inputs["candidates"]
        for client_id in CLIENT_IDS
    ]
    inputs["aggregate_mape"] = {
        "proposal": 0.391,
        "beta": 0.390,
        "anchor": 0.391,
        "alpha": 0.390,
    }

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "accepted"
    assert decision.diagnostics["best_legal_candidate_id"] == "alpha"


def test_gate_accepts_requested_stronger_anchor_itself():
    inputs = _valid_inputs()
    inputs["requested_candidate_id"] = "anchor"
    inputs["aggregate_mape"].update(anchor=0.399, best=0.399)

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "accepted"
    assert decision.selected_candidate_id == "anchor"


def test_gate_diagnostics_are_candidate_decision_immutable_compatible():
    decision = select_with_gate(**_valid_inputs())

    assert decision.diagnostics["legal_candidate_ids"] == (
        "anchor",
        "best",
        "proposal",
    )
    with pytest.raises(TypeError):
        decision.diagnostics["l1_distance"] = 0.0
    with pytest.raises(FrozenInstanceError):
        decision.gate_status = "changed"


def test_gate_ignores_illegal_non_requested_candidate_when_finding_best():
    inputs = _valid_inputs()
    inputs["candidates"]["best"] = _candidate(
        "best",
        server_optimizer="invalid",
    )
    inputs["aggregate_mape"]["best"] = 0.1

    decision = select_with_gate(**inputs)

    assert decision.gate_status == "accepted"
    assert decision.diagnostics["best_legal_candidate_id"] in {
        "anchor",
        "proposal",
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda inputs: inputs.update(stronger_anchor_id="missing"),
        lambda inputs: inputs["candidates"].update(
            anchor=_candidate("anchor", server_optimizer="invalid")
        ),
    ],
)
def test_gate_requires_an_existing_legal_stronger_anchor(mutator):
    inputs = _valid_inputs()
    mutator(inputs)

    with pytest.raises(ValueError):
        select_with_gate(**inputs)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda inputs: inputs.update(votes=[]),
        lambda inputs: inputs["candidates"].update(
            wrong_key=inputs["candidates"].pop("best")
        ),
        lambda inputs: inputs["votes"].pop(),
        lambda inputs: inputs["votes"].append(_vote("c1", "unknown")),
        lambda inputs: inputs["aggregate_mape"].pop("best"),
        lambda inputs: inputs["aggregate_mape"].update(extra=0.4),
        lambda inputs: inputs["aggregate_mape"].update(best=float("nan")),
        lambda inputs: inputs["aggregate_mape"].update(best=float("inf")),
    ],
)
def test_gate_raises_infrastructure_errors_instead_of_returning_anchor(
    mutator,
):
    inputs = _valid_inputs()
    mutator(inputs)

    with pytest.raises((TypeError, ValueError)):
        select_with_gate(**inputs)


@pytest.mark.parametrize(
    "previous_weights",
    [
        {"c1": 0.3, "c2": 0.7},
        {"c1": 0.3, "c2": 0.4, "c3": float("nan")},
        {"c1": 0.3, "c2": 0.4, "c3": float("inf")},
        {"c1": 0.3, "c2": 0.4, "c3": 0.4},
        {"c1": 0.3, "c2": 0.4, "c3": True},
    ],
)
def test_gate_validates_previous_weights(previous_weights):
    inputs = _valid_inputs()
    inputs["previous_weights"] = previous_weights

    with pytest.raises((TypeError, ValueError)):
        select_with_gate(**inputs)


class _ValueErrorCandidate(CandidateAction):
    def validate(self, client_ids):
        raise ValueError("subclass schema error")


class _RuntimeErrorCandidate(CandidateAction):
    def validate(self, client_ids):
        raise RuntimeError("subclass runtime error")


@pytest.mark.parametrize(
    "client_mode",
    ["replaced", "missing", "extra"],
)
def test_gate_binds_every_candidate_vote_to_expected_clients(client_mode):
    inputs = _valid_inputs()
    if client_mode == "replaced":
        replacements = {"c1": "x", "c2": "y", "c3": "z"}
        inputs["votes"] = [
            _vote(replacements[vote.client_id], vote.candidate_id)
            for vote in inputs["votes"]
        ]
    elif client_mode == "missing":
        inputs["votes"] = [
            vote for vote in inputs["votes"] if vote.client_id != "c3"
        ]
    else:
        inputs["votes"].extend(
            _vote("c4", candidate_id)
            for candidate_id in inputs["candidates"]
        )

    with pytest.raises(
        ValueError,
        match="vote client IDs must exactly match previous_weights client IDs",
    ):
        select_with_gate(**inputs)


class _ClientId(str):
    pass


def test_gate_requires_exact_string_vote_client_ids():
    inputs = _valid_inputs()
    inputs["votes"] = [
        _vote(
            _ClientId(vote.client_id)
            if vote.client_id == "c1"
            else vote.client_id,
            vote.candidate_id,
        )
        for vote in inputs["votes"]
    ]

    with pytest.raises((TypeError, ValueError)):
        select_with_gate(**inputs)


@pytest.mark.parametrize(
    "candidate_type",
    [_ValueErrorCandidate, _RuntimeErrorCandidate],
)
@pytest.mark.parametrize(
    "candidate_id",
    ["anchor", "best", "proposal"],
)
def test_gate_rejects_action_subclasses_before_any_validation_catch(
    candidate_type,
    candidate_id,
):
    inputs = _valid_inputs()
    inputs["candidates"][candidate_id] = candidate_type(
        candidate_id=candidate_id,
        weights=PREVIOUS_WEIGHTS,
        server_optimizer="fedyogi",
        server_lr_scale=1.0,
        update_clip_norm=1.0,
        source="test",
        rationale="test",
    )

    with pytest.raises(TypeError, match="exact CandidateAction"):
        select_with_gate(**inputs)
