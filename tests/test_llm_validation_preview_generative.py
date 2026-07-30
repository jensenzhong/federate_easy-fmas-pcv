import json
import tempfile

import pytest


def _planner(llm_client):
    from src.federated_learning.llm_planner import LLMPlanner

    tmp = tempfile.TemporaryDirectory()
    planner = LLMPlanner(
        config={"scene_c": {"llm": {"temperature": 0.0}, "strategies": [{"name": "size_only"}]}},
        llm_client=llm_client,
        log_dir=tmp.name,
    )
    planner._tmp = tmp
    return planner


def _candidate_preview():
    return {
        "size_anchor": {
            "source": "size_anchor",
            "weights": {"Client 1": 0.34, "Client 2": 0.34, "Client 3": 0.32},
            "server_lr_scale": 1.0,
            "score": 0.520,
            "client_gap": 0.20,
            "update_norm": 1.0,
            "validation_metrics": {"mape": 0.520, "rmse": 100.0, "mae": 80.0, "mpe": -0.10},
        },
        "candidate_best": {
            "source": "local_grid",
            "weights": {"Client 1": 0.29, "Client 2": 0.39, "Client 3": 0.32},
            "server_lr_scale": 1.0,
            "score": 0.500,
            "client_gap": 0.15,
            "update_norm": 1.1,
            "validation_metrics": {"mape": 0.500, "rmse": 98.0, "mae": 77.0, "mpe": -0.08},
        },
        "coherence_prior": {
            "source": "coherence_prior",
            "weights": {"Client 1": 0.38, "Client 2": 0.35, "Client 3": 0.27},
            "server_lr_scale": 1.0,
            "score": 0.501,
            "client_gap": 0.10,
            "update_norm": 0.9,
            "validation_metrics": {"mape": 0.501, "rmse": 96.0, "mae": 76.0, "mpe": -0.05},
        },
    }


def test_parse_validation_preview_generative_response_builds_mixture_weights():
    planner = _planner(object())
    decision = planner.parse_validation_preview_generative_response(
        json.dumps({
            "selected_candidate_ids": ["candidate_best", "coherence_prior"],
            "mixture_weights": {"candidate_best": 0.75, "coherence_prior": 0.25},
            "server_lr_scale": 0.75,
            "decision_type": "validation_improvement",
            "reasoning": "candidate_best has best MAPE; coherence_prior lowers RMSE and gap",
            "risk": "small validation gap",
        }),
        _candidate_preview(),
        score_tolerance=0.002,
    )

    assert decision["fallback_used"] is False
    assert decision["selected_candidate_ids"] == ["candidate_best", "coherence_prior"]
    assert decision["projected_weights"]["Client 1"] == pytest.approx(0.3125)
    assert decision["projected_weights"]["Client 2"] == pytest.approx(0.38)
    assert decision["projected_weights"]["Client 3"] == pytest.approx(0.3075)
    assert decision["server_lr_scale"] == pytest.approx(0.75)


def test_parse_validation_preview_generative_response_rejects_unknown_candidate():
    planner = _planner(object())

    decision = planner.parse_validation_preview_generative_response(
        json.dumps({
            "selected_candidate_ids": ["missing"],
            "mixture_weights": {"missing": 1.0},
            "server_lr_scale": 1.0,
            "decision_type": "validation_improvement",
        }),
        _candidate_preview(),
        score_tolerance=0.002,
    )

    assert decision["fallback_used"] is True
    assert decision["selected_candidate_ids"] == ["candidate_best"]
    assert decision["projected_weights"] == _candidate_preview()["candidate_best"]["weights"]


def test_choose_validation_preview_generative_strategy_uses_sanitized_prompt_and_logs():
    class CapturingClient:
        def __init__(self):
            self.prompt = ""

        def generate(self, prompt, **kwargs):
            self.prompt = prompt
            return json.dumps({
                "selected_candidate_ids": ["candidate_best", "coherence_prior"],
                "mixture_weights": {"candidate_best": 0.5, "coherence_prior": 0.5},
                "server_lr_scale": 1.0,
                "decision_type": "stability",
                "reasoning": "near-best mixture improves stability",
                "risk": "validation estimate variance",
            })

    client = CapturingClient()
    planner = _planner(client)

    decision = planner.choose_validation_preview_generative_strategy(
        history_round_metrics=[
            {
                "round": 0,
                "global_val": {"mape": 0.55},
                "test_hidden": {"true_value": [1], "predicted_value": [2]},
            }
        ],
        current_round=1,
        num_rounds=20,
        candidate_preview=_candidate_preview(),
        client_summaries={"Client 1": {"summary_text": "stable"}},
        coherence_diagnostics={"Client 1": {"cosine_to_mean_update": 0.8}},
        score_tolerance=0.002,
    )

    assert decision["fallback_used"] is False
    assert "candidate_preview" in client.prompt
    assert "client_summaries" in client.prompt
    assert "coherence_diagnostics" in client.prompt
    lowered = client.prompt.lower()
    assert "test_hidden" not in lowered
    assert "true_value" not in lowered
    assert "predicted_value" not in lowered
