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


def _diagnostics():
    return {
        "client_1": {
            "sample_size_weight": 0.5,
            "cosine_to_mean_update": 0.8,
            "update_norm": 1.0,
            "val_mape": 0.4,
            "val_mpe": -0.02,
        },
        "client_2": {
            "sample_size_weight": 0.5,
            "cosine_to_mean_update": 0.1,
            "update_norm": 1.0,
            "val_mape": 0.5,
            "val_mpe": 0.03,
        },
    }


def test_parse_generative_weight_response_normalizes_valid_json():
    planner = _planner(object())
    response = json.dumps({
        "aggregation_weights": {"client_1": 3, "client_2": 1},
        "server_lr_scale": 1.7,
        "control_action": "global_underfit_recovery",
        "anchor_l1_limit": 0.18,
        "decision_type": "coherence_driven",
        "reasoning": "client_1 is more coherent",
        "risk": "watch concentration",
    })

    decision = planner.parse_generative_weight_response(response, ["client_1", "client_2"])

    assert decision["fallback_used"] is False
    assert decision["aggregation_weights"]["client_1"] == pytest.approx(0.75)
    assert decision["aggregation_weights"]["client_2"] == pytest.approx(0.25)
    assert decision["server_lr_scale"] == 1.5
    assert decision["control_action"] == "global_underfit_recovery"
    assert decision["anchor_l1_limit"] == pytest.approx(0.18)
    assert decision["decision_type"] == "coherence_driven"


def test_parse_generative_weight_response_rejects_missing_or_unknown_clients():
    planner = _planner(object())

    missing = planner.parse_generative_weight_response(
        json.dumps({"aggregation_weights": {"client_1": 1.0}}),
        ["client_1", "client_2"],
    )
    unknown = planner.parse_generative_weight_response(
        json.dumps({"aggregation_weights": {"client_1": 0.5, "client_3": 0.5}}),
        ["client_1", "client_2"],
    )

    assert missing["fallback_used"] is True
    assert unknown["fallback_used"] is True


def test_choose_generated_weights_uses_llm_then_constraints():
    class CapturingClient:
        def __init__(self):
            self.prompt = ""

        def generate(self, prompt, **kwargs):
            self.prompt = prompt
            return json.dumps({
                "aggregation_weights": {"client_1": 0.7, "client_2": 0.3},
                "server_lr_scale": 1.5,
                "control_action": "coherence_shift",
                "anchor_l1_limit": 0.2,
                "decision_type": "coherence_driven",
                "reasoning": "client_1 has stronger update coherence",
                "risk": "monitor weight concentration",
            })

    client = CapturingClient()
    planner = _planner(client)
    decision = planner.choose_generated_weights(
        history_round_metrics=[],
        current_round=0,
        num_rounds=20,
        client_summaries={
            "client_1": {"summary_text": "Client 1: strong coherence."},
            "client_2": {"summary_text": "Client 2: weak coherence."},
        },
        coherence_diagnostics=_diagnostics(),
        previous_weights={"client_1": 0.5, "client_2": 0.5},
    )

    assert decision["fallback_used"] is False
    assert sum(decision["projected_weights"].values()) == pytest.approx(1.0)
    assert decision["server_lr_scale"] == pytest.approx(1.5)
    assert decision["constraint_status"]["anchor_l1_limit_used"] == pytest.approx(0.2)
    assert "client_summaries" in client.prompt
    assert "coherence_diagnostics" in client.prompt
    assert "test" not in client.prompt.lower()


def test_choose_generated_weights_does_not_snap_nonbalanced_control_to_size_prior():
    class ControlClient:
        def generate(self, prompt, **kwargs):
            return json.dumps({
                "aggregation_weights": {"client_1": 0.62, "client_2": 0.38},
                "server_lr_scale": 1.5,
                "control_action": "coherence_shift",
                "anchor_l1_limit": 0.25,
                "decision_type": "coherence_driven",
                "reasoning": "client_1 has stronger coherence and no drift warning",
                "risk": "bounded by anchor and previous-weight limits",
            })

    planner = _planner(ControlClient())
    decision = planner.choose_generated_weights(
        history_round_metrics=[],
        current_round=4,
        num_rounds=20,
        client_summaries={},
        coherence_diagnostics=_diagnostics(),
        previous_weights={"client_1": 0.5, "client_2": 0.5},
    )

    assert decision["projected_weights"]["client_1"] > 0.55
    assert decision["server_lr_scale"] == pytest.approx(1.5)
    assert decision["control_action"] == "coherence_shift"


def test_balanced_llm_decision_activates_global_underfit_recovery_when_evidence_is_clear():
    class BalancedClient:
        def generate(self, prompt, **kwargs):
            return json.dumps({
                "aggregation_weights": {"client_1": 0.5, "client_2": 0.5},
                "server_lr_scale": 1.0,
                "control_action": "balanced",
                "anchor_l1_limit": 0.05,
                "decision_type": "balanced",
                "reasoning": "weights are similar",
                "risk": "none",
            })

    diagnostics = {
        "client_1": {
            "sample_size_weight": 0.5,
            "cosine_to_mean_update": 0.9,
            "update_norm": 1.0,
            "val_mape": 0.95,
            "val_mpe": -0.82,
        },
        "client_2": {
            "sample_size_weight": 0.5,
            "cosine_to_mean_update": 0.8,
            "update_norm": 1.1,
            "val_mape": 0.93,
            "val_mpe": -0.79,
        },
    }
    planner = _planner(BalancedClient())
    decision = planner.choose_generated_weights(
        history_round_metrics=[],
        current_round=3,
        num_rounds=20,
        client_summaries={},
        coherence_diagnostics=diagnostics,
        previous_weights={"client_1": 0.5, "client_2": 0.5},
    )

    assert decision["control_action"] == "global_underfit_recovery"
    assert decision["decision_type"] == "stability_recovery"
    assert decision["server_lr_scale"] == pytest.approx(1.5)
    assert decision["constraint_status"]["control_activation"] == "global_underfit_recovery"


def test_balanced_llm_decision_not_activated_on_final_round():
    class BalancedClient:
        def generate(self, prompt, **kwargs):
            return json.dumps({
                "aggregation_weights": {"client_1": 0.5, "client_2": 0.5},
                "server_lr_scale": 1.0,
                "control_action": "balanced",
                "anchor_l1_limit": 0.05,
                "decision_type": "balanced",
                "reasoning": "final round should avoid step shock",
                "risk": "late change",
            })

    diagnostics = {
        "client_1": {"sample_size_weight": 0.5, "cosine_to_mean_update": 0.9, "update_norm": 1.0, "val_mape": 0.95, "val_mpe": -0.8},
        "client_2": {"sample_size_weight": 0.5, "cosine_to_mean_update": 0.8, "update_norm": 1.0, "val_mape": 0.92, "val_mpe": -0.7},
    }
    planner = _planner(BalancedClient())
    decision = planner.choose_generated_weights(
        history_round_metrics=[],
        current_round=19,
        num_rounds=20,
        client_summaries={},
        coherence_diagnostics=diagnostics,
        previous_weights={"client_1": 0.5, "client_2": 0.5},
    )

    assert decision["control_action"] == "balanced"
    assert decision["server_lr_scale"] == pytest.approx(1.0)


def test_choose_generated_weights_falls_back_to_coherence_on_api_failure():
    class FailingClient:
        def generate(self, *args, **kwargs):
            raise RuntimeError("api down")

    planner = _planner(FailingClient())
    decision = planner.choose_generated_weights(
        history_round_metrics=[],
        current_round=0,
        num_rounds=20,
        client_summaries={},
        coherence_diagnostics=_diagnostics(),
        previous_weights=None,
    )

    assert decision["fallback_used"] is True
    assert decision["fallback_source"] == "coherence_baseline"
    assert decision["projected_weights"]["client_1"] > decision["projected_weights"]["client_2"]


def test_generative_prompt_uses_robust_prior_instead_of_uniform_default():
    planner = _planner(object())

    prompt = planner.build_generative_weight_prompt(
        history_round_metrics=[],
        current_round=0,
        num_rounds=20,
        client_summaries={},
        coherence_diagnostics=_diagnostics(),
        previous_weights=None,
    )

    assert "coherence_prior_weights" in prompt
    assert "size_prior_weights" in prompt
    assert "robust_prior_weights" in prompt
    assert '"client_1": 0.503' in prompt
    assert '"client_2": 0.496' in prompt
    assert "Do not return exact uniform weights" in prompt
    assert "control_action" in prompt
    assert "global_underfit_recovery" in prompt
    assert "server_lr_scale" in prompt
    assert "validation overfitting" in prompt
    assert "size prior is the generalization anchor" in prompt
