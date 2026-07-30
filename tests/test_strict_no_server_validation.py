import inspect


def test_strict_final_round_training_uses_client_reported_path_only():
    from src.federated_learning.mas_agents import CentralAgent

    source = inspect.getsource(CentralAgent.run_training_strict_final_round)

    assert "_apply_aggregated_round_client_reported" in source
    assert "build_candidate_validation_preview" not in source
    assert "build_continuous_candidate_preview" not in source
    assert "_evaluate_state_on_global_val" not in source
    assert "self.evaluate_global()" not in source


def test_strict_final_round_training_documents_final_round_checkpoint():
    from src.federated_learning.mas_agents import CentralAgent

    source = inspect.getsource(CentralAgent.run_training_strict_final_round)

    assert "final_round" in source
    assert "best checkpoint" not in source.lower()
