import math

import torch


def _state(weight_value, *, counter=1):
    return {
        "linear.weight": torch.tensor([float(weight_value)], dtype=torch.float32),
        "counter": torch.tensor(counter, dtype=torch.long),
    }


def _metrics(samples=10, val_mape=0.5, val_mpe=0.0):
    return {"n_samples": samples, "val_mape": val_mape, "val_mpe": val_mpe}


def test_same_direction_updates_have_positive_mean_coherence():
    from src.federated_learning.coherence_diagnostics import compute_coherence_diagnostics

    diagnostics = compute_coherence_diagnostics(
        global_state=_state(0.0),
        client_states={"client_1": _state(1.0), "client_2": _state(2.0)},
        client_metrics={"client_1": _metrics(10), "client_2": _metrics(30)},
    )

    assert diagnostics["client_1"]["cosine_to_mean_update"] == 1.0
    assert diagnostics["client_2"]["cosine_to_mean_update"] == 1.0
    assert diagnostics["client_2"]["sample_size_weight"] == 0.75


def test_opposite_direction_updates_have_negative_pairwise_coherence():
    from src.federated_learning.coherence_diagnostics import compute_coherence_diagnostics

    diagnostics = compute_coherence_diagnostics(
        global_state=_state(0.0),
        client_states={"client_1": _state(1.0), "client_2": _state(-1.0)},
        client_metrics={"client_1": _metrics(), "client_2": _metrics()},
    )

    assert diagnostics["client_1"]["pairwise_mean_cosine"] == -1.0
    assert diagnostics["client_2"]["pairwise_mean_cosine"] == -1.0


def test_zero_update_and_non_float_state_do_not_create_nan():
    from src.federated_learning.coherence_diagnostics import compute_coherence_diagnostics

    diagnostics = compute_coherence_diagnostics(
        global_state=_state(0.0, counter=5),
        client_states={"client_1": _state(0.0, counter=99)},
        client_metrics={"client_1": _metrics(val_mape=0.42, val_mpe=-0.03)},
    )

    row = diagnostics["client_1"]
    assert row["update_norm"] == 0.0
    assert row["cosine_to_mean_update"] == 0.0
    assert row["cosine_to_previous_global_update"] == 0.0
    assert not any(math.isnan(float(value)) for value in row.values())
    assert row["val_mape"] == 0.42
    assert row["val_mpe"] == -0.03


def test_diagnostic_output_contains_required_fields():
    from src.federated_learning.coherence_diagnostics import compute_coherence_diagnostics

    diagnostics = compute_coherence_diagnostics(
        global_state=_state(0.0),
        client_states={"client_1": _state(1.0)},
        client_metrics={"client_1": _metrics()},
    )

    assert set(diagnostics["client_1"]) == {
        "update_norm",
        "cosine_to_mean_update",
        "pairwise_mean_cosine",
        "cosine_to_previous_global_update",
        "drift_from_mean_update",
        "sample_size_weight",
        "val_mape",
        "val_mpe",
    }
