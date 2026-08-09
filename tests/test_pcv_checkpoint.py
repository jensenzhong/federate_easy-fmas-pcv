import copy
import random

import numpy as np
import pytest
import torch

from src.federated_learning.pcv.checkpoint import (
    CheckpointFormatError,
    CheckpointRestoreError,
    ResumeApprovalRequired,
    ResumeMismatchError,
    build_checkpoint_payload,
    capture_rng_state,
    load_checkpoint,
    restore_checkpoint,
    restore_rng_state,
    save_checkpoint,
    validate_resume,
)


class _Optimizer:
    def __init__(self, state=None):
        self.state = state or {"name": "test", "momentum": torch.tensor([2.0])}

    def get_optimizer_state(self):
        return self.state

    def load_optimizer_state(self, state):
        self.state = state


def _payload(**overrides):
    model = overrides.pop("model", torch.nn.Linear(2, 1))
    optimizer = overrides.pop("server_optimizer", _Optimizer())
    values = {
        "round_index": 3,
        "freeze_id": "freeze-a",
        "method": "fmas",
        "training_seed": 17,
        "llm_rep": 2,
        "model": model,
        "server_optimizer": optimizer,
        "previous_weights": {"client-a": 0.6, "client-b": 0.4},
        "best_validation": {"mape": 0.25, "round": 2},
        "best_model_state": model.state_dict(),
        "partition_sha256": "partition-hash",
        "config_sha256": "config-hash",
        "prompt_hashes": {"planner": "prompt-a", "reviewer": "prompt-b"},
    }
    values.update(overrides)
    return build_checkpoint_payload(**values)


def _resume_kwargs(**overrides):
    values = {
        "user_approved_resume": True,
        "requested_freeze_id": "freeze-a",
        "requested_method": "fmas",
        "requested_training_seed": 17,
        "requested_llm_rep": 2,
        "requested_partition_sha256": "partition-hash",
        "requested_config_sha256": "config-hash",
        "requested_prompt_hashes": {"planner": "prompt-a", "reviewer": "prompt-b"},
    }
    values.update(overrides)
    return values


def _assert_rng_states_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy"][0] == right["numpy"][0]
    assert np.array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])
    if left["torch_cuda"] is None or right["torch_cuda"] is None:
        assert left["torch_cuda"] is right["torch_cuda"]
    else:
        assert len(left["torch_cuda"]) == len(right["torch_cuda"])
        assert all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))


def test_rng_state_round_trip_is_exact():
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(1).item())
    assert actual == expected


def test_rng_capture_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def unexpected_cuda_access():
        raise AssertionError("CUDA RNG access would initialize CUDA")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", unexpected_cuda_access)
    assert capture_rng_state()["torch_cuda"] is None


def test_rng_state_is_detached_from_mutable_runtime_state():
    state = capture_rng_state()
    cpu_copy = state["torch_cpu"].clone()
    numpy_copy = state["numpy"][1].copy()
    random.random()
    np.random.rand()
    torch.rand(1)
    assert torch.equal(state["torch_cpu"], cpu_copy)
    assert np.array_equal(state["numpy"][1], numpy_copy)


def test_resume_requires_explicit_user_approval_flag():
    with pytest.raises(ResumeApprovalRequired):
        validate_resume(
            checkpoint_freeze_id="freeze-a",
            requested_freeze_id="freeze-a",
            user_approved_resume=False,
        )


def test_legacy_resume_check_accepts_only_an_explicit_true():
    with pytest.raises(ResumeApprovalRequired):
        validate_resume(
            checkpoint_freeze_id="freeze-a",
            requested_freeze_id="freeze-a",
            user_approved_resume=1,
        )
    assert (
        validate_resume(
            checkpoint_freeze_id="freeze-a",
            requested_freeze_id="freeze-a",
            user_approved_resume=True,
        )
        is None
    )


def test_payload_has_complete_exact_schema_and_clones_all_mutable_state():
    class TensorSubclass(torch.Tensor):
        pass

    source = torch.arange(2.0)
    subclass = torch.Tensor._make_subclass(TensorSubclass, source, False)
    model = torch.nn.Linear(2, 1)
    optimizer = _Optimizer({"name": "test", "nested": {"tensor": subclass}})
    previous_weights = {"client-a": subclass}
    best_validation = {"metric": [subclass]}
    best_model_state = {"weight": subclass}
    prompt_hashes = {"planner": "prompt-a"}

    payload = _payload(
        model=model,
        server_optimizer=optimizer,
        previous_weights=previous_weights,
        best_validation=best_validation,
        best_model_state=best_model_state,
        prompt_hashes=prompt_hashes,
    )

    assert set(payload) == {
        "schema_version",
        "last_complete_round",
        "freeze_id",
        "method",
        "training_seed",
        "llm_rep",
        "global_model_state",
        "server_optimizer_state",
        "previous_weights",
        "best_validation",
        "best_model_state",
        "rng_state",
        "partition_sha256",
        "config_sha256",
        "prompt_hashes",
    }
    assert payload["schema_version"] == 1
    assert payload["last_complete_round"] == 3

    cloned_tensors = (
        payload["server_optimizer_state"]["nested"]["tensor"],
        payload["previous_weights"]["client-a"],
        payload["best_validation"]["metric"][0],
        payload["best_model_state"]["weight"],
    )
    assert all(type(value) is torch.Tensor for value in cloned_tensors)
    assert all(value.data_ptr() != source.data_ptr() for value in cloned_tensors)
    assert len({value.data_ptr() for value in cloned_tensors}) == len(cloned_tensors)
    assert all(
        saved.data_ptr() != model.state_dict()[name].data_ptr()
        for name, saved in payload["global_model_state"].items()
    )

    with torch.no_grad():
        model.weight.add_(10)
    source.add_(10)
    optimizer.state["new"] = "mutation"
    previous_weights["new"] = 1.0
    best_validation["new"] = 1.0
    best_model_state["new"] = source
    prompt_hashes["new"] = "mutation"
    assert "new" not in payload["server_optimizer_state"]
    assert "new" not in payload["previous_weights"]
    assert "new" not in payload["best_validation"]
    assert "new" not in payload["best_model_state"]
    assert "new" not in payload["prompt_hashes"]
    assert not torch.equal(payload["global_model_state"]["weight"], model.weight)
    assert all(torch.equal(value, torch.arange(2.0)) for value in cloned_tensors)


@pytest.mark.parametrize("bad_round", [-1, True, 1.5, "1"])
def test_payload_rejects_invalid_round(bad_round):
    with pytest.raises(CheckpointFormatError):
        _payload(round_index=bad_round)


def test_save_and_load_checkpoint_atomically(tmp_path):
    payload = _payload()
    target = save_checkpoint(tmp_path, payload)
    assert target == tmp_path / "last_complete.pt"
    assert target.is_file()
    assert not list(tmp_path.glob(".last_complete.pt.*.tmp"))

    loaded = load_checkpoint(target)
    assert loaded["last_complete_round"] == 3
    assert loaded["prompt_hashes"] == payload["prompt_hashes"]
    assert torch.equal(
        loaded["global_model_state"]["weight"],
        payload["global_model_state"]["weight"],
    )
    assert loaded["global_model_state"]["weight"].device.type == "cpu"


def test_atomic_replace_failure_preserves_old_checkpoint_and_cleans_temp(tmp_path, monkeypatch):
    import src.federated_learning.pcv.checkpoint as checkpoint_module

    target = tmp_path / "last_complete.pt"
    target.write_bytes(b"old checkpoint")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_checkpoint(tmp_path, _payload())
    assert target.read_bytes() == b"old checkpoint"
    assert not list(tmp_path.glob(".last_complete.pt.*.tmp"))


def test_torch_save_failure_preserves_old_checkpoint_and_cleans_temp(tmp_path, monkeypatch):
    import src.federated_learning.pcv.checkpoint as checkpoint_module

    target = tmp_path / "last_complete.pt"
    target.write_bytes(b"old checkpoint")

    def fail_save(payload, handle):
        handle.write(b"partial")
        raise RuntimeError("save failed")

    monkeypatch.setattr(checkpoint_module.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="save failed"):
        save_checkpoint(tmp_path, _payload())
    assert target.read_bytes() == b"old checkpoint"
    assert not list(tmp_path.glob(".last_complete.pt.*.tmp"))


@pytest.mark.parametrize("contents", [b"not a checkpoint", b"PK\x03\x04truncated"])
def test_load_rejects_corrupt_or_truncated_checkpoint(tmp_path, contents):
    path = tmp_path / "broken.pt"
    path.write_bytes(contents)
    with pytest.raises(CheckpointFormatError):
        load_checkpoint(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("method"), "fields"),
        (lambda row: row.__setitem__("extra", 1), "fields"),
        (lambda row: row.__setitem__("schema_version", 2), "schema_version"),
        (lambda row: row.__setitem__("last_complete_round", -1), "last_complete_round"),
        (lambda row: row.__setitem__("last_complete_round", True), "last_complete_round"),
        (lambda row: row.__setitem__("training_seed", "17"), "training_seed"),
        (lambda row: row.__setitem__("prompt_hashes", {"planner": 1}), "prompt_hashes"),
        (lambda row: row.__setitem__("global_model_state", []), "global_model_state"),
    ],
)
def test_load_rejects_wrong_schema_fields_and_types(tmp_path, mutation, message):
    payload = _payload()
    mutation(payload)
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(CheckpointFormatError, match=message):
        load_checkpoint(path)


def test_load_uses_cpu_map_location_and_explicit_current_weights_only_semantics(tmp_path, monkeypatch):
    import src.federated_learning.pcv.checkpoint as checkpoint_module

    path = save_checkpoint(tmp_path, _payload())
    original_load = torch.load
    observed = {}

    def recording_load(*args, **kwargs):
        observed.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.torch, "load", recording_load)
    load_checkpoint(path)
    assert torch.device(observed["map_location"]).type == "cpu"
    assert observed["weights_only"] is False


def test_validate_resume_requires_an_existing_path_and_returns_next_round(tmp_path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        validate_resume(resume_checkpoint=missing, **_resume_kwargs())

    target = save_checkpoint(tmp_path, _payload())
    assert validate_resume(resume_checkpoint=target, **_resume_kwargs()) == 4


@pytest.mark.parametrize(
    ("requested_field", "bad_value"),
    [
        ("requested_freeze_id", "freeze-b"),
        ("requested_method", "fedavg"),
        ("requested_training_seed", 18),
        ("requested_llm_rep", 3),
        ("requested_partition_sha256", "other-partition"),
        ("requested_config_sha256", "other-config"),
        ("requested_prompt_hashes", {"planner": "prompt-a"}),
        (
            "requested_prompt_hashes",
            {"planner": "prompt-a", "reviewer": "prompt-b", "extra": "prompt-c"},
        ),
    ],
)
def test_validate_resume_requires_exact_metadata_and_hash_mapping(requested_field, bad_value):
    kwargs = _resume_kwargs(**{requested_field: bad_value})
    with pytest.raises(ResumeMismatchError):
        validate_resume(checkpoint=_payload(), **kwargs)


def test_validate_resume_requires_all_expected_metadata():
    kwargs = _resume_kwargs()
    del kwargs["requested_config_sha256"]
    with pytest.raises(CheckpointFormatError, match="requested_config_sha256"):
        validate_resume(checkpoint=_payload(), **kwargs)


def test_validate_resume_rejects_ambiguous_or_mixed_input_modes(tmp_path):
    payload = _payload()
    target = save_checkpoint(tmp_path, payload)
    with pytest.raises(CheckpointFormatError, match="exactly one"):
        validate_resume(checkpoint=payload, resume_checkpoint=target, **_resume_kwargs())
    with pytest.raises(CheckpointFormatError, match="legacy"):
        validate_resume(
            checkpoint=payload,
            checkpoint_freeze_id="freeze-a",
            **_resume_kwargs(),
        )


def test_mismatch_fails_before_model_optimizer_or_rng_are_modified(monkeypatch):
    import src.federated_learning.pcv.checkpoint as checkpoint_module

    saved_model = torch.nn.Linear(2, 1)
    payload = _payload(model=saved_model)
    active_model = torch.nn.Linear(2, 1)
    active_optimizer = _Optimizer({"name": "active", "momentum": torch.tensor([9.0])})
    model_before = copy.deepcopy(active_model.state_dict())
    optimizer_before = copy.deepcopy(active_optimizer.state)
    rng_before = capture_rng_state()
    monkeypatch.setattr(
        checkpoint_module,
        "restore_rng_state",
        lambda state: (_ for _ in ()).throw(AssertionError("RNG restoration must not run")),
    )

    with pytest.raises(ResumeMismatchError):
        restore_checkpoint(
            checkpoint=payload,
            model=active_model,
            server_optimizer=active_optimizer,
            **_resume_kwargs(requested_method="wrong"),
        )

    assert all(torch.equal(active_model.state_dict()[key], value) for key, value in model_before.items())
    assert active_optimizer.state["name"] == optimizer_before["name"]
    assert torch.equal(active_optimizer.state["momentum"], optimizer_before["momentum"])
    _assert_rng_states_equal(capture_rng_state(), rng_before)


def test_restore_checkpoint_restores_model_optimizer_and_rng_exactly():
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    saved_model = torch.nn.Linear(2, 1)
    payload = _payload(model=saved_model)
    expected_random = (random.random(), np.random.rand(), torch.rand(1).item())

    active_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        active_model.weight.fill_(100)
        active_model.bias.fill_(100)
    active_optimizer = _Optimizer({"name": "active", "momentum": torch.tensor([9.0])})
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    start_round = restore_checkpoint(
        checkpoint=payload,
        model=active_model,
        server_optimizer=active_optimizer,
        **_resume_kwargs(),
    )

    assert start_round == 4
    assert all(
        torch.equal(active_model.state_dict()[key], value)
        for key, value in payload["global_model_state"].items()
    )
    assert active_optimizer.state["name"] == "test"
    assert torch.equal(active_optimizer.state["momentum"], torch.tensor([2.0]))
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected_random


def test_rng_restore_failure_rolls_back_model_optimizer_and_rng(monkeypatch):
    import src.federated_learning.pcv.checkpoint as checkpoint_module

    payload = _payload()
    active_model = torch.nn.Linear(2, 1)
    active_optimizer = _Optimizer({"name": "active", "momentum": torch.tensor([9.0])})
    model_before = copy.deepcopy(active_model.state_dict())
    optimizer_before = copy.deepcopy(active_optimizer.state)
    rng_before = capture_rng_state()
    real_restore = checkpoint_module.restore_rng_state

    def fail_only_checkpoint_state(state):
        if state is payload["rng_state"]:
            raise RuntimeError("rng restore failed")
        real_restore(state)

    monkeypatch.setattr(checkpoint_module, "restore_rng_state", fail_only_checkpoint_state)
    with pytest.raises(CheckpointRestoreError, match="rng restore failed"):
        restore_checkpoint(
            checkpoint=payload,
            model=active_model,
            server_optimizer=active_optimizer,
            **_resume_kwargs(),
        )

    assert all(torch.equal(active_model.state_dict()[key], value) for key, value in model_before.items())
    assert active_optimizer.state["name"] == optimizer_before["name"]
    assert torch.equal(active_optimizer.state["momentum"], optimizer_before["momentum"])
    _assert_rng_states_equal(capture_rng_state(), rng_before)
