import json
import random

import numpy as np
import pytest
import torch

from src.federated_learning.pcv import agents as agents_module
from src.federated_learning.pcv.agents import DeepSeekCallError
from src.federated_learning.pcv.checkpoint import capture_rng_state
from src.federated_learning.pcv.engine import (
    ExperimentPaused,
    ExperimentRuntimeError,
    PCVEngine,
    RoundResult,
)
from src.federated_learning.pcv.schemas import (
    CandidateAction,
    CandidateDecision,
    ClientTelemetry,
    LocalCandidateVote,
)
from src.federated_learning.pcv.telemetry import AppendOnlyTelemetry
from src.federated_learning.server_optimizers import FedYogiServerOptimizer


CLIENTS = ("client_01", "client_02", "client_03")
COUNTS = {"client_01": 2, "client_02": 3, "client_03": 5}


def _state_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        if isinstance(left[key], dict):
            _state_equal(left[key], right[key])
        elif isinstance(left[key], torch.Tensor):
            assert type(left[key]) is torch.Tensor
            assert torch.equal(left[key], right[key])
        else:
            assert left[key] == right[key]


def _rng_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy"]["bit_generator"] == right["numpy"]["bit_generator"]
    assert torch.equal(left["numpy"]["keys"], right["numpy"]["keys"])
    assert left["numpy"]["position"] == right["numpy"]["position"]
    assert left["numpy"]["has_gauss"] == right["numpy"]["has_gauss"]
    assert left["numpy"]["cached_gaussian"] == right["numpy"]["cached_gaussian"]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])
    assert left["cuda_initialized"] is right["cuda_initialized"]
    assert left["cuda_device_count"] == right["cuda_device_count"]
    if left["torch_cuda"] is None:
        assert right["torch_cuda"] is None
    else:
        assert all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))


def _proposal(candidate_id, weights, source="agent"):
    return CandidateAction(
        candidate_id=candidate_id,
        weights=weights,
        server_optimizer="fedyogi",
        server_lr_scale=0.75,
        update_clip_norm=0.5,
        source=source,
        rationale="fake deterministic response",
    )


class FakeAgent:
    def __init__(self, fail_role=None):
        self.roles = []
        self.payloads = []
        self.fail_role = fail_role

    def __call__(self, *, role, payload):
        self.roles.append(role)
        self.payloads.append((role, payload))
        if role == self.fail_role:
            raise DeepSeekCallError("connection", role, "Bearer never-log-me")
        if role == "diagnostic":
            return {
                "state_summary": "fake",
                "risks": ["none"],
                "priorities": ["validation"],
            }
        if role == "performance_proposer":
            return [_proposal("performance_01", {"client_01": 0.21, "client_02": 0.29, "client_03": 0.50})]
        if role == "stability_proposer":
            return [_proposal("stability_01", {"client_01": 0.19, "client_02": 0.31, "client_03": 0.50})]
        if role == "balance_proposer":
            return [_proposal("balance_01", {"client_01": 0.20, "client_02": 0.31, "client_03": 0.49})]
        if role == "critic":
            return {
                "accepted_candidate_ids": [
                    candidate["candidate_id"] for candidate in payload["candidates"]
                ],
                "rejected": [],
            }
        if role == "single_proposer":
            return {
                "diagnostic": {
                    "state_summary": "single fake",
                    "risks": ["none"],
                    "priorities": ["validation"],
                },
                "proposals": [
                    _proposal("single_01", {"client_01": 0.21, "client_02": 0.30, "client_03": 0.49})
                ],
            }
        if role == "coordinator":
            return {"requested_candidate_id": "anchor_fedyogi"}
        raise AssertionError(role)


def _train_clients(*, round_index, global_state, training_config, seed):
    assert training_config == {"local_epochs": 2}
    assert seed == 40 + round_index
    return {
        client_id: {
            key: value.detach().clone() + float(index)
            for key, value in global_state.items()
        }
        for index, client_id in enumerate(CLIENTS, start=1)
    }


def _collect_telemetry(*, round_index, global_state, local_updates):
    del round_index, global_state, local_updates
    return {
        client_id: ClientTelemetry(
            client_id=client_id,
            sample_count=COUNTS[client_id],
            train_loss=0.2 + index / 100,
            val_mape=0.3 + index / 100,
            val_rmse=1.0 + index,
            update_norm=float(index),
            cosine_to_mean=0.9 - index / 10,
            cosine_to_previous=0.8 - index / 10,
        )
        for index, client_id in enumerate(CLIENTS)
    }


def _evaluate_candidates(*, candidate_states, anchor_candidate_ids):
    assert set(anchor_candidate_ids).issubset(candidate_states)
    ordered = list(candidate_states)
    scores = {
        candidate_id: (
            0.10
            if candidate_id == "anchor_fedyogi"
            else 0.11
            if candidate_id == "anchor_fedavg"
            else 0.12 + ordered.index(candidate_id) / 1000
        )
        for candidate_id in ordered
    }
    ranked = sorted(ordered, key=lambda candidate_id: (scores[candidate_id], candidate_id))
    return [
        LocalCandidateVote(
            client_id=client_id,
            candidate_id=candidate_id,
            sample_count=COUNTS[client_id],
            val_mape=scores[candidate_id],
            val_rmse=scores[candidate_id] * 10,
            relative_mape=0.0,
            relative_rmse=0.0,
            rank=ranked.index(candidate_id) + 1,
            confidence=0.9,
            catastrophic_degradation=False,
        )
        for client_id in CLIENTS
        for candidate_id in ordered
    ]


def _engine(tmp_path, method="FMAS_PCV_FEDYOGI", **overrides):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = FedYogiServerOptimizer(server_lr=0.1, update_clip_norm=2.0)
    values = dict(
        method=method,
        model=model,
        server_optimizer=optimizer,
        sample_counts=COUNTS,
        train_clients=_train_clients,
        collect_telemetry=_collect_telemetry,
        evaluate_candidates=_evaluate_candidates,
        training_config={"local_epochs": 2},
        training_seed=40,
        last_complete_round=2,
        agent_orchestrator=FakeAgent(),
        single_agent=FakeAgent(),
        pause_report_path=tmp_path / "PAUSED.json",
    )
    values.update(overrides)
    return PCVEngine(**values)


def test_checkpoint_writer_requires_one_transactionally_managed_path(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_path"):
        _engine(tmp_path, checkpoint_writer=lambda payload: None)


def test_checkpoint_writer_must_report_the_managed_checkpoint_path(tmp_path):
    checkpoint_path = tmp_path / "last_complete.pt"
    checkpoint_path.write_bytes(b"old-checkpoint")

    def writer(path, payload):
        del payload
        path.write_bytes(b"new-checkpoint")
        return tmp_path / "unmanaged.pt"

    engine = _engine(
        tmp_path,
        checkpoint_path=checkpoint_path,
        checkpoint_writer=writer,
    )

    with pytest.raises(ExperimentPaused) as paused:
        engine.run_round(3)

    assert paused.value.failure.exception_type == "ValueError"
    assert checkpoint_path.read_bytes() == b"old-checkpoint"
    assert engine.last_complete_round == 2


@pytest.mark.parametrize(
    ("method", "calls", "candidate_min", "candidate_max"),
    [
        ("FEDAVG_STRICT", 0, 1, 1),
        ("FEDYOGI_STRICT", 0, 1, 1),
        ("DPCV_FEDYOGI", 0, 2, 8),
        ("SA_PCV_FEDYOGI", 2, 2, 8),
        ("FMAS_PCV_FEDYOGI", 6, 2, 8),
    ],
)
def test_method_call_counts_candidate_budgets_and_common_round_contract(
    tmp_path, method, calls, candidate_min, candidate_max
):
    orchestrator = FakeAgent()
    single = FakeAgent()
    engine = _engine(
        tmp_path,
        method,
        agent_orchestrator=orchestrator,
        single_agent=single,
    )

    result = engine.run_round(3)

    assert isinstance(result, RoundResult)
    assert result.agent_call_count == calls
    assert candidate_min <= len(result.candidate_ids) <= candidate_max
    assert len(result.candidate_ids) == len(set(result.candidate_ids))
    assert result.selected_candidate_id in result.candidate_ids
    assert engine.last_complete_round == 3
    assert engine.best_validation == {"mape": result.aggregate_mape[result.selected_candidate_id], "round": 3}
    if method == "SA_PCV_FEDYOGI":
        assert single.roles == ["single_proposer", "coordinator"]
        assert orchestrator.roles == []
    elif method == "FMAS_PCV_FEDYOGI":
        assert orchestrator.roles == [
            "diagnostic",
            "performance_proposer",
            "stability_proposer",
            "balance_proposer",
            "critic",
            "coordinator",
        ]
        assert single.roles == []
    else:
        assert orchestrator.roles == single.roles == []

    record = engine.telemetry_records[-1]
    assert record["candidate_scores"] == {
        candidate_id: result.aggregate_mape[candidate_id]
        for candidate_id in sorted(result.candidate_ids)
    }
    assert record["selected_action"]["candidate_id"] == result.selected_candidate_id
    assert record["selected_action"]["server_optimizer"] in {"fedavg", "fedyogi"}
    assert set(record["selected_action"]) == {
        "candidate_id",
        "source",
        "server_optimizer",
        "server_lr_scale",
        "update_clip_norm",
        "weights",
    }


def test_fmas_staged_calls_use_the_strict_agent_payload_contract(tmp_path):
    agent = FakeAgent()
    engine = _engine(tmp_path, agent_orchestrator=agent)

    engine.run_round(3)

    payloads = dict(agent.payloads)
    base = {"round_index", "clients"}
    assert set(payloads["diagnostic"]) == base
    for role in ("performance_proposer", "stability_proposer", "balance_proposer"):
        assert set(payloads[role]) == base | {"diagnostic"}
    assert set(payloads["critic"]) == base | {"diagnostic", "candidates"}
    assert set(payloads["coordinator"]) == base | {
        "diagnostic",
        "candidates",
        "critique",
        "anchor_candidate_ids",
        "client_votes",
    }
    assert all(type(candidate) is dict for candidate in payloads["critic"]["candidates"])
    assert len(payloads["coordinator"]["client_votes"]) == (
        len(CLIENTS) * len(payloads["coordinator"]["candidates"])
    )
    for role, payload in agent.payloads:
        agents_module._assert_context_payload_safe(role, payload)


@pytest.mark.parametrize(
    ("action_alias", "critic_candidate_count", "coordinator_candidate_count"),
    [(True, 5, 7), (False, 6, 8)],
)
def test_fmas_deduplicates_action_aliases_before_critic(
    tmp_path, action_alias, critic_candidate_count, coordinator_candidate_count
):
    class RoundEightAgent(FakeAgent):
        def __call__(self, *, role, payload):
            if role == "performance_proposer":
                self.roles.append(role)
                self.payloads.append((role, payload))
                first = _proposal(
                    "perf_c1_equal" if action_alias else "perf_c1_unique",
                    (
                        {"client_01": 0.20, "client_02": 0.31, "client_03": 0.49}
                        if action_alias
                        else {"client_01": 0.22, "client_02": 0.29, "client_03": 0.49}
                    ),
                    source="performance_proposer",
                )
                return [
                    first,
                    _proposal("perf_c2", {"client_01": 0.21, "client_02": 0.30, "client_03": 0.49}),
                ]
            if role == "stability_proposer":
                self.roles.append(role)
                self.payloads.append((role, payload))
                return [
                    _proposal("stability_c1", {"client_01": 0.19, "client_02": 0.31, "client_03": 0.50}),
                    _proposal("stability_c2", {"client_01": 0.18, "client_02": 0.32, "client_03": 0.50}),
                ]
            if role == "balance_proposer":
                self.roles.append(role)
                self.payloads.append((role, payload))
                return [
                    _proposal("balance_round8_equal", {"client_01": 0.20, "client_02": 0.31, "client_03": 0.49}),
                    _proposal("balance_c2", {"client_01": 0.20, "client_02": 0.32, "client_03": 0.48}),
                ]
            return super().__call__(role=role, payload=payload)

    agent = RoundEightAgent()
    engine = _engine(tmp_path, agent_orchestrator=agent)

    result = engine.run_round(3)

    payloads = dict(agent.payloads)
    critic_ids = [candidate["candidate_id"] for candidate in payloads["critic"]["candidates"]]
    coordinator_ids = [
        candidate["candidate_id"] for candidate in payloads["coordinator"]["candidates"]
    ]
    assert len(critic_ids) == critic_candidate_count
    assert len(coordinator_ids) == coordinator_candidate_count
    assert set(payloads["coordinator"]["critique"]["accepted_candidate_ids"]) == set(critic_ids)
    assert set(critic_ids) == set(coordinator_ids) - {"anchor_fedavg", "anchor_fedyogi"}
    assert agent.roles == [
        "diagnostic",
        "performance_proposer",
        "stability_proposer",
        "balance_proposer",
        "critic",
        "coordinator",
    ]
    assert result.agent_call_count == 6
    assert engine.last_complete_round == 3


def test_single_agent_coordinator_uses_strict_evidence_shape(tmp_path):
    single = FakeAgent()
    engine = _engine(tmp_path, method="SA_PCV_FEDYOGI", single_agent=single)

    engine.run_round(3)

    payloads = dict(single.payloads)
    assert set(payloads["single_proposer"]) == {"round_index", "clients"}
    assert set(payloads["coordinator"]) == {
        "round_index",
        "clients",
        "diagnostic",
        "candidates",
        "critique",
        "anchor_candidate_ids",
        "client_votes",
    }
    assert payloads["coordinator"]["critique"] == {
        "accepted_candidate_ids": ["single_01"],
        "rejected": [],
    }
    agents_module._assert_context_payload_safe("coordinator", payloads["coordinator"])


def test_agent_failure_does_not_commit_incomplete_round(tmp_path):
    failing = FakeAgent(fail_role="diagnostic")
    engine = _engine(tmp_path, agent_orchestrator=failing)
    original_state = {key: value.clone() for key, value in engine.global_state.items()}

    with pytest.raises(ExperimentPaused) as paused:
        engine.run_round(round_index=3)

    assert paused.value.failure.category == "connection"
    _state_equal(engine.global_state, original_state)
    assert engine.last_complete_round == 2
    report = json.loads((tmp_path / "PAUSED.json").read_text(encoding="utf-8"))
    assert report["failure"]["category"] == "connection"
    assert "never-log-me" not in (tmp_path / "PAUSED.json").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("stage", "method_name"),
    [
        ("train", "_train_clients"),
        ("telemetry", "_collect_client_telemetry"),
        ("agent", "_build_method_proposals"),
        ("preview", "_preview_candidates"),
        ("vote", "_evaluate_on_clients"),
        ("gate", "_gate"),
        ("checkpoint", "_write_checkpoint"),
    ],
)
def test_every_round_stage_failure_restores_complete_runtime_and_artifacts(
    tmp_path, monkeypatch, stage, method_name
):
    checkpoint = tmp_path / "last_complete.pt"
    checkpoint.write_bytes(b"last complete checkpoint")
    telemetry_path = tmp_path / "rounds.jsonl"
    telemetry_path.write_bytes(b'{"old":true}\n')
    engine = _engine(
        tmp_path,
        checkpoint_path=checkpoint,
        checkpoint_writer=lambda path, payload: path,
        telemetry_sink=AppendOnlyTelemetry(telemetry_path),
    )
    optimizer_before = engine.server_optimizer.get_optimizer_state()
    global_before = {key: value.clone() for key, value in engine.global_state.items()}
    model_before = {key: value.clone() for key, value in engine.model.state_dict().items()}
    previous_before = dict(engine.previous_weights)
    best_before = dict(engine.best_validation)
    best_model_before = {key: value.clone() for key, value in engine.best_model_state.items()}
    rng_before = capture_rng_state()
    checkpoint_before = checkpoint.read_bytes()
    telemetry_before = telemetry_path.read_bytes()

    def fail(*args, **kwargs):
        del args, kwargs
        with torch.no_grad():
            engine.model.weight.add_(100)
        engine.global_state["weight"].add_(100)
        engine.server_optimizer.m["weight"] = torch.tensor([999.0])
        engine.previous_weights["client_01"] = 999.0
        engine.best_validation["mape"] = -1.0
        engine.best_model_state["weight"].add_(100)
        engine.last_complete_round = 999
        engine.telemetry_records.append({"bad": True})
        engine.pending_telemetry.append({"bad": True})
        random.random()
        np.random.rand()
        torch.rand(2)
        if stage == "checkpoint":
            checkpoint.write_bytes(b"corrupt new checkpoint")
        raise RuntimeError(f"{stage} Bearer secret-token")

    monkeypatch.setattr(engine, method_name, fail)
    with pytest.raises(ExperimentPaused) as paused:
        engine.run_round(3)

    assert isinstance(paused.value.failure, ExperimentRuntimeError)
    _state_equal(engine.global_state, global_before)
    _state_equal(engine.model.state_dict(), model_before)
    _state_equal(engine.server_optimizer.get_optimizer_state(), optimizer_before)
    assert engine.previous_weights == previous_before
    assert engine.best_validation == best_before
    _state_equal(engine.best_model_state, best_model_before)
    assert engine.last_complete_round == 2
    assert engine.telemetry_records == []
    assert engine.pending_telemetry == []
    _rng_equal(capture_rng_state(), rng_before)
    assert checkpoint.read_bytes() == checkpoint_before
    assert telemetry_path.read_bytes() == telemetry_before
    raw_report = (tmp_path / "PAUSED.json").read_text(encoding="utf-8")
    assert "secret-token" not in raw_report


def test_checkpoint_is_written_only_after_runtime_commit(tmp_path):
    observations = []

    def writer(path, payload):
        observations.append(
            (
                engine.last_complete_round,
                engine.model.weight.detach().clone(),
                payload["last_complete_round"],
            )
        )
        return path

    engine = _engine(
        tmp_path,
        checkpoint_path=tmp_path / "last_complete.pt",
        checkpoint_writer=writer,
    )
    before = engine.model.weight.detach().clone()

    result = engine.run_round(3)

    assert result.checkpoint_path == tmp_path / "last_complete.pt"
    assert observations[0][0] == observations[0][2] == 3
    assert not torch.equal(observations[0][1], before)


def test_fedavg_commit_clears_fedyogi_moments_and_records_reset(tmp_path):
    optimizer = FedYogiServerOptimizer(server_lr=0.1)
    optimizer.m["weight"] = torch.tensor([[3.0]])
    optimizer.v["weight"] = torch.tensor([[4.0]])
    engine = _engine(tmp_path, "FEDAVG_STRICT", server_optimizer=optimizer)

    result = engine.run_round(3)

    assert engine.server_optimizer.m == {}
    assert engine.server_optimizer.v == {}
    assert engine.telemetry_records[-1]["fedyogi_moments_reset"] is True
    assert result.selected_candidate_id == "anchor_fedavg"


def test_fedavg_candidate_preview_applies_lr_scale_and_clip(tmp_path):
    engine = _engine(tmp_path)
    candidate = CandidateAction(
        candidate_id="damped_fedavg",
        weights={"client_01": 0.2, "client_02": 0.3, "client_03": 0.5},
        server_optimizer="fedavg",
        server_lr_scale=0.5,
        update_clip_norm=0.25,
        source="stability_proposer",
        rationale="exercise registered FedAvg controls",
    )
    local_updates = {
        client_id: {"weight": torch.tensor([[value]])}
        for client_id, value in zip(CLIENTS, (3.0, 4.0, 5.0))
    }

    preview = engine._preview_candidates([candidate], local_updates)[candidate.candidate_id]

    assert preview.telemetry["effective_server_lr"] == pytest.approx(0.5)
    assert preview.telemetry["update_clipped"] is True
    assert preview.telemetry["update_norm"] == pytest.approx(0.25)
    torch.testing.assert_close(preview.model_state["weight"], torch.tensor([[1.25]]))


def test_fedyogi_commit_uses_selected_preview_without_polluting_optimizer_config(tmp_path):
    optimizer = FedYogiServerOptimizer(server_lr=0.2, update_clip_norm=2.0)
    engine = _engine(tmp_path, "FEDYOGI_STRICT", server_optimizer=optimizer)

    result = engine.run_round(3)

    assert optimizer.server_lr == pytest.approx(0.2)
    assert optimizer.update_clip_norm == 2.0
    assert optimizer.m and optimizer.v
    assert result.optimizer_telemetry["server_lr_scale"] == 1.0


def test_strict_fedyogi_anchor_has_no_implicit_update_clip(tmp_path):
    engine = _engine(tmp_path, "FEDYOGI_STRICT")

    result = engine.run_round(3)

    assert result.optimizer_telemetry["update_clipped"] is False
    assert engine.fedyogi_clip_norm is None
    assert result.optimizer_telemetry["update_norm"] <= 1.0


def test_fail_stop_agent_does_not_call_remaining_roles(tmp_path):
    agent = FakeAgent(fail_role="stability_proposer")
    engine = _engine(tmp_path, agent_orchestrator=agent)

    with pytest.raises(ExperimentPaused) as paused:
        engine.run_round(3)

    assert paused.value.failure.category == "connection"
    assert agent.roles == ["diagnostic", "performance_proposer", "stability_proposer"]
    assert not set(agent.roles) & {"balance_proposer", "critic", "coordinator"}


def test_pause_report_failure_never_overwrites_original_failure(tmp_path, monkeypatch):
    agent = FakeAgent(fail_role="diagnostic")
    checkpoint_directory = tmp_path / "checkpoint"
    checkpoint_directory.mkdir()
    engine = _engine(
        tmp_path,
        agent_orchestrator=agent,
        checkpoint_path=checkpoint_directory / "last_complete.pt",
        pause_report_path=tmp_path / "primary" / "PAUSED.json",
        pause_report_writer=lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("report failed with Bearer secret-report-token")
        ),
    )

    with pytest.raises(ExperimentPaused) as paused:
        engine.run_round(3)

    assert isinstance(paused.value.failure, DeepSeekCallError)
    assert paused.value.failure.category == "connection"
    assert paused.value.report_path == checkpoint_directory / "PAUSED.json"
    assert paused.value.report_path.is_file()
    assert paused.value.report_error == "OSError"
    report = json.loads(paused.value.report_path.read_text(encoding="utf-8"))
    assert report["primary_report_error"] == {"exception_type": "OSError"}
    assert "secret-report-token" not in paused.value.report_path.read_text(encoding="utf-8")
    assert engine.last_complete_round == 2


@pytest.mark.parametrize(
    "method_name",
    [
        "_train_clients",
        "_collect_client_telemetry",
        "_build_method_proposals",
        "_preview_candidates",
        "_evaluate_on_clients",
        "_gate",
        "_write_checkpoint",
    ],
)
def test_callback_experiment_paused_is_rewrapped_for_current_round(
    tmp_path, monkeypatch, method_name
):
    engine = _engine(tmp_path)
    stale_report = tmp_path / "PAUSED.json"
    stale_report.write_text(
        '{"status":"paused","failed_round":1,"last_complete_round":0}',
        encoding="utf-8",
    )
    original_failure = DeepSeekCallError(
        "connection", "diagnostic", "Bearer callback-secret"
    )
    original_pause = ExperimentPaused(original_failure, stale_report)

    def fail(*args, **kwargs):
        del args, kwargs
        engine.global_state["weight"].add_(99)
        engine.last_complete_round = 99
        raise original_pause

    monkeypatch.setattr(engine, method_name, fail)
    with pytest.raises(ExperimentPaused) as paused:
        engine.run_round(3)

    assert paused.value is not original_pause
    assert paused.value.failure is not original_failure
    assert paused.value.failure.category == "connection"
    assert paused.value.report_path == tmp_path / "PAUSED.json"
    assert paused.value.report_path.is_file()
    assert paused.value.__cause__ is original_pause
    report_text = paused.value.report_path.read_text(encoding="utf-8")
    assert json.loads(report_text)["failed_round"] == 3
    assert "callback-secret" not in report_text
    assert engine.last_complete_round == 2


def test_invalid_nested_pause_failure_is_sanitized_and_rewrapped(tmp_path):
    engine = _engine(tmp_path)
    nested = ExperimentPaused.__new__(ExperimentPaused)
    RuntimeError.__init__(nested, "invalid nested pause")
    nested.failure = "Bearer invalid nested failure"
    nested.report_path = tmp_path / "stale.json"
    engine.train_clients = lambda **kwargs: (_ for _ in ()).throw(nested)

    with pytest.raises(ExperimentPaused) as paused:
        engine.run_round(3)

    assert isinstance(paused.value.failure, ExperimentRuntimeError)
    assert paused.value.failure.category == "runtime"
    assert paused.value.report_path.is_file()
    assert "invalid nested failure" not in paused.value.report_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "bad_sink",
    [
        lambda record: None,
        type("AppendWithoutPath", (), {"append": lambda self, record: None})(),
    ],
)
def test_non_transactional_telemetry_sink_is_rejected(tmp_path, bad_sink):
    with pytest.raises(TypeError, match="transactional|AppendOnlyTelemetry|path"):
        _engine(tmp_path, telemetry_sink=bad_sink)


def test_transactional_telemetry_path_must_be_a_regular_file(tmp_path):
    directory = tmp_path / "not-a-telemetry-file"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        _engine(tmp_path, telemetry_sink=AppendOnlyTelemetry(directory))


def test_telemetry_append_then_failure_leaves_no_round_record(
    tmp_path, monkeypatch
):
    telemetry_path = tmp_path / "rounds.jsonl"
    telemetry_path.write_bytes(b'{"old":true}\n')
    engine = _engine(tmp_path, telemetry_sink=AppendOnlyTelemetry(telemetry_path))
    append = engine._append_telemetry

    def append_then_fail(record):
        append(record)
        raise OSError("failure after telemetry append")

    monkeypatch.setattr(engine, "_append_telemetry", append_then_fail)
    with pytest.raises(ExperimentPaused):
        engine.run_round(3)

    assert telemetry_path.read_bytes() == b'{"old":true}\n'
    assert engine.telemetry_records == []
    assert engine.last_complete_round == 2


def test_checkpoint_failure_after_replace_restores_checkpoint_and_telemetry(tmp_path):
    telemetry_path = tmp_path / "rounds.jsonl"
    telemetry_path.write_bytes(b'{"old":true}\n')
    checkpoint_path = tmp_path / "last_complete.pt"
    checkpoint_path.write_bytes(b"old-checkpoint")

    def replace_then_fail(path, payload):
        del payload
        path.write_bytes(b"new-checkpoint")
        raise OSError("failure after checkpoint replace")

    engine = _engine(
        tmp_path,
        telemetry_sink=AppendOnlyTelemetry(telemetry_path),
        checkpoint_path=checkpoint_path,
        checkpoint_writer=replace_then_fail,
    )
    with pytest.raises(ExperimentPaused):
        engine.run_round(3)

    assert checkpoint_path.read_bytes() == b"old-checkpoint"
    assert telemetry_path.read_bytes() == b'{"old":true}\n'
    assert engine.telemetry_records == []
    assert engine.last_complete_round == 2


def test_callback_state_is_cloned_and_tensor_subclass_fails_closed(tmp_path):
    shared = torch.tensor([[5.0]])

    def aliased_train(**kwargs):
        del kwargs
        shared_state = {"weight": shared}
        return {client_id: shared_state for client_id in CLIENTS}

    engine = _engine(tmp_path, "FEDAVG_STRICT", train_clients=aliased_train)
    engine.run_round(3)
    assert shared.item() == 5.0

    class LeakyTensor(torch.Tensor):
        leaked_dataset = object()

    leaky = torch.Tensor._make_subclass(LeakyTensor, torch.tensor([[2.0]]), False)
    engine = _engine(
        tmp_path,
        "FEDAVG_STRICT",
        train_clients=lambda **kwargs: {
            client_id: {"weight": leaky} for client_id in CLIENTS
        },
    )
    with pytest.raises(ExperimentPaused) as paused:
        engine.run_round(3)
    assert isinstance(paused.value.__cause__, TypeError)
    assert engine.last_complete_round == 2


def test_invalid_rounds_and_method_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="invalid formal method"):
        _engine(tmp_path, "NOT_A_METHOD")

    engine = _engine(tmp_path, "FEDAVG_STRICT")
    with pytest.raises(ExperimentPaused) as gap:
        engine.run_round(4)
    assert gap.value.failure.exception_type == "ValueError"
    engine.run_round(3)
    with pytest.raises(ExperimentPaused) as duplicate:
        engine.run_round(3)
    assert duplicate.value.failure.exception_type == "ValueError"
    assert engine.last_complete_round == 3


def test_vote_ids_stronger_anchor_and_selected_state_are_verified(tmp_path):
    def incomplete_votes(**kwargs):
        votes = _evaluate_candidates(**kwargs)
        return [vote for vote in votes if vote.candidate_id != "anchor_fedavg"]

    engine = _engine(tmp_path, evaluate_candidates=incomplete_votes)
    with pytest.raises(ExperimentPaused) as incomplete:
        engine.run_round(3)
    assert incomplete.value.failure.exception_type == "ValueError"

    def false_stronger(**kwargs):
        votes = _evaluate_candidates(**kwargs)
        scores = {}
        for candidate_id in kwargs["candidate_states"]:
            scores[candidate_id] = next(
                vote.val_mape for vote in votes if vote.candidate_id == candidate_id
            )
        return votes, scores, "anchor_fedavg"

    engine = _engine(tmp_path, evaluate_candidates=false_stronger)
    with pytest.raises(ExperimentPaused) as stronger:
        engine.run_round(3)
    assert stronger.value.failure.exception_type == "ValueError"

    def missing_selection(**kwargs):
        return CandidateDecision(
            requested_candidate_id=kwargs["requested_candidate_id"],
            selected_candidate_id="missing-state",
            gate_status="accepted",
            rationale="fake",
        )

    engine = _engine(tmp_path, gate_selector=missing_selection)
    with pytest.raises(ExperimentPaused) as selected:
        engine.run_round(3)
    assert selected.value.failure.exception_type == "ValueError"
