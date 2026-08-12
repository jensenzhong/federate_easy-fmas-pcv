import hashlib
import builtins
import json
import math
from pathlib import Path
from threading import Thread

import pytest
import requests

from src.federated_learning.pcv import telemetry as telemetry_module
from src.federated_learning.pcv.agents import (
    DeepSeekCallError,
    MultiAgentOrchestrator,
    OrchestrationResult,
    StrictDeepSeekClient,
    load_prompt_hashes,
    validate_coordinator_response,
    validate_critic_response,
    validate_diagnostic_response,
    validate_proposer_response,
)
from src.federated_learning.pcv.protocol import PrivacyViolation
from src.federated_learning.pcv.runtime import StagedDeepSeekAgent
from src.federated_learning.pcv.schemas import CandidateAction
from src.federated_learning.pcv.telemetry import AppendOnlyTelemetry


PROMPT_DIR = Path(__file__).parents[1] / "configs" / "prompts"
ROLE_NAMES = (
    "diagnostic",
    "performance_proposer",
    "stability_proposer",
    "balance_proposer",
    "critic",
    "coordinator",
)


@pytest.fixture
def safe_payload():
    return {
        "round_index": 1,
        "clients": [
            {
                "client_id": "client-a",
                "sample_count": 12,
                "train_loss": 0.2,
                "val_mape": 0.3,
                "val_rmse": 1.1,
                "update_norm": 0.4,
                "cosine_to_mean": 0.8,
                "cosine_to_previous": 0.7,
            },
            {
                "client_id": "client-b",
                "sample_count": 8,
                "train_loss": 0.3,
                "val_mape": 0.4,
                "val_rmse": 1.2,
                "update_norm": 0.5,
                "cosine_to_mean": 0.6,
                "cosine_to_previous": 0.5,
            },
        ],
    }


class FakeResponse:
    def __init__(
        self,
        *,
        content=None,
        status_code=200,
        json_error=None,
        http_error_message="sensitive server error detail",
    ):
        self.status_code = status_code
        self._json_error = json_error
        self._http_error_message = http_error_message
        self._body = (
            {"choices": [{"message": {"content": content}}]}
            if content is not None
            else {"choices": [{"message": {}}]}
        )

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(
                self._http_error_message,
                response=self,
            )
            raise error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body


class FakeSession:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0
        self.last_args = None
        self.last_kwargs = None

    def post(self, *args, **kwargs):
        self.calls += 1
        self.last_args = args
        self.last_kwargs = kwargs
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.payloads = []

    def post(self, *args, **kwargs):
        del args
        self.calls += 1
        self.payloads.append(
            json.loads(kwargs["json"]["messages"][1]["content"])
        )
        return FakeResponse(content=self.responses.pop(0))


def make_client(session, *, api_key="test-only-secret", telemetry=None):
    return StrictDeepSeekClient(
        api_key=api_key,
        model_name="deepseek-chat",
        base_url="https://example.invalid/v1/",
        timeout_seconds=2,
        session=session,
        telemetry=telemetry,
    )


def _candidate_context():
    return {
        "candidate_id": "performance_01",
        "weights": {"client-a": 0.6, "client-b": 0.4},
        "server_optimizer": "fedyogi",
        "server_lr_scale": 1.0,
        "update_clip_norm": None,
        "source": "performance_proposer",
        "rationale": "bounded aggregate-only evidence",
    }


def _vote_context(client_id):
    return {
        "client_id": client_id,
        "candidate_id": "performance_01",
        "sample_count": 1,
        "val_mape": 0.2,
        "val_rmse": 0.3,
        "relative_mape": 0.0,
        "relative_rmse": 0.0,
        "rank": 1,
        "confidence": 1.0,
        "catastrophic_degradation": False,
    }


def test_staged_sa_real_response_shape_reaches_coordinator_without_mutating_dto(
    safe_payload,
):
    single_response = json.dumps(
        {
            "diagnostic": {
                "state_summary": "heterogeneous but bounded",
                "risks": ["large relative update"],
                "priorities": ["validation gate"],
            },
            "candidates": [_candidate_context()],
        }
    )
    coordinator_response = json.dumps(
        {
            "selected_candidate_id": "performance_01",
            "rationale": "best local vote",
            "risk_acknowledgement": "gate retains authority",
        }
    )
    session = SequenceSession([single_response, coordinator_response])
    agent = StagedDeepSeekAgent(make_client(session), PROMPT_DIR)

    single = agent.call(role="single_proposer", payload=safe_payload)
    diagnostic = single["diagnostic"]
    agent.call(
        role="coordinator",
        payload={
            **safe_payload,
            "diagnostic": diagnostic,
            "candidates": [_candidate_context()],
            "critique": {
                "accepted_candidate_ids": ["performance_01"],
                "rejected": [],
            },
            "anchor_candidate_ids": [],
            "client_votes": [
                _vote_context("client-a"),
                _vote_context("client-b"),
            ],
        },
    )

    assert session.calls == 2
    assert type(session.payloads[1]["diagnostic"]["risks"]) is list
    assert type(diagnostic["risks"]) is tuple
    assert diagnostic["risks"] == ("large relative update",)


def test_single_proposer_prompt_requires_every_action_to_repeat_all_fields():
    prompt = (PROMPT_DIR / "single_proposer.md").read_text(encoding="utf-8")

    for field in (
        "candidate_id",
        "weights",
        "server_optimizer",
        "server_lr_scale",
        "update_clip_norm",
        "source",
        "rationale",
    ):
        assert f"`{field}`" in prompt
    assert "applies separately to the first and second action" in prompt
    assert '`"source":"performance_proposer"`' in prompt
    assert "inside every action" in prompt
    assert "first character of your response must be `{`" in prompt
    assert "Do not use Markdown" in prompt
    assert "code fences" in prompt


def test_staged_fmas_real_dto_shapes_reach_all_later_roles_as_json(
    safe_payload,
):
    candidate = _candidate_context()
    session = SequenceSession(
        [
            json.dumps(
                {
                    "state_summary": "heterogeneous but bounded",
                    "risks": ["large relative update"],
                    "priorities": ["validation gate"],
                }
            ),
            json.dumps(
                {
                    "candidates": [candidate],
                }
            ),
            json.dumps(
                {
                    "accepted_candidate_ids": ["performance_01"],
                    "rejected": [],
                }
            ),
            json.dumps(
                {
                    "selected_candidate_id": "performance_01",
                    "rationale": "best local vote",
                    "risk_acknowledgement": "gate retains authority",
                }
            ),
        ]
    )
    agent = StagedDeepSeekAgent(make_client(session), PROMPT_DIR)

    diagnostic = agent.call(role="diagnostic", payload=safe_payload)
    proposals = agent.call(
        role="performance_proposer",
        payload={**safe_payload, "diagnostic": diagnostic},
    )
    assert proposals[0].candidate_id == "performance_01"
    critique = agent.call(
        role="critic",
        payload={
            **safe_payload,
            "diagnostic": diagnostic,
            "candidates": [candidate],
        },
    )
    agent.call(
        role="coordinator",
        payload={
            **safe_payload,
            "diagnostic": diagnostic,
            "candidates": [candidate],
            "critique": critique,
            "anchor_candidate_ids": [],
            "client_votes": [
                _vote_context("client-a"),
                _vote_context("client-b"),
            ],
        },
    )

    assert session.calls == 4
    assert type(session.payloads[1]["diagnostic"]["risks"]) is list
    assert type(session.payloads[2]["diagnostic"]["risks"]) is list
    assert type(session.payloads[3]["diagnostic"]["risks"]) is list
    assert type(session.payloads[3]["critique"]["accepted_candidate_ids"]) is list
    assert type(diagnostic["risks"]) is tuple
    assert type(critique["accepted_candidate_ids"]) is tuple


@pytest.mark.parametrize(
    "case",
    ["bytes", "set", "custom", "nan", "positive_inf", "negative_inf", "key", "cycle"],
)
def test_outbound_json_boundary_fails_closed_before_post(safe_payload, case):
    session = FakeSession(AssertionError("post must not run"))
    invalid_values = {
        "bytes": b"private bytes",
        "set": {"not", "json"},
        "custom": object(),
        "nan": float("nan"),
        "positive_inf": float("inf"),
        "negative_inf": -float("inf"),
    }
    risks = []
    if case == "cycle":
        risks.append(risks)
    elif case == "key":
        risks = ["bounded"]
    else:
        risks = [invalid_values[case]]
    diagnostic = {
        "state_summary": "stable",
        "risks": risks,
        "priorities": ["validation"],
    }
    if case == "key":
        diagnostic[1] = "not a JSON object key"

    with pytest.raises((TypeError, ValueError)):
        make_client(session).generate_json(
            "performance_proposer",
            "Return JSON.",
            {**safe_payload, "diagnostic": diagnostic},
            lambda value: value,
        )

    assert session.calls == 0


def test_outbound_json_boundary_rejects_custom_mapping_before_post(safe_payload):
    class CustomMapping(dict):
        pass

    session = FakeSession(AssertionError("post must not run"))
    diagnostic = CustomMapping(
        state_summary="stable",
        risks=["bounded"],
        priorities=["validation"],
    )

    with pytest.raises(TypeError, match="unsupported CustomMapping"):
        make_client(session).generate_json(
            "performance_proposer",
            "Return JSON.",
            {**safe_payload, "diagnostic": diagnostic},
            lambda value: value,
        )

    assert session.calls == 0


def test_missing_api_key_is_authentication_preflight_and_makes_no_call():
    session = FakeSession(AssertionError("post must not run"))
    with pytest.raises(DeepSeekCallError) as error:
        make_client(session, api_key="  ")
    assert error.value.category == "authentication"
    assert error.value.role == "preflight"
    assert session.calls == 0


@pytest.mark.parametrize(
    "exception",
    [
        ConnectionError("offline"),
        requests.ConnectionError("offline"),
    ],
)
def test_connection_failure_has_one_attempt_and_preserves_role(
    safe_payload,
    exception,
):
    session = FakeSession(exception)
    client = make_client(session)
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "diagnostic",
            "Return JSON.",
            safe_payload,
            lambda value: value,
        )
    assert session.calls == 1
    assert error.value.category == "connection"
    assert error.value.role == "diagnostic"


@pytest.mark.parametrize(
    "exception",
    [requests.Timeout("slow"), TimeoutError("slow")],
)
def test_timeout_failure_has_one_attempt_and_is_connection_category(
    safe_payload,
    exception,
):
    session = FakeSession(exception)
    client = make_client(session)
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "coordinator",
            "Return JSON.",
            safe_payload,
            lambda value: value,
        )
    assert session.calls == 1
    assert error.value.category == "connection"
    assert error.value.role == "coordinator"


@pytest.mark.parametrize("status", [401, 403])
def test_http_authentication_failures_are_explicit(safe_payload, status):
    session = FakeSession(FakeResponse(content="{}", status_code=status))
    client = make_client(session)
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "critic", "Return JSON.", safe_payload, lambda value: value
        )
    assert session.calls == 1
    assert error.value.category == "authentication"
    assert error.value.role == "critic"


def test_other_http_failure_is_explicit(safe_payload):
    session = FakeSession(FakeResponse(content="{}", status_code=500))
    client = make_client(session)
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "diagnostic", "Return JSON.", safe_payload, lambda value: value
        )
    assert session.calls == 1
    assert error.value.category == "http"


def test_invalid_http_json_is_schema_failure(safe_payload):
    session = FakeSession(
        FakeResponse(json_error=ValueError("invalid HTTP response JSON"))
    )
    client = make_client(session)
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "diagnostic", "Return JSON.", safe_payload, lambda value: value
        )
    assert session.calls == 1
    assert error.value.category == "schema"


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_provider_envelope_and_content_failures_are_schema_with_one_post(
    safe_payload,
    document,
):
    class EnvelopeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return document

    session = FakeSession(EnvelopeResponse())
    client = make_client(session)
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "diagnostic", "Return JSON.", safe_payload, lambda value: value
        )
    assert session.calls == 1
    assert error.value.category == "schema"
    assert error.value.role == "diagnostic"


@pytest.mark.parametrize("content", ["not json", "[]", "NaN"])
def test_malformed_or_non_object_content_is_schema_failure(
    safe_payload,
    content,
):
    session = FakeSession(FakeResponse(content=content))
    client = make_client(session)
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "critic", "Return JSON.", safe_payload, lambda value: value
        )
    assert session.calls == 1
    assert error.value.category == "schema"
    assert error.value.role == "critic"


def test_validator_error_stops_without_repair_or_fallback(safe_payload):
    session = FakeSession(FakeResponse(content='{"unexpected":true}'))
    client = make_client(session)

    def reject(_value):
        raise ValueError("validator rejected response")

    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json("critic", "Return JSON.", safe_payload, reject)
    assert session.calls == 1
    assert error.value.category == "schema"


def test_payload_privacy_is_checked_before_the_only_post(safe_payload):
    safe_payload["clients"][0]["labels"] = [1.0]
    session = FakeSession(FakeResponse(content="{}"))
    client = make_client(session)
    with pytest.raises(PrivacyViolation):
        client.generate_json(
            "diagnostic", "Return JSON.", safe_payload, lambda value: value
        )
    assert session.calls == 0


def test_success_makes_exactly_one_post_and_uses_strict_request(safe_payload):
    session = FakeSession(FakeResponse(content='{"ok":true}'))
    client = make_client(session)
    parsed = client.generate_json(
        "diagnostic", "Return JSON.", safe_payload, lambda value: value
    )
    assert parsed == {"ok": True}
    assert session.calls == 1
    assert session.last_args == (
        "https://example.invalid/v1/chat/completions",
    )
    request = session.last_kwargs
    assert request["timeout"] == 2
    assert request["headers"] == {
        "Authorization": "Bearer test-only-secret",
        "Content-Type": "application/json",
    }
    assert request["json"]["model"] == "deepseek-chat"
    assert request["json"]["response_format"] == {"type": "json_object"}
    assert request["json"]["stream"] is False
    assert len(request["json"]["messages"]) == 2


def test_complementary_json_objects_are_canonicalized_once_and_audited(
    tmp_path,
    safe_payload,
):
    response_text = (
        '{"state_summary":"Round 11 aggregate telemetry is unstable."}\n\n'
        '{"risks":["directional disagreement"]}\n\n'
        '{"priorities":["stabilize aggregation"]}'
    )
    path = tmp_path / "agent_calls.jsonl"
    session = FakeSession(FakeResponse(content=response_text))

    result = make_client(
        session,
        telemetry=AppendOnlyTelemetry(path),
    ).generate_json(
        "diagnostic",
        "Return JSON.",
        safe_payload,
        validate_diagnostic_response,
    )

    assert session.calls == 1
    assert result["risks"] == ("directional disagreement",)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["response_text"] == response_text
    assert record["parsed_response"] == {
        "state_summary": "Round 11 aggregate telemetry is unstable.",
        "risks": ["directional disagreement"],
        "priorities": ["stabilize aggregation"],
    }
    assert record["canonicalization_applied"] is True
    assert record["canonicalization_rule"] == "complementary-json-objects-v1"


def test_strict_single_json_records_no_canonicalization(tmp_path, safe_payload):
    path = tmp_path / "agent_calls.jsonl"
    response = {
        "state_summary": "stable",
        "risks": ["bounded"],
        "priorities": ["monitor"],
    }
    session = FakeSession(FakeResponse(content=json.dumps(response)))

    make_client(session, telemetry=AppendOnlyTelemetry(path)).generate_json(
        "diagnostic",
        "Return JSON.",
        safe_payload,
        validate_diagnostic_response,
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["canonicalization_applied"] is False
    assert record["canonicalization_rule"] is None


@pytest.mark.parametrize(
    "response_text",
    [
        '```json\n{"state_summary":"stable","risks":[],"priorities":[]}\n```',
        (
            "I will now return the requested aggregate-only JSON.\n"
            '{"state_summary":"stable","risks":[],"priorities":[]}'
        ),
    ],
)
def test_json_output_mode_still_fails_closed_if_provider_returns_outer_text(
    safe_payload,
    response_text,
):
    session = FakeSession(FakeResponse(content=response_text))

    with pytest.raises(DeepSeekCallError) as error:
        make_client(session).generate_json(
            "diagnostic",
            "Return raw JSON.",
            safe_payload,
            validate_diagnostic_response,
        )

    assert session.calls == 1
    assert error.value.category == "schema"


def test_real_identical_coordinator_key_is_collapsed_once_and_audited(
    tmp_path,
    safe_payload,
):
    response_text = (
        '{"selected_candidate_id":"perf_fedavg_balanced",'
        '"rationale":"The aggregate client votes rank perf_fedavg_balanced first '
        'on all three clients, with consistent small relative improvements in both '
        'val_mape and val_rmse over anchor_fedavg. It uses balanced aggregation with '
        'moderate clipping, reducing sensitivity to any single client while '
        'maintaining standard FedAvg stability. The accepted critique includes this '
        'candidate, and the safety gate retains final authority.",'
        '"risk_acknowledgement":"I acknowledge that the deterministic safety gate '
        'is the final authority and may override this selection if it detects any '
        'risk not captured in the aggregate client votes or diagnostic information.",'
        '"selected_candidate_id":"perf_fedavg_balanced"}'
    )
    path = tmp_path / "agent_calls.jsonl"
    session = FakeSession(FakeResponse(content=response_text))

    result = make_client(
        session,
        telemetry=AppendOnlyTelemetry(path),
    ).generate_json(
        "coordinator",
        "Return JSON.",
        safe_payload,
        lambda value: validate_coordinator_response(
            value,
            candidate_ids=("perf_fedavg_balanced",),
        ),
    )

    assert session.calls == 1
    assert result["selected_candidate_id"] == "perf_fedavg_balanced"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["response_text"] == response_text
    assert record["parsed_response"]["selected_candidate_id"] == (
        "perf_fedavg_balanced"
    )
    assert record["canonicalization_applied"] is True
    assert record["canonicalization_rule"] == "identical-duplicate-key-v1"


def test_fragment_and_identical_duplicate_rules_are_composed_deterministically(
    tmp_path,
    safe_payload,
):
    response_text = (
        '{"state_summary":"ok"}\n'
        '{"state_summary":"ok","risks":[]}\n'
        '{"priorities":[]}'
    )
    path = tmp_path / "agent_calls.jsonl"
    session = FakeSession(FakeResponse(content=response_text))

    make_client(session, telemetry=AppendOnlyTelemetry(path)).generate_json(
        "diagnostic",
        "Return JSON.",
        safe_payload,
        validate_diagnostic_response,
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["canonicalization_applied"] is True
    assert record["canonicalization_rule"] == (
        "complementary-json-objects-v1+identical-duplicate-key-v1"
    )


def test_identical_duplicate_is_audited_when_canonical_object_fails_schema(
    tmp_path,
    safe_payload,
):
    response_text = (
        '{"state_summary":"ok","risks":[],"priorities":[],'
        '"unknown":{"nested":[1,true]},'
        '"unknown":{"nested":[1,true]}}'
    )
    path = tmp_path / "agent_calls.jsonl"
    session = FakeSession(FakeResponse(content=response_text))

    with pytest.raises(DeepSeekCallError):
        make_client(session, telemetry=AppendOnlyTelemetry(path)).generate_json(
            "diagnostic",
            "Return JSON.",
            safe_payload,
            validate_diagnostic_response,
        )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["response_text"] == response_text
    assert record["parsed_response"]["state_summary"] == "ok"
    assert record["parsed_response"]["unknown"] == {"nested": [1, True]}
    assert record["failure_category"] == "schema"
    assert record["canonicalization_applied"] is True
    assert record["canonicalization_rule"] == "identical-duplicate-key-v1"


@pytest.mark.parametrize(
    "content",
    [
        '{"state_summary":"ok","state_summary":"different",'
        '"risks":[],"priorities":[]}',
        '{"state_summary":"ok","risks":[1],"risks":[true],'
        '"priorities":[]}',
        '{"state_summary":"ok","risks":[1],"risks":[1.0],'
        '"priorities":[]}',
        '{"state_summary":"ok","risks":[{"x":1}],'
        '"risks":[{"x":2}],"priorities":[]}',
        '{"state_summary":"ok","STATE_SUMMARY":"ok",'
        '"risks":[],"priorities":[]}',
        '{"state_summary":"ok","risks":[NaN],"risks":[NaN],'
        '"priorities":[]}',
    ],
)
def test_duplicate_key_canonicalization_rejects_nonidentical_or_unsafe_values(
    safe_payload,
    content,
):
    session = FakeSession(FakeResponse(content=content))

    with pytest.raises(DeepSeekCallError) as error:
        make_client(session).generate_json(
            "diagnostic",
            "Return JSON.",
            safe_payload,
            validate_diagnostic_response,
        )

    assert session.calls == 1
    assert error.value.category == "schema"


@pytest.mark.parametrize(
    "content",
    [
        '{"state_summary":"ok","risks":[-0.0],"risks":[0.0],'
        '"priorities":[]}',
        '{"state_summary":"ok"}\n{"risks":[-0.0]}\n'
        '{"risks":[0.0]}\n{"priorities":[]}',
    ],
)
def test_signed_zero_duplicate_values_are_not_exactly_identical(
    safe_payload,
    content,
):
    session = FakeSession(FakeResponse(content=content))

    with pytest.raises(DeepSeekCallError) as error:
        make_client(session).generate_json(
            "diagnostic",
            "Return JSON.",
            safe_payload,
            lambda value: value,
        )

    assert session.calls == 1
    assert error.value.category == "schema"


def test_exponent_overflow_fails_schema_and_audit_remains_serializable(
    tmp_path,
    safe_payload,
):
    response_text = (
        '{"state_summary":"ok","risks":[1e400],"risks":[1e400],'
        '"priorities":[]}'
    )
    path = tmp_path / "agent_calls.jsonl"
    session = FakeSession(FakeResponse(content=response_text))

    with pytest.raises(DeepSeekCallError) as error:
        make_client(session, telemetry=AppendOnlyTelemetry(path)).generate_json(
            "diagnostic",
            "Return JSON.",
            safe_payload,
            validate_diagnostic_response,
        )

    assert session.calls == 1
    assert error.value.category == "schema"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["response_text"] == response_text
    assert record["parsed_response"] is None
    assert record["failure_category"] == "schema"
    assert record["canonicalization_applied"] is False
    assert record["canonicalization_rule"] is None


def test_canonicalized_object_that_fails_schema_is_audited_truthfully(
    tmp_path,
    safe_payload,
):
    response_text = '{"state_summary":"ok"}\n{"risks":[]}\n{"unknown":1}'
    path = tmp_path / "agent_calls.jsonl"
    session = FakeSession(FakeResponse(content=response_text))

    with pytest.raises(DeepSeekCallError) as error:
        make_client(session, telemetry=AppendOnlyTelemetry(path)).generate_json(
            "diagnostic",
            "Return JSON.",
            safe_payload,
            validate_diagnostic_response,
        )

    assert session.calls == 1
    assert error.value.category == "schema"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["response_text"] == response_text
    assert record["parsed_response"] == {
        "state_summary": "ok",
        "risks": [],
        "unknown": 1,
    }
    assert record["canonicalization_applied"] is True
    assert record["canonicalization_rule"] == "complementary-json-objects-v1"


@pytest.mark.parametrize(
    "content",
    [
        '{"state_summary":"ok","STATE_SUMMARY":"again",'
        '"risks":[],"priorities":[]}',
        '{"state_summary":"ok"}\n'
        '{"risks":[],"RISKS":["again"]}\n'
        '{"priorities":[]}',
    ],
)
def test_normalized_key_collisions_fail_closed_in_strict_and_fragment_objects(
    safe_payload,
    content,
):
    session = FakeSession(FakeResponse(content=content))

    with pytest.raises(DeepSeekCallError) as error:
        make_client(session).generate_json(
            "diagnostic",
            "Return JSON.",
            safe_payload,
            validate_diagnostic_response,
        )

    assert session.calls == 1
    assert error.value.category == "schema"


@pytest.mark.parametrize(
    "content",
    [
        '{"state_summary":"ok"}\n["not an object"]\n'
        '{"risks":[]}\n{"priorities":[]}',
        '{"state_summary":"ok"}\ntrue\n{"risks":[]}\n{"priorities":[]}',
        '{"state_summary":"ok"}\ngarbage\n{"risks":[]}\n{"priorities":[]}',
        '{"state_summary":"ok"}\n{"state_summary":"again"}\n'
        '{"risks":[]}\n{"priorities":[]}',
        '{"state_summary":"ok","state_summary":"again"}\n'
        '{"risks":[]}\n{"priorities":[]}',
        '{"state_summary":"ok"}\n{"STATE_SUMMARY":"again"}\n'
        '{"risks":[]}\n{"priorities":[]}',
        '{"state_summary":"ok"}\n{"risks":[]}\n{"unknown":1}',
        '{"state_summary":"ok"}\n{"risks":"wrong"}\n{"priorities":[]}',
    ],
)
def test_json_object_canonicalization_fails_closed_without_retry(
    safe_payload,
    content,
):
    session = FakeSession(FakeResponse(content=content))

    with pytest.raises(DeepSeekCallError) as error:
        make_client(session).generate_json(
            "diagnostic",
            "Return JSON.",
            safe_payload,
            validate_diagnostic_response,
        )

    assert session.calls == 1
    assert error.value.category == "schema"


def test_failures_never_disclose_key_headers_or_payload(safe_payload):
    secret = "key-that-must-never-appear"
    session = FakeSession(FakeResponse(content="not json"))
    client = make_client(session, api_key=secret)
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "diagnostic", "Return JSON.", safe_payload, lambda value: value
        )
    message = str(error.value)
    assert secret not in message
    assert "Authorization" not in message
    assert "client-a" not in message


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"state_summary": "ok", "risks": [], "priorities": [], "extra": 1},
        {"state_summary": "ok", "risks": "none", "priorities": []},
        {"state_summary": True, "risks": [], "priorities": []},
    ],
)
def test_diagnostic_validator_requires_exact_schema(value):
    with pytest.raises(ValueError):
        validate_diagnostic_response(value)


def test_diagnostic_validator_returns_a_deep_defensive_immutable_value():
    source = {
        "state_summary": "stable",
        "risks": ["oscillation"],
        "priorities": ["stability"],
    }
    result = validate_diagnostic_response(source)
    source["risks"].append("source mutation")
    assert result["risks"] == ("oscillation",)
    with pytest.raises(TypeError):
        result["state_summary"] = "mutated"
    with pytest.raises(AttributeError):
        result["risks"].append("direct mutation")


def proposal(candidate_id="performance_01", **changes):
    value = {
        "candidate_id": candidate_id,
        "weights": {"client-a": 0.6, "client-b": 0.4},
        "server_optimizer": "fedyogi",
        "server_lr_scale": 1.0,
        "update_clip_norm": 1.0,
        "source": "performance_proposer",
        "rationale": "legal aggregate-only action",
    }
    value.update(changes)
    return value


def test_proposer_validator_returns_validated_candidate_actions():
    result = validate_proposer_response(
        {"candidates": [proposal()]},
        client_ids=("client-a", "client-b"),
        role="performance_proposer",
    )
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], CandidateAction)


def test_proposer_validator_normalizes_three_decimal_llm_rounding_without_mutating_raw():
    raw_weights = {"client-a": 0.333, "client-b": 0.333, "client-c": 0.333}
    value = {"candidates": [proposal(weights=raw_weights)]}

    result = validate_proposer_response(
        value,
        client_ids=("client-a", "client-b", "client-c"),
        role="performance_proposer",
    )

    normalized = result[0].weights
    assert sum(normalized.values()) == pytest.approx(1.0, abs=1e-15)
    assert normalized["client-a"] / normalized["client-b"] == 1.0
    assert raw_weights == {"client-a": 0.333, "client-b": 0.333, "client-c": 0.333}


def test_proposer_rounding_normalization_does_not_change_raw_response_telemetry(tmp_path):
    raw_response = {
        "candidates": [
            proposal(
                weights={"client-a": 0.333, "client-b": 0.333, "client-c": 0.333}
            )
        ]
    }
    telemetry = AppendOnlyTelemetry(tmp_path / "calls.jsonl")
    session = FakeSession(FakeResponse(content=json.dumps(raw_response)))

    result = make_client(session, telemetry=telemetry).generate_json(
        "performance_proposer",
        "Return JSON.",
        {"round_index": 1, "clients": []},
        lambda value: validate_proposer_response(
            value,
            client_ids=("client-a", "client-b", "client-c"),
            role="performance_proposer",
        ),
    )

    record = json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))
    assert record["parsed_response"] == raw_response
    assert sum(result[0].weights.values()) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(
    "weights",
    [
        {"client-a": 0.33, "client-b": 0.33, "client-c": 0.33},
        {"client-a": -0.001, "client-b": 0.501, "client-c": 0.5},
        {"client-a": 0.04995, "client-b": 0.4745, "client-c": 0.4745},
        {"client-a": 0.8001, "client-b": 0.1001, "client-c": 0.1001},
        {"client-a": 0.5, "client-b": 0.5},
        {"client-a": float("nan"), "client-b": 0.5, "client-c": 0.5},
        {"client-a": float("inf"), "client-b": 0.0, "client-c": 0.0},
        {"client-a": 10**1000, "client-b": 0.0, "client-c": 0.0},
        {"client-a": 1e308, "client-b": 1e308, "client-c": 1e308},
        {"client-a": 0.0, "client-b": 0.0, "client-c": 0.0},
    ],
)
def test_proposer_validator_rejects_weights_outside_rounding_contract(weights):
    with pytest.raises(ValueError):
        validate_proposer_response(
            {"candidates": [proposal(weights=weights)]},
            client_ids=("client-a", "client-b", "client-c"),
            role="performance_proposer",
        )


def test_proposer_weight_rounding_tolerance_has_closed_two_sided_boundary():
    tolerance = 3 * 0.0005 + math.ulp(1.0)
    lower = 1.0 - tolerance
    upper = 1.0 + tolerance
    for accepted_total in (lower, upper):
        accepted = {
            "client-a": 0.3,
            "client-b": 0.3,
            "client-c": accepted_total - 0.6,
        }
        validate_proposer_response(
            {"candidates": [proposal(weights=accepted)]},
            client_ids=("client-a", "client-b", "client-c"),
            role="performance_proposer",
        )

    for rejected_total in (
        math.nextafter(lower, 0.0),
        math.nextafter(upper, math.inf),
    ):
        rejected = {
            "client-a": 0.3,
            "client-b": 0.3,
            "client-c": rejected_total - 0.6,
        }
        with pytest.raises(ValueError, match="sum to one"):
            validate_proposer_response(
                {"candidates": [proposal(weights=rejected)]},
                client_ids=("client-a", "client-b", "client-c"),
                role="performance_proposer",
            )


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"candidates": [], "extra": True},
        {"candidates": [proposal(), proposal("p2"), proposal("p3")]},
        {"candidates": [proposal(), proposal()]},
        {"candidates": [proposal(extra_field="forbidden")]},
        {"candidates": [proposal(weights={"client-a": True, "client-b": 0.0})]},
        {"candidates": [proposal(server_lr_scale=True)]},
        {"candidates": [proposal(source="stability_proposer")]},
    ],
)
def test_proposer_validator_rejects_budget_duplicates_and_invalid_fields(value):
    with pytest.raises(ValueError):
        validate_proposer_response(
            value,
            client_ids=("client-a", "client-b"),
            role="performance_proposer",
        )


def test_critic_validator_requires_known_unique_candidate_ids():
    valid = {
        "accepted_candidate_ids": ["p1"],
        "rejected": [{"candidate_id": "p2", "reason": "duplicate action"}],
    }
    validated = validate_critic_response(valid, candidate_ids=("p1", "p2"))
    assert validated["accepted_candidate_ids"] == ("p1",)
    assert validated["rejected"][0]["candidate_id"] == "p2"
    bad_values = [
        {**valid, "extra": 1},
        {"accepted_candidate_ids": ["unknown"], "rejected": []},
        {"accepted_candidate_ids": ["p1", "p1"], "rejected": []},
        {
            "accepted_candidate_ids": ["p1"],
            "rejected": [{"candidate_id": "p1", "reason": "both"}],
        },
        {
            "accepted_candidate_ids": [],
            "rejected": [{"candidate_id": "p2", "reason": False}],
        },
        {
            "accepted_candidate_ids": [],
            "rejected": [
                {"candidate_id": "p2", "reason": "no", "extra": True}
            ],
        },
    ]
    for bad in bad_values:
        with pytest.raises(ValueError):
            validate_critic_response(bad, candidate_ids=("p1", "p2"))


def test_critic_validator_returns_a_deep_defensive_immutable_value():
    source = {
        "accepted_candidate_ids": ["p1"],
        "rejected": [{"candidate_id": "p2", "reason": "duplicate"}],
    }
    result = validate_critic_response(source, candidate_ids=("p1", "p2"))
    source["accepted_candidate_ids"].append("p2")
    source["rejected"][0]["reason"] = "source mutation"
    assert result["accepted_candidate_ids"] == ("p1",)
    assert result["rejected"][0]["reason"] == "duplicate"
    with pytest.raises(TypeError):
        result["rejected"][0]["reason"] = "direct mutation"


def test_coordinator_validator_requires_known_admissible_id_and_exact_types():
    valid = {
        "selected_candidate_id": "p1",
        "rationale": "best client-local aggregate vote",
        "risk_acknowledgement": "deterministic gate retains authority",
    }
    assert validate_coordinator_response(valid, candidate_ids=("p1",)) == valid
    bad_values = [
        {**valid, "extra": 1},
        {**valid, "selected_candidate_id": "unknown"},
        {**valid, "rationale": True},
        {**valid, "risk_acknowledgement": ""},
    ]
    for bad in bad_values:
        with pytest.raises(ValueError):
            validate_coordinator_response(bad, candidate_ids=("p1",))


def test_coordinator_validator_returns_a_defensive_immutable_value():
    source = {
        "selected_candidate_id": "p1",
        "rationale": "best vote",
        "risk_acknowledgement": "gate retains authority",
    }
    result = validate_coordinator_response(source, candidate_ids=("p1",))
    source["rationale"] = "source mutation"
    assert result["rationale"] == "best vote"
    with pytest.raises(TypeError):
        result["rationale"] = "direct mutation"


def test_orchestration_result_deep_freezes_constructor_inputs():
    diagnostic = {"state_summary": "ok", "risks": ["r"], "priorities": ["p"]}
    critique = {"accepted_candidate_ids": ["p1"], "rejected": []}
    coordination = {
        "selected_candidate_id": "p1",
        "rationale": "best",
        "risk_acknowledgement": "gate",
    }
    result = OrchestrationResult(
        diagnostic=diagnostic,
        proposals=(),
        critique=critique,
        coordination=coordination,
    )
    diagnostic["risks"].append("source mutation")
    critique["accepted_candidate_ids"].append("source mutation")
    coordination["rationale"] = "source mutation"
    assert result.diagnostic["risks"] == ("r",)
    assert result.critique["accepted_candidate_ids"] == ("p1",)
    assert result.coordination["rationale"] == "best"
    with pytest.raises(TypeError):
        result.diagnostic["state_summary"] = "direct mutation"


class RecordingAgentClient:
    def __init__(self, *, duplicate_mode=None):
        self.roles = []
        self.payloads = []
        self.duplicate_mode = duplicate_mode

    def generate_json(
        self,
        role,
        system_prompt,
        payload,
        response_validator,
        **kwargs,
    ):
        self.roles.append(role)
        self.payloads.append((role, payload))
        if role == "diagnostic":
            value = {
                "state_summary": "updates differ",
                "risks": ["oscillation"],
                "priorities": ["stability"],
            }
        elif role.endswith("proposer"):
            action_variants = {
                "performance_proposer": {
                    "weights": {"client-a": 0.6, "client-b": 0.4},
                    "server_lr_scale": 1.0,
                    "update_clip_norm": 1.0,
                },
                "stability_proposer": {
                    "weights": {"client-a": 0.55, "client-b": 0.45},
                    "server_lr_scale": 0.75,
                    "update_clip_norm": 0.5,
                },
                "balance_proposer": {
                    "weights": {"client-a": 0.5, "client-b": 0.5},
                    "server_lr_scale": 1.25,
                    "update_clip_norm": 2.0,
                },
            }
            if self.duplicate_mode == "alias" and role == "stability_proposer":
                action_variants[role] = action_variants["performance_proposer"]
            candidate_id = f"{role}_01"
            if self.duplicate_mode == "id" and role == "stability_proposer":
                candidate_id = "performance_proposer_01"
            value = {
                "candidates": [
                    proposal(
                        candidate_id=candidate_id,
                        source=role,
                        **action_variants[role],
                    )
                ]
            }
        elif role == "critic":
            value = {
                "accepted_candidate_ids": [
                    "performance_proposer_01",
                    "stability_proposer_01",
                ],
                "rejected": [
                    {
                        "candidate_id": "balance_proposer_01",
                        "reason": "client-b degradation risk",
                    }
                ],
            }
        else:
            value = {
                "selected_candidate_id": "performance_proposer_01",
                "rationale": "best admissible vote summary",
                "risk_acknowledgement": "safety gate retains final authority",
            }
        return response_validator(value)


def vote_record(client_id, candidate_id):
    return {
        "client_id": client_id,
        "candidate_id": candidate_id,
        "sample_count": 10,
        "val_mape": 0.3,
        "val_rmse": 1.0,
        "relative_mape": -0.01,
        "relative_rmse": -0.01,
        "rank": 1,
        "confidence": 0.8,
        "catastrophic_degradation": False,
    }


def complete_votes(candidate_ids):
    return [
        vote_record(client_id, candidate_id)
        for client_id in ("client-a", "client-b")
        for candidate_id in candidate_ids
    ]


def test_orchestrator_calls_all_six_real_roles_once_in_fixed_round_order(
    safe_payload,
):
    client = RecordingAgentClient()
    orchestrator = MultiAgentOrchestrator(client, PROMPT_DIR)
    result = orchestrator.run_round(
        telemetry_payload=safe_payload,
        anchor_candidates=(),
        client_votes=complete_votes(
            ("performance_proposer_01", "stability_proposer_01")
        ),
    )
    assert client.roles == list(ROLE_NAMES)
    assert result.coordination["selected_candidate_id"] == (
        "performance_proposer_01"
    )
    assert len(result.proposals) == 3
    coordinator_payload = client.payloads[-1][1]
    assert coordinator_payload["critique"] == {
        "accepted_candidate_ids": [
            "performance_proposer_01",
            "stability_proposer_01",
        ],
        "rejected": [
            {
                "candidate_id": "balance_proposer_01",
                "reason": "client-b degradation risk",
            }
        ],
    }
    assert coordinator_payload["anchor_candidate_ids"] == []


def test_orchestrator_marks_anchors_separately_from_real_critic_evidence(
    safe_payload,
):
    anchor = CandidateAction(
        candidate_id="anchor_fedavg",
        weights={"client-a": 0.5, "client-b": 0.5},
        server_optimizer="fedavg",
        server_lr_scale=1.0,
        update_clip_norm=None,
        source="anchor",
        rationale="deterministic anchor",
    )
    client = RecordingAgentClient()
    result = MultiAgentOrchestrator(client, PROMPT_DIR).run_round(
        telemetry_payload=safe_payload,
        anchor_candidates=(anchor,),
        client_votes=complete_votes(
            (
                "anchor_fedavg",
                "performance_proposer_01",
                "stability_proposer_01",
            )
        ),
    )
    assert result.coordination["selected_candidate_id"] == "performance_proposer_01"
    coordinator_payload = client.payloads[-1][1]
    assert coordinator_payload["anchor_candidate_ids"] == ["anchor_fedavg"]
    assert "anchor_fedavg" not in coordinator_payload["critique"][
        "accepted_candidate_ids"
    ]
    assert coordinator_payload["critique"]["rejected"][0]["reason"] == (
        "client-b degradation risk"
    )


@pytest.mark.parametrize(
    "invalid_votes",
    [
        [],
        complete_votes(
            ("performance_proposer_01", "stability_proposer_01")
        )[:-1],
        complete_votes(
            ("performance_proposer_01", "stability_proposer_01")
        )
        + [vote_record("client-a", "performance_proposer_01")],
        complete_votes(
            ("performance_proposer_01", "stability_proposer_01")
        )
        + [vote_record("unknown-client", "performance_proposer_01")],
        complete_votes(
            ("performance_proposer_01", "stability_proposer_01")
        )
        + [vote_record("client-a", "unknown-candidate")],
    ],
)
def test_orchestrator_rejects_incomplete_extra_or_duplicate_vote_matrix(
    safe_payload,
    invalid_votes,
):
    client = RecordingAgentClient()
    with pytest.raises(DeepSeekCallError) as error:
        MultiAgentOrchestrator(client, PROMPT_DIR).run_round(
            telemetry_payload=safe_payload,
            anchor_candidates=(),
            client_votes=invalid_votes,
        )
    assert error.value.category == "schema"
    assert error.value.role == "orchestrator"
    assert client.roles == list(ROLE_NAMES[:-1])


@pytest.mark.parametrize("duplicate_mode", ["alias", "id"])
def test_orchestrator_rejects_action_aliases_and_duplicate_ids_before_critic(
    safe_payload,
    duplicate_mode,
):
    client = RecordingAgentClient(duplicate_mode=duplicate_mode)
    with pytest.raises(DeepSeekCallError) as error:
        MultiAgentOrchestrator(client, PROMPT_DIR).run_round(
            telemetry_payload=safe_payload,
            anchor_candidates=(),
            client_votes=(),
        )
    assert error.value.category == "schema"
    assert error.value.role == "stability_proposer"
    assert client.roles == [
        "diagnostic",
        "performance_proposer",
        "stability_proposer",
    ]


def test_append_only_telemetry_explicitly_flushes_and_fsyncs(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
    real_open = builtins.open
    real_fsync = telemetry_module.os.fsync
    output_spies = []
    fsynced_descriptors = []

    class FlushSpy:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.descriptor = wrapped.fileno()
            self.flush_calls = 0

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def write(self, value):
            return self.wrapped.write(value)

        def flush(self):
            self.flush_calls += 1
            return self.wrapped.flush()

        def fileno(self):
            return self.wrapped.fileno()

    def open_spy(file, mode="r", *args, **kwargs):
        stream = real_open(file, mode, *args, **kwargs)
        if Path(file) == path and mode == "a+b":
            spy = FlushSpy(stream)
            output_spies.append(spy)
            return spy
        return stream

    def fsync_spy(descriptor):
        fsynced_descriptors.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(builtins, "open", open_spy)
    monkeypatch.setattr(telemetry_module.os, "fsync", fsync_spy)
    writer.append(
        {
            "request_hash": "r",
            "prompt_hash": "p",
            "model": "deepseek-chat",
            "role": "diagnostic",
            "response_text": "{}",
            "parsed_response": {},
            "timing_seconds": 0.1,
            "candidate_decision": None,
            "failure_category": None,
        }
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["role"] == "diagnostic"
    assert len(output_spies) == 1
    assert output_spies[0].flush_calls == 1
    assert output_spies[0].descriptor in fsynced_descriptors


def test_telemetry_serialization_failure_cannot_leave_a_partial_line(tmp_path):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
    with pytest.raises((TypeError, ValueError)):
        writer.append({"role": "diagnostic", "bad": object()})
    assert not path.exists() or path.read_bytes() == b""


def test_partial_os_write_rolls_back_to_last_complete_record(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
    writer.append({"index": "old"})
    old_bytes = path.read_bytes()
    real_write = telemetry_module.os.write

    def partial_write(descriptor, value):
        prefix = value[: max(1, len(value) // 3)]
        real_write(descriptor, prefix)
        return len(prefix)

    monkeypatch.setattr(telemetry_module.os, "write", partial_write)
    with pytest.raises(OSError):
        writer.append({"index": "must-rollback", "padding": "x" * 100})
    assert path.read_bytes() == old_bytes


def test_flush_failure_rolls_back_to_last_complete_record(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
    writer.append({"index": "old"})
    old_bytes = path.read_bytes()
    real_open = builtins.open

    class FlushFailure:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def flush(self):
            raise OSError("simulated flush failure")

    def failing_open(file, mode="r", *args, **kwargs):
        stream = real_open(file, mode, *args, **kwargs)
        if Path(file) == path and mode == "a+b":
            return FlushFailure(stream)
        return stream

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(OSError, match="flush"):
        writer.append({"index": "must-rollback"})
    assert path.read_bytes() == old_bytes


def test_fsync_failure_rolls_back_to_last_complete_record(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
    writer.append({"index": "old"})
    old_bytes = path.read_bytes()

    def fail_fsync(_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(telemetry_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync"):
        writer.append({"index": "must-rollback"})
    assert path.read_bytes() == old_bytes


def test_next_append_recovers_a_crash_left_partial_suffix(tmp_path):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
    writer.append({"index": "old"})
    with open(path, "ab") as stream:
        stream.write(b'{"index":"crash-fragment"')
        stream.flush()
    writer.append({"index": "new"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["index"] for line in lines] == ["old", "new"]


def test_telemetry_refuses_secret_bearing_fields(tmp_path):
    writer = AppendOnlyTelemetry(tmp_path / "agent_calls.jsonl")
    for secret_field in ("api_key", "authorization", "headers", "payload"):
        with pytest.raises(ValueError):
            writer.append({secret_field: "secret"})


@pytest.mark.parametrize(
    "secret_field",
    [
        "request_headers_backup",
        "client_secret_note",
        "access_token_cache",
        "nested_authorization_value",
        "ＡＰＩ＿ＫＥＹ",
    ],
)
def test_telemetry_rejects_normalized_sensitive_key_fragments(
    tmp_path,
    secret_field,
):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
    with pytest.raises(ValueError):
        writer.append({"safe": {secret_field: "must-not-write"}})
    assert not path.exists()


def test_telemetry_recursively_redacts_known_and_header_secrets(tmp_path):
    path = tmp_path / "agent_calls.jsonl"
    known_secret = "fake-api-key-never-log"
    bearer_secret = "standalone-bearer-token"
    header_secret = "plain-authorization-value"
    writer = AppendOnlyTelemetry(path, known_secrets=(known_secret,))
    writer.append(
        {
            "response_text": (
                f"echo={known_secret}; Bearer {bearer_secret}; "
                f"Authorization: {header_secret}"
            ),
            "parsed_response": {
                "nested": [
                    {"value": known_secret},
                    f"Bearer {bearer_secret}",
                    f"authorization={header_secret}",
                ]
            },
        }
    )
    raw = path.read_text(encoding="utf-8")
    assert known_secret not in raw
    assert bearer_secret not in raw
    assert header_secret not in raw
    assert "[REDACTED]" in raw
    record = json.loads(raw)
    assert record["parsed_response"]["nested"][0]["value"] == "[REDACTED]"


def test_telemetry_removes_entire_basic_and_bearer_credentials(tmp_path):
    path = tmp_path / "agent_calls.jsonl"
    basic_value = "QWxhZGRpbjpvcGVuIHNlc2FtZQ=="
    bearer_value = "header.payload.signature"
    writer = AppendOnlyTelemetry(path)
    writer.append(
        {
            "response_text": (
                f"Authorization: Basic {basic_value}; "
                f"Authorization: Bearer {bearer_value}; "
                f"Basic {basic_value}; Bearer {bearer_value}"
            )
        }
    )
    raw = path.read_text(encoding="utf-8")
    assert basic_value not in raw
    assert bearer_value not in raw
    assert "Basic" not in raw
    assert "Bearer" not in raw
    assert raw.count("[REDACTED]") >= 4


def test_concurrent_telemetry_appends_never_interleave_lines(tmp_path):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
    threads = [
        Thread(target=writer.append, args=({"index": index},))
        for index in range(30)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(record["index"] for record in records) == list(range(30))


def test_client_telemetry_records_call_without_secrets(tmp_path, safe_payload):
    path = tmp_path / "agent_calls.jsonl"
    session = FakeSession(FakeResponse(content='{"ok":true}'))
    client = make_client(session, telemetry=AppendOnlyTelemetry(path))
    assert client.generate_json(
        "diagnostic", "Return JSON.", safe_payload, lambda value: value
    ) == {"ok": True}
    record = json.loads(path.read_text(encoding="utf-8"))
    assert set(
        (
            "request_hash",
            "prompt_hash",
            "model",
            "role",
            "response_text",
            "parsed_response",
            "timing_seconds",
            "candidate_decision",
            "failure_category",
        )
    ) <= set(record)
    serialized = json.dumps(record, sort_keys=True)
    assert "test-only-secret" not in serialized
    assert "Authorization" not in serialized
    assert "client-a" not in serialized


def test_client_registers_key_and_redacts_response_text_and_nested_values(
    tmp_path,
    safe_payload,
):
    path = tmp_path / "agent_calls.jsonl"
    api_key = "fake-client-key-never-log"
    bearer_token = "fake-response-bearer"
    response = {
        "echo": api_key,
        "nested": {"value": f"Bearer {bearer_token}"},
    }
    session = FakeSession(FakeResponse(content=json.dumps(response)))
    client = make_client(
        session,
        api_key=api_key,
        telemetry=AppendOnlyTelemetry(path),
    )
    assert client.generate_json(
        "diagnostic", "Return JSON.", safe_payload, lambda value: value
    ) == response
    raw = path.read_text(encoding="utf-8")
    assert api_key not in raw
    assert bearer_token not in raw
    assert "[REDACTED]" in raw
    record = json.loads(raw)
    assert "[REDACTED]" in record["response_text"]
    assert record["parsed_response"]["echo"] == "[REDACTED]"


def test_http_exception_detail_is_logged_only_after_value_redaction(
    tmp_path,
    safe_payload,
):
    path = tmp_path / "agent_calls.jsonl"
    api_key = "fake-http-key-never-log"
    bearer_token = "fake-http-bearer"
    header_value = "fake-http-authorization"
    session = FakeSession(
        FakeResponse(
            content="{}",
            status_code=500,
            http_error_message=(
                f"key={api_key}; Bearer {bearer_token}; "
                f"Authorization: {header_value}"
            ),
        )
    )
    client = make_client(
        session,
        api_key=api_key,
        telemetry=AppendOnlyTelemetry(path),
    )
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "critic", "Return JSON.", safe_payload, lambda value: value
        )
    assert session.calls == 1
    assert error.value.category == "http"
    assert api_key not in str(error.value)
    raw = path.read_text(encoding="utf-8")
    assert api_key not in raw
    assert bearer_token not in raw
    assert header_value not in raw
    assert "[REDACTED]" in raw


@pytest.mark.parametrize(
    ("outcome", "expected_category"),
    [
        (ConnectionError("offline"), "connection"),
        (FakeResponse(content="{}", status_code=500), "http"),
        (FakeResponse(content="not json"), "schema"),
    ],
)
def test_telemetry_failure_never_masks_primary_deepseek_failure(
    safe_payload,
    outcome,
    expected_category,
):
    class BrokenTelemetry:
        def register_secret(self, _secret):
            return None

        def append(self, _record):
            raise OSError("disk failed with Bearer logger-secret")

    session = FakeSession(outcome)
    client = make_client(session, telemetry=BrokenTelemetry())
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "critic", "Return JSON.", safe_payload, lambda value: value
        )
    assert session.calls == 1
    assert error.value.category == expected_category
    assert error.value.role == "critic"
    assert "logger-secret" not in str(error.value)


def test_prompt_files_are_complete_hashable_and_do_not_request_private_data():
    hashes = load_prompt_hashes(PROMPT_DIR)
    assert tuple(hashes) == ROLE_NAMES
    for role, digest in hashes.items():
        path = PROMPT_DIR / f"{role}.md"
        content = path.read_text(encoding="utf-8")
        assert digest == hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert len(digest) == 64
        assert "Allowed input fields:" in content
        assert "Exact JSON output schema:" in content
        assert "exactly one complete JSON object" in content
        assert "Never split" in content
        assert "no prose, markdown" in content
        assert "must not invent" in content.casefold()
        assert "must not request" in content.casefold()
        allowed_section = content.split("Allowed input fields:", 1)[1].split(
            "Unavailable information:", 1
        )[0]
        requested_inputs = allowed_section.casefold()
        for forbidden in (
            "test_mape",
            "raw_features",
            "labels",
            "predictions",
        ):
            assert forbidden not in requested_inputs
