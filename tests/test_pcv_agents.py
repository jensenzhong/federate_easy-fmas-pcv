import hashlib
import json
from pathlib import Path
from threading import Thread

import pytest
import requests

from src.federated_learning.pcv.agents import (
    DeepSeekCallError,
    MultiAgentOrchestrator,
    StrictDeepSeekClient,
    load_prompt_hashes,
    validate_coordinator_response,
    validate_critic_response,
    validate_diagnostic_response,
    validate_proposer_response,
)
from src.federated_learning.pcv.protocol import PrivacyViolation
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
    def __init__(self, *, content=None, status_code=200, json_error=None):
        self.status_code = status_code
        self._json_error = json_error
        self._body = (
            {"choices": [{"message": {"content": content}}]}
            if content is not None
            else {"choices": [{"message": {}}]}
        )

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(
                "sensitive server error detail",
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


def make_client(session, *, api_key="test-only-secret", telemetry=None):
    return StrictDeepSeekClient(
        api_key=api_key,
        model_name="deepseek-chat",
        base_url="https://example.invalid/v1/",
        timeout_seconds=2,
        session=session,
        telemetry=telemetry,
    )


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
def test_timeout_failure_has_one_attempt_and_distinct_category(
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
    assert error.value.category == "timeout"
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


def test_invalid_http_json_is_response_failure(safe_payload):
    session = FakeSession(
        FakeResponse(json_error=ValueError("invalid HTTP response JSON"))
    )
    client = make_client(session)
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            "diagnostic", "Return JSON.", safe_payload, lambda value: value
        )
    assert session.calls == 1
    assert error.value.category == "response"


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
    assert request["json"]["stream"] is False
    assert len(request["json"]["messages"]) == 2


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
    assert validate_critic_response(valid, candidate_ids=("p1", "p2")) == valid
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


class RecordingAgentClient:
    def __init__(self):
        self.roles = []

    def generate_json(
        self,
        role,
        system_prompt,
        payload,
        response_validator,
        **kwargs,
    ):
        self.roles.append(role)
        if role == "diagnostic":
            value = {
                "state_summary": "updates differ",
                "risks": ["oscillation"],
                "priorities": ["stability"],
            }
        elif role.endswith("proposer"):
            value = {
                "candidates": [
                    proposal(
                        candidate_id=f"{role}_01",
                        source=role,
                    )
                ]
            }
        elif role == "critic":
            value = {
                "accepted_candidate_ids": [
                    "performance_proposer_01",
                    "stability_proposer_01",
                    "balance_proposer_01",
                ],
                "rejected": [],
            }
        else:
            value = {
                "selected_candidate_id": "performance_proposer_01",
                "rationale": "best admissible vote summary",
                "risk_acknowledgement": "safety gate retains final authority",
            }
        return response_validator(value)


def test_orchestrator_calls_all_six_real_roles_once_in_fixed_round_order(
    safe_payload,
):
    client = RecordingAgentClient()
    orchestrator = MultiAgentOrchestrator(client, PROMPT_DIR)
    result = orchestrator.run_round(
        telemetry_payload=safe_payload,
        anchor_candidates=(),
        client_votes=(),
    )
    assert client.roles == list(ROLE_NAMES)
    assert result.coordination["selected_candidate_id"] == (
        "performance_proposer_01"
    )
    assert len(result.proposals) == 3


def test_append_only_telemetry_is_valid_jsonl_and_fsynced_contract(tmp_path):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
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


def test_telemetry_serialization_failure_cannot_leave_a_partial_line(tmp_path):
    path = tmp_path / "agent_calls.jsonl"
    writer = AppendOnlyTelemetry(path)
    with pytest.raises((TypeError, ValueError)):
        writer.append({"role": "diagnostic", "bad": object()})
    assert not path.exists() or path.read_bytes() == b""


def test_telemetry_refuses_secret_bearing_fields(tmp_path):
    writer = AppendOnlyTelemetry(tmp_path / "agent_calls.jsonl")
    for secret_field in ("api_key", "authorization", "headers", "payload"):
        with pytest.raises(ValueError):
            writer.append({secret_field: "secret"})


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
