"""Strict DeepSeek calls and FMAS-PCV proposal/critique orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any

import requests

from .protocol import assert_prompt_payload_safe
from .schemas import CandidateAction, LocalCandidateVote
from .telemetry import sanitize_telemetry_value


ROLE_NAMES = (
    "diagnostic",
    "performance_proposer",
    "stability_proposer",
    "balance_proposer",
    "critic",
    "coordinator",
)
PROPOSER_ROLES = ROLE_NAMES[1:4]
_ACTION_FIELDS = frozenset(
    {
        "candidate_id",
        "weights",
        "server_optimizer",
        "server_lr_scale",
        "update_clip_norm",
        "source",
        "rationale",
    }
)
_VOTE_FIELDS = frozenset(
    {
        "client_id",
        "candidate_id",
        "sample_count",
        "val_mape",
        "val_rmse",
        "relative_mape",
        "relative_rmse",
        "rank",
        "confidence",
        "catastrophic_degradation",
    }
)


class DeepSeekCallError(RuntimeError):
    """Sanitized fail-stop error for one DeepSeek role call."""

    def __init__(self, category: str, role: str, message: str):
        super().__init__(f"{category} failure in {role}: {message}")
        self.category = category
        self.role = role


def _exact_object(value: Any, fields: frozenset[str], *, context: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact JSON object")
    if set(value) != fields:
        raise ValueError(f"{context} must contain exactly {sorted(fields)}")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{context} field names must be exact strings")
    return value


def _nonempty_string(value: Any, *, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{context} must be a non-empty exact string")
    return value


def _string_list(value: Any, *, context: str) -> list[str]:
    if type(value) is not list:
        raise ValueError(f"{context} must be an exact JSON array")
    output = []
    for index, item in enumerate(value):
        output.append(_nonempty_string(item, context=f"{context}[{index}]"))
    return output


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _candidate_id_tuple(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    if type(candidate_ids) not in (tuple, list):
        raise ValueError("candidate_ids must be an exact tuple or list")
    values = tuple(candidate_ids)
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("candidate IDs must be non-empty exact strings")
    if len(set(values)) != len(values):
        raise ValueError("candidate IDs must be unique")
    return values


def _client_id_tuple(client_ids: Sequence[str]) -> tuple[str, ...]:
    values = _candidate_id_tuple(client_ids)
    if not values:
        raise ValueError("client_ids must not be empty")
    return values


def validate_diagnostic_response(value: Any) -> Mapping[str, Any]:
    result = _exact_object(
        value,
        frozenset({"state_summary", "risks", "priorities"}),
        context="diagnostic response",
    )
    state_summary = _nonempty_string(
        result["state_summary"], context="state_summary"
    )
    risks = tuple(_string_list(result["risks"], context="risks"))
    priorities = tuple(
        _string_list(result["priorities"], context="priorities")
    )
    return MappingProxyType(
        {
            "state_summary": state_summary,
            "risks": risks,
            "priorities": priorities,
        }
    )


def _candidate_from_json(
    value: Any,
    *,
    client_ids: tuple[str, ...],
    expected_source: str | None,
) -> CandidateAction:
    action = _exact_object(value, _ACTION_FIELDS, context="candidate action")
    if type(action["weights"]) is not dict:
        raise ValueError("candidate weights must be an exact JSON object")
    if any(type(key) is not str for key in action["weights"]):
        raise ValueError("candidate weight client IDs must be exact strings")
    if expected_source is not None and action["source"] != expected_source:
        raise ValueError("candidate source must match its proposer role")
    candidate = CandidateAction(
        candidate_id=action["candidate_id"],
        weights=action["weights"],
        server_optimizer=action["server_optimizer"],
        server_lr_scale=action["server_lr_scale"],
        update_clip_norm=action["update_clip_norm"],
        source=action["source"],
        rationale=action["rationale"],
    )
    candidate.validate(client_ids)
    return candidate


def validate_proposer_response(
    value: Any,
    *,
    client_ids: Sequence[str],
    role: str,
) -> tuple[CandidateAction, ...]:
    if role not in PROPOSER_ROLES:
        raise ValueError("unknown proposer role")
    clients = _client_id_tuple(client_ids)
    result = _exact_object(
        value,
        frozenset({"candidates"}),
        context=f"{role} response",
    )
    candidates_json = result["candidates"]
    if type(candidates_json) is not list:
        raise ValueError("candidates must be an exact JSON array")
    if len(candidates_json) > 2:
        raise ValueError("each proposer may return at most two candidates")
    candidates = tuple(
        _candidate_from_json(
            candidate,
            client_ids=clients,
            expected_source=role,
        )
        for candidate in candidates_json
    )
    ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs are forbidden")
    return candidates


def validate_critic_response(
    value: Any,
    *,
    candidate_ids: Sequence[str],
) -> Mapping[str, Any]:
    known_ids = _candidate_id_tuple(candidate_ids)
    known_set = set(known_ids)
    result = _exact_object(
        value,
        frozenset({"accepted_candidate_ids", "rejected"}),
        context="critic response",
    )
    accepted = _string_list(
        result["accepted_candidate_ids"],
        context="accepted_candidate_ids",
    )
    if len(set(accepted)) != len(accepted):
        raise ValueError("accepted candidate IDs must be unique")
    if not set(accepted) <= known_set:
        raise ValueError("critic accepted an unknown candidate ID")

    rejected_json = result["rejected"]
    if type(rejected_json) is not list:
        raise ValueError("rejected must be an exact JSON array")
    rejected_ids = []
    rejected_values = []
    for index, rejected in enumerate(rejected_json):
        item = _exact_object(
            rejected,
            frozenset({"candidate_id", "reason"}),
            context=f"rejected[{index}]",
        )
        rejected_id = _nonempty_string(
            item["candidate_id"],
            context=f"rejected[{index}].candidate_id",
        )
        reason = _nonempty_string(
            item["reason"], context=f"rejected[{index}].reason"
        )
        rejected_ids.append(rejected_id)
        rejected_values.append(
            MappingProxyType(
                {"candidate_id": rejected_id, "reason": reason}
            )
        )
    if len(set(rejected_ids)) != len(rejected_ids):
        raise ValueError("rejected candidate IDs must be unique")
    if not set(rejected_ids) <= known_set:
        raise ValueError("critic rejected an unknown candidate ID")
    if set(accepted) & set(rejected_ids):
        raise ValueError("a candidate cannot be both accepted and rejected")
    if set(accepted) | set(rejected_ids) != known_set:
        raise ValueError("critic must classify every provided candidate")
    return MappingProxyType(
        {
            "accepted_candidate_ids": tuple(accepted),
            "rejected": tuple(rejected_values),
        }
    )


def validate_coordinator_response(
    value: Any,
    *,
    candidate_ids: Sequence[str],
) -> Mapping[str, str]:
    known_ids = _candidate_id_tuple(candidate_ids)
    result = _exact_object(
        value,
        frozenset(
            {
                "selected_candidate_id",
                "rationale",
                "risk_acknowledgement",
            }
        ),
        context="coordinator response",
    )
    selected_id = _nonempty_string(
        result["selected_candidate_id"],
        context="selected_candidate_id",
    )
    if selected_id not in set(known_ids):
        raise ValueError("coordinator selected an unknown candidate ID")
    rationale = _nonempty_string(result["rationale"], context="rationale")
    risk_acknowledgement = _nonempty_string(
        result["risk_acknowledgement"],
        context="risk_acknowledgement",
    )
    return MappingProxyType(
        {
            "selected_candidate_id": selected_id,
            "rationale": rationale,
            "risk_acknowledgement": risk_acknowledgement,
        }
    )


def _strict_json_object(content: str) -> dict[str, Any]:
    def reject_constant(value: str):
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    parsed = json.loads(
        content,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    if type(parsed) is not dict:
        raise ValueError("agent content must be exactly one JSON object")
    return parsed


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _base_prompt_payload(payload: Any) -> dict[str, Any]:
    if type(payload) is not dict:
        return payload
    if "round_index" not in payload or "clients" not in payload:
        return payload
    return {
        "round_index": payload["round_index"],
        "clients": payload["clients"],
    }


def _assert_context_payload_safe(role: str, payload: Any) -> None:
    if type(payload) is not dict:
        raise ValueError("agent payload must be an exact dictionary")
    allowed_fields = {
        "diagnostic": {"round_index", "clients"},
        "performance_proposer": {"round_index", "clients", "diagnostic"},
        "stability_proposer": {"round_index", "clients", "diagnostic"},
        "balance_proposer": {"round_index", "clients", "diagnostic"},
        "critic": {"round_index", "clients", "diagnostic", "candidates"},
        "coordinator": {
            "round_index",
            "clients",
            "diagnostic",
            "candidates",
            "critique",
            "anchor_candidate_ids",
            "client_votes",
        },
    }.get(role, {"round_index", "clients"})
    if not set(payload) <= allowed_fields:
        raise ValueError("agent payload contains an unapproved role field")
    if "diagnostic" in payload:
        validate_diagnostic_response(payload["diagnostic"])
    if "candidates" in payload:
        if type(payload["candidates"]) is not list:
            raise ValueError("candidate context must be an exact JSON array")
        if len(payload["candidates"]) > 8:
            raise ValueError("candidate context exceeds the round budget")
        client_ids = tuple(client["client_id"] for client in payload["clients"])
        for candidate in payload["candidates"]:
            _candidate_from_json(
                candidate,
                client_ids=client_ids,
                expected_source=None,
            )
    if "critique" in payload:
        admissible_candidate_ids = tuple(
            candidate["candidate_id"] for candidate in payload["candidates"]
        )
        anchor_ids_json = payload.get("anchor_candidate_ids", [])
        if type(anchor_ids_json) is not list:
            raise ValueError("anchor_candidate_ids must be an exact JSON array")
        anchor_ids = _candidate_id_tuple(anchor_ids_json)
        if not set(anchor_ids) <= set(admissible_candidate_ids):
            raise ValueError("anchor_candidate_ids must refer to candidates")
        critique_json = _exact_object(
            payload["critique"],
            frozenset({"accepted_candidate_ids", "rejected"}),
            context="critic context",
        )
        accepted_ids = _string_list(
            critique_json["accepted_candidate_ids"],
            context="critic context accepted_candidate_ids",
        )
        if type(critique_json["rejected"]) is not list:
            raise ValueError("critic context rejected must be an exact JSON array")
        rejected_ids = []
        for index, rejected in enumerate(critique_json["rejected"]):
            item = _exact_object(
                rejected,
                frozenset({"candidate_id", "reason"}),
                context=f"critic context rejected[{index}]",
            )
            rejected_ids.append(item["candidate_id"])
        critic_candidate_ids = tuple([*accepted_ids, *rejected_ids])
        validate_critic_response(
            payload["critique"],
            candidate_ids=critic_candidate_ids,
        )
        if set(anchor_ids) & set(critic_candidate_ids):
            raise ValueError("anchors must not be represented as critic evidence")
        if set(accepted_ids) != set(admissible_candidate_ids) - set(anchor_ids):
            raise ValueError("critic accepted IDs must match non-anchor candidates")
    if "client_votes" in payload:
        _validate_vote_context(
            payload["client_votes"],
            client_ids=tuple(client["client_id"] for client in payload["clients"]),
            candidate_ids=tuple(
                candidate["candidate_id"] for candidate in payload["candidates"]
            ),
        )


def _finite_exact_number(value: Any, *, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite exact number")
    return float(value)


def _validate_vote_context(
    votes: Any,
    *,
    client_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
) -> None:
    if type(votes) is not list:
        raise ValueError("client_votes must be an exact JSON array")
    known_clients = set(client_ids)
    known_candidates = set(candidate_ids)
    seen_pairs = set()
    for index, vote in enumerate(votes):
        item = _exact_object(vote, _VOTE_FIELDS, context=f"client_votes[{index}]")
        client_id = _nonempty_string(item["client_id"], context="vote client_id")
        candidate_id = _nonempty_string(
            item["candidate_id"], context="vote candidate_id"
        )
        if client_id not in known_clients or candidate_id not in known_candidates:
            raise ValueError("vote refers to an unknown client or candidate")
        pair = (client_id, candidate_id)
        if pair in seen_pairs:
            raise ValueError("duplicate client/candidate vote")
        seen_pairs.add(pair)
        for field in (
            "val_mape",
            "val_rmse",
            "relative_mape",
            "relative_rmse",
            "confidence",
        ):
            _finite_exact_number(item[field], context=f"vote {field}")
        for field in ("sample_count", "rank"):
            if type(item[field]) is not int or item[field] <= 0:
                raise ValueError(f"vote {field} must be a positive exact integer")
        if type(item["catastrophic_degradation"]) is not bool:
            raise ValueError("vote catastrophic_degradation must be an exact boolean")


def _validate_complete_vote_matrix(
    votes: list[dict[str, Any]],
    *,
    client_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
) -> None:
    if not votes:
        raise ValueError("client vote matrix must not be empty")
    _validate_vote_context(
        votes,
        client_ids=client_ids,
        candidate_ids=candidate_ids,
    )
    expected_pairs = {
        (client_id, candidate_id)
        for client_id in client_ids
        for candidate_id in candidate_ids
    }
    actual_pairs = {
        (vote["client_id"], vote["candidate_id"])
        for vote in votes
    }
    if len(votes) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise ValueError(
            "client votes must contain exactly one vote per client/candidate pair"
        )


class StrictDeepSeekClient:
    """Make exactly one HTTP POST per call and fail without repair or fallback."""

    def __init__(
        self,
        api_key,
        model_name,
        base_url,
        timeout_seconds,
        session,
        telemetry=None,
    ):
        if type(api_key) is not str or not api_key.strip():
            raise DeepSeekCallError(
                "authentication",
                "preflight",
                "API key is missing",
            )
        if type(model_name) is not str or not model_name.strip():
            raise ValueError("model_name must be a non-empty exact string")
        if type(base_url) is not str or not base_url.strip():
            raise ValueError("base_url must be a non-empty exact string")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if not callable(getattr(session, "post", None)):
            raise TypeError("session must provide a callable post method")
        if telemetry is not None and not callable(getattr(telemetry, "append", None)):
            raise TypeError("telemetry must provide a callable append method")
        self._api_key = api_key
        self.model_name = model_name
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.session = session
        self.telemetry = telemetry
        if self.telemetry is not None:
            register_secret = getattr(self.telemetry, "register_secret", None)
            if callable(register_secret):
                register_secret(api_key)

    def _record(
        self,
        *,
        request_hash: str,
        prompt_hash: str,
        role: str,
        response_text: str | None,
        parsed_response: dict[str, Any] | None,
        started: float,
        failure_category: str | None,
        failure_detail: str | None,
    ) -> None:
        if self.telemetry is None:
            return
        candidate_decision = None
        if parsed_response is not None and "selected_candidate_id" in parsed_response:
            candidate_decision = parsed_response["selected_candidate_id"]
        record = sanitize_telemetry_value(
            {
                "request_hash": request_hash,
                "prompt_hash": prompt_hash,
                "model": self.model_name,
                "role": role,
                "response_text": response_text,
                "parsed_response": parsed_response,
                "timing_seconds": max(0.0, time.perf_counter() - started),
                "candidate_decision": candidate_decision,
                "failure_category": failure_category,
                "failure_detail": failure_detail,
            },
            known_secrets=(self._api_key,),
        )
        self.telemetry.append(record)

    def _fail(
        self,
        category: str,
        role: str,
        message: str,
        *,
        request_hash: str,
        prompt_hash: str,
        response_text: str | None,
        parsed_response: dict[str, Any] | None,
        started: float,
        failure_detail: str | None = None,
    ) -> None:
        try:
            self._record(
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                role=role,
                response_text=response_text,
                parsed_response=parsed_response,
                started=started,
                failure_category=category,
                failure_detail=failure_detail,
            )
        except Exception:
            # Audit storage is best effort on the failure path. It must never
            # replace or reclassify the primary provider/transport failure.
            pass
        raise DeepSeekCallError(category, role, message) from None

    def generate_json(
        self,
        role,
        system_prompt,
        payload,
        response_validator: Callable[[dict[str, Any]], Any],
    ):
        # This aggregate-only check intentionally precedes request construction.
        assert_prompt_payload_safe(_base_prompt_payload(payload))
        _assert_context_payload_safe(role, payload)
        if type(role) is not str or not role.strip():
            raise ValueError("role must be a non-empty exact string")
        if type(system_prompt) is not str or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty exact string")
        if not callable(response_validator):
            raise TypeError("response_validator must be callable")

        request_hash = _canonical_json_hash(payload)
        prompt_hash = sha256(system_prompt.encode("utf-8")).hexdigest()
        started = time.perf_counter()
        response_text = None
        parsed_response = None
        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                },
            ],
            "temperature": 0.8,
            "stream": False,
        }
        try:
            response = self.session.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except (requests.Timeout, TimeoutError) as error:
            self._fail(
                "connection",
                role,
                "request timed out",
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                response_text=None,
                parsed_response=None,
                started=started,
                failure_detail=str(error),
            )
        except requests.HTTPError as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            category = "authentication" if status in (401, 403) else "http"
            self._fail(
                category,
                role,
                f"HTTP status {status if type(status) is int else 'unknown'}",
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                response_text=None,
                parsed_response=None,
                started=started,
                failure_detail=str(error),
            )
        except (requests.RequestException, ConnectionError, OSError) as error:
            self._fail(
                "connection",
                role,
                "request could not connect",
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                response_text=None,
                parsed_response=None,
                started=started,
                failure_detail=str(error),
            )
        except Exception as error:
            self._fail(
                "connection",
                role,
                "request transport failed",
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                response_text=None,
                parsed_response=None,
                started=started,
                failure_detail=str(error),
            )

        try:
            response_document = response.json()
            if type(response_document) is not dict:
                raise ValueError("HTTP response must be a JSON object")
            choices = response_document["choices"]
            if type(choices) is not list or len(choices) < 1:
                raise ValueError("HTTP response choices are missing")
            first_choice = choices[0]
            if type(first_choice) is not dict:
                raise ValueError("HTTP response choice must be an object")
            message = first_choice["message"]
            if type(message) is not dict:
                raise ValueError("HTTP response message must be an object")
            response_text = message["content"]
            if type(response_text) is not str:
                raise ValueError("HTTP response content must be a string")
        except Exception as error:
            self._fail(
                "schema",
                role,
                "provider response envelope is invalid",
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                response_text=None,
                parsed_response=None,
                started=started,
                failure_detail=str(error),
            )

        try:
            parsed_response = _strict_json_object(response_text)
            validated = response_validator(parsed_response)
        except Exception as error:
            self._fail(
                "schema",
                role,
                "agent JSON failed exact schema validation",
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                response_text=response_text,
                parsed_response=parsed_response,
                started=started,
                failure_detail=str(error),
            )

        self._record(
            request_hash=request_hash,
            prompt_hash=prompt_hash,
            role=role,
            response_text=response_text,
            parsed_response=parsed_response,
            started=started,
            failure_category=None,
            failure_detail=None,
        )
        return validated


def _candidate_to_json(candidate: CandidateAction) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "weights": dict(candidate.weights),
        "server_optimizer": candidate.server_optimizer,
        "server_lr_scale": candidate.server_lr_scale,
        "update_clip_norm": candidate.update_clip_norm,
        "source": candidate.source,
        "rationale": candidate.rationale,
    }


def _candidate_execution_signature(candidate: CandidateAction) -> tuple[Any, ...]:
    return (
        tuple(
            sorted(
                (client_id, round(float(weight), 12))
                for client_id, weight in candidate.weights.items()
            )
        ),
        candidate.server_optimizer,
        round(float(candidate.server_lr_scale), 12),
        (
            None
            if candidate.update_clip_norm is None
            else round(float(candidate.update_clip_norm), 12)
        ),
    )


def _vote_to_json(vote: LocalCandidateVote | Mapping[str, Any]) -> dict[str, Any]:
    if type(vote) is LocalCandidateVote:
        return {field: getattr(vote, field) for field in _VOTE_FIELDS}
    if type(vote) is dict:
        return dict(vote)
    raise TypeError("client votes must be LocalCandidateVote or exact dictionaries")


def load_prompt_hashes(prompt_dir: str | Path) -> dict[str, str]:
    directory = Path(prompt_dir)
    hashes = {}
    for role in ROLE_NAMES:
        content = (directory / f"{role}.md").read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"prompt file is empty for role {role}")
        hashes[role] = sha256(content.encode("utf-8")).hexdigest()
    return hashes


@dataclass(frozen=True)
class OrchestrationResult:
    diagnostic: Mapping[str, Any]
    proposals: tuple[CandidateAction, ...]
    critique: Mapping[str, Any]
    coordination: Mapping[str, str]

    def __post_init__(self) -> None:
        proposals = tuple(self.proposals)
        if any(type(candidate) is not CandidateAction for candidate in proposals):
            raise TypeError("proposals must contain exact CandidateAction values")
        object.__setattr__(self, "diagnostic", _deep_freeze_json(self.diagnostic))
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(self, "critique", _deep_freeze_json(self.critique))
        object.__setattr__(self, "coordination", _deep_freeze_json(self.coordination))


class MultiAgentOrchestrator:
    """Run the six FMAS roles exactly once in their frozen round order."""

    def __init__(self, client: StrictDeepSeekClient, prompt_dir: str | Path):
        if not callable(getattr(client, "generate_json", None)):
            raise TypeError("client must provide generate_json")
        self.client = client
        directory = Path(prompt_dir)
        self.prompts = {
            role: (directory / f"{role}.md").read_text(encoding="utf-8")
            for role in ROLE_NAMES
        }
        if any(not prompt.strip() for prompt in self.prompts.values()):
            raise ValueError("all six role prompts must be non-empty")

    def _call(self, role: str, payload: dict[str, Any], validator: Callable):
        try:
            return self.client.generate_json(
                role,
                self.prompts[role],
                payload,
                validator,
            )
        except DeepSeekCallError:
            raise
        except Exception:
            raise DeepSeekCallError(
                "schema", role, "agent response failed exact validation"
            ) from None

    def run_round(
        self,
        *,
        telemetry_payload: dict[str, Any],
        anchor_candidates: Sequence[CandidateAction],
        client_votes: Sequence[LocalCandidateVote | Mapping[str, Any]],
    ) -> OrchestrationResult:
        assert_prompt_payload_safe(telemetry_payload)
        clients = _client_id_tuple(
            tuple(
                client["client_id"] for client in telemetry_payload["clients"]
            )
        )
        anchors = tuple(anchor_candidates)
        if len(anchors) > 2:
            raise DeepSeekCallError(
                "schema", "orchestrator", "anchor candidate budget exceeds two"
            )
        for anchor in anchors:
            if type(anchor) is not CandidateAction:
                raise DeepSeekCallError(
                    "schema", "orchestrator", "anchor must be CandidateAction"
                )
            try:
                anchor.validate(clients)
            except Exception:
                raise DeepSeekCallError(
                    "schema", "orchestrator", "anchor candidate is invalid"
                ) from None
        anchor_ids = tuple(anchor.candidate_id for anchor in anchors)
        if len(set(anchor_ids)) != len(anchor_ids):
            raise DeepSeekCallError(
                "schema", "orchestrator", "duplicate anchor candidate IDs"
            )
        seen_signatures: dict[tuple[Any, ...], str] = {}
        for anchor in anchors:
            signature = _candidate_execution_signature(anchor)
            if signature in seen_signatures:
                raise DeepSeekCallError(
                    "schema", "orchestrator", "anchor action aliases another anchor"
                )
            seen_signatures[signature] = anchor.candidate_id

        base_payload = {
            "round_index": telemetry_payload["round_index"],
            "clients": telemetry_payload["clients"],
        }
        diagnostic = self._call(
            "diagnostic",
            base_payload,
            validate_diagnostic_response,
        )
        diagnostic_json = _thaw_json(diagnostic)

        proposals: list[CandidateAction] = []
        seen_ids = set(anchor_ids)
        for role in PROPOSER_ROLES:
            role_payload = {**base_payload, "diagnostic": diagnostic_json}
            role_proposals = self._call(
                role,
                role_payload,
                lambda value, role=role: validate_proposer_response(
                    value,
                    client_ids=clients,
                    role=role,
                ),
            )
            for candidate in role_proposals:
                if candidate.candidate_id in seen_ids:
                    raise DeepSeekCallError(
                        "schema", role, "duplicate candidate ID across round proposals"
                    )
                signature = _candidate_execution_signature(candidate)
                if signature in seen_signatures:
                    raise DeepSeekCallError(
                        "schema",
                        role,
                        "candidate action aliases an existing round action",
                    )
                seen_ids.add(candidate.candidate_id)
                seen_signatures[signature] = candidate.candidate_id
                proposals.append(candidate)
        if len(anchors) + len(proposals) > 8:
            raise DeepSeekCallError(
                "schema", "orchestrator", "round candidate budget exceeds eight"
            )

        proposal_json = [_candidate_to_json(candidate) for candidate in proposals]
        critique = self._call(
            "critic",
            {
                **base_payload,
                "diagnostic": diagnostic_json,
                "candidates": proposal_json,
            },
            lambda value: validate_critic_response(
                value,
                candidate_ids=tuple(
                    candidate.candidate_id for candidate in proposals
                ),
            ),
        )
        accepted_ids = tuple(critique["accepted_candidate_ids"])
        accepted_set = set(accepted_ids)
        admissible = anchors + tuple(
            candidate
            for candidate in proposals
            if candidate.candidate_id in accepted_set
        )
        if not admissible:
            raise DeepSeekCallError(
                "schema", "critic", "critic left no candidate for coordination"
            )

        votes_json = [_vote_to_json(vote) for vote in client_votes]
        admissible_ids = tuple(
            candidate.candidate_id for candidate in admissible
        )
        try:
            _validate_complete_vote_matrix(
                votes_json,
                client_ids=clients,
                candidate_ids=admissible_ids,
            )
        except Exception:
            raise DeepSeekCallError(
                "schema",
                "orchestrator",
                "client vote matrix is incomplete or invalid",
            ) from None
        all_candidate_json = [_candidate_to_json(candidate) for candidate in admissible]
        coordination = self._call(
            "coordinator",
            {
                **base_payload,
                "diagnostic": diagnostic_json,
                "candidates": all_candidate_json,
                "critique": _thaw_json(critique),
                "anchor_candidate_ids": list(anchor_ids),
                "client_votes": votes_json,
            },
            lambda value: validate_coordinator_response(
                value,
                candidate_ids=admissible_ids,
            ),
        )
        return OrchestrationResult(
            diagnostic=diagnostic,
            proposals=tuple(proposals),
            critique=critique,
            coordination=coordination,
        )


__all__ = [
    "DeepSeekCallError",
    "MultiAgentOrchestrator",
    "OrchestrationResult",
    "ROLE_NAMES",
    "StrictDeepSeekClient",
    "load_prompt_hashes",
    "validate_coordinator_response",
    "validate_critic_response",
    "validate_diagnostic_response",
    "validate_proposer_response",
]
