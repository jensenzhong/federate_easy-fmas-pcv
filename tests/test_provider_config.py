from src.federated_learning.pcv.provider_config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_TIMEOUT_SECONDS,
    deepseek_protocol_config,
)


def test_frozen_deepseek_transport_contract_uses_120_second_timeout():
    assert DEEPSEEK_TIMEOUT_SECONDS == 120
    assert deepseek_protocol_config() == {
        "model": DEEPSEEK_MODEL,
        "base_url": DEEPSEEK_BASE_URL,
        "temperature": DEEPSEEK_TEMPERATURE,
        "timeout_seconds": 120,
    }
