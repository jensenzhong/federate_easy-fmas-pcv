# DeepSeek V4 Flash Provider Switch

Date: 2026-08-31

## Approved Runtime Contract

- Base URL: `https://api.deepseek.com`
- Model: `deepseek-v4-flash`
- Credential source: process-only `DEEPSEEK_API_KEY`
- Temperature: `0.8`
- Timeout: `60` seconds
- Retry and fallback: disabled

The model setting applies to every real DeepSeek request in the strict study:
preflight, SA proposer/coordinator calls, and all FMAS diagnostic, proposer,
critic, and coordinator calls. A run whose provenance names another model is
rejected rather than mixed with the current matrix.

## Research Impact

This is a provider-model configuration change requested before the replacement
seed-42 matrix began. It does not change the client partition, preprocessing,
model architecture, local training, candidate definitions, voting, safety
gate, checkpoint rule, metrics, or Locked Test policy. Only protocol-approved
aggregate train/controller-validation telemetry remains eligible for prompts.

The partial `c0902ff` matrix remains invalidated and audit-only. The failed
`859a404` preflight is also retained as failure evidence. A successful new
preflight and all nine replacement development runs must share the commit that
freezes this provider configuration.

## Failure Discipline

Authentication, connection, timeout, rate-limit, HTTP, model, parsing, schema,
or runtime failures stop execution immediately. The runner must not retry,
change models, synthesize a response, or continue training after such a failure.
