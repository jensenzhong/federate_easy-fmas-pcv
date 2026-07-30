# FMAS-PCV Strict Federated Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one auditable strict-federated FMAS-PCV research pipeline that uses real-time DeepSeek multi-agent proposal, critique, client-local voting, and fail-stop experiment execution.

**Architecture:** A fixed client-local partition and train-only federated preprocessing feed a modular PCV engine. Deterministic anchors, DeepSeek agents, client-local candidate evaluation, voting, and a deterministic safety gate remain separate components with typed schemas and append-only telemetry. A single canonical runner reads `study_manifest.yaml`, never loads locked test data during training, and writes immutable run directories.

**Tech Stack:** Python 3.13, PyTorch, pandas, scikit-learn, PyYAML, requests, pytest, existing FedYogi server optimizer.

---

## Scope and execution rules

- Work in the dedicated `codex/fmas-pcv` worktree created in Task 0 after the current source state is explicitly approved and checkpointed.
- Do not run a real DeepSeek request until Task 11's user-approval gate.
- A DeepSeek authentication, connection, HTTP, timeout, schema, parsing, or agent runtime failure stops the run immediately. No retry, alternate model, fake response, hidden heuristic, or continued training is permitted.
- No automatic retry is permitted in preflight, development, resume, or formal execution.
- Do not inspect or report locked-test metrics during development.
- Do not move or delete historical results until the dry-run archive manifest is reviewed and approved.
- Do not add methods, actions, hyperparameters, or experimental branches that are absent from the approved design.

## File map

### Create

- `study_manifest.yaml`: single source of truth for method roles, phases, seeds, data protocol, and paper-eligible runs.
- `configs/development_seed42.yaml`: development-only frozen comparison settings.
- `configs/formal_frozen.yaml`: formal protocol template populated only after the development gate.
- `configs/methods/fedavg_strict.yaml`
- `configs/methods/fedyogi_strict.yaml`
- `configs/methods/dpcv_fedyogi.yaml`
- `configs/methods/sa_pcv_fedyogi.yaml`
- `configs/methods/fmas_pcv_fedyogi.yaml`
- `configs/prompts/diagnostic.md`
- `configs/prompts/performance_proposer.md`
- `configs/prompts/stability_proposer.md`
- `configs/prompts/balance_proposer.md`
- `configs/prompts/critic.md`
- `configs/prompts/coordinator.md`
- `src/study_manifest.py`: typed manifest loader and phase/method gates.
- `src/federated_learning/pcv/__init__.py`
- `src/federated_learning/pcv/schemas.py`: immutable action, telemetry, vote, metric, and failure contracts.
- `src/federated_learning/pcv/protocol.py`: privacy allowlist, partition manifest, test lock, and prompt audit.
- `src/federated_learning/pcv/candidates.py`: anchors, candidate validation, deduplication, and state aggregation.
- `src/federated_learning/pcv/client_evaluation.py`: local validation/test metric sufficient statistics and votes.
- `src/federated_learning/pcv/voting.py`: candidate aggregation, ranking, and deterministic reference score.
- `src/federated_learning/pcv/gate.py`: trust-region and degradation checks.
- `src/federated_learning/pcv/agents.py`: strict DeepSeek client and multi-agent orchestration.
- `src/federated_learning/pcv/telemetry.py`: append-only JSONL records, prompt hashes, and failure reports.
- `src/federated_learning/pcv/engine.py`: one-round and multi-round PCV execution.
- `src/federated_learning/pcv/checkpoint.py`: exact last-complete-round checkpoint and resume validation.
- `experiments/run_strict_federated.py`: only formal/development experiment entry point.
- `scripts/run_fmas_development.py`: seed-42 comparison and three LLM trajectory launcher.
- `scripts/create_strict_partition_manifest.py`: one-time deterministic client-local partition manifest generator.
- `scripts/freeze_fmas_protocol.py`: freeze manifest, configs, prompts, partition hash, and commit.
- `scripts/archive_legacy_results.py`: checksum-first dry-run and approved archive mover.
- `tests/test_study_manifest.py`
- `tests/test_strict_partition_manifest.py`
- `tests/test_pcv_protocol.py`
- `tests/test_pcv_schemas.py`
- `tests/test_pcv_candidates.py`
- `tests/test_pcv_client_evaluation.py`
- `tests/test_pcv_voting.py`
- `tests/test_pcv_gate.py`
- `tests/test_pcv_agents.py`
- `tests/test_pcv_checkpoint.py`
- `tests/test_pcv_engine.py`
- `tests/test_strict_runner.py`
- `tests/test_no_plaintext_api_keys.py`
- `tests/test_archive_legacy_results.py`

### Modify

- `.gitignore`: ignore nested worktrees, local credentials, and runtime-only pause files.
- `configs/config.yaml`: remove plaintext API credentials; preserve historical settings without making them formal defaults.
- `src/data_preprocessing.py`: add manifest-driven strict partition loading and train-only preprocessing.
- `src/federated_learning/server_optimizers.py`: allow per-candidate clip override without preview side effects.
- `src/experiment_names.py`: retain legacy aliases but expose the five formal keys in one canonical group.
- `scripts/run_multi_seed.py`: mark as legacy/non-formal and reject new formal FMAS keys.
- `scripts/statistical_analysis.py`: consume only a frozen manifest batch and aggregate LLM repetitions within seed.
- `scripts/generate_paper_tables.py`: consume only paper-eligible frozen results.
- `scripts/generate_paper_figures.py`: consume only paper-eligible frozen results.
- `README.md`: describe the single strict formal pipeline.
- `PROJECT_STATUS.md`: report development gate status without claiming unrun results.

## Task 0: Secure the current state and create an isolated worktree

**Files:**
- Create: `tests/test_no_plaintext_api_keys.py`
- Modify: `configs/config.yaml`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing credential test**

```python
# tests/test_no_plaintext_api_keys.py
from pathlib import Path
import re


SECRET_PATTERN = re.compile(r"""(?i)(api[_-]?key\s*:\s*["']?sk-[A-Za-z0-9_-]{16,})""")


def test_repository_configs_contain_no_plaintext_api_keys():
    offenders = []
    for path in [Path("configs/config.yaml"), *Path("configs").glob("**/*.yaml")]:
        if path.exists() and SECRET_PATTERN.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert offenders == [], f"plaintext API credentials found in: {offenders}"
```

- [ ] **Step 2: Run the test and verify it fails without printing the credential**

Run:

```powershell
python -m pytest tests/test_no_plaintext_api_keys.py -q
```

Expected: FAIL naming `configs/config.yaml`; the assertion must not include the secret value.

- [ ] **Step 3: Remove the plaintext key and require the environment variable**

Replace the DeepSeek provider block with:

```yaml
deepseek:
  model_name: "deepseek-chat"
  base_url: "https://api.deepseek.com"
  api_key_env: "DEEPSEEK_API_KEY"
```

Add to `.gitignore`:

```gitignore
.worktrees/
.env
.env.*
results/**/PAUSED.json
```

- [ ] **Step 4: Re-run the credential test**

Run:

```powershell
python -m pytest tests/test_no_plaintext_api_keys.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Stop for mandatory user action**

Report that a plaintext DeepSeek credential was found and removed from the project configuration. Ask the user to place the replacement credential only in `DEEPSEEK_API_KEY`; never copy the credential from chat into a file or command. Credential rotation is recommended but is not an execution blocker because the user supplied a replacement. Do not call DeepSeek until the user confirms the environment variable is available.

- [ ] **Step 6: Prepare a source-only checkpoint for review**

Run:

```powershell
git add -- .gitignore README.md PROJECT_STATUS.md requirements.txt pytest.ini configs docs experiments scripts src tests
git diff --cached --check
git diff --cached --name-only
```

Expected: no `results/`, `Data/`, model files, API credentials, or generated logs are staged.

- [ ] **Step 7: Stop for staged-file approval**

Show the staged-file list and ask the user to approve the checkpoint. Do not commit until explicitly approved.

- [ ] **Step 8: Commit the approved pre-FMAS source checkpoint**

Run:

```powershell
git commit -m "chore: checkpoint pre-FMAS research implementation"
```

Expected: commit succeeds with source, tests, configs, and documentation only.

- [ ] **Step 9: Create the dedicated worktree**

Run:

```powershell
git worktree add ".worktrees/fmas-pcv" -b "codex/fmas-pcv"
git -C ".worktrees/fmas-pcv" status --short --branch
```

Expected: branch `codex/fmas-pcv`; clean worktree.

All remaining tasks run with working directory `.worktrees/fmas-pcv`.

## Task 1: Establish the single study manifest and method registry

**Files:**
- Create: `study_manifest.yaml`
- Create: `src/study_manifest.py`
- Create: `tests/test_study_manifest.py`
- Modify: `src/experiment_names.py`

- [ ] **Step 1: Write failing manifest tests**

```python
# tests/test_study_manifest.py
from pathlib import Path
import pytest

from src.study_manifest import load_study_manifest


def test_manifest_exposes_exact_formal_method_order():
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    assert manifest.formal_methods == (
        "FEDAVG_STRICT",
        "FEDYOGI_STRICT",
        "DPCV_FEDYOGI",
        "SA_PCV_FEDYOGI",
        "FMAS_PCV_FEDYOGI",
    )


def test_seed_42_is_development_only():
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    assert manifest.development_seed == 42
    assert 42 not in manifest.formal_seeds


def test_unfrozen_manifest_cannot_name_paper_eligible_batch():
    manifest = load_study_manifest(Path("study_manifest.yaml"))
    assert manifest.formal_frozen is False
    assert manifest.paper_eligible_freeze_ids == ()
```

- [ ] **Step 2: Run the tests and verify import failure**

Run:

```powershell
python -m pytest tests/test_study_manifest.py -q
```

Expected: FAIL because `src.study_manifest` does not exist.

- [ ] **Step 3: Create the manifest**

```yaml
# study_manifest.yaml
schema_version: 1
project_key: fmas_pcv_strict_federated
stage: development
development_seed: 42
split_seed: 20260730
formal_seeds: [314, 2718, 2025, 3407, 9001]
formal_frozen: false
paper_eligible_freeze_ids: []
data_protocol:
  train_ratio: 0.70
  controller_validation_ratio: 0.15
  locked_test_ratio: 0.15
  target_quantile_bins: 5
  preprocessing_fit_partition: train
methods:
  FEDAVG_STRICT: {role: baseline, formal: true, uses_llm: false}
  FEDYOGI_STRICT: {role: baseline, formal: true, uses_llm: false}
  DPCV_FEDYOGI: {role: ablation, formal: true, uses_llm: false}
  SA_PCV_FEDYOGI: {role: ablation, formal: true, uses_llm: true}
  FMAS_PCV_FEDYOGI: {role: primary, formal: true, uses_llm: true}
legacy_methods:
  - C
  - MAS_ADAPTIVE
  - VG_FEDYOGI_TR
  - MAS_VG_FEDYOGI_TR
  - COHERENCE_FEDYOGI_TR
  - LLM_GCA_FEDYOGI_TR
  - STRICT_COHERENCE_FEDYOGI_TR
  - LLM_STRICT_GCA_FEDYOGI_TR
  - VP_GCA_FEDYOGI_TR
  - LLM_VP_GCA_FEDYOGI_TR
```

- [ ] **Step 4: Implement the typed loader**

```python
# src/study_manifest.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


FORMAL_METHOD_ORDER = (
    "FEDAVG_STRICT",
    "FEDYOGI_STRICT",
    "DPCV_FEDYOGI",
    "SA_PCV_FEDYOGI",
    "FMAS_PCV_FEDYOGI",
)


@dataclass(frozen=True)
class StudyManifest:
    schema_version: int
    stage: str
    development_seed: int
    split_seed: int
    formal_seeds: tuple[int, ...]
    formal_frozen: bool
    paper_eligible_freeze_ids: tuple[str, ...]
    data_protocol: dict[str, Any]
    methods: dict[str, dict[str, Any]]

    @property
    def formal_methods(self) -> tuple[str, ...]:
        return tuple(key for key in FORMAL_METHOD_ORDER if self.methods[key]["formal"])


def load_study_manifest(path: Path) -> StudyManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    methods = raw["methods"]
    missing = set(FORMAL_METHOD_ORDER) - set(methods)
    if missing:
        raise ValueError(f"manifest missing formal methods: {sorted(missing)}")
    formal_seeds = tuple(int(seed) for seed in raw["formal_seeds"])
    development_seed = int(raw["development_seed"])
    if development_seed in formal_seeds:
        raise ValueError("development seed must not appear in formal seeds")
    return StudyManifest(
        schema_version=int(raw["schema_version"]),
        stage=str(raw["stage"]),
        development_seed=development_seed,
        split_seed=int(raw["split_seed"]),
        formal_seeds=formal_seeds,
        formal_frozen=bool(raw["formal_frozen"]),
        paper_eligible_freeze_ids=tuple(raw["paper_eligible_freeze_ids"]),
        data_protocol=dict(raw["data_protocol"]),
        methods=methods,
    )
```

In `src/experiment_names.py`, add the five formal display names and a separate `FORMAL_EXPERIMENT_ORDER = list(FORMAL_METHOD_ORDER)`. Do not delete legacy aliases.

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_study_manifest.py tests/test_multi_seed_config.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add study_manifest.yaml src/study_manifest.py src/experiment_names.py tests/test_study_manifest.py
git commit -m "feat: add canonical FMAS study manifest"
```

## Task 2: Build fixed client-local partitions and train-only preprocessing

**Files:**
- Create: `src/federated_learning/pcv/__init__.py`
- Create: `src/federated_learning/pcv/protocol.py`
- Create: `tests/test_strict_partition_manifest.py`
- Modify: `src/data_preprocessing.py`

- [ ] **Step 1: Write failing partition tests**

```python
# tests/test_strict_partition_manifest.py
import pandas as pd

from src.federated_learning.pcv.protocol import (
    PartitionRatios,
    build_partition_manifest,
)


def _frame():
    rows = []
    for client in ("Client 1", "Client 2", "Client 3"):
        for index in range(40):
            rows.append({"Client": client, "ContAmnt": float(index + 1), "source_index": len(rows)})
    return pd.DataFrame(rows)


def test_partition_is_client_local_disjoint_and_complete():
    frame = _frame()
    manifest = build_partition_manifest(
        frame,
        client_column="Client",
        target_column="ContAmnt",
        source_index_column="source_index",
        dataset_sha256="0" * 64,
        split_seed=20260730,
        ratios=PartitionRatios(0.70, 0.15, 0.15),
        quantile_bins=5,
    )
    assert len(manifest) == len(frame)
    assert manifest["row_id"].is_unique
    assert set(manifest["partition"]) == {"train", "controller_validation", "locked_test"}
    for _, rows in manifest.groupby("client_id"):
        assert abs((rows["partition"] == "train").mean() - 0.70) <= 0.05


def test_partition_does_not_change_with_training_seed():
    frame = _frame()
    kwargs = dict(
        client_column="Client",
        target_column="ContAmnt",
        source_index_column="source_index",
        dataset_sha256="1" * 64,
        split_seed=20260730,
        ratios=PartitionRatios(0.70, 0.15, 0.15),
        quantile_bins=5,
    )
    left = build_partition_manifest(frame, **kwargs)
    right = build_partition_manifest(frame, **kwargs)
    pd.testing.assert_frame_equal(left, right)
```

- [ ] **Step 2: Run the tests and verify import failure**

Run:

```powershell
python -m pytest tests/test_strict_partition_manifest.py -q
```

Expected: FAIL because `pcv.protocol` does not exist.

- [ ] **Step 3: Implement deterministic partition creation**

```python
# src/federated_learning/pcv/protocol.py
from dataclasses import dataclass
from hashlib import sha256
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PartitionRatios:
    train: float
    controller_validation: float
    locked_test: float

    def validate(self) -> None:
        if abs(self.train + self.controller_validation + self.locked_test - 1.0) > 1e-9:
            raise ValueError("partition ratios must sum to one")


def _row_id(dataset_sha256: str, source_index: int) -> str:
    return sha256(f"{dataset_sha256}:{source_index}".encode("utf-8")).hexdigest()[:24]


def build_partition_manifest(
    frame: pd.DataFrame,
    *,
    client_column: str,
    target_column: str,
    source_index_column: str,
    dataset_sha256: str,
    split_seed: int,
    ratios: PartitionRatios,
    quantile_bins: int,
) -> pd.DataFrame:
    ratios.validate()
    output = []
    for client_id, client_frame in frame.groupby(client_column, sort=True):
        local = client_frame.copy()
        ranks = local[target_column].rank(method="first")
        local["_target_bin"] = pd.qcut(ranks, q=min(quantile_bins, len(local)), labels=False)
        train, holdout = train_test_split(
            local,
            test_size=1.0 - ratios.train,
            random_state=split_seed,
            stratify=local["_target_bin"],
        )
        holdout_bins = holdout["_target_bin"]
        test_fraction = ratios.locked_test / (ratios.controller_validation + ratios.locked_test)
        validation, test = train_test_split(
            holdout,
            test_size=test_fraction,
            random_state=split_seed,
            stratify=holdout_bins,
        )
        for partition, rows in (
            ("train", train),
            ("controller_validation", validation),
            ("locked_test", test),
        ):
            for source_index in rows[source_index_column].astype(int):
                output.append({
                    "row_id": _row_id(dataset_sha256, source_index),
                    "source_index": source_index,
                    "client_id": str(client_id),
                    "partition": partition,
                    "dataset_sha256": dataset_sha256,
                })
    return pd.DataFrame(output).sort_values(["client_id", "source_index"]).reset_index(drop=True)
```

- [ ] **Step 4: Add manifest-driven strict partition loading**

In `src/data_preprocessing.py`, add `load_strict_partition_frames(config, partition_manifest_path)` that:

1. verifies the dataset SHA-256 recorded beside the partition manifest;
2. joins by `source_index`;
3. computes federated feature statistics from `partition == "train"` only;
4. returns client-local train, controller-validation, and locked-test frames without exposing them to a server agent;
5. never creates a server global-validation dataset.

Use this implementation shape:

```python
@dataclass(frozen=True)
class StrictPartitionFrames:
    client_frames: dict[str, dict[str, pd.DataFrame]]
    preprocessor: DataPreprocessor
    dataset_sha256: str
    partition_sha256: str


def load_strict_partition_frames(
    config: dict,
    partition_manifest_path: str,
) -> StrictPartitionFrames:
    data_cfg = config["scene_c"]["data"]
    raw_path = Path(data_cfg["raw_csv"])
    raw_bytes = raw_path.read_bytes()
    dataset_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    frame = pd.read_csv(raw_path).rename(columns=data_cfg.get("rename_map", {}))
    frame = frame[data_cfg["feature_columns"] + [data_cfg["target_column"], data_cfg["client_column"]]].copy()
    frame["source_index"] = frame.index.astype(int)

    manifest_path = Path(partition_manifest_path)
    partition_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = pd.read_csv(manifest_path)
    expected_dataset_sha256 = manifest["dataset_sha256"].drop_duplicates().tolist()
    if expected_dataset_sha256 != [dataset_sha256]:
        raise ValueError("partition manifest dataset hash does not match canonical data")
    merged = frame.merge(
        manifest[["source_index", "client_id", "partition"]],
        on="source_index",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(frame):
        raise ValueError("partition manifest does not cover the canonical dataset exactly")

    feature_columns = data_cfg["feature_columns"]
    train_only = merged[merged["partition"] == "train"]
    local_stats = [
        compute_local_stats(client_frame, feature_columns)
        for _, client_frame in train_only.groupby("client_id", sort=True)
    ]
    global_stats = aggregate_federated_stats(local_stats)
    preprocessor = DataPreprocessor(
        feature_scaler=config["preprocessing"]["scaler"],
        target_transform=config["preprocessing"]["target_transform"],
        random_seed=int(config["preprocessing"]["random_seed"]),
    )
    preprocessor.set_global_stats(
        mean=global_stats["mean"],
        std=global_stats["std"],
        var=global_stats["var"],
        n_samples=global_stats["count"],
        feature_columns=feature_columns,
    )
    client_frames = {
        str(client_id): {
            partition: rows.drop(columns=["client_id", "partition"]).copy()
            for partition, rows in client_rows.groupby("partition", sort=True)
        }
        for client_id, client_rows in merged.groupby("client_id", sort=True)
    }
    return StrictPartitionFrames(
        client_frames=client_frames,
        preprocessor=preprocessor,
        dataset_sha256=dataset_sha256,
        partition_sha256=partition_sha256,
    )
```

Add imports for `dataclass` and `hashlib`. Add a test using an extreme value placed only in controller validation and assert that the fitted mean and scale equal train-only statistics.

- [ ] **Step 5: Add the one-time manifest generator**

```python
# scripts/create_strict_partition_manifest.py
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

from src.federated_learning.pcv.protocol import PartitionRatios, build_partition_manifest
from src.study_manifest import load_study_manifest
from src.utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", default="study_manifest.yaml")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="results/manifests/strict_partition_v1.csv")
    args = parser.parse_args()
    output = Path(args.output)
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"partition output already exists: {output}")
    study = load_study_manifest(Path(args.study_manifest))
    config = load_config(args.config)
    data_cfg = config["scene_c"]["data"]
    raw_path = Path(data_cfg["raw_csv"])
    dataset_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    frame = pd.read_csv(raw_path).rename(columns=data_cfg.get("rename_map", {}))
    frame["source_index"] = frame.index.astype(int)
    protocol = study.data_protocol
    manifest = build_partition_manifest(
        frame,
        client_column=data_cfg["client_column"],
        target_column=data_cfg["target_column"],
        source_index_column="source_index",
        dataset_sha256=dataset_sha256,
        split_seed=study.split_seed,
        ratios=PartitionRatios(
            protocol["train_ratio"],
            protocol["controller_validation_ratio"],
            protocol["locked_test_ratio"],
        ),
        quantile_bins=int(protocol["target_quantile_bins"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    partition_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps({
            "dataset_sha256": dataset_sha256,
            "partition_sha256": partition_sha256,
            "split_seed": study.split_seed,
            "rows": len(manifest),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Extend `StudyManifest` in Task 1 with a `data_protocol: dict[str, Any]` field populated from YAML.

- [ ] **Step 6: Run partition and fairness tests**

Run:

```powershell
python -m pytest tests/test_strict_partition_manifest.py tests/test_fair_centralized_split.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Generate and inspect the development partition manifest**

Run:

```powershell
python scripts/create_strict_partition_manifest.py
Import-Csv results/manifests/strict_partition_v1.csv | Group-Object client_id,partition | Select-Object Name,Count
```

Expected: 688 total rows, three clients, all three partitions for every client, and no overwrite of an existing manifest.

- [ ] **Step 8: Commit code; keep the generated manifest as a protocol artifact**

```powershell
git add src/federated_learning/pcv/__init__.py src/federated_learning/pcv/protocol.py src/data_preprocessing.py scripts/create_strict_partition_manifest.py tests/test_strict_partition_manifest.py results/manifests/strict_partition_v1.csv results/manifests/strict_partition_v1.json
git commit -m "feat: add fixed strict client data protocol"
```

## Task 3: Define privacy-safe schemas and the locked-test boundary

**Files:**
- Create: `src/federated_learning/pcv/schemas.py`
- Create: `tests/test_pcv_schemas.py`
- Create: `tests/test_pcv_protocol.py`
- Modify: `src/federated_learning/pcv/protocol.py`

- [ ] **Step 1: Write failing schema and privacy tests**

```python
# tests/test_pcv_protocol.py
import pytest

from src.federated_learning.pcv.protocol import (
    PrivacyViolation,
    TestPartitionLocked,
    assert_prompt_payload_safe,
    require_test_unlock,
)


def test_prompt_rejects_raw_or_test_fields():
    for payload in (
        {"raw_features": [1.0]},
        {"labels": [2.0]},
        {"row_predictions": [3.0]},
        {"test_mape": 0.4},
    ):
        with pytest.raises(PrivacyViolation):
            assert_prompt_payload_safe(payload)


def test_prompt_accepts_approved_aggregate_fields():
    assert_prompt_payload_safe({
        "round_index": 2,
        "clients": [{
            "client_id": "client_01",
            "sample_count": 100,
            "train_loss": 0.3,
            "val_mape": 0.4,
            "val_rmse": 1.2,
            "update_norm": 0.8,
        }],
    })


def test_locked_test_requires_frozen_formal_unlock():
    with pytest.raises(TestPartitionLocked):
        require_test_unlock(phase="development", formal_frozen=False, explicit_unlock=False)
```

```python
# tests/test_pcv_schemas.py
import pytest

from src.federated_learning.pcv.schemas import CandidateAction


def test_candidate_action_requires_legal_weights_and_actions():
    legal = CandidateAction(
        candidate_id="performance_01",
        weights={"client_01": 0.3, "client_02": 0.4, "client_03": 0.3},
        server_optimizer="fedyogi",
        server_lr_scale=1.0,
        update_clip_norm=1.0,
        source="performance_proposer",
        rationale="aligned update with lower reported validation error",
    )
    legal.validate(("client_01", "client_02", "client_03"))
    illegal = CandidateAction(
        candidate_id="bad",
        weights={"client_01": 0.95, "client_02": 0.03, "client_03": 0.02},
        server_optimizer="unknown",
        server_lr_scale=3.0,
        update_clip_norm=7.0,
        source="bad",
        rationale="bad",
    )
    with pytest.raises(ValueError):
        illegal.validate(("client_01", "client_02", "client_03"))
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/test_pcv_protocol.py tests/test_pcv_schemas.py -q
```

Expected: FAIL because the contracts do not exist.

- [ ] **Step 3: Implement immutable schemas**

```python
# src/federated_learning/pcv/schemas.py
from dataclasses import asdict, dataclass, field
from typing import Any
import math


ALLOWED_LR_SCALES = (0.50, 0.75, 1.00, 1.25)
ALLOWED_CLIP_NORMS = (None, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class ClientTelemetry:
    client_id: str
    sample_count: int
    train_loss: float
    val_mape: float
    val_rmse: float
    update_norm: float
    cosine_to_mean: float
    cosine_to_previous: float

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateAction:
    candidate_id: str
    weights: dict[str, float]
    server_optimizer: str
    server_lr_scale: float
    update_clip_norm: float | None
    source: str
    rationale: str

    def validate(self, client_ids: tuple[str, ...]) -> None:
        if set(self.weights) != set(client_ids):
            raise ValueError("candidate weights must match client ids")
        if any(not math.isfinite(value) for value in self.weights.values()):
            raise ValueError("candidate weights must be finite")
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("candidate weights must sum to one")
        if any(value < 0.05 or value > 0.80 for value in self.weights.values()):
            raise ValueError("candidate weight outside [0.05, 0.80]")
        if self.server_optimizer not in {"fedavg", "fedyogi"}:
            raise ValueError("invalid server_optimizer")
        if self.server_lr_scale not in ALLOWED_LR_SCALES:
            raise ValueError("invalid server_lr_scale")
        if self.update_clip_norm not in ALLOWED_CLIP_NORMS:
            raise ValueError("invalid update_clip_norm")


@dataclass(frozen=True)
class LocalCandidateVote:
    client_id: str
    candidate_id: str
    sample_count: int
    val_mape: float
    val_rmse: float
    relative_mape: float
    relative_rmse: float
    rank: int
    confidence: float
    catastrophic_degradation: bool


@dataclass(frozen=True)
class CandidateDecision:
    requested_candidate_id: str
    selected_candidate_id: str
    gate_status: str
    rationale: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Implement recursive prompt auditing and test unlock**

```python
# additions to src/federated_learning/pcv/protocol.py
class PrivacyViolation(RuntimeError):
    pass


class TestPartitionLocked(RuntimeError):
    pass


FORBIDDEN_PROMPT_KEYS = {
    "raw_features", "raw_labels", "labels", "row_predictions",
    "predictions", "residuals", "test_mape", "test_rmse",
    "test_mae", "test_r2", "locked_test",
}


def assert_prompt_payload_safe(payload) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_PROMPT_KEYS:
                raise PrivacyViolation(f"prohibited prompt field: {key}")
            assert_prompt_payload_safe(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_prompt_payload_safe(value)


def require_test_unlock(*, phase: str, formal_frozen: bool, explicit_unlock: bool) -> None:
    if phase != "formal_evaluate" or not formal_frozen or not explicit_unlock:
        raise TestPartitionLocked("locked test is unavailable before frozen formal evaluation")
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_pcv_protocol.py tests/test_pcv_schemas.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/federated_learning/pcv/protocol.py src/federated_learning/pcv/schemas.py tests/test_pcv_protocol.py tests/test_pcv_schemas.py
git commit -m "feat: enforce FMAS privacy and action contracts"
```

## Task 4: Implement deterministic anchors and candidate construction

**Files:**
- Create: `src/federated_learning/pcv/candidates.py`
- Create: `tests/test_pcv_candidates.py`
- Modify: `src/federated_learning/server_optimizers.py`

- [ ] **Step 1: Write failing candidate tests**

```python
# tests/test_pcv_candidates.py
import torch

from src.federated_learning.pcv.candidates import (
    build_anchor_candidates,
    deduplicate_candidates,
    weighted_average_state,
)
from src.federated_learning.pcv.schemas import CandidateAction


def test_anchor_candidates_are_always_present():
    anchors = build_anchor_candidates(
        sample_counts={"client_01": 2, "client_02": 3, "client_03": 5},
        fedyogi_lr_scale=1.0,
        fedyogi_clip_norm=1.0,
    )
    assert [item.candidate_id for item in anchors] == ["anchor_fedavg", "anchor_fedyogi"]


def test_weighted_average_state_matches_candidate_weights():
    states = {
        "client_01": {"w": torch.tensor([1.0])},
        "client_02": {"w": torch.tensor([3.0])},
        "client_03": {"w": torch.tensor([5.0])},
    }
    result = weighted_average_state(
        states,
        {"client_01": 0.2, "client_02": 0.3, "client_03": 0.5},
    )
    assert torch.allclose(result["w"], torch.tensor([3.6]))


def test_candidate_deduplication_keeps_eight_or_fewer():
    repeated = [
        CandidateAction(
            candidate_id=f"c{i}",
            weights={"client_01": .3, "client_02": .4, "client_03": .3},
            server_optimizer="fedyogi",
            server_lr_scale=1.0,
            update_clip_norm=1.0,
            source="p",
            rationale="r",
        )
        for i in range(10)
    ]
    assert len(deduplicate_candidates(repeated, budget=8)) == 1
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/test_pcv_candidates.py -q
```

Expected: FAIL because `pcv.candidates` does not exist.

- [ ] **Step 3: Implement anchors, state averaging, and deduplication**

Implement:

```python
def build_anchor_candidates(
    sample_counts: dict[str, int],
    fedyogi_lr_scale: float,
    fedyogi_clip_norm: float | None,
) -> list[CandidateAction]:
    total = sum(sample_counts.values())
    size_weights = {client_id: count / total for client_id, count in sample_counts.items()}
    return [
        CandidateAction(
            "anchor_fedavg", size_weights, "fedavg", 1.0, None, "anchor", "strict FedAvg anchor"
        ),
        CandidateAction(
            "anchor_fedyogi",
            size_weights,
            "fedyogi",
            fedyogi_lr_scale,
            fedyogi_clip_norm,
            "anchor",
            "strict FedYogi anchor",
        ),
    ]
```

`deduplicate_candidates` keys on rounded weights, server optimizer, LR scale, and clip norm, preserves anchors first, validates all actions, and truncates to eight.

`weighted_average_state` skips floating-point arithmetic for non-floating tensors and clones every output tensor.

Add `build_deterministic_candidates` with this fixed order:

1. strict FedAvg size-weight anchor;
2. strict FedYogi size-weight anchor;
3. uniform weights under FedYogi;
4. previous accepted weights under FedYogi;
5. positive-coherence weights under FedYogi;
6. inverse-validation-MAPE weights under FedYogi;
7. validation-error-compensation weights under FedYogi;
8. a 50/50 blend of size weights and previous accepted weights under FedYogi.

Every derived weight vector is projected to `[0.05, 0.80]`. Missing previous weights reuse the size weights; missing or non-finite telemetry rejects the affected derived candidate instead of inventing a value.

- [ ] **Step 4: Add side-effect-free clip override**

Extend `FedYogiServerOptimizer.preview_step` and `step` with:

```python
_UNSET = object()


update_clip_norm_override: float | None | object = _UNSET
```

The preview saves and restores optimizer state in `finally`. The override applies only to that candidate and does not modify `self.update_clip_norm`. Add a regression test comparing optimizer state before and after previews with different clip values.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_pcv_candidates.py tests/test_checkpoint_state_copy.py tests/test_server_optimizers.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/federated_learning/pcv/candidates.py src/federated_learning/server_optimizers.py tests/test_pcv_candidates.py tests/test_server_optimizers.py
git commit -m "feat: add strict PCV anchors and candidate states"
```

## Task 5: Implement client-local candidate evaluation and sufficient statistics

**Files:**
- Create: `src/federated_learning/pcv/client_evaluation.py`
- Create: `tests/test_pcv_client_evaluation.py`
- Modify: `src/federated_learning/pcv/protocol.py`

- [ ] **Step 1: Write failing metric tests**

```python
# tests/test_pcv_client_evaluation.py
import math

from src.federated_learning.pcv.client_evaluation import (
    MetricSums,
    aggregate_metric_sums,
    build_vote,
)


def test_metric_sums_aggregate_without_predictions_or_labels():
    combined = aggregate_metric_sums([
        MetricSums(2, ape_sum=0.4, se_sum=5.0, ae_sum=3.0, y_sum=6.0, y_sq_sum=20.0),
        MetricSums(1, ape_sum=0.1, se_sum=4.0, ae_sum=2.0, y_sum=4.0, y_sq_sum=16.0),
    ])
    assert combined["mape"] == 0.5 / 3
    assert combined["rmse"] == math.sqrt(3.0)
    assert combined["mae"] == 5.0 / 3


def test_vote_marks_more_than_five_percent_client_degradation():
    vote = build_vote(
        client_id="client_01",
        candidate_id="candidate_01",
        sample_count=30,
        candidate_mape=0.43,
        candidate_rmse=10.0,
        anchor_mape=0.40,
        anchor_rmse=9.0,
        rank=3,
        confidence=0.8,
    )
    assert vote.catastrophic_degradation is True
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/test_pcv_client_evaluation.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement metric sufficient statistics**

```python
# src/federated_learning/pcv/client_evaluation.py
from dataclasses import dataclass
import math
import numpy as np

from .schemas import LocalCandidateVote


@dataclass(frozen=True)
class MetricSums:
    n: int
    ape_sum: float
    se_sum: float
    ae_sum: float
    y_sum: float
    y_sq_sum: float


def compute_metric_sums(y_true, y_pred, epsilon: float = 1e-8) -> MetricSums:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("metric inputs must be non-empty and have equal shape")
    error = prediction - truth
    denominator = np.maximum(np.abs(truth), epsilon)
    return MetricSums(
        n=int(truth.size),
        ape_sum=float(np.sum(np.abs(error) / denominator)),
        se_sum=float(np.sum(error * error)),
        ae_sum=float(np.sum(np.abs(error))),
        y_sum=float(np.sum(truth)),
        y_sq_sum=float(np.sum(truth * truth)),
    )


def aggregate_metric_sums(items: list[MetricSums]) -> dict[str, float]:
    n = sum(item.n for item in items)
    if n <= 0:
        raise ValueError("metric aggregation requires observations")
    ape = sum(item.ape_sum for item in items)
    se = sum(item.se_sum for item in items)
    ae = sum(item.ae_sum for item in items)
    y_sum = sum(item.y_sum for item in items)
    y_sq_sum = sum(item.y_sq_sum for item in items)
    denominator = y_sq_sum - (y_sum * y_sum / n)
    return {
        "mape": ape / n,
        "rmse": math.sqrt(se / n),
        "mae": ae / n,
        "r2": 1.0 - se / denominator if denominator > 0 else 0.0,
    }


def build_vote(
    *,
    client_id: str,
    candidate_id: str,
    sample_count: int,
    candidate_mape: float,
    candidate_rmse: float,
    anchor_mape: float,
    anchor_rmse: float,
    rank: int,
    confidence: float,
) -> LocalCandidateVote:
    relative_mape = (candidate_mape - anchor_mape) / max(abs(anchor_mape), 1e-12)
    relative_rmse = (candidate_rmse - anchor_rmse) / max(abs(anchor_rmse), 1e-12)
    return LocalCandidateVote(
        client_id=client_id,
        candidate_id=candidate_id,
        sample_count=sample_count,
        val_mape=candidate_mape,
        val_rmse=candidate_rmse,
        relative_mape=relative_mape,
        relative_rmse=relative_rmse,
        rank=rank,
        confidence=confidence,
        catastrophic_degradation=relative_mape > 0.05,
    )
```

All metrics are computed after inverse-transforming the target to the original currency scale. `build_vote` computes relative changes against the stronger anchor and marks catastrophic degradation when candidate MAPE exceeds anchor MAPE by more than 5% relative.

- [ ] **Step 4: Implement `ClientDataVault`**

Add to `protocol.py` a class that owns private train, controller-validation, and test datasets. Dependencies are injected as callables so the server engine receives results rather than tensors:

```python
from collections.abc import Callable


class ClientDataVault:
    def __init__(
        self,
        *,
        client_id: str,
        train_dataset,
        controller_validation_dataset,
        locked_test_dataset,
        train_fn: Callable,
        telemetry_fn: Callable,
        metric_sums_fn: Callable,
    ):
        self.client_id = client_id
        self.__train_dataset = train_dataset
        self.__controller_validation_dataset = controller_validation_dataset
        self.__locked_test_dataset = locked_test_dataset
        self.__train_fn = train_fn
        self.__telemetry_fn = telemetry_fn
        self.__metric_sums_fn = metric_sums_fn

    def train_local(self, global_state, training_config, seed):
        return self.__train_fn(self.__train_dataset, global_state, training_config, seed)

    def controller_telemetry(self, model_state) -> ClientTelemetry:
        return self.__telemetry_fn(
            self.client_id,
            self.__controller_validation_dataset,
            model_state,
        )

    def evaluate_candidates(
        self,
        candidate_states: dict[str, dict],
        stronger_anchor_id: str,
    ) -> list[LocalCandidateVote]:
        metric_sums = {
            candidate_id: self.__metric_sums_fn(
                self.__controller_validation_dataset,
                model_state,
            )
            for candidate_id, model_state in candidate_states.items()
        }
        metrics = {
            candidate_id: {
                "n": sums.n,
                **aggregate_metric_sums([sums]),
            }
            for candidate_id, sums in metric_sums.items()
        }
        ranked = sorted(metrics, key=lambda key: metrics[key]["mape"])
        anchor = metrics[stronger_anchor_id]
        return [
            build_vote(
                client_id=self.client_id,
                candidate_id=candidate_id,
                sample_count=int(candidate_metrics["n"]),
                candidate_mape=float(candidate_metrics["mape"]),
                candidate_rmse=float(candidate_metrics["rmse"]),
                anchor_mape=float(anchor["mape"]),
                anchor_rmse=float(anchor["rmse"]),
                rank=ranked.index(candidate_id) + 1,
                confidence=1.0 / len(ranked),
            )
            for candidate_id, candidate_metrics in metrics.items()
        ]

    def final_test_sums(self, model_state, unlock_context) -> MetricSums:
        require_test_unlock(**unlock_context)
        return self.__metric_sums_fn(self.__locked_test_dataset, model_state)
```

No property or method returns the underlying validation/test tensors to the server engine. `final_test_sums` calls `require_test_unlock`.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_pcv_client_evaluation.py tests/test_pcv_protocol.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/federated_learning/pcv/client_evaluation.py src/federated_learning/pcv/protocol.py tests/test_pcv_client_evaluation.py tests/test_pcv_protocol.py
git commit -m "feat: add private client candidate evaluation"
```

## Task 6: Implement voting and the deterministic safety gate

**Files:**
- Create: `src/federated_learning/pcv/voting.py`
- Create: `src/federated_learning/pcv/gate.py`
- Create: `tests/test_pcv_voting.py`
- Create: `tests/test_pcv_gate.py`

- [ ] **Step 1: Write failing voting and gate tests**

```python
# tests/test_pcv_gate.py
from src.federated_learning.pcv.gate import select_with_gate
from src.federated_learning.pcv.schemas import CandidateAction, LocalCandidateVote


def _candidate(candidate_id, weights):
    return CandidateAction(candidate_id, weights, "fedyogi", 1.0, 1.0, "test", "test")


def test_gate_rejects_client_catastrophe_even_when_global_score_is_best():
    candidates = {
        "anchor_fedyogi": _candidate("anchor_fedyogi", {"c1": .3, "c2": .4, "c3": .3}),
        "proposal": _candidate("proposal", {"c1": .4, "c2": .3, "c3": .3}),
    }
    votes = [
        LocalCandidateVote("c1", "proposal", 10, .35, 1.0, -.1, -.1, 1, .9, False),
        LocalCandidateVote("c2", "proposal", 10, .60, 1.0, .2, .0, 2, .9, True),
    ]
    decision = select_with_gate(
        requested_candidate_id="proposal",
        candidates=candidates,
        votes=votes,
        aggregate_mape={"anchor_fedyogi": .40, "proposal": .39},
        previous_weights={"c1": .3, "c2": .4, "c3": .3},
        stronger_anchor_id="anchor_fedyogi",
    )
    assert decision.selected_candidate_id == "anchor_fedyogi"
    assert decision.gate_status == "rejected_client_degradation"
```

```python
# tests/test_pcv_voting.py
from src.federated_learning.pcv.voting import aggregate_candidate_votes
from src.federated_learning.pcv.schemas import LocalCandidateVote


def test_vote_aggregation_is_weighted_by_validation_count():
    votes = [
        LocalCandidateVote("c1", "x", 90, .30, 10, -.1, 0, 1, .9, False),
        LocalCandidateVote("c2", "x", 10, .70, 10, .1, 0, 2, .9, False),
    ]
    result = aggregate_candidate_votes(votes)
    assert result["x"].weighted_mape == .34
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/test_pcv_voting.py tests/test_pcv_gate.py -q
```

Expected: FAIL because voting and gate modules do not exist.

- [ ] **Step 3: Implement vote aggregation**

Create `CandidateVoteSummary` with candidate id, weighted MAPE, weighted RMSE, mean rank, minimum confidence, and catastrophic-client count. Group votes by candidate and weight metrics by validation sample count.

- [ ] **Step 4: Implement exact gate order**

`select_with_gate` checks, in order:

1. requested candidate exists;
2. action schema is legal;
3. no client catastrophic degradation;
4. L1 distance from previous weights is at most `0.35`;
5. aggregate MAPE is within `0.002` absolute of the best legal candidate;
6. aggregate MAPE is no more than `0.001` absolute worse than the stronger anchor.

Any rejection selects the stronger anchor and returns a unique gate status. Infrastructure or parsing exceptions are never converted into an anchor decision.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_pcv_voting.py tests/test_pcv_gate.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/federated_learning/pcv/voting.py src/federated_learning/pcv/gate.py tests/test_pcv_voting.py tests/test_pcv_gate.py
git commit -m "feat: add PCV voting and deterministic safety gate"
```

## Task 7: Implement fail-stop DeepSeek and multi-agent orchestration

**Files:**
- Create: `src/federated_learning/pcv/agents.py`
- Create: `src/federated_learning/pcv/telemetry.py`
- Create: `tests/test_pcv_agents.py`
- Create: `configs/prompts/diagnostic.md`
- Create: `configs/prompts/performance_proposer.md`
- Create: `configs/prompts/stability_proposer.md`
- Create: `configs/prompts/balance_proposer.md`
- Create: `configs/prompts/critic.md`
- Create: `configs/prompts/coordinator.md`

- [ ] **Step 1: Write failing fail-stop tests with a fake HTTP session**

```python
# tests/test_pcv_agents.py
import pytest

from src.federated_learning.pcv.agents import DeepSeekCallError, StrictDeepSeekClient


class FailingSession:
    calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        raise ConnectionError("offline")


def test_deepseek_failure_has_no_retry():
    session = FailingSession()
    client = StrictDeepSeekClient(
        api_key="test-only",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com",
        timeout_seconds=2,
        session=session,
    )
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json(
            role="diagnostic",
            system_prompt="Return JSON.",
            payload={"round_index": 1, "clients": []},
            response_validator=lambda value: value,
        )
    assert session.calls == 1
    assert error.value.category == "connection"


def test_schema_failure_stops_instead_of_repairing_or_falling_back():
    class InvalidJsonSession:
        def post(self, *args, **kwargs):
            class Response:
                def raise_for_status(self): pass
                def json(self):
                    return {"choices": [{"message": {"content": "not json"}}]}
            return Response()

    client = StrictDeepSeekClient("test-only", "deepseek-chat", "https://api.deepseek.com", 2, InvalidJsonSession())
    with pytest.raises(DeepSeekCallError) as error:
        client.generate_json("critic", "Return JSON.", {"round_index": 1}, lambda value: value)
    assert error.value.category == "schema"
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/test_pcv_agents.py -q
```

Expected: FAIL because strict agent client does not exist.

- [ ] **Step 3: Implement the single-attempt strict client**

```python
# core contract in src/federated_learning/pcv/agents.py
class DeepSeekCallError(RuntimeError):
    def __init__(self, category: str, role: str, message: str):
        super().__init__(f"{category} failure in {role}: {message}")
        self.category = category
        self.role = role


class StrictDeepSeekClient:
    def __init__(self, api_key, model_name, base_url, timeout_seconds, session):
        if not api_key:
            raise DeepSeekCallError("authentication", "preflight", "DEEPSEEK_API_KEY is missing")
        self.api_key = api_key
        self.model_name = model_name
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.session = session

    def generate_json(self, role, system_prompt, payload, response_validator):
        assert_prompt_payload_safe(payload)
        try:
            response = self.session.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
                    ],
                    "temperature": 0.8,
                    "stream": False,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            category = "authentication" if status in (401, 403) else "http"
            raise DeepSeekCallError(category, role, str(exc)) from exc
        except (requests.RequestException, ConnectionError) as exc:
            raise DeepSeekCallError("connection", role, str(exc)) from exc
        try:
            content = response.json()["choices"][0]["message"]["content"]
            return response_validator(json.loads(content))
        except Exception as exc:
            raise DeepSeekCallError("schema", role, str(exc)) from exc
```

There is no loop, sleep, repair request, alternate provider, or fallback.

- [ ] **Step 4: Implement append-only telemetry**

`telemetry.py` writes one JSON object per line using flush plus `os.fsync`. It stores request hash, prompt hash, model, role, response text, parsed response, timing, candidate decisions, and failure category. It never stores the API key or authorization headers.

- [ ] **Step 5: Implement exact agent response validators and orchestrator**

The diagnostic agent returns:

```json
{"state_summary":"client updates disagree in direction", "risks":["oscillation"], "priorities":["stability"]}
```

Each proposer returns at most two legal `CandidateAction` objects. The critic returns:

```json
{"accepted_candidate_ids":["performance_01"], "rejected":[{"candidate_id":"stability_02","reason":"duplicate action"}]}
```

The coordinator returns:

```json
{"selected_candidate_id":"performance_01", "rationale":"best admissible client vote summary", "risk_acknowledgement":"safety gate retains final authority"}
```

Any missing key, unknown candidate id, extra action field, or invalid candidate raises `DeepSeekCallError(category="schema")`.

- [ ] **Step 6: Write the six approved prompt files**

Each system prompt states:

- its one role;
- the allowed input fields;
- that raw data, labels, predictions, and test metrics are unavailable;
- the exact JSON output schema;
- that it must not invent clients or actions;
- that it must not request a new tool, model, hyperparameter, or data field.

Add tests that hash all prompt files and assert that `test_mape`, `raw_features`, `labels`, and `predictions` do not appear as requested inputs.

- [ ] **Step 7: Run tests**

Run:

```powershell
python -m pytest tests/test_pcv_agents.py tests/test_pcv_protocol.py -q
```

Expected: all tests pass without any network request.

- [ ] **Step 8: Commit**

```powershell
git add src/federated_learning/pcv/agents.py src/federated_learning/pcv/telemetry.py configs/prompts tests/test_pcv_agents.py
git commit -m "feat: add fail-stop DeepSeek multi-agent orchestration"
```

## Task 8: Add exact checkpoints and user-approved resume

**Files:**
- Create: `src/federated_learning/pcv/checkpoint.py`
- Create: `tests/test_pcv_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint tests**

```python
# tests/test_pcv_checkpoint.py
import random
import numpy as np
import pytest
import torch

from src.federated_learning.pcv.checkpoint import (
    ResumeApprovalRequired,
    capture_rng_state,
    restore_rng_state,
    validate_resume,
)


def test_rng_state_round_trip_is_exact():
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(1).item())
    assert actual == expected


def test_resume_requires_explicit_user_approval_flag():
    with pytest.raises(ResumeApprovalRequired):
        validate_resume(
            checkpoint_freeze_id="freeze-a",
            requested_freeze_id="freeze-a",
            user_approved_resume=False,
        )
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/test_pcv_checkpoint.py -q
```

Expected: FAIL because checkpoint module does not exist.

- [ ] **Step 3: Implement checkpoint payload**

Checkpoint fields:

```python
{
    "schema_version": 1,
    "last_complete_round": round_index,
    "freeze_id": freeze_id,
    "method": method,
    "training_seed": training_seed,
    "llm_rep": llm_rep,
    "global_model_state": cloned_state,
    "server_optimizer_state": optimizer.get_optimizer_state(),
    "previous_weights": previous_weights,
    "best_validation": best_validation,
    "best_model_state": best_model_state,
    "rng_state": capture_rng_state(),
    "partition_sha256": partition_sha256,
    "config_sha256": config_sha256,
    "prompt_hashes": prompt_hashes,
}
```

Write to a temporary file, call `fsync`, then atomically replace `last_complete.pt`.

- [ ] **Step 4: Implement resume validation**

Resume requires:

- `--resume-checkpoint`;
- `--user-approved-resume`;
- exact freeze, method, seed, partition, config, and prompt hashes;
- restoration from `last_complete_round + 1`.

A mismatch raises and does not modify any state.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_pcv_checkpoint.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/federated_learning/pcv/checkpoint.py tests/test_pcv_checkpoint.py
git commit -m "feat: add exact FMAS pause and resume checkpoints"
```

## Task 9: Implement the PCV round engine

**Files:**
- Create: `src/federated_learning/pcv/engine.py`
- Create: `tests/test_pcv_engine.py`

- [ ] **Step 1: Write failing engine failure-semantics test**

```python
# tests/test_pcv_engine.py
import pytest

from src.federated_learning.pcv.agents import DeepSeekCallError
from src.federated_learning.pcv.engine import ExperimentPaused, PCVEngine


def test_agent_failure_does_not_commit_incomplete_round(fake_engine_dependencies):
    engine = PCVEngine(**fake_engine_dependencies)
    original_state = {key: value.clone() for key, value in engine.global_state.items()}
    engine.agent_orchestrator = lambda *args, **kwargs: (_ for _ in ()).throw(
        DeepSeekCallError("connection", "diagnostic", "offline")
    )
    with pytest.raises(ExperimentPaused) as paused:
        engine.run_round(round_index=3)
    assert paused.value.failure.category == "connection"
    for key in original_state:
        assert engine.global_state[key].equal(original_state[key])
    assert engine.last_complete_round == 2
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/test_pcv_engine.py -q
```

Expected: FAIL because engine module does not exist.

- [ ] **Step 3: Implement one-round transaction**

`PCVEngine.run_round` performs all local training, telemetry, agent calls, candidate previews, votes, coordination, and gate checks against cloned working state. It commits global model, optimizer, previous weights, best-validation checkpoint, telemetry, and last-complete checkpoint only after the entire round succeeds.

The method skeleton is:

```python
def run_round(self, round_index: int) -> RoundResult:
    snapshot = self._snapshot_runtime()
    try:
        local_updates, client_telemetry = self._train_clients(round_index)
        anchors = self._build_anchors(local_updates)
        proposals = self._build_method_proposals(round_index, client_telemetry, anchors)
        candidates = deduplicate_candidates([*anchors, *proposals], budget=8)
        candidate_states = self._preview_candidates(candidates, local_updates)
        votes = self._evaluate_on_clients(candidate_states, anchors)
        requested_id = self._coordinate(round_index, candidates, votes)
        decision = self._gate(requested_id, candidates, votes)
        self._commit_round(round_index, decision, candidate_states)
        return self._round_result(round_index, decision)
    except DeepSeekCallError as failure:
        self._restore_runtime(snapshot)
        report_path = self._write_pause_report(round_index, failure)
        raise ExperimentPaused(failure, report_path) from failure
    except Exception as failure:
        self._restore_runtime(snapshot)
        wrapped = ExperimentRuntimeError(type(failure).__name__, str(failure))
        report_path = self._write_pause_report(round_index, wrapped)
        raise ExperimentPaused(wrapped, report_path) from failure
```

Do not catch `ExperimentPaused` in a way that continues the loop.

`_preview_candidates` uses `FedAvgServerOptimizer` for `candidate.server_optimizer == "fedavg"` and a cloned `FedYogiServerOptimizer` state for `candidate.server_optimizer == "fedyogi"`. It applies the candidate LR scale and clip override without changing the real optimizer. `_commit_round` loads the selected FedYogi preview state for a FedYogi candidate. Selecting the FedAvg anchor clears FedYogi moment tensors before the next round so stale moments cannot be applied to a new FedAvg global state; the reset is recorded in telemetry.

- [ ] **Step 4: Implement method-specific proposal behavior**

- `FEDAVG_STRICT`: execute FedAvg anchor only; no DeepSeek call.
- `FEDYOGI_STRICT`: execute FedYogi anchor only; no DeepSeek call.
- `DPCV_FEDYOGI`: the eight fixed deterministic candidates from Task 4 plus client voting; no DeepSeek call.
- `SA_PCV_FEDYOGI`: one combined diagnostic/proposal call and one coordinator call.
- `FMAS_PCV_FEDYOGI`: diagnostic, three proposers, critic, and coordinator calls.

All five methods still use the same data, rounds, local epochs, checkpoint rule, and metric code.

- [ ] **Step 5: Add tests for method call counts and candidate budgets**

Assert per round:

- `FEDAVG_STRICT`: 0 calls, 1 candidate;
- `FEDYOGI_STRICT`: 0 calls, 1 candidate;
- `DPCV_FEDYOGI`: 0 calls, at most 8 candidates;
- `SA_PCV_FEDYOGI`: 2 calls, at most 8 candidates;
- `FMAS_PCV_FEDYOGI`: 6 calls, at most 8 candidates.

- [ ] **Step 6: Run tests**

Run:

```powershell
python -m pytest tests/test_pcv_engine.py tests/test_pcv_candidates.py tests/test_pcv_agents.py tests/test_pcv_checkpoint.py -q
```

Expected: all tests pass without network access.

- [ ] **Step 7: Commit**

```powershell
git add src/federated_learning/pcv/engine.py tests/test_pcv_engine.py
git commit -m "feat: add transactional FMAS-PCV training engine"
```

## Task 10: Add canonical configs, runner, and immutable outputs

**Files:**
- Create: `configs/development_seed42.yaml`
- Create: `configs/formal_frozen.yaml`
- Create: `configs/methods/*.yaml`
- Create: `experiments/run_strict_federated.py`
- Create: `tests/test_strict_runner.py`
- Modify: `scripts/run_multi_seed.py`

- [ ] **Step 1: Write failing runner tests**

```python
# tests/test_strict_runner.py
from pathlib import Path
import pytest

from experiments.run_strict_federated import build_parser, resolve_run_directory


def test_runner_accepts_only_formal_methods():
    parser = build_parser()
    args = parser.parse_args(["--method", "FMAS_PCV_FEDYOGI", "--phase", "development"])
    assert args.method == "FMAS_PCV_FEDYOGI"
    with pytest.raises(SystemExit):
        parser.parse_args(["--method", "LLM_GCA_FEDYOGI_TR", "--phase", "development"])


def test_run_directory_never_overwrites(tmp_path):
    first = resolve_run_directory(tmp_path, "dev-a")
    first.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        resolve_run_directory(tmp_path, "dev-a")
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/test_strict_runner.py -q
```

Expected: FAIL because runner does not exist.

- [ ] **Step 3: Create method configs**

Every method config contains these fixed fields:

```yaml
num_rounds: 20
local_epochs: 20
batch_size: 32
client_learning_rate: 0.0005
checkpoint_metric: aggregated_client_val_mape
candidate_budget: 8
min_client_weight: 0.05
max_client_weight: 0.80
weight_l1_limit: 0.35
best_candidate_tolerance: 0.002
anchor_mape_tolerance: 0.001
catastrophic_client_relative_mape: 0.05
```

The exact differences are:

- `FEDAVG_STRICT`: `server_optimizer: fedavg`, `proposal_mode: anchor_only`, `deepseek_roles: []`;
- `FEDYOGI_STRICT`: `server_optimizer: fedyogi`, `proposal_mode: anchor_only`, `deepseek_roles: []`;
- `DPCV_FEDYOGI`: `server_optimizer: fedyogi`, `proposal_mode: deterministic`, `deepseek_roles: []`;
- `SA_PCV_FEDYOGI`: `server_optimizer: fedyogi`, `proposal_mode: single_agent`, `deepseek_roles: [single_proposer, coordinator]`;
- `FMAS_PCV_FEDYOGI`: `server_optimizer: fedyogi`, `proposal_mode: multi_agent`, `deepseek_roles: [diagnostic, performance_proposer, stability_proposer, balance_proposer, critic, coordinator]`.

- [ ] **Step 4: Implement the canonical runner**

Parser arguments:

```text
--method
--phase development|formal_train|formal_evaluate
--training-seed
--llm-rep
--run-id
--freeze-id
--resume-checkpoint
--user-approved-resume
--unlock-test
--preflight-only
```

The runner:

1. loads and validates `study_manifest.yaml`;
2. refuses formal phases when the manifest is not frozen;
3. loads the fixed partition manifest;
4. constructs client vaults;
5. reads `DEEPSEEK_API_KEY` only for LLM methods;
6. creates an immutable run directory;
7. records provenance before training;
8. exits nonzero and prints the pause-report path on `ExperimentPaused`;
9. never evaluates test data in development or formal training;
10. evaluates locked test only in `formal_evaluate` with all three unlock conditions.

- [ ] **Step 5: Mark the old runner as non-formal**

Add a visible warning to `scripts/run_multi_seed.py` and reject the five new formal keys with:

```python
raise RuntimeError(
    "FMAS formal methods must use experiments/run_strict_federated.py and study_manifest.yaml"
)
```

Do not break historical key execution.

- [ ] **Step 6: Run tests**

Run:

```powershell
python -m pytest tests/test_strict_runner.py tests/test_study_manifest.py tests/test_strict_no_server_validation.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```powershell
git add configs/development_seed42.yaml configs/formal_frozen.yaml configs/methods experiments/run_strict_federated.py scripts/run_multi_seed.py tests/test_strict_runner.py
git commit -m "feat: add canonical strict federated runner"
```

## Task 11: Verify offline, then perform a user-approved DeepSeek preflight

**Files:**
- Modify: `tests/test_pcv_agents.py`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Run the entire offline test suite**

Run:

```powershell
python -m pytest -q
```

Expected: every test passes; no DeepSeek request is made.

- [ ] **Step 2: Run static privacy and secret scans**

Run:

```powershell
python -m pytest tests/test_no_plaintext_api_keys.py tests/test_pcv_protocol.py tests/test_strict_partition_manifest.py -q
git grep -l -E "api_key:[[:space:]]*['\"]?sk-" -- . ':!docs/superpowers'
```

Expected: tests pass and `git grep` returns no matches.

- [ ] **Step 3: Stop for user approval before the first real API call**

Report:

- offline test result;
- exact model name and endpoint;
- prompt role `preflight`;
- that no training will run;
- that any authentication, network, HTTP, timeout, or schema failure will stop immediately.

Wait for explicit approval.

- [ ] **Step 4: Run exactly one preflight request**

Use a dedicated command:

```powershell
python experiments/run_strict_federated.py --phase development --method FMAS_PCV_FEDYOGI --training-seed 42 --run-id deepseek-preflight --preflight-only
```

The prompt requests exactly:

```json
{"status":"ready","model":"deepseek-chat"}
```

Expected: one valid response, recorded with prompt and response hashes. No training data, client telemetry, model state, or test information is included.

- [ ] **Step 5: Handle the preflight outcome**

If it fails: stop all experiment work, preserve the error report, and notify the user. Do not retry.

If it succeeds: report success and wait for explicit user approval before any seed-42 training.

- [ ] **Step 6: Update status and commit the verified offline implementation**

Document only verified facts in `PROJECT_STATUS.md`, then:

```powershell
git add PROJECT_STATUS.md tests/test_pcv_agents.py
git commit -m "test: verify FMAS offline safety gates"
```

## Task 12: Run the approved seed-42 development matrix

**Files:**
- Create: `scripts/run_fmas_development.py`
- Create: `tests/test_fmas_development_runner.py`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Write failing matrix test**

```python
# tests/test_fmas_development_runner.py
from scripts.run_fmas_development import build_run_matrix


def test_development_matrix_is_predeclared():
    matrix = build_run_matrix(training_seed=42)
    assert [(row.method, row.llm_rep) for row in matrix] == [
        ("FEDAVG_STRICT", 0),
        ("FEDYOGI_STRICT", 0),
        ("DPCV_FEDYOGI", 0),
        ("SA_PCV_FEDYOGI", 1),
        ("SA_PCV_FEDYOGI", 2),
        ("SA_PCV_FEDYOGI", 3),
        ("FMAS_PCV_FEDYOGI", 1),
        ("FMAS_PCV_FEDYOGI", 2),
        ("FMAS_PCV_FEDYOGI", 3),
    ]
```

- [ ] **Step 2: Implement the fixed matrix launcher**

The launcher runs sequentially, never in parallel. Before every LLM run it verifies that the previous run is complete. On any nonzero exit or `PAUSED.json`, it stops immediately and returns the same nonzero status.

- [ ] **Step 3: Run launcher tests**

Run:

```powershell
python -m pytest tests/test_fmas_development_runner.py -q
```

Expected: all tests pass using mocked subprocesses.

- [ ] **Step 4: Stop for explicit training approval**

Show:

- exact nine-run matrix;
- development config hash;
- partition hash;
- prompt hashes;
- current Git commit;
- output root.

Wait for explicit approval.

- [ ] **Step 5: Execute the development matrix**

Run:

```powershell
python scripts/run_fmas_development.py --training-seed 42 --config configs/development_seed42.yaml
```

Expected: sequential execution. Any DeepSeek or runtime failure stops the matrix and triggers immediate user reporting.

- [ ] **Step 6: Evaluate the development gate using validation only**

Produce `results/development/seed42/development_gate.json` containing:

- strongest strict baseline validation metrics;
- each LLM trajectory's validation metrics;
- relative MAPE improvement;
- RMSE increase ratio;
- R2 difference;
- pass/fail for each trajectory;
- count of passing FMAS trajectories.

Do not load locked test data.

- [ ] **Step 7: Report and stop**

If fewer than two FMAS trajectories pass: report the true outcome and stop. Do not tune or add a new idea automatically.

If at least two pass: report the evidence and ask whether to freeze the method.

- [ ] **Step 8: Commit code and status, never generated run data**

```powershell
git add scripts/run_fmas_development.py tests/test_fmas_development_runner.py PROJECT_STATUS.md
git commit -m "feat: add approved FMAS development matrix"
```

## Task 13: Freeze the successful protocol

**Files:**
- Create: `scripts/freeze_fmas_protocol.py`
- Create: `tests/test_freeze_fmas_protocol.py`
- Modify: `study_manifest.yaml`
- Modify: `configs/formal_frozen.yaml`

- [ ] **Step 1: Write failing freeze tests**

Test that freezing refuses:

- a dirty Git tree;
- a failed development gate;
- seed 42 in formal seeds;
- missing prompt or partition hashes;
- missing DeepSeek model and temperature;
- a freeze id that already exists.

- [ ] **Step 2: Implement freeze-id generation**

The freeze payload includes:

```text
git_commit
partition_sha256
development_config_sha256
formal_config_sha256
method_config_sha256s
prompt_sha256s
deepseek_model
deepseek_temperature
formal_seeds
development_gate_sha256
```

Compute:

```python
freeze_id = sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 3: Run freeze tests**

Run:

```powershell
python -m pytest tests/test_freeze_fmas_protocol.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Stop for freeze approval**

Show the complete freeze payload and formal seed list. Wait for explicit approval.

- [ ] **Step 5: Freeze without running formal experiments**

Run:

```powershell
python scripts/freeze_fmas_protocol.py --development-gate results/development/seed42/development_gate.json
```

Expected:

- `study_manifest.yaml` changes to `stage: formal_ready` and `formal_frozen: true`;
- `configs/formal_frozen.yaml` contains exact frozen values;
- `results/manifests/<freeze_id>.json` is created;
- no model training and no test evaluation occur.

- [ ] **Step 6: Commit**

```powershell
git add study_manifest.yaml configs/formal_frozen.yaml scripts/freeze_fmas_protocol.py tests/test_freeze_fmas_protocol.py
git commit -m "chore: freeze approved FMAS formal protocol"
```

## Task 14: Isolate statistics and paper artifacts to frozen batches

**Files:**
- Modify: `scripts/statistical_analysis.py`
- Modify: `scripts/generate_paper_tables.py`
- Modify: `scripts/generate_paper_figures.py`
- Create: `tests/test_fmas_formal_statistics.py`

- [ ] **Step 1: Write failing frozen-batch tests**

Create fixtures with deterministic baselines and three LLM repetitions per seed. Assert:

- LLM repetitions are averaged within training seed;
- paired tests receive five observations, not fifteen;
- seed 42 is rejected;
- legacy scenario rows are rejected;
- a freeze id not marked paper-eligible is rejected;
- Holm-corrected significance is reported.

- [ ] **Step 2: Implement frozen result loading**

Add:

```python
def load_frozen_formal_results(
    results_root: Path,
    manifest: StudyManifest,
    freeze_id: str,
) -> pd.DataFrame:
    if freeze_id not in manifest.paper_eligible_freeze_ids:
        raise ValueError(f"freeze id is not paper eligible: {freeze_id}")
    batch_root = results_root / "formal" / freeze_id
    metric_paths = sorted(batch_root.glob("*/*/*/metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"no formal metrics found under {batch_root}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    frame = pd.DataFrame(rows)
    required_columns = {
        "freeze_id", "method", "training_seed", "llm_rep", "success",
        "test_unlocked", "test_mape", "test_rmse", "test_mae", "test_r2",
        "partition_sha256", "config_sha256", "prompt_bundle_sha256",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"formal results missing columns: {sorted(missing)}")
    if set(frame["freeze_id"]) != {freeze_id}:
        raise ValueError("formal result freeze id mismatch")
    if not frame["success"].astype(bool).all() or not frame["test_unlocked"].astype(bool).all():
        raise ValueError("formal result batch contains incomplete or non-evaluated runs")
    if 42 in set(frame["training_seed"].astype(int)):
        raise ValueError("development seed 42 is forbidden in formal statistics")
    if set(frame["method"]) - set(manifest.formal_methods):
        raise ValueError("formal result batch contains a legacy method")
    expected_seeds = set(manifest.formal_seeds)
    for method in manifest.formal_methods:
        method_rows = frame[frame["method"] == method]
        if set(method_rows["training_seed"].astype(int)) != expected_seeds:
            raise ValueError(f"formal seed coverage mismatch for {method}")
        expected_reps = {1, 2, 3} if manifest.methods[method]["uses_llm"] else {0}
        for seed in expected_seeds:
            reps = set(method_rows[method_rows["training_seed"] == seed]["llm_rep"].astype(int))
            if reps != expected_reps:
                raise ValueError(f"LLM repetition coverage mismatch for {method} seed {seed}")
    return frame.sort_values(["method", "training_seed", "llm_rep"]).reset_index(drop=True)
```

It validates provenance hashes, required methods, formal seeds, repetition counts, success states, and locked-test evaluation status.

- [ ] **Step 3: Implement hierarchical aggregation**

For LLM methods:

```python
per_seed = raw.groupby(["method", "training_seed"], as_index=False).mean(numeric_only=True)
```

Pair `per_seed` against deterministic baselines on `training_seed`. Report mean, standard deviation, confidence interval, paired t-test, Wilcoxon, paired effect size, wins out of five, and Holm-adjusted p-values.

The generated claim-status field is:

- `stable_improvement` only when FMAS-PCV wins on at least four of five seeds;
- `significant_improvement` only when `stable_improvement` is true and the Holm-adjusted primary MAPE p-value is below `0.05`;
- `mean_improvement_trend` when mean MAPE improves without both gates;
- `no_supported_improvement` when mean MAPE does not improve.

- [ ] **Step 4: Restrict tables and figures**

Tables and figures require `--freeze-id`. They write only under:

```text
results/paper/<freeze_id>/tables/
results/paper/<freeze_id>/figures/
```

Missing required results cause a hard failure, not a skipped panel.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_fmas_formal_statistics.py tests/test_statistical_analysis.py tests/test_generate_paper_tables.py tests/test_generate_paper_figures.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add scripts/statistical_analysis.py scripts/generate_paper_tables.py scripts/generate_paper_figures.py tests/test_fmas_formal_statistics.py
git commit -m "feat: restrict paper evidence to frozen FMAS batches"
```

## Task 15: Add checksum-first historical result archival

**Files:**
- Create: `scripts/archive_legacy_results.py`
- Create: `tests/test_archive_legacy_results.py`
- Modify: `README.md`
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Write failing archive tests**

Use a temporary result tree and assert:

- default mode is dry-run;
- manifest contains original path, destination, size, mtime, and SHA-256;
- formal/development/manifests/paper directories are excluded;
- execute mode refuses without `--user-approved`;
- a copied/moved file is checksum-verified before the source is removed;
- interrupted moves remain recoverable.

- [ ] **Step 2: Implement dry-run classification**

Classify as legacy:

- old top-level scenario CSVs;
- `results/multi_seed` rows not associated with a new freeze id;
- smoke, pilot, diagnostic, tuning, convergence, and old ablation outputs;
- old figure and paper-table outputs.

Never classify:

- `results/development`;
- `results/formal`;
- `results/manifests`;
- `results/paper`;
- the archive manifest itself.

- [ ] **Step 3: Run archive tests**

Run:

```powershell
python -m pytest tests/test_archive_legacy_results.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Generate a real dry-run manifest**

Run:

```powershell
python scripts/archive_legacy_results.py --results-root results --dry-run
```

Expected: writes only an archive proposal manifest; moves nothing.

- [ ] **Step 5: Stop for explicit archive approval**

Show file count, total bytes, destination, and exclusions. Wait for explicit approval. Do not execute archive movement automatically.

- [ ] **Step 6: Execute only after approval**

Run:

```powershell
python scripts/archive_legacy_results.py --results-root results --execute --user-approved
```

Expected: every moved item is checksum-verified and recoverable from `results/archive/<archive_id>/`.

- [ ] **Step 7: Update project docs and commit**

README documents only the strict pipeline. PROJECT_STATUS reports archived history and the current approved phase without inventing results.

README must state the threat model precisely: this repository is a single-machine simulation with logical client/server separation. The server code path receives only approved updates and aggregates, but the implementation does not claim process isolation, secure aggregation, differential privacy, or a cryptographic privacy guarantee.

```powershell
git add scripts/archive_legacy_results.py tests/test_archive_legacy_results.py README.md PROJECT_STATUS.md
git commit -m "chore: isolate legacy experiment outputs"
```

## Task 16: Final offline verification and implementation handoff

**Files:**
- Modify: `PROJECT_STATUS.md`

- [ ] **Step 1: Run full tests**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run integrity checks**

Run:

```powershell
git diff --check
python -m pytest tests/test_no_plaintext_api_keys.py tests/test_pcv_protocol.py tests/test_strict_partition_manifest.py tests/test_pcv_checkpoint.py tests/test_strict_runner.py -q
git status --short
```

Expected: no secret, privacy, partition, checkpoint, or runner failures. The only permitted uncommitted paths are generated files under `results/development/`, `results/manifests/`, and an explicitly approved `results/archive/`; none may be staged.

- [ ] **Step 3: Verify the approved specification is covered**

Record in `PROJECT_STATUS.md`:

- data protocol implemented and tested;
- five-method registry active;
- fail-stop DeepSeek policy tested;
- development gate status;
- formal freeze status;
- whether archive execution occurred;
- exact remaining user approvals.

- [ ] **Step 4: Commit final implementation status**

```powershell
git add PROJECT_STATUS.md
git commit -m "docs: report FMAS-PCV implementation status"
```

- [ ] **Step 5: Stop before formal multi-seed work**

Do not run formal training or unlock test data as part of this implementation plan. Present the verified implementation and development-gate evidence to the user. Formal multi-seed execution requires a separate explicit approval after freeze review.

## Plan self-review

- Spec coverage: data privacy, fixed partition, train-only preprocessing, five-method registry, real-time multi-agent calls, local voting, deterministic gate, exact checkpoints, failure stop/reporting, immutable outputs, development gate, freeze gate, hierarchical statistics, claim discipline, and reversible archive are each mapped to a task.
- Completeness scan: every planned code change names its file, interface, failing test, verification command, and commit boundary.
- Type consistency: `CandidateAction`, `ClientTelemetry`, `LocalCandidateVote`, `CandidateDecision`, `MetricSums`, `StudyManifest`, `DeepSeekCallError`, and checkpoint fields have one canonical definition and consistent names across tasks.
- Scope boundary: formal multi-seed execution and locked-test evaluation are intentionally excluded. They begin only after a successful seed-42 development gate, protocol freeze, and separate user approval.
