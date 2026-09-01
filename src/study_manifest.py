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
    if type(raw) is not dict:
        raise ValueError("study manifest must contain one mapping")
    if type(raw.get("formal_frozen")) is not bool:
        raise ValueError("formal_frozen must be an exact boolean")
    if type(raw.get("paper_eligible_freeze_ids")) is not list or any(
        type(item) is not str or not item for item in raw["paper_eligible_freeze_ids"]
    ):
        raise ValueError("paper_eligible_freeze_ids must contain non-empty strings")
    methods = raw["methods"]
    missing = set(FORMAL_METHOD_ORDER) - set(methods)
    if missing:
        raise ValueError(f"manifest missing formal methods: {sorted(missing)}")
    formal_seeds = tuple(int(seed) for seed in raw["formal_seeds"])
    development_seed = int(raw["development_seed"])
    if development_seed in formal_seeds:
        raise ValueError("development seed must not appear in formal seeds")
    stage = str(raw["stage"])
    formal_frozen = raw["formal_frozen"]
    freeze_ids = tuple(raw["paper_eligible_freeze_ids"])
    if formal_frozen:
        if stage != "formal_ready" or not freeze_ids:
            raise ValueError("a frozen study must be formal_ready and paper eligible")
    elif stage != "development" or freeze_ids:
        raise ValueError("an unfrozen study must remain in development without freeze ids")
    return StudyManifest(
        schema_version=int(raw["schema_version"]),
        stage=stage,
        development_seed=development_seed,
        split_seed=int(raw["split_seed"]),
        formal_seeds=formal_seeds,
        formal_frozen=formal_frozen,
        paper_eligible_freeze_ids=freeze_ids,
        data_protocol=dict(raw["data_protocol"]),
        methods=methods,
    )
