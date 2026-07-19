from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import PipelineSettings
from .scoring import CombinationRule, KeywordGroup, ScoringConfig


@dataclass(frozen=True, slots=True)
class AppConfig:
    pipeline: PipelineSettings
    scoring: ScoringConfig


def load_config(path: Path) -> AppConfig:
    data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    pipeline: dict[str, Any] = data.get("pipeline", {})
    scoring: dict[str, Any] = data.get("scoring", {})
    return AppConfig(
        pipeline=PipelineSettings(
            minimum_score=float(pipeline.get("minimum_score", 1.0)),
            top_n=int(pipeline.get("top_n", 3)),
            candidate_pool_size=int(pipeline.get("candidate_pool_size", 10)),
            diversity_penalty=float(pipeline.get("diversity_penalty", 0.25)),
            dry_run=True,
        ),
        scoring=ScoringConfig(
            source_weights=_float_mapping(scoring.get("source_weights", {}), "source_weights"),
            keyword_weights=_float_mapping(scoring.get("keyword_weights", {}), "keyword_weights"),
            freshness_window_hours=float(scoring.get("freshness_window_hours", 72.0)),
            relevance_weight=float(scoring.get("relevance_weight", 0.60)),
            freshness_weight=float(scoring.get("freshness_weight", 0.25)),
            popularity_weight=float(scoring.get("popularity_weight", 0.15)),
            keyword_groups=_keyword_groups(scoring.get("keyword_groups", [])),
            combination_rules=_combination_rules(scoring.get("combination_rules", [])),
            noise_terms=_string_tuple(scoring.get("noise_terms", []), "noise_terms"),
            noise_penalty=float(scoring.get("noise_penalty", 0.0)),
            title_stopwords=_string_tuple(scoring.get("title_stopwords", []), "title_stopwords"),
            relevance_scale=float(scoring.get("relevance_scale", 1.0)),
        ),
    )


def _float_mapping(value: object, field_name: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a TOML table")
    return {str(key): float(item) for key, item in value.items()}


def _keyword_groups(value: object) -> tuple[KeywordGroup, ...]:
    rows = _table_array(value, "keyword_groups")
    return tuple(
        KeywordGroup(
            name=_required_string(row.get("name"), f"keyword_groups[{index}].name"),
            terms=_string_tuple(row.get("terms"), f"keyword_groups[{index}].terms"),
            term_weight=float(row.get("term_weight", 0.0)),
            max_score=float(row.get("max_score", 0.0)),
        )
        for index, row in enumerate(rows)
    )


def _combination_rules(value: object) -> tuple[CombinationRule, ...]:
    rows = _table_array(value, "combination_rules")
    return tuple(
        CombinationRule(
            name=_required_string(row.get("name"), f"combination_rules[{index}].name"),
            required_groups=frozenset(
                _string_tuple(
                    row.get("required_groups"),
                    f"combination_rules[{index}].required_groups",
                )
            ),
            bonus=float(row.get("bonus", 0.0)),
        )
        for index, row in enumerate(rows)
    )


def _table_array(value: object, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} must be an array of TOML tables")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be an array of strings")
    return tuple(value)


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
