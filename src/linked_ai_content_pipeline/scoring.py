from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from .models import Material, ScoreBreakdown, ScoredMaterial


@dataclass(frozen=True, slots=True)
class KeywordGroup:
    name: str
    terms: tuple[str, ...]
    term_weight: float
    max_score: float

    def __post_init__(self) -> None:
        if not self.name or not self.terms:
            raise ValueError("keyword group name and terms must not be empty")
        if self.term_weight <= 0 or self.max_score <= 0:
            raise ValueError("keyword group weights must be positive")


@dataclass(frozen=True, slots=True)
class CombinationRule:
    name: str
    required_groups: frozenset[str]
    bonus: float

    def __post_init__(self) -> None:
        if not self.name or not self.required_groups:
            raise ValueError("combination rule name and groups must not be empty")
        if self.bonus <= 0:
            raise ValueError("combination rule bonus must be positive")


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    source_weights: Mapping[str, float]
    keyword_weights: Mapping[str, float]
    freshness_window_hours: float = 72.0
    relevance_weight: float = 0.60
    freshness_weight: float = 0.25
    popularity_weight: float = 0.15
    keyword_groups: tuple[KeywordGroup, ...] = ()
    combination_rules: tuple[CombinationRule, ...] = ()
    noise_terms: tuple[str, ...] = ()
    noise_penalty: float = 0.0
    title_stopwords: tuple[str, ...] = ()
    relevance_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.freshness_window_hours <= 0:
            raise ValueError("freshness_window_hours must be positive")
        total = self.relevance_weight + self.freshness_weight + self.popularity_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("component weights must add up to 1")
        if self.noise_penalty < 0:
            raise ValueError("noise_penalty must not be negative")
        if self.relevance_scale <= 0:
            raise ValueError("relevance_scale must be positive")
        known_groups = {group.name for group in self.keyword_groups}
        unknown_groups = {
            group
            for rule in self.combination_rules
            for group in rule.required_groups
            if group not in known_groups
        }
        if unknown_groups:
            raise ValueError(f"unknown keyword groups: {sorted(unknown_groups)}")


def score_material(
    material: Material,
    config: ScoringConfig,
    *,
    now: datetime | None = None,
) -> ScoredMaterial:
    reference_time = now or datetime.now(UTC)
    published_at = material.published_at
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    age_hours = max(0.0, (reference_time - published_at).total_seconds() / 3600)
    freshness_ratio = max(0.0, 1.0 - age_hours / config.freshness_window_hours)
    source_weight = config.source_weights.get(material.source, 0.0)
    relevance_points = 0.0
    reasons = [f"source_weight:{source_weight:.2f}", f"freshness:{freshness_ratio:.2f}"]
    searchable = f"{material.title} {material.body}".casefold()
    for keyword, weight in config.keyword_weights.items():
        if _contains_term(searchable, keyword):
            relevance_points += weight
            reasons.append(f"keyword:{keyword}:{weight:.2f}")

    matched_groups: list[str] = []
    for group in config.keyword_groups:
        matches = tuple(term for term in group.terms if _contains_term(searchable, term))
        if not matches:
            continue
        group_score = min(group.max_score, len(matches) * group.term_weight)
        relevance_points += group_score
        matched_groups.append(group.name)
        reasons.append(f"group:{group.name}:{group_score:.2f}")

    matched_group_set = frozenset(matched_groups)
    applied_bonuses: list[str] = []
    for rule in config.combination_rules:
        if rule.required_groups <= matched_group_set:
            relevance_points += rule.bonus
            applied_bonuses.append(rule.name)
            reasons.append(f"combination:{rule.name}:{rule.bonus:.2f}")

    matched_noise = tuple(term for term in config.noise_terms if _contains_term(searchable, term))
    applied_noise_penalty = len(matched_noise) * config.noise_penalty
    if applied_noise_penalty:
        relevance_points -= applied_noise_penalty
        reasons.append(f"noise_penalty:{applied_noise_penalty:.2f}")

    title = material.title.casefold()
    stopword_blocked = any(_contains_term(title, term) for term in config.title_stopwords)
    if stopword_blocked:
        relevance_points = 0.0
        reasons.append("title_stopword:blocked")

    relevance = min(1.0, max(0.0, relevance_points / config.relevance_scale))
    popularity = min(1.0, max(0.0, material.popularity))
    score = (
        relevance * config.relevance_weight
        + freshness_ratio * config.freshness_weight
        + popularity * config.popularity_weight
    ) * source_weight
    breakdown = ScoreBreakdown(
        relevance=round(relevance, 4),
        freshness=round(freshness_ratio, 4),
        popularity=round(popularity, 4),
        source_weight=round(source_weight, 4),
        matched_groups=tuple(matched_groups),
        combination_bonuses=tuple(applied_bonuses),
        noise_penalty=round(applied_noise_penalty, 4),
        stopword_blocked=stopword_blocked,
    )
    return ScoredMaterial(material, round(score, 4), tuple(reasons), breakdown)


def _contains_term(text: str, term: str) -> bool:
    normalized = term.casefold().strip()
    if not normalized:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text) is not None


def rank_materials(
    materials: tuple[Material, ...],
    config: ScoringConfig,
    *,
    minimum_score: float,
    limit: int,
    now: datetime | None = None,
) -> tuple[ScoredMaterial, ...]:
    eligible = (
        item
        for item in (score_material(material, config, now=now) for material in materials)
        if item.score >= minimum_score
    )
    return tuple(
        sorted(
            eligible,
            key=lambda item: (-item.score, -item.material.published_at.timestamp()),
        )[:limit]
    )
