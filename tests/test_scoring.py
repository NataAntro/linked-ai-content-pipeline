from __future__ import annotations

from datetime import UTC, datetime

import pytest

from linked_ai_content_pipeline.models import Material
from linked_ai_content_pipeline.scoring import (
    CombinationRule,
    KeywordGroup,
    ScoringConfig,
    rank_materials,
    score_material,
)


def _material(
    identifier: str, title: str, published_hour: int, popularity: float = 0.0
) -> Material:
    return Material(
        source="trusted",
        external_id=identifier,
        title=title,
        body="Technology news content",
        canonical_url=f"https://example.test/{identifier}",
        published_at=datetime(2026, 7, 17, published_hour, tzinfo=UTC),
        content_fingerprint=identifier,
        popularity=popularity,
    )


def test_scoring_uses_explainable_weighted_formula() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    config = ScoringConfig(
        source_weights={"trusted": 1.0},
        keyword_weights={"AI": 2.0},
        freshness_window_hours=24,
    )
    result = score_material(_material("one", "AI update", 6, popularity=0.4), config, now=now)
    assert result.score == 0.8475
    assert result.breakdown.relevance == 1.0
    assert result.breakdown.freshness == 0.75
    assert result.breakdown.popularity == 0.4
    assert "keyword:AI:2.00" in result.reasons


def test_keyword_matching_respects_term_boundaries() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    config = ScoringConfig(
        source_weights={"trusted": 1.0},
        keyword_weights={"AI": 1.0},
        freshness_window_hours=24,
    )

    result = score_material(_material("daily", "Daily update", 12), config, now=now)

    assert result.breakdown.relevance == 0.0


def test_rank_filters_and_limits_results() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    config = ScoringConfig({"trusted": 1.0}, {"AI": 1.0}, freshness_window_hours=24)
    ranked = rank_materials(
        (_material("older", "General", 0), _material("best", "AI news", 11)),
        config,
        minimum_score=0.5,
        limit=1,
        now=now,
    )
    assert [item.material.external_id for item in ranked] == ["best"]


def test_grouped_relevance_caps_terms_and_explains_combination_bonus() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    config = ScoringConfig(
        source_weights={"trusted": 1.0},
        keyword_weights={},
        freshness_window_hours=24,
        keyword_groups=(
            KeywordGroup("ai", ("AI", "LLM"), term_weight=0.3, max_score=0.3),
            KeywordGroup(
                "reliability",
                ("reliability", "testing"),
                term_weight=0.2,
                max_score=0.4,
            ),
        ),
        combination_rules=(
            CombinationRule("ai_reliability", frozenset({"ai", "reliability"}), 0.2),
        ),
        relevance_scale=0.9,
    )

    result = score_material(
        _material("grouped", "AI and LLM reliability testing", 12), config, now=now
    )

    assert result.breakdown.relevance == 1.0
    assert result.breakdown.matched_groups == ("ai", "reliability")
    assert result.breakdown.combination_bonuses == ("ai_reliability",)
    assert "group:ai:0.30" in result.reasons
    assert "combination:ai_reliability:0.20" in result.reasons


def test_noise_penalty_and_title_stopword_are_visible_in_breakdown() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    config = ScoringConfig(
        source_weights={"trusted": 1.0},
        keyword_weights={"AI": 1.0},
        freshness_window_hours=24,
        noise_terms=("sponsored",),
        noise_penalty=0.25,
        title_stopwords=("weekly digest",),
    )

    result = score_material(
        _material("blocked", "Weekly digest: sponsored AI", 12), config, now=now
    )

    assert result.breakdown.relevance == 0.0
    assert result.breakdown.noise_penalty == 0.25
    assert result.breakdown.stopword_blocked is True
    assert "title_stopword:blocked" in result.reasons


def test_score_is_monotonic_for_popularity_and_freshness() -> None:
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    config = ScoringConfig(
        source_weights={"trusted": 1.0},
        keyword_weights={"AI": 1.0},
        freshness_window_hours=24,
    )

    popularity_scores = [
        score_material(_material(str(value), "AI", 6, popularity=value), config, now=now).score
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    freshness_scores = [
        score_material(_material(str(hour), "AI", hour), config, now=now).score
        for hour in (0, 3, 6, 9, 12)
    ]

    assert popularity_scores == sorted(popularity_scores)
    assert freshness_scores == sorted(freshness_scores)


def test_scoring_config_rejects_invalid_invariants() -> None:
    with pytest.raises(ValueError):
        ScoringConfig({"trusted": 1.0}, {}, freshness_window_hours=0)
    with pytest.raises(ValueError):
        ScoringConfig({"trusted": 1.0}, {}, relevance_weight=0.5)
    with pytest.raises(ValueError):
        ScoringConfig({"trusted": 1.0}, {}, noise_penalty=-0.1)
    with pytest.raises(ValueError):
        ScoringConfig({"trusted": 1.0}, {}, relevance_scale=0)
