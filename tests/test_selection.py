from __future__ import annotations

from datetime import UTC, datetime

from linked_ai_content_pipeline.models import Material, ScoreBreakdown, ScoredMaterial
from linked_ai_content_pipeline.selection import select_diverse_materials


def _candidate(identifier: str, text: str, score: float) -> ScoredMaterial:
    material = Material(
        source="fixture",
        external_id=identifier,
        title=text,
        body=text,
        canonical_url=f"https://example.test/{identifier}",
        published_at=datetime(2026, 7, 19, tzinfo=UTC),
        content_fingerprint=identifier,
    )
    return ScoredMaterial(material, score, (), ScoreBreakdown(1.0, 1.0, 0.0, 1.0))


def test_diversity_selection_prefers_new_signal_over_near_duplicate() -> None:
    candidates = (
        _candidate("leader", "AI agents need reliable evaluation", 0.90),
        _candidate("duplicate", "Reliable evaluation for AI agents", 0.89),
        _candidate("different", "Spatial interfaces change product design", 0.82),
    )

    result = select_diverse_materials(candidates, limit=2, similarity_penalty=0.4)

    assert [item.material.external_id for item in result.materials] == ["leader", "different"]
    assert result.audit[0].maximum_similarity == 0.0
    assert result.audit[1].adjusted_score == 0.82


def test_diversity_selection_is_stable_when_scores_are_equal() -> None:
    candidates = (
        _candidate("first", "one topic", 0.8),
        _candidate("second", "another topic", 0.8),
    )

    result = select_diverse_materials(candidates, limit=1, similarity_penalty=0.25)

    assert result.materials[0].material.external_id == "first"


def test_diversity_audit_preserves_selection_invariants_for_larger_pool() -> None:
    candidates = tuple(
        _candidate(f"item-{index}", f"topic {index % 3} signal {index}", 1.0 - index / 100)
        for index in range(20)
    )

    result = select_diverse_materials(candidates, limit=7, similarity_penalty=0.3)

    identifiers = [item.material.external_id for item in result.materials]
    assert len(identifiers) == len(set(identifiers)) == 7
    assert [entry.selected_order for entry in result.audit] == list(range(1, 8))
    assert all(entry.adjusted_score <= entry.base_score for entry in result.audit)
