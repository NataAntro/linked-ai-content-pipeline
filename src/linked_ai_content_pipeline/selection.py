from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ScoredMaterial

_TOKEN = re.compile(r"[\w+#.-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class DiversityDecision:
    material_id: str
    selected_order: int
    base_score: float
    maximum_similarity: float
    adjusted_score: float


@dataclass(frozen=True, slots=True)
class DiversitySelection:
    materials: tuple[ScoredMaterial, ...]
    audit: tuple[DiversityDecision, ...]


def select_diverse_materials(
    candidates: tuple[ScoredMaterial, ...],
    *,
    limit: int,
    similarity_penalty: float,
) -> DiversitySelection:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not 0.0 <= similarity_penalty <= 1.0:
        raise ValueError("similarity_penalty must be between 0 and 1")

    remaining = list(enumerate(candidates))
    selected: list[ScoredMaterial] = []
    selected_tokens: list[frozenset[str]] = []
    audit: list[DiversityDecision] = []

    while remaining and len(selected) < limit:
        evaluated: list[tuple[float, float, int, float, frozenset[str], ScoredMaterial]] = []
        for original_index, candidate in remaining:
            tokens = _material_tokens(candidate)
            maximum_similarity = max(
                (_jaccard(tokens, previous) for previous in selected_tokens),
                default=0.0,
            )
            adjusted_score = candidate.score - similarity_penalty * maximum_similarity
            evaluated.append(
                (
                    adjusted_score,
                    candidate.score,
                    -original_index,
                    maximum_similarity,
                    tokens,
                    candidate,
                )
            )

        best = max(evaluated, key=lambda item: item[:3])
        adjusted_score, _, negative_index, maximum_similarity, tokens, candidate = best
        original_index = -negative_index
        remaining = [item for item in remaining if item[0] != original_index]
        selected.append(candidate)
        selected_tokens.append(tokens)
        audit.append(
            DiversityDecision(
                material_id=f"{candidate.material.source}:{candidate.material.external_id}",
                selected_order=len(selected),
                base_score=candidate.score,
                maximum_similarity=round(maximum_similarity, 4),
                adjusted_score=round(adjusted_score, 4),
            )
        )

    return DiversitySelection(tuple(selected), tuple(audit))


def _material_tokens(candidate: ScoredMaterial) -> frozenset[str]:
    text = f"{candidate.material.title} {candidate.material.body}".casefold()
    return frozenset(_TOKEN.findall(text))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0
