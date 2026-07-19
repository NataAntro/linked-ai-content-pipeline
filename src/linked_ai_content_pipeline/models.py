from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RawMaterial:
    source: str
    external_id: str
    title: str
    body: str
    url: str
    published_at: datetime
    popularity: float = 0.0


@dataclass(frozen=True, slots=True)
class Material:
    source: str
    external_id: str
    title: str
    body: str
    canonical_url: str
    published_at: datetime
    content_fingerprint: str
    popularity: float = 0.0


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    relevance: float
    freshness: float
    popularity: float
    source_weight: float
    matched_groups: tuple[str, ...] = ()
    combination_bonuses: tuple[str, ...] = ()
    noise_penalty: float = 0.0
    stopword_blocked: bool = False


@dataclass(frozen=True, slots=True)
class ScoredMaterial:
    material: Material
    score: float
    reasons: tuple[str, ...]
    breakdown: ScoreBreakdown


@dataclass(frozen=True, slots=True)
class Draft:
    material_id: str
    headline: str
    body: str


@dataclass(frozen=True, slots=True)
class CollectionFailure:
    source: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class DraftFailure:
    material_id: str
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class CollectionReport:
    materials: tuple[RawMaterial, ...]
    failures: tuple[CollectionFailure, ...]
