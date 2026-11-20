from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from .agent import AgentTraceEntry, BoundedEditorialAgent
from .collection import collect_concurrently
from .collectors import Collector
from .editorial import (
    EditorialAuditEntry,
    EditorialChain,
    EditorialRejection,
    default_editorial_chain,
)
from .models import CollectionFailure, Draft, DraftFailure, ScoredMaterial
from .normalization import normalize_and_deduplicate
from .providers import DraftProvider
from .scoring import ScoringConfig, rank_materials
from .selection import DiversityDecision, select_diverse_materials
from .sinks import DraftSink


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    minimum_score: float = 1.0
    top_n: int = 3
    candidate_pool_size: int = 10
    diversity_penalty: float = 0.25
    dry_run: bool = True


@dataclass(frozen=True, slots=True)
class PipelineResult:
    collected_count: int
    unique_count: int
    selected: tuple[ScoredMaterial, ...]
    selection_audit: tuple[DiversityDecision, ...]
    drafts: tuple[Draft, ...]
    collection_failures: tuple[CollectionFailure, ...]
    draft_failures: tuple[DraftFailure, ...]
    editorial_audit: tuple[EditorialAuditEntry, ...]
    editorial_rejections: tuple[EditorialRejection, ...]
    agent_trace: tuple[AgentTraceEntry, ...]
    persisted: bool


class ContentPipeline:
    def __init__(
        self,
        collectors: Sequence[Collector],
        provider: DraftProvider,
        scoring: ScoringConfig,
        settings: PipelineSettings,
        sink: DraftSink | None = None,
        editorial: EditorialChain | None = None,
        agent: BoundedEditorialAgent | None = None,
    ) -> None:
        if not collectors:
            raise ValueError("at least one collector is required")
        if settings.top_n < 1:
            raise ValueError("top_n must be positive")
        if settings.candidate_pool_size < settings.top_n:
            raise ValueError("candidate_pool_size must not be smaller than top_n")
        if not 0.0 <= settings.diversity_penalty <= 1.0:
            raise ValueError("diversity_penalty must be between 0 and 1")
        if not settings.dry_run and sink is None:
            raise ValueError("a sink is required when dry_run is disabled")
        self._collectors = tuple(collectors)
        self._provider = provider
        self._scoring = scoring
        self._settings = settings
        self._sink = sink
        self._editorial = editorial or default_editorial_chain()
        self._agent = agent

    async def run(self, *, now: datetime | None = None) -> PipelineResult:
        collection = await collect_concurrently(self._collectors)
        unique = normalize_and_deduplicate(collection.materials)
        candidates = rank_materials(
            unique,
            self._scoring,
            minimum_score=self._settings.minimum_score,
            limit=self._settings.candidate_pool_size,
            now=now,
        )
        selection = select_diverse_materials(
            candidates,
            limit=self._settings.top_n,
            similarity_penalty=self._settings.diversity_penalty,
        )
        selected = selection.materials
        drafts: list[Draft] = []
        audit: list[EditorialAuditEntry] = []
        rejections: list[EditorialRejection] = []
        agent_trace: list[AgentTraceEntry] = []
        draft_failures: list[DraftFailure] = []
        if self._agent is None:
            generated = await asyncio.gather(
                *(self._provider.generate(item) for item in selected),
                return_exceptions=True,
            )
            for material, generated_draft in zip(selected, generated, strict=True):
                if isinstance(generated_draft, BaseException):
                    if not isinstance(generated_draft, Exception):
                        raise generated_draft
                    draft_failures.append(
                        self._draft_failure(material, "generate", generated_draft)
                    )
                    continue
                try:
                    review = await self._editorial.review(
                        generated_draft,
                        material=material,
                        accepted_drafts=tuple(drafts),
                    )
                except Exception as error:
                    draft_failures.append(self._draft_failure(material, "review", error))
                    continue
                audit.extend(review.audit)
                if review.rejection is not None:
                    rejections.append(review.rejection)
                elif review.draft is not None:
                    drafts.append(review.draft)
        else:
            for material in selected:
                try:
                    result = await self._agent.run(material, accepted_drafts=tuple(drafts))
                except Exception as error:
                    draft_failures.append(self._draft_failure(material, "agent", error))
                    continue
                audit.extend(result.audit)
                agent_trace.extend(result.trace)
                if result.rejection is not None:
                    rejections.append(result.rejection)
                elif result.draft is not None:
                    drafts.append(result.draft)
        approved_drafts = tuple(drafts)
        persisted = False
        if not self._settings.dry_run:
            sink = self._sink
            if sink is None:
                raise RuntimeError("pipeline invariant violated: persistence sink is missing")
            sink.write(approved_drafts)
            persisted = True
        return PipelineResult(
            collected_count=len(collection.materials),
            unique_count=len(unique),
            selected=selected,
            selection_audit=selection.audit,
            drafts=approved_drafts,
            collection_failures=collection.failures,
            draft_failures=tuple(draft_failures),
            editorial_audit=tuple(audit),
            editorial_rejections=tuple(rejections),
            agent_trace=tuple(agent_trace),
            persisted=persisted,
        )

    @staticmethod
    def _draft_failure(
        material: ScoredMaterial,
        stage: str,
        error: Exception,
    ) -> DraftFailure:
        source = material.material
        return DraftFailure(
            material_id=f"{source.source}:{source.external_id}",
            stage=stage,
            error_type=type(error).__name__,
            message=str(error),
        )
