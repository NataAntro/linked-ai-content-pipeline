from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .editorial import EditorialAuditEntry, EditorialChain, EditorialRejection
from .models import Draft, ScoredMaterial
from .rag import SearchHit, SemanticRetriever


class ContextualDraftProvider(Protocol):
    async def generate(
        self,
        material: ScoredMaterial,
        context: tuple[SearchHit, ...],
    ) -> Draft: ...

    async def revise(
        self,
        draft: Draft,
        material: ScoredMaterial,
        context: tuple[SearchHit, ...],
        rejection: EditorialRejection,
    ) -> Draft: ...


@dataclass(frozen=True, slots=True)
class AgentTraceEntry:
    step: int
    action: str
    detail: str


@dataclass(frozen=True, slots=True)
class AgentResult:
    draft: Draft | None
    context: tuple[SearchHit, ...]
    audit: tuple[EditorialAuditEntry, ...]
    trace: tuple[AgentTraceEntry, ...]
    rejection: EditorialRejection | None = None


class BoundedEditorialAgent:
    """Retrieve -> draft -> review loop with explicit budgets and a complete trace."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        provider: ContextualDraftProvider,
        editorial: EditorialChain,
        *,
        retrieval_limit: int = 3,
        max_revision_rounds: int = 1,
        max_steps: int = 8,
    ) -> None:
        if retrieval_limit < 1:
            raise ValueError("retrieval_limit must be positive")
        if max_revision_rounds < 0:
            raise ValueError("max_revision_rounds must not be negative")
        if max_steps < 3:
            raise ValueError("max_steps must be at least 3")
        self._retriever = retriever
        self._provider = provider
        self._editorial = editorial
        self._retrieval_limit = retrieval_limit
        self._max_revision_rounds = max_revision_rounds
        self._max_steps = max_steps

    async def run(
        self,
        material: ScoredMaterial,
        *,
        accepted_drafts: tuple[Draft, ...],
    ) -> AgentResult:
        trace: list[AgentTraceEntry] = []
        audit: list[EditorialAuditEntry] = []

        def record(action: str, detail: str) -> None:
            if len(trace) >= self._max_steps:
                raise RuntimeError("agent step budget exhausted")
            trace.append(AgentTraceEntry(len(trace) + 1, action, detail))

        source = material.material
        query = f"{source.title}\n{source.body}"
        context = await self._retriever.retrieve(query, limit=self._retrieval_limit)
        record("retrieve_context", f"retrieved={len(context)}")
        current = await self._provider.generate(material, context)
        record("generate_draft", f"material={current.material_id}")

        for revision_round in range(self._max_revision_rounds + 1):
            review = await self._editorial.review(
                current,
                material=material,
                accepted_drafts=accepted_drafts,
            )
            audit.extend(review.audit)
            if review.draft is not None:
                record("complete", f"revision_rounds={revision_round}")
                return AgentResult(review.draft, context, tuple(audit), tuple(trace))
            rejection = review.rejection
            if rejection is None:
                raise RuntimeError("editorial review returned neither draft nor rejection")
            record("review_rejected", f"stage={rejection.stage}; reason={rejection.reason}")
            if revision_round >= self._max_revision_rounds:
                record("stop", "revision budget exhausted")
                return AgentResult(None, context, tuple(audit), tuple(trace), rejection)
            current = await self._provider.revise(
                current,
                material,
                context,
                rejection,
            )
            record("revise_draft", f"round={revision_round + 1}")
        raise RuntimeError("agent loop exhausted unexpectedly")


class DeterministicContextDraftProvider:
    """Offline provider that makes retrieved citations observable in the result."""

    async def generate(
        self,
        material: ScoredMaterial,
        context: tuple[SearchHit, ...],
    ) -> Draft:
        source = material.material
        citations = "\n".join(
            f"[{index}] {hit.document.source_url}" for index, hit in enumerate(context, start=1)
        )
        return Draft(
            material_id=f"{source.source}:{source.external_id}",
            headline=f"Что меняет {source.title}",
            body=f"{source.body[:240].rstrip()}\n\nКонтекст:\n{citations}".strip(),
        )

    async def revise(
        self,
        draft: Draft,
        material: ScoredMaterial,
        context: tuple[SearchHit, ...],
        rejection: EditorialRejection,
    ) -> Draft:
        del material, context
        headline = draft.headline
        if rejection.stage == "anti_slop_gate":
            headline = headline.replace("революционный прорыв", "практическое изменение")
        return Draft(draft.material_id, headline, draft.body)
