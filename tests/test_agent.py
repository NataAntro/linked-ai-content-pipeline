from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from linked_ai_content_pipeline.agent import (
    BoundedEditorialAgent,
    DeterministicContextDraftProvider,
)
from linked_ai_content_pipeline.editorial import EditorialRejection, default_editorial_chain
from linked_ai_content_pipeline.models import Draft, Material, ScoreBreakdown, ScoredMaterial
from linked_ai_content_pipeline.rag import (
    HashEmbeddingProvider,
    InMemoryVectorStore,
    KnowledgeDocument,
    SearchHit,
    SemanticRetriever,
)


def test_agent_retrieves_context_and_records_bounded_trace() -> None:
    async def scenario() -> None:
        retriever = SemanticRetriever(HashEmbeddingProvider(64), InMemoryVectorStore())
        await retriever.index(
            [
                KnowledgeDocument(
                    "context",
                    "Reliable AI agents use validation and bounded retries",
                    "https://example.test/context",
                )
            ]
        )
        material = ScoredMaterial(
            Material(
                "fixture",
                "one",
                "Reliable AI agents",
                "Validation and bounded retries make agent behavior observable.",
                "https://example.test/news",
                datetime(2026, 7, 20, tzinfo=UTC),
                "fingerprint",
            ),
            1.0,
            (),
            ScoreBreakdown(1.0, 1.0, 0.0, 1.0),
        )
        agent = BoundedEditorialAgent(
            retriever,
            DeterministicContextDraftProvider(),
            default_editorial_chain(),
            max_steps=6,
        )

        result = await agent.run(material, accepted_drafts=())

        assert result.draft is not None
        assert "https://example.test/context" in result.draft.body
        assert [entry.action for entry in result.trace] == [
            "retrieve_context",
            "generate_draft",
            "complete",
        ]
        assert result.context[0].document.id == "context"

    asyncio.run(scenario())


def test_agent_revises_rejected_draft_once_and_rechecks_all_gates() -> None:
    class RepairingProvider:
        async def generate(
            self,
            material: ScoredMaterial,
            context: tuple[SearchHit, ...],
        ) -> Draft:
            del material, context
            return Draft("fixture:one", "Революционный прорыв", "Проверяемый текст")

        async def revise(
            self,
            draft: Draft,
            material: ScoredMaterial,
            context: tuple[SearchHit, ...],
            rejection: EditorialRejection,
        ) -> Draft:
            del material, context, rejection
            return Draft(draft.material_id, "Практическое изменение", draft.body)

    async def scenario() -> None:
        retriever = SemanticRetriever(HashEmbeddingProvider(32), InMemoryVectorStore())
        await retriever.index(
            [KnowledgeDocument("context", "AI reliability", "https://example.test/context")]
        )
        material = ScoredMaterial(
            Material(
                "fixture",
                "one",
                "AI reliability",
                "Bounded recovery",
                "https://example.test/news",
                datetime(2026, 7, 20, tzinfo=UTC),
                "fingerprint",
            ),
            1.0,
            (),
            ScoreBreakdown(1.0, 1.0, 0.0, 1.0),
        )
        agent = BoundedEditorialAgent(
            retriever,
            RepairingProvider(),
            default_editorial_chain(),
            max_steps=7,
        )

        result = await agent.run(material, accepted_drafts=())

        assert result.draft is not None
        assert result.draft.headline == "Практическое изменение"
        assert [entry.action for entry in result.trace] == [
            "retrieve_context",
            "generate_draft",
            "review_rejected",
            "revise_draft",
            "complete",
        ]

    asyncio.run(scenario())
