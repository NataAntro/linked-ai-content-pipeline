from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from linked_ai_content_pipeline.rag import KnowledgeDocument, QdrantVectorStore


@pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason="QDRANT_URL is not configured")
def test_qdrant_adapter_against_live_service() -> None:
    async def scenario() -> None:
        store = QdrantVectorStore(
            os.environ["QDRANT_URL"],
            "editorial_context_ci",
            3,
        )
        for attempt in range(20):
            try:
                await store.initialize()
                break
            except httpx.HTTPError:
                if attempt == 19:
                    raise
                await asyncio.sleep(0.25)
        document = KnowledgeDocument(
            "live-qdrant",
            "Observable retrieval",
            "https://example.test/live-qdrant",
        )
        await store.upsert([document], [[1.0, 0.0, 0.0]])
        hits = await store.query([1.0, 0.0, 0.0], limit=1)
        await store.close()

        assert hits[0].document == document
        assert hits[0].score > 0.99

    asyncio.run(scenario())
