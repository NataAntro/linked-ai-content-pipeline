from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from linked_ai_content_pipeline.rag import (
    HashEmbeddingProvider,
    InMemoryVectorStore,
    KnowledgeDocument,
    OpenAIEmbeddingProvider,
    QdrantVectorStore,
    SemanticRetriever,
    TextChunker,
    retrieval_runtime_from_environment,
)


def test_semantic_retriever_returns_relevant_traceable_context() -> None:
    async def scenario() -> None:
        retriever = SemanticRetriever(HashEmbeddingProvider(64), InMemoryVectorStore())
        await retriever.index(
            [
                KnowledgeDocument(
                    "reliability",
                    "AI agents need retry budgets, validation and observable traces",
                    "https://example.test/reliability",
                ),
                KnowledgeDocument(
                    "design",
                    "Typography and color systems for mobile interfaces",
                    "https://example.test/design",
                ),
            ]
        )

        hits = await retriever.retrieve("validation and retry for AI agents", limit=1)

        assert hits[0].document.id == "reliability"
        assert hits[0].score > 0

    asyncio.run(scenario())


def test_chunker_bounds_size_preserves_overlap_and_source() -> None:
    source = KnowledgeDocument(
        "long",
        " ".join(f"token-{index}" for index in range(60)),
        "https://example.test/long",
    )

    chunks = TextChunker(max_characters=96, overlap_characters=16).split(source)

    assert len(chunks) > 2
    assert all(len(chunk.text) <= 96 for chunk in chunks)
    assert all(chunk.source_url == source.source_url for chunk in chunks)
    assert [chunk.id for chunk in chunks] == [
        f"long#chunk-{index}" for index in range(1, len(chunks) + 1)
    ]
    assert chunks[0].text.split()[-1] in chunks[1].text


def test_embedding_adapter_handles_empty_input_and_rejects_wrong_dimensions() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIEmbeddingProvider(
            "https://api.example.test/v1/embeddings",
            "secret",
            "embedding-model",
            3,
            client=client,
        )

        assert await provider.embed([]) == ()
        with pytest.raises(ValueError, match="unexpected dimensions"):
            await provider.embed(["first"])
        await client.aclose()

    asyncio.run(scenario())


def test_qdrant_adapter_uses_idempotent_upsert_and_query_api() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/points/query"):
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {
                                "score": 0.91,
                                "payload": {
                                    "document_id": "doc-1",
                                    "text": "Retry budgets",
                                    "source_url": "https://example.test/one",
                                },
                            }
                        ]
                    }
                },
            )
        return httpx.Response(200, json={"status": "ok"})

    async def scenario() -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://qdrant.local:6333",
        )
        store = QdrantVectorStore(
            "http://qdrant.local:6333",
            "editorial_context",
            3,
            client=client,
        )
        document = KnowledgeDocument(
            "doc-1",
            "Retry budgets",
            "https://example.test/one",
        )
        await store.initialize()
        await store.upsert([document], [[1.0, 0.0, 0.0]])
        hits = await store.query([1.0, 0.0, 0.0], limit=1)
        await client.aclose()

        assert hits[0].document == document
        assert [request.method for request in requests] == ["PUT", "PUT", "POST"]
        assert requests[-1].url.path.endswith("/points/query")

    asyncio.run(scenario())


def test_openai_embedding_adapter_preserves_input_order_and_dimensions() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["model"] == "embedding-model"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 1, 0]},
                    {"index": 0, "embedding": [1, 0, 0]},
                ]
            },
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIEmbeddingProvider(
            "https://api.example.test/v1/embeddings",
            "secret",
            "embedding-model",
            3,
            client=client,
        )

        vectors = await provider.embed(["first", "second"])
        await client.aclose()

        assert vectors == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    asyncio.run(scenario())


def test_external_retrieval_runtime_wires_embeddings_to_qdrant() -> None:
    embedding_requests: list[httpx.Request] = []
    qdrant_requests: list[httpx.Request] = []

    async def embedding_handler(request: httpx.Request) -> httpx.Response:
        embedding_requests.append(request)
        body = json.loads(request.content)
        vectors = [
            {"index": index, "embedding": [1.0, 0.0, 0.0]}
            for index, _text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": vectors})

    async def qdrant_handler(request: httpx.Request) -> httpx.Response:
        qdrant_requests.append(request)
        if request.url.path.endswith("/points/query"):
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {
                                "score": 0.93,
                                "payload": {
                                    "document_id": "agent-reliability",
                                    "text": "Bounded retries and traceable decisions",
                                    "source_url": "https://example.test/agent-reliability",
                                },
                            }
                        ]
                    }
                },
            )
        return httpx.Response(200, json={"status": "ok"})

    async def scenario() -> None:
        embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(embedding_handler))
        qdrant_client = httpx.AsyncClient(
            transport=httpx.MockTransport(qdrant_handler),
            base_url="http://qdrant.local:6333",
        )
        runtime = await retrieval_runtime_from_environment(
            "external",
            {
                "EMBEDDING_ENDPOINT": "https://api.example.test/v1/embeddings",
                "EMBEDDING_MODEL": "embedding-model",
                "EMBEDDING_DIMENSIONS": "3",
                "EMBEDDING_API_KEY": "embedding-secret",
                "QDRANT_URL": "http://qdrant.local:6333",
                "QDRANT_COLLECTION": "editorial_context_test",
                "QDRANT_API_KEY": "qdrant-secret",
            },
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
        )
        document = KnowledgeDocument(
            "agent-reliability",
            "Bounded retries and traceable decisions",
            "https://example.test/agent-reliability",
        )
        await runtime.retriever.index([document])
        hits = await runtime.retriever.retrieve("reliable AI agent", limit=1)
        await runtime.close()
        await embedding_client.aclose()
        await qdrant_client.aclose()

        assert hits[0].document == document
        assert len(embedding_requests) == 2
        assert [request.method for request in qdrant_requests] == ["PUT", "PUT", "POST"]
        assert qdrant_requests[0].headers["api-key"] == "qdrant-secret"

    asyncio.run(scenario())


def test_external_retrieval_runtime_validates_configuration() -> None:
    with pytest.raises(ValueError, match="missing external retrieval settings"):
        asyncio.run(retrieval_runtime_from_environment("external", {}))

    environment = {
        "EMBEDDING_ENDPOINT": "https://api.example.test/v1/embeddings",
        "EMBEDDING_MODEL": "embedding-model",
        "EMBEDDING_DIMENSIONS": "not-a-number",
        "EMBEDDING_API_KEY": "embedding-secret",
        "QDRANT_URL": "http://qdrant.local:6333",
    }
    with pytest.raises(ValueError, match="must be an integer"):
        asyncio.run(retrieval_runtime_from_environment("external", environment))
