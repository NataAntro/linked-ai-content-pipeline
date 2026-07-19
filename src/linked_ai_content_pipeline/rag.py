from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

import httpx

from .http_client import request_with_retry


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    id: str
    text: str
    source_url: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    document: KnowledgeDocument
    score: float


class TextChunker:
    """Deterministic, source-preserving chunker for long editorial documents."""

    def __init__(self, max_characters: int = 800, overlap_characters: int = 120) -> None:
        if max_characters < 64:
            raise ValueError("max_characters must be at least 64")
        if not 0 <= overlap_characters < max_characters:
            raise ValueError("overlap_characters must be smaller than max_characters")
        self._max_characters = max_characters
        self._overlap_characters = overlap_characters

    def split(self, document: KnowledgeDocument) -> tuple[KnowledgeDocument, ...]:
        text = re.sub(r"\s+", " ", document.text).strip()
        if len(text) <= self._max_characters:
            return (KnowledgeDocument(document.id, text, document.source_url),)
        chunks: list[KnowledgeDocument] = []
        start = 0
        while start < len(text):
            hard_end = min(len(text), start + self._max_characters)
            end = hard_end
            if hard_end < len(text):
                boundary = text.rfind(" ", start + self._max_characters // 2, hard_end)
                if boundary > start:
                    end = boundary
            chunks.append(
                KnowledgeDocument(
                    id=f"{document.id}#chunk-{len(chunks) + 1}",
                    text=text[start:end].strip(),
                    source_url=document.source_url,
                )
            )
            if end == len(text):
                break
            start = max(start + 1, end - self._overlap_characters)
        return tuple(chunks)

    def split_all(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> tuple[KnowledgeDocument, ...]:
        return tuple(chunk for document in documents for chunk in self.split(document))


class EmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class VectorStore(Protocol):
    async def upsert(
        self,
        documents: Sequence[KnowledgeDocument],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    async def query(self, vector: Sequence[float], *, limit: int) -> tuple[SearchHit, ...]: ...


class HashEmbeddingProvider:
    """Deterministic local embedding used for tests and offline demonstrations."""

    _tokens = re.compile(r"[\w-]+", re.UNICODE)

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimensions
        tokens = self._tokens.findall(text.casefold())
        features = list(tokens)
        features.extend(
            f"char:{token[index : index + 3]}"
            for token in tokens
            if len(token) >= 5
            for index in range(len(token) - 2)
        )
        features.extend(f"bigram:{left}:{right}" for left, right in pairwise(tokens))
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return tuple(values)
        return tuple(value / norm for value in values)


class OpenAIEmbeddingProvider:
    """Async adapter for OpenAI-compatible embedding endpoints."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        dimensions: int,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(20.0),
        )
        self._owns_client = client is None
        self._endpoint = endpoint
        self._headers = {"authorization": f"Bearer {api_key}"}
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        response = await request_with_retry(
            self._client,
            "POST",
            self._endpoint,
            headers=self._headers,
            json={"model": self._model, "input": list(texts), "dimensions": self._dimensions},
        )
        response.raise_for_status()
        body = response.json()
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ValueError("embedding response has an invalid number of vectors")
        ordered = sorted(rows, key=_embedding_index)
        vectors = tuple(_embedding_vector(row, self._dimensions) for row in ordered)
        return vectors

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class InMemoryVectorStore:
    """Exact cosine search adapter with the same port as the Qdrant implementation."""

    def __init__(self) -> None:
        self._points: dict[str, tuple[KnowledgeDocument, tuple[float, ...]]] = {}

    async def upsert(
        self,
        documents: Sequence[KnowledgeDocument],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")
        for document, vector in zip(documents, vectors, strict=True):
            self._points[document.id] = (document, tuple(vector))

    async def query(self, vector: Sequence[float], *, limit: int) -> tuple[SearchHit, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_vector = tuple(vector)
        ranked = sorted(
            (
                SearchHit(document, _cosine_similarity(query_vector, stored_vector))
                for document, stored_vector in self._points.values()
            ),
            key=lambda hit: (-hit.score, hit.document.id),
        )
        return tuple(ranked[:limit])


class QdrantVectorStore:
    """Small async Qdrant REST adapter; payload keeps every hit traceable to its source."""

    def __init__(
        self,
        base_url: str,
        collection: str,
        dimensions: int,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._headers = {"api-key": api_key} if api_key else None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(10.0),
        )
        self._owns_client = client is None
        self._collection = collection
        self._dimensions = dimensions

    async def initialize(self) -> None:
        response = await request_with_retry(
            self._client,
            "PUT",
            f"/collections/{self._collection}",
            headers=self._headers,
            json={"vectors": {"size": self._dimensions, "distance": "Cosine"}},
        )
        if response.status_code not in {200, 409}:
            response.raise_for_status()

    async def upsert(
        self,
        documents: Sequence[KnowledgeDocument],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")
        points = [
            {
                "id": str(uuid5(NAMESPACE_URL, document.id)),
                "vector": list(vector),
                "payload": {
                    "document_id": document.id,
                    "text": document.text,
                    "source_url": document.source_url,
                },
            }
            for document, vector in zip(documents, vectors, strict=True)
        ]
        response = await request_with_retry(
            self._client,
            "PUT",
            f"/collections/{self._collection}/points",
            headers=self._headers,
            params={"wait": "true"},
            json={"points": points},
        )
        response.raise_for_status()

    async def query(self, vector: Sequence[float], *, limit: int) -> tuple[SearchHit, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        response = await request_with_retry(
            self._client,
            "POST",
            f"/collections/{self._collection}/points/query",
            headers=self._headers,
            json={"query": list(vector), "limit": limit, "with_payload": True},
        )
        response.raise_for_status()
        body = response.json()
        rows = body.get("result", {}).get("points", [])
        if not isinstance(rows, list):
            raise ValueError("Qdrant response does not contain result.points")
        return tuple(_parse_qdrant_hit(row) for row in rows)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class SemanticRetriever:
    def __init__(self, embeddings: EmbeddingProvider, store: VectorStore) -> None:
        self._embeddings = embeddings
        self._store = store

    async def index(self, documents: Sequence[KnowledgeDocument]) -> None:
        vectors = await self._embeddings.embed([document.text for document in documents])
        await self._store.upsert(documents, vectors)

    async def retrieve(self, query: str, *, limit: int = 3) -> tuple[SearchHit, ...]:
        (vector,) = await self._embeddings.embed([query])
        return await self._store.query(vector, limit=limit)


@dataclass(slots=True)
class RetrievalRuntime:
    """Owns a configured retriever and the network adapters behind it."""

    retriever: SemanticRetriever
    embeddings: OpenAIEmbeddingProvider | None = None
    store: QdrantVectorStore | None = None

    async def close(self) -> None:
        if self.store is not None:
            await self.store.close()
        if self.embeddings is not None:
            await self.embeddings.close()


async def retrieval_runtime_from_environment(
    mode: str,
    environment: Mapping[str, str],
    *,
    embedding_client: httpx.AsyncClient | None = None,
    qdrant_client: httpx.AsyncClient | None = None,
) -> RetrievalRuntime:
    """Build either the offline retriever or the complete embeddings + Qdrant path."""

    if mode == "local":
        return RetrievalRuntime(SemanticRetriever(HashEmbeddingProvider(), InMemoryVectorStore()))
    if mode != "external":
        raise ValueError(f"unsupported retrieval provider: {mode}")

    required = (
        "EMBEDDING_ENDPOINT",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_API_KEY",
        "QDRANT_URL",
    )
    missing = [name for name in required if not environment.get(name, "").strip()]
    if missing:
        raise ValueError(f"missing external retrieval settings: {', '.join(missing)}")
    try:
        dimensions = int(environment["EMBEDDING_DIMENSIONS"])
    except ValueError as error:
        raise ValueError("EMBEDDING_DIMENSIONS must be an integer") from error
    if dimensions < 1:
        raise ValueError("EMBEDDING_DIMENSIONS must be positive")

    embeddings = OpenAIEmbeddingProvider(
        environment["EMBEDDING_ENDPOINT"],
        environment["EMBEDDING_API_KEY"],
        environment["EMBEDDING_MODEL"],
        dimensions,
        client=embedding_client,
    )
    store = QdrantVectorStore(
        environment["QDRANT_URL"],
        environment.get("QDRANT_COLLECTION", "linked_editorial_context"),
        dimensions,
        api_key=environment.get("QDRANT_API_KEY") or None,
        client=qdrant_client,
    )
    try:
        await store.initialize()
    except BaseException:
        await store.close()
        await embeddings.close()
        raise
    return RetrievalRuntime(SemanticRetriever(embeddings, store), embeddings, store)


def load_knowledge_documents(path: Path) -> tuple[KnowledgeDocument, ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("knowledge fixture must contain a JSON array")
    documents: list[KnowledgeDocument] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("knowledge document must be an object")
        try:
            documents.append(
                KnowledgeDocument(
                    id=str(row["id"]),
                    text=str(row["text"]),
                    source_url=str(row["source_url"]),
                )
            )
        except KeyError as error:
            raise ValueError("knowledge document is missing a required field") from error
    return tuple(documents)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _parse_qdrant_hit(row: object) -> SearchHit:
    if not isinstance(row, dict):
        raise ValueError("Qdrant point must be an object")
    typed_row: dict[str, Any] = row
    payload = typed_row.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Qdrant point payload is missing")
    try:
        document = KnowledgeDocument(
            id=str(payload["document_id"]),
            text=str(payload["text"]),
            source_url=str(payload["source_url"]),
        )
        score = float(typed_row["score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Qdrant point payload is invalid") from error
    return SearchHit(document, score)


def _embedding_index(row: object) -> int:
    if not isinstance(row, dict) or not isinstance(row.get("index"), int):
        raise ValueError("embedding response row has no numeric index")
    return int(row["index"])


def _embedding_vector(row: object, dimensions: int) -> tuple[float, ...]:
    if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
        raise ValueError("embedding response row has no vector")
    try:
        vector = tuple(float(value) for value in row["embedding"])
    except (TypeError, ValueError) as error:
        raise ValueError("embedding vector contains a non-numeric value") from error
    if len(vector) != dimensions:
        raise ValueError("embedding vector has unexpected dimensions")
    return vector
