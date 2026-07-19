from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .rag import SearchHit


class Retriever(Protocol):
    async def retrieve(self, query: str, *, limit: int) -> tuple[SearchHit, ...]: ...


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    query: str
    expected_document_ids: frozenset[str]
    excluded_document_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    query: str
    ranked_document_ids: tuple[str, ...]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    hard_negative_hits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    case_count: int
    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    hard_negative_rate: float
    cases: tuple[RetrievalCaseResult, ...]


def load_retrieval_cases(path: Path) -> tuple[RetrievalCase, ...]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("retrieval evaluation fixture must be a non-empty JSON array")
    cases: list[RetrievalCase] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("retrieval evaluation case must be an object")
        query = row.get("query")
        expected = row.get("expected_document_ids")
        excluded = row.get("excluded_document_ids", [])
        if not isinstance(query, str) or not query.strip():
            raise ValueError("retrieval query must be a non-empty string")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(item, str) and item for item in expected)
        ):
            raise ValueError("expected_document_ids must be a non-empty string array")
        if not isinstance(excluded, list) or not all(
            isinstance(item, str) and item for item in excluded
        ):
            raise ValueError("excluded_document_ids must be a string array")
        if set(expected).intersection(excluded):
            raise ValueError("expected and excluded document ids must not overlap")
        cases.append(RetrievalCase(query.strip(), frozenset(expected), frozenset(excluded)))
    return tuple(cases)


async def evaluate_retrieval(
    retriever: Retriever,
    cases: tuple[RetrievalCase, ...],
    *,
    limit: int,
) -> RetrievalEvaluation:
    if not cases:
        raise ValueError("at least one retrieval case is required")
    if limit < 1:
        raise ValueError("limit must be positive")
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    hard_negative_rates: list[float] = []
    results: list[RetrievalCaseResult] = []
    for case in cases:
        hits = await retriever.retrieve(case.query, limit=limit)
        ranked_ids = [hit.document.id.split("#chunk-", 1)[0] for hit in hits]
        matched = case.expected_document_ids.intersection(ranked_ids)
        recall = len(matched) / len(case.expected_document_ids)
        precision = len(matched) / limit
        first_rank = next(
            (index for index, item in enumerate(ranked_ids, start=1) if item in matched),
            None,
        )
        reciprocal_rank = 0.0 if first_rank is None else 1 / first_rank
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, document_id in enumerate(ranked_ids, start=1)
            if document_id in case.expected_document_ids
        )
        ideal_hits = min(len(case.expected_document_ids), limit)
        ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
        hard_negative_hits = tuple(
            document_id for document_id in ranked_ids if document_id in case.excluded_document_ids
        )
        hard_negative_rate = len(hard_negative_hits) / max(1, len(case.excluded_document_ids))
        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(reciprocal_rank)
        ndcgs.append(ndcg)
        hard_negative_rates.append(hard_negative_rate)
        results.append(
            RetrievalCaseResult(
                query=case.query,
                ranked_document_ids=tuple(ranked_ids),
                recall_at_k=recall,
                precision_at_k=precision,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg,
                hard_negative_hits=hard_negative_hits,
            )
        )
    return RetrievalEvaluation(
        case_count=len(cases),
        recall_at_k=sum(recalls) / len(recalls),
        precision_at_k=sum(precisions) / len(precisions),
        hit_rate_at_k=sum(value > 0 for value in recalls) / len(recalls),
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks),
        ndcg_at_k=sum(ndcgs) / len(ndcgs),
        hard_negative_rate=sum(hard_negative_rates) / len(hard_negative_rates),
        cases=tuple(results),
    )
