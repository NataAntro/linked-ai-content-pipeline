from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from linked_ai_content_pipeline.rag import KnowledgeDocument, SearchHit
from linked_ai_content_pipeline.rag_eval import main
from linked_ai_content_pipeline.retrieval_evaluation import (
    RetrievalCase,
    evaluate_retrieval,
    load_retrieval_cases,
)


class FixtureRetriever:
    async def retrieve(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        del limit
        ids = ("noise", "expected#chunk-1") if query == "hit-second" else ("noise",)
        return tuple(
            SearchHit(
                KnowledgeDocument(item, item, f"https://example.test/{item}"),
                1.0 / index,
            )
            for index, item in enumerate(ids, start=1)
        )


def test_retrieval_evaluation_calculates_recall_and_mrr() -> None:
    report = asyncio.run(
        evaluate_retrieval(
            FixtureRetriever(),
            (
                RetrievalCase("hit-second", frozenset({"expected"})),
                RetrievalCase("miss", frozenset({"missing"})),
            ),
            limit=2,
        )
    )

    assert report.case_count == 2
    assert report.recall_at_k == 0.5
    assert report.hit_rate_at_k == 0.5
    assert report.precision_at_k == 0.25
    assert report.mean_reciprocal_rank == 0.25
    assert report.ndcg_at_k == 0.31546487678572877


def test_retrieval_fixture_has_explicit_expected_documents(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.json"
    fixture.write_text(
        '[{"query":"agent reliability","expected_document_ids":["agent"]}]',
        encoding="utf-8",
    )
    cases = load_retrieval_cases(fixture)
    assert cases == (RetrievalCase("agent reliability", frozenset({"agent"})),)

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_retrieval_cases(empty)


def test_rag_evaluation_command_reports_reproducible_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = Path(__file__).parents[1]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "linked-rag-eval",
            "--knowledge",
            str(project / "fixtures/knowledge.json"),
            "--cases",
            str(project / "fixtures/retrieval-evaluation.json"),
            "--limit",
            "3",
        ],
    )

    assert main() == 0
    output = capsys.readouterr().out
    assert '"case_count": 10' in output
    assert '"recall_at_k"' in output
    assert '"ndcg_at_k"' in output
    assert '"passed": true' in output


def test_retrieval_evaluation_validates_inputs(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('[{"query":"","expected_document_ids":[]}]', encoding="utf-8")
    with pytest.raises(ValueError, match="query"):
        load_retrieval_cases(invalid)

    with pytest.raises(ValueError, match="at least one"):
        asyncio.run(evaluate_retrieval(FixtureRetriever(), (), limit=1))
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(
            evaluate_retrieval(
                FixtureRetriever(),
                (RetrievalCase("miss", frozenset({"expected"})),),
                limit=0,
            )
        )
