from __future__ import annotations

import asyncio
from pathlib import Path

from linked_ai_content_pipeline.draft_evaluation import (
    DraftEvaluationPolicy,
    evaluate_draft,
)
from linked_ai_content_pipeline.llm_eval import evaluate_generation_fixture
from linked_ai_content_pipeline.models import Draft
from linked_ai_content_pipeline.rag import KnowledgeDocument


def _context() -> tuple[KnowledgeDocument, ...]:
    return (
        KnowledgeDocument(
            "one",
            "Трасса решений делает обработку проверяемой.",
            "https://example.test/context/one",
        ),
        KnowledgeDocument(
            "two",
            "Ограниченный бюджет останавливает цикл.",
            "https://example.test/context/two",
        ),
    )


def test_draft_evaluation_accepts_grounded_cited_result() -> None:
    draft = Draft(
        "evaluation:one",
        "Проверяемая обработка",
        (
            "Трасса решений делает обработку проверяемой, а ограниченный бюджет "
            "останавливает цикл.\nhttps://example.test/context/one\n"
            "https://example.test/context/two"
        ),
    )

    result = evaluate_draft(
        draft,
        source_text="Проверяемая обработка использует трассу решений и бюджет.",
        source_url="https://example.test/source",
        context=_context(),
        policy=DraftEvaluationPolicy(required_terms=("трасса", "бюджет")),
    )

    assert result.passed is True
    assert result.citation_precision == 1.0
    assert result.citation_recall == 1.0
    assert result.unsupported_numbers == ()


def test_draft_evaluation_rejects_unknown_citation_and_unsupported_claims() -> None:
    draft = Draft(
        "evaluation:one",
        "Революционный прорыв",
        "Эффективность выросла на 87%. https://unknown.test/invented",
    )

    result = evaluate_draft(
        draft,
        source_text="Система проверяет результат.",
        source_url="https://example.test/source",
        context=_context(),
        policy=DraftEvaluationPolicy(required_terms=("бюджет",)),
    )

    assert result.passed is False
    assert result.citation_precision == 0.0
    assert result.citation_recall == 0.0
    assert result.unsupported_numbers == ("87",)
    assert result.forbidden_phrases == ("революционный прорыв",)
    assert len(result.issues) >= 5


def test_generation_evaluation_runs_each_case_multiple_times() -> None:
    project = Path(__file__).parents[1]
    report = asyncio.run(
        evaluate_generation_fixture(
            project / "fixtures/generation-evaluation.json",
            runs_per_case=3,
        )
    )

    assert report["case_count"] == 6
    assert report["passed_cases"] == 6
    assert report["mean_stability_rate"] == 1.0
    assert all(len(case["runs"]) == 3 for case in report["cases"])
