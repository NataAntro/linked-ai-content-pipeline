from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent import DeterministicContextDraftProvider
from .draft_evaluation import DraftEvaluationPolicy, evaluate_draft
from .llm import OpenAIContextDraftProvider, external_draft_provider_from_environment
from .models import Material, ScoreBreakdown, ScoredMaterial
from .rag import KnowledgeDocument, SearchHit


async def evaluate_generation_fixture(
    fixture_path: Path,
    *,
    provider_mode: str = "local",
    runs_per_case: int = 3,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not 1 <= runs_per_case <= 10:
        raise ValueError("runs_per_case must be between 1 and 10")
    if provider_mode not in {"local", "external"}:
        raise ValueError(f"unsupported LLM evaluation provider: {provider_mode}")
    fixtures = _load_cases(fixture_path)
    provider = (
        DeterministicContextDraftProvider()
        if provider_mode == "local"
        else external_draft_provider_from_environment(environment or os.environ)
    )
    case_reports: list[dict[str, Any]] = []
    try:
        for row in fixtures:
            material = _material(row)
            context = _context(row)
            runs: list[dict[str, Any]] = []
            for run in range(1, runs_per_case + 1):
                started = time.perf_counter()
                try:
                    draft = await provider.generate(material, context)
                    evaluation = evaluate_draft(
                        draft,
                        source_text=f"{material.material.title} {material.material.body}",
                        source_url=material.material.canonical_url,
                        context=tuple(hit.document for hit in context),
                        policy=DraftEvaluationPolicy(
                            required_terms=tuple(row.get("required_terms", [])),
                        ),
                    )
                    runs.append(
                        {
                            "run": run,
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                            "evaluation": asdict(evaluation),
                        }
                    )
                except Exception as error:  # Failure becomes evaluation evidence.
                    runs.append(
                        {
                            "run": run,
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                            "evaluation": {"passed": False, "issues": [str(error)]},
                        }
                    )
            successful = sum(bool(run["evaluation"]["passed"]) for run in runs)
            latencies = sorted(int(run["latency_ms"]) for run in runs)
            case_reports.append(
                {
                    "id": row["id"],
                    "runs": runs,
                    "stability_rate": successful / runs_per_case,
                    "p95_latency_ms": latencies[max(0, round(0.95 * len(latencies)) - 1)],
                    "passed": successful / runs_per_case >= 2 / 3,
                }
            )
    finally:
        if isinstance(provider, OpenAIContextDraftProvider):
            await provider.close()
    passed_cases = sum(bool(case["passed"]) for case in case_reports)
    return {
        "evaluation_version": "linked-draft-eval-v1",
        "provider": provider_mode,
        "runs_per_case": runs_per_case,
        "case_count": len(case_reports),
        "passed_cases": passed_cases,
        "case_pass_rate": passed_cases / len(case_reports),
        "mean_stability_rate": sum(float(case["stability_rate"]) for case in case_reports)
        / len(case_reports),
        "cases": case_reports,
        "passed": passed_cases == len(case_reports),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Linked contextual LLM drafts")
    parser.add_argument("--cases", type=Path, default=Path("fixtures/generation-evaluation.json"))
    parser.add_argument(
        "--provider",
        choices=("local", "external"),
        default=os.environ.get("LINKED_LLM_EVAL_PROVIDER", "local"),
    )
    parser.add_argument("--runs", type=int, default=int(os.environ.get("LLM_EVAL_RUNS", "3")))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = asyncio.run(
        evaluate_generation_fixture(
            args.cases, provider_mode=args.provider, runs_per_case=args.runs
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def _load_cases(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("generation evaluation fixture must be a non-empty array")
    required = {"id", "title", "body", "url", "context"}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("generation evaluation case is missing required fields")
        if not isinstance(row["context"], list) or not row["context"]:
            raise ValueError("generation evaluation context must be a non-empty array")
    return rows


def _material(row: Mapping[str, Any]) -> ScoredMaterial:
    material = Material(
        source="evaluation",
        external_id=str(row["id"]),
        title=str(row["title"]),
        body=str(row["body"]),
        canonical_url=str(row["url"]),
        published_at=datetime(2026, 7, 20, tzinfo=UTC),
        content_fingerprint=str(row["id"]),
    )
    return ScoredMaterial(material, 1.0, (), ScoreBreakdown(1.0, 1.0, 0.0, 1.0))


def _context(row: Mapping[str, Any]) -> tuple[SearchHit, ...]:
    return tuple(
        SearchHit(
            KnowledgeDocument(
                id=str(item["id"]),
                text=str(item["text"]),
                source_url=str(item["source_url"]),
            ),
            score=max(0.0, 1.0 - index * 0.1),
        )
        for index, item in enumerate(row["context"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
