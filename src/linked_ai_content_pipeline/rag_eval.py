from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from .rag import (
    TextChunker,
    load_knowledge_documents,
    retrieval_runtime_from_environment,
)
from .retrieval_evaluation import RetrievalEvaluation, evaluate_retrieval, load_retrieval_cases


async def evaluate_fixture(
    knowledge_path: Path,
    cases_path: Path,
    *,
    limit: int = 3,
    retrieval_provider: str = "local",
    environment: Mapping[str, str] | None = None,
) -> RetrievalEvaluation:
    runtime = await retrieval_runtime_from_environment(
        retrieval_provider,
        environment if environment is not None else os.environ,
    )
    try:
        documents = TextChunker().split_all(load_knowledge_documents(knowledge_path))
        await runtime.retriever.index(documents)
        return await evaluate_retrieval(
            runtime.retriever,
            load_retrieval_cases(cases_path),
            limit=limit,
        )
    finally:
        await runtime.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Linked semantic retrieval")
    parser.add_argument("--knowledge", type=Path, default=Path("fixtures/knowledge.json"))
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("fixtures/retrieval-evaluation.json"),
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-mrr", type=float, default=0.60)
    parser.add_argument("--max-hard-negative-rate", type=float, default=0.35)
    parser.add_argument(
        "--retrieval-provider",
        choices=("local", "external"),
        default=os.environ.get("LINKED_RETRIEVAL_PROVIDER", "local"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = asyncio.run(
        evaluate_fixture(
            args.knowledge,
            args.cases,
            limit=args.limit,
            retrieval_provider=args.retrieval_provider,
        )
    )
    payload = {
        "evaluation_version": "linked-retrieval-eval-v2",
        **asdict(report),
        "thresholds": {
            "min_recall": args.min_recall,
            "min_mrr": args.min_mrr,
            "max_hard_negative_rate": args.max_hard_negative_rate,
        },
    }
    passed = (
        report.recall_at_k >= args.min_recall
        and report.mean_reciprocal_rank >= args.min_mrr
        and report.hard_negative_rate <= args.max_hard_negative_rate
    )
    print(json.dumps({**payload, "passed": passed}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
