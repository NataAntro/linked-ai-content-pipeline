from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from .agent import (
    BoundedEditorialAgent,
    ContextualDraftProvider,
    DeterministicContextDraftProvider,
)
from .collectors import Collector, FailingCollector, FixtureCollector
from .config import load_config
from .editorial import default_editorial_chain
from .llm import OpenAIContextDraftProvider, external_draft_provider_from_environment
from .pipeline import ContentPipeline
from .providers import DeterministicDraftProvider
from .rag import (
    TextChunker,
    load_knowledge_documents,
    retrieval_runtime_from_environment,
)
from .sinks import JsonDraftSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linked content pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.toml"))
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/materials.json"))
    parser.add_argument("--knowledge", type=Path, default=Path("fixtures/knowledge.json"))
    parser.add_argument(
        "--write-drafts",
        type=Path,
        metavar="PATH",
        help="Явно отключить dry-run и записать JSON-черновики в PATH",
    )
    parser.add_argument(
        "--without-failure-probe",
        action="store_true",
        help="Не подключать controlled-failure collector",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("local", "external"),
        default=os.environ.get("LINKED_LLM_PROVIDER", "local"),
        help="Провайдер генерации контекстного черновика",
    )
    parser.add_argument(
        "--retrieval-provider",
        choices=("local", "external"),
        default=os.environ.get("LINKED_RETRIEVAL_PROVIDER", "local"),
        help="Провайдер эмбеддингов и векторного хранилища",
    )
    return parser


async def run_pipeline(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    collectors: list[Collector] = [FixtureCollector("tech_feed", args.fixture)]
    if not args.without_failure_probe:
        collectors.append(FailingCollector("unstable_feed"))
    settings = replace(config.pipeline, dry_run=args.write_drafts is None)
    sink = JsonDraftSink(args.write_drafts) if args.write_drafts else None
    retrieval = await retrieval_runtime_from_environment(
        args.retrieval_provider,
        os.environ,
    )
    external_provider: OpenAIContextDraftProvider | None = None
    try:
        documents = TextChunker().split_all(load_knowledge_documents(args.knowledge))
        await retrieval.retriever.index(documents)
        contextual_provider: ContextualDraftProvider = DeterministicContextDraftProvider()
        if args.llm_provider == "external":
            external_provider = external_draft_provider_from_environment(os.environ)
            contextual_provider = external_provider
        agent = BoundedEditorialAgent(
            retrieval.retriever,
            contextual_provider,
            default_editorial_chain(),
        )
        result = await ContentPipeline(
            collectors=collectors,
            provider=DeterministicDraftProvider(),
            scoring=config.scoring,
            settings=settings,
            sink=sink,
            agent=agent,
        ).run()
    finally:
        if external_provider is not None:
            await external_provider.close()
        await retrieval.close()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def main() -> int:
    return asyncio.run(run_pipeline(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
