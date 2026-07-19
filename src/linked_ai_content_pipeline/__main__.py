from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path

from .collectors import Collector, FailingCollector, FixtureCollector
from .config import load_config
from .pipeline import ContentPipeline
from .providers import DeterministicDraftProvider
from .sinks import JsonDraftSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linked content pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.toml"))
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/materials.json"))
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
    return parser


async def run_pipeline(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    collectors: list[Collector] = [FixtureCollector("tech_feed", args.fixture)]
    if not args.without_failure_probe:
        collectors.append(FailingCollector("unstable_feed"))
    settings = replace(config.pipeline, dry_run=args.write_drafts is None)
    sink = JsonDraftSink(args.write_drafts) if args.write_drafts else None
    result = await ContentPipeline(
        collectors=collectors,
        provider=DeterministicDraftProvider(),
        scoring=config.scoring,
        settings=settings,
        sink=sink,
    ).run()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


def main() -> int:
    return asyncio.run(run_pipeline(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
