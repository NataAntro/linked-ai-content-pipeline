from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from linked_ai_content_pipeline.collection import collect_concurrently
from linked_ai_content_pipeline.collectors import Collector, FailingCollector
from linked_ai_content_pipeline.models import RawMaterial


class SuccessfulCollector:
    name = "working"

    async def collect(self) -> tuple[RawMaterial, ...]:
        await asyncio.sleep(0)
        return (
            RawMaterial(
                source=self.name,
                external_id="one",
                title="AI infrastructure update",
                body="Safe fixture",
                url="https://example.test/one",
                published_at=datetime(2026, 7, 16, tzinfo=UTC),
            ),
        )


class CoordinatedCollector:
    def __init__(self, name: str, started: list[str], ready: asyncio.Event) -> None:
        self.name = name
        self._started = started
        self._ready = ready

    async def collect(self) -> tuple[RawMaterial, ...]:
        self._started.append(self.name)
        if len(self._started) == 2:
            self._ready.set()
        await asyncio.wait_for(self._ready.wait(), timeout=0.1)
        return ()


def test_failure_of_one_collector_does_not_cancel_others() -> None:
    collectors: list[Collector] = [SuccessfulCollector(), FailingCollector("broken", "expected")]
    report = asyncio.run(collect_concurrently(collectors))

    assert [item.external_id for item in report.materials] == ["one"]
    assert len(report.failures) == 1
    assert report.failures[0].source == "broken"
    assert report.failures[0].error_type == "RuntimeError"


def test_collectors_are_started_concurrently() -> None:
    async def scenario() -> None:
        started: list[str] = []
        ready = asyncio.Event()
        collectors: list[Collector] = [
            CoordinatedCollector("one", started, ready),
            CoordinatedCollector("two", started, ready),
        ]
        report = await collect_concurrently(collectors)
        assert report.failures == ()
        assert started == ["one", "two"]

    asyncio.run(scenario())
