from __future__ import annotations

import asyncio
from collections.abc import Sequence

from .collectors import Collector
from .models import CollectionFailure, CollectionReport, RawMaterial


async def collect_concurrently(collectors: Sequence[Collector]) -> CollectionReport:
    """Fan out collectors while preserving successful results after a partial failure."""

    outcomes = await asyncio.gather(
        *(collector.collect() for collector in collectors),
        return_exceptions=True,
    )
    materials: list[RawMaterial] = []
    failures: list[CollectionFailure] = []
    for collector, outcome in zip(collectors, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            failures.append(
                CollectionFailure(
                    source=collector.name,
                    error_type=type(outcome).__name__,
                    message=str(outcome),
                )
            )
        else:
            materials.extend(outcome)
    return CollectionReport(tuple(materials), tuple(failures))
