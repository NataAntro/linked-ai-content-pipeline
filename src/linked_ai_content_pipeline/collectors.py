from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .models import RawMaterial


class Collector(Protocol):
    @property
    def name(self) -> str: ...

    async def collect(self) -> Sequence[RawMaterial]: ...


class FixtureCollector:
    """Asynchronous adapter over a local JSON dataset."""

    def __init__(self, name: str, fixture_path: Path, delay_seconds: float = 0.0) -> None:
        self._name = name
        self._fixture_path = fixture_path
        self._delay_seconds = delay_seconds

    @property
    def name(self) -> str:
        return self._name

    async def collect(self) -> Sequence[RawMaterial]:
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        payload: object = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("fixture root must be a JSON array")
        return tuple(self._parse_item(item) for item in payload)

    def _parse_item(self, item: object) -> RawMaterial:
        if not isinstance(item, Mapping):
            raise ValueError("fixture item must be a JSON object")
        return RawMaterial(
            source=self.name,
            external_id=_required_string(item, "id"),
            title=_required_string(item, "title"),
            body=_required_string(item, "body"),
            url=_required_string(item, "url"),
            published_at=datetime.fromisoformat(_required_string(item, "published_at")),
            popularity=_optional_ratio(item, "popularity"),
        )


class FailingCollector:
    """Controlled failure adapter for testing resilience."""

    def __init__(self, name: str, message: str = "collector is temporarily unavailable") -> None:
        self._name = name
        self._message = message

    @property
    def name(self) -> str:
        return self._name

    async def collect(self) -> Sequence[RawMaterial]:
        raise RuntimeError(self._message)


def _required_string(item: Mapping[object, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_ratio(item: Mapping[object, object], key: str) -> float:
    value = item.get(key, 0.0)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return ratio
