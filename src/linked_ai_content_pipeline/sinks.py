from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .models import Draft


class DraftSink(Protocol):
    def write(self, drafts: Sequence[Draft]) -> None: ...


class JsonDraftSink:
    """Optional local sink used only when dry-run is explicitly disabled."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def write(self, drafts: Sequence[Draft]) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps([asdict(draft) for draft in drafts], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
