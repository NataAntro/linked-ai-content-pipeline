from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from linked_ai_content_pipeline.__main__ import build_parser, main


def test_parser_defaults_to_safe_dry_run() -> None:
    args = build_parser().parse_args([])

    assert args.config == Path("configs/pipeline.toml")
    assert args.fixture == Path("fixtures/materials.json")
    assert args.write_drafts is None
    assert args.without_failure_probe is False


def test_main_runs_fixture_pipeline_without_external_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "pipeline.toml"
    config.write_text(
        """
[pipeline]
minimum_score = 0.0
top_n = 1

[scoring]
freshness_window_hours = 8760.0

[scoring.source_weights]
tech_feed = 1.0

[scoring.keyword_weights]
""".strip(),
        encoding="utf-8",
    )
    fixture = tmp_path / "materials.json"
    fixture.write_text(
        """[{"id":"one","title":"AI reliability","body":"Body",
        "url":"https://example.test/one","published_at":"2026-07-19T10:00:00+00:00"}]""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "linked-pipeline",
            "--config",
            str(config),
            "--fixture",
            str(fixture),
            "--without-failure-probe",
        ],
    )

    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["collected_count"] == 1
    assert result["persisted"] is False
    assert result["collection_failures"] == []
