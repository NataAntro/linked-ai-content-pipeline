from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from linked_ai_content_pipeline.collectors import Collector, FailingCollector, FixtureCollector
from linked_ai_content_pipeline.models import Draft, ScoredMaterial
from linked_ai_content_pipeline.pipeline import ContentPipeline, PipelineSettings
from linked_ai_content_pipeline.providers import DeterministicDraftProvider
from linked_ai_content_pipeline.scoring import ScoringConfig
from linked_ai_content_pipeline.sinks import JsonDraftSink


class ExplodingSink:
    def write(self, drafts: Sequence[Draft]) -> None:
        raise AssertionError("dry-run must not write drafts")


class PartiallyFailingProvider:
    def __init__(self) -> None:
        self._delegate = DeterministicDraftProvider()

    async def generate(self, material: ScoredMaterial) -> Draft:
        if material.material.external_id == "broken":
            raise ConnectionError("temporary provider failure")
        return await self._delegate.generate(material)


def test_pipeline_generates_preview_and_keeps_dry_run_side_effect_free(tmp_path: Path) -> None:
    fixture = tmp_path / "materials.json"
    fixture.write_text(
        """[
          {
            "id": "safe-1",
            "title": "AI reliability",
            "body": "Article body",
            "url": "https://example.test/safe",
            "published_at": "2026-07-17T09:00:00+00:00"
          }
        ]""",
        encoding="utf-8",
    )
    collectors: list[Collector] = [
        FixtureCollector("fixture", fixture),
        FailingCollector("broken"),
    ]
    pipeline = ContentPipeline(
        collectors=collectors,
        provider=DeterministicDraftProvider(),
        scoring=ScoringConfig({"fixture": 1.0}, {"AI": 1.0}, freshness_window_hours=24),
        settings=PipelineSettings(minimum_score=0.5, top_n=1, dry_run=True),
        sink=ExplodingSink(),
    )
    result = asyncio.run(pipeline.run(now=datetime(2026, 7, 17, 12, tzinfo=UTC)))
    assert result.collected_count == 1
    assert result.unique_count == 1
    assert len(result.drafts) == 1
    assert len(result.selection_audit) == 1
    assert result.persisted is False
    assert [failure.source for failure in result.collection_failures] == ["broken"]
    assert [entry.stage for entry in result.editorial_audit] == [
        "clarity_editor",
        "anti_slop_gate",
        "similarity_gate",
    ]
    assert result.editorial_rejections == ()


def test_explicit_non_dry_run_writes_local_json(tmp_path: Path) -> None:
    fixture = tmp_path / "one.json"
    fixture.write_text(
        """[{"id":"one","title":"AI","body":"Article","url":"https://example.test/one",
        "published_at":"2026-07-17T09:00:00+00:00"}]""",
        encoding="utf-8",
    )
    output = tmp_path / "drafts.json"
    pipeline = ContentPipeline(
        collectors=[FixtureCollector("fixture", fixture)],
        provider=DeterministicDraftProvider(),
        scoring=ScoringConfig({"fixture": 1.0}, {"AI": 1.0}, freshness_window_hours=24),
        settings=PipelineSettings(minimum_score=0.5, top_n=1, dry_run=False),
        sink=JsonDraftSink(output),
    )

    result = asyncio.run(pipeline.run(now=datetime(2026, 7, 17, 12, tzinfo=UTC)))

    assert result.persisted is True
    assert '"material_id": "fixture:one"' in output.read_text(encoding="utf-8")


def test_draft_provider_failure_is_isolated_to_one_selected_material(tmp_path: Path) -> None:
    fixture = tmp_path / "two.json"
    fixture.write_text(
        """[
          {"id":"good","title":"AI agents","body":"Reliable agent tracing",
           "url":"https://example.test/good","published_at":"2026-07-17T10:00:00+00:00"},
          {"id":"broken","title":"AI retrieval","body":"Vector retrieval quality",
           "url":"https://example.test/broken","published_at":"2026-07-17T09:00:00+00:00"}
        ]""",
        encoding="utf-8",
    )
    pipeline = ContentPipeline(
        collectors=[FixtureCollector("fixture", fixture)],
        provider=PartiallyFailingProvider(),
        scoring=ScoringConfig({"fixture": 1.0}, {"AI": 1.0}, freshness_window_hours=24),
        settings=PipelineSettings(minimum_score=0.5, top_n=2, dry_run=True),
    )

    result = asyncio.run(pipeline.run(now=datetime(2026, 7, 17, 12, tzinfo=UTC)))

    assert [draft.material_id for draft in result.drafts] == ["fixture:good"]
    assert len(result.draft_failures) == 1
    assert result.draft_failures[0].material_id == "fixture:broken"
    assert result.draft_failures[0].stage == "generate"
    assert result.draft_failures[0].error_type == "ConnectionError"
