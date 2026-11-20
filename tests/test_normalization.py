from __future__ import annotations

from datetime import UTC, datetime

from linked_ai_content_pipeline.models import RawMaterial
from linked_ai_content_pipeline.normalization import canonicalize_url, normalize_and_deduplicate


def _raw(identifier: str, url: str, title: str = "AI   pipeline") -> RawMaterial:
    return RawMaterial(
        source="fixture",
        external_id=identifier,
        title=title,
        body="Нормализация\nтекста",
        url=url,
        published_at=datetime(2026, 7, 16, tzinfo=UTC),
    )


def test_canonical_url_removes_tracking_and_sorts_query() -> None:
    value = canonicalize_url("HTTPS://EXAMPLE.TEST/story/?b=2&utm_source=private&a=1#fragment")
    assert value == "https://example.test/story?a=1&b=2"


def test_deduplication_uses_url_and_normalized_content() -> None:
    materials = normalize_and_deduplicate(
        [
            _raw("one", "https://example.test/story?utm_medium=demo"),
            _raw("two", "https://example.test/story?utm_source=copy"),
            _raw("three", "https://example.test/other", title="AI pipeline"),
            _raw("four", "https://example.test/unique", title="Другая тема"),
        ]
    )
    assert [item.external_id for item in materials] == ["one", "four"]


def test_nfkc_normalization_deduplicates_unicode_compatibility_forms() -> None:
    materials = normalize_and_deduplicate(
        [
            _raw("fullwidth", "https://example.test/fullwidth", title="ＡＩ pipeline"),
            _raw("ascii", "https://example.test/ascii", title="AI pipeline"),
        ]
    )

    assert [item.external_id for item in materials] == ["fullwidth"]
    assert materials[0].title == "AI pipeline"
