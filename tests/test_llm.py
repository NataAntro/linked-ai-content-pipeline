from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime

import httpx
import pytest

from linked_ai_content_pipeline.editorial import EditorialRejection
from linked_ai_content_pipeline.llm import (
    OpenAIContextDraftProvider,
    external_draft_provider_from_environment,
)
from linked_ai_content_pipeline.models import Draft, Material, ScoreBreakdown, ScoredMaterial
from linked_ai_content_pipeline.rag import KnowledgeDocument, SearchHit


def _material() -> ScoredMaterial:
    return ScoredMaterial(
        Material(
            "fixture",
            "one",
            "Надёжные AI-агенты",
            "Наблюдаемая трасса и лимиты делают поведение системы предсказуемым.",
            "https://example.test/news",
            datetime(2026, 7, 20, tzinfo=UTC),
            "fingerprint",
        ),
        1.0,
        (),
        ScoreBreakdown(1.0, 1.0, 0.0, 1.0),
    )


def test_openai_context_provider_uses_strict_schema_and_traceable_context() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "output_text": json.dumps(
                    {
                        "headline": "Надёжность AI-агентов",
                        "body": "Лимиты шагов и трасса решений делают поведение проверяемым.",
                    },
                    ensure_ascii=False,
                )
            },
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIContextDraftProvider(
            "https://api.example.test/v1/responses",
            "secret",
            "model-test",
            client=client,
        )
        context = (
            SearchHit(
                KnowledgeDocument(
                    "doc",
                    "Проверяемый контекст",
                    "https://example.test/context",
                ),
                0.9,
            ),
        )
        generated = await provider.generate(_material(), context)
        revised = await provider.revise(
            Draft(generated.material_id, "Старый заголовок", "Старый текст черновика"),
            _material(),
            context,
            EditorialRejection("fixture:one", "anti_slop_gate", "обнаружен штамп"),
        )
        await client.aclose()

        assert generated.material_id == "fixture:one"
        assert revised.headline == "Надёжность AI-агентов"
        assert requests[0]["model"] == "model-test"
        assert requests[0]["text"]["format"]["strict"] is True  # type: ignore[index]
        assert "https://example.test/context" in str(requests[0]["input"])
        assert "anti_slop_gate" in str(requests[1]["instructions"])

    asyncio.run(scenario())


def test_openai_context_provider_rejects_malformed_payload() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"output_text": '{"headline":"коротко"}'})

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIContextDraftProvider("https://example.test", "key", "model", client=client)
        with pytest.raises(ValueError, match="headline and body"):
            await provider.generate(_material(), ())
        await client.aclose()

    asyncio.run(scenario())


def test_external_provider_configuration_requires_every_setting() -> None:
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        external_draft_provider_from_environment(
            {"LLM_ENDPOINT": "https://example.test", "LLM_MODEL": "model"}
        )


def test_external_provider_configuration_builds_owned_client() -> None:
    async def scenario() -> None:
        provider = external_draft_provider_from_environment(
            {
                "LLM_ENDPOINT": "https://example.test",
                "LLM_API_KEY": "secret",
                "LLM_MODEL": "model",
            }
        )
        await provider.close()

    asyncio.run(scenario())


def test_openai_context_provider_redacts_direct_identifiers_and_bounds_context() -> None:
    captured_input = ""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_input
        captured_input = str(json.loads(request.content)["input"])
        return httpx.Response(
            200,
            json={
                "output_text": json.dumps(
                    {"headline": "Проверяемый черновик", "body": "Достаточно длинный текст."}
                )
            },
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIContextDraftProvider("https://example.test", "key", "model", client=client)
        material = _material()
        private_material = replace(
            material,
            material=replace(
                material.material,
                body="Email person@example.test, телефон +7 (999) 123-45-67. " + "x" * 3_000,
            ),
        )
        await provider.generate(private_material, ())
        await client.aclose()

    asyncio.run(scenario())
    assert "person@example.test" not in captured_input
    assert "+7 (999)" not in captured_input
    assert "[REDACTED_EMAIL]" in captured_input
    assert "[REDACTED_PHONE]" in captured_input
    assert len(captured_input) < 3_000


@pytest.mark.parametrize(
    ("response_body", "message"),
    [
        ([], "must be an object"),
        ({"output": []}, "does not contain structured output"),
        (
            {
                "output": [
                    {
                        "content": [
                            {
                                "text": json.dumps(
                                    {"headline": "bad", "body": "достаточно длинный текст"}
                                )
                            }
                        ]
                    }
                ]
            },
            "headline has invalid length",
        ),
    ],
)
def test_openai_context_provider_validates_response_shapes(
    response_body: object,
    message: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=response_body)

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIContextDraftProvider("https://example.test", "key", "model", client=client)
        with pytest.raises(ValueError, match=message):
            await provider.generate(_material(), ())
        await provider.close()
        await client.aclose()

    asyncio.run(scenario())
