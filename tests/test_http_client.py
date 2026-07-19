from __future__ import annotations

import asyncio

import httpx
import pytest

from linked_ai_content_pipeline.http_client import RetryPolicy, request_with_retry


def test_retry_policy_retries_transient_status_and_honors_retry_after() -> None:
    requests = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(429, headers={"retry-after": "0.4"})
        return httpx.Response(200, json={"status": "ok"})

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await request_with_retry(
                client,
                "GET",
                "https://api.example.test/resource",
                policy=RetryPolicy(max_attempts=3, max_delay_seconds=1.0),
                sleep=record_sleep,
            )
        assert response.status_code == 200

    asyncio.run(scenario())
    assert requests == 2
    assert delays == [0.4]


def test_retry_policy_bounds_connect_failures() -> None:
    requests = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise httpx.ConnectError("temporarily unavailable", request=request)

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(httpx.ConnectError):
                await request_with_retry(
                    client,
                    "GET",
                    "https://api.example.test/resource",
                    policy=RetryPolicy(
                        max_attempts=3,
                        base_delay_seconds=0.1,
                        max_delay_seconds=0.15,
                    ),
                    sleep=record_sleep,
                )

    asyncio.run(scenario())
    assert requests == 3
    assert delays == [0.1, 0.15]


def test_retry_policy_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="max_delay_seconds"):
        RetryPolicy(base_delay_seconds=2.0, max_delay_seconds=1.0)
