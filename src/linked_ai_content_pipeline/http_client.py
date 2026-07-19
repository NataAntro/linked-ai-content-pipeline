from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

Sleep = Callable[[float], Awaitable[None]]
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must not be negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must not be smaller than base delay")


DEFAULT_RETRY_POLICY = RetryPolicy()


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    sleep: Sleep = asyncio.sleep,
    **kwargs: Any,
) -> httpx.Response:
    """Retry only bounded, transient failures and honor numeric Retry-After."""

    for attempt in range(1, policy.max_attempts + 1):
        try:
            response = await client.request(method, url, **kwargs)
        except _RETRYABLE_TRANSPORT_ERRORS:
            if attempt == policy.max_attempts:
                raise
            await sleep(_backoff_delay(policy, attempt))
            continue
        if response.status_code not in _TRANSIENT_STATUSES or attempt == policy.max_attempts:
            return response
        await sleep(_response_delay(response, policy, attempt))
    raise AssertionError("retry loop exhausted without a response")


def _response_delay(
    response: httpx.Response,
    policy: RetryPolicy,
    attempt: int,
) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), policy.max_delay_seconds)
        except ValueError:
            pass
    return _backoff_delay(policy, attempt)


def _backoff_delay(policy: RetryPolicy, attempt: int) -> float:
    return min(
        policy.base_delay_seconds * (2.0 ** (attempt - 1)),
        policy.max_delay_seconds,
    )
