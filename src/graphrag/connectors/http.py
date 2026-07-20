"""HTTP-помощник с бэкоффом под rate limits (JIRA/Confluence REST)."""

from __future__ import annotations

import time

import httpx


def get_json_with_backoff(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    max_retries: int = 4,
    base_delay: float = 0.5,
    sleep=time.sleep,
) -> dict:
    """GET с ретраями на 429/5xx. Экспоненциальный бэкофф.

    `sleep` инъектируется для тестов (чтобы не ждать реально).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.get(url, params=params)
        except httpx.TransportError as e:
            # Транспортные сбои (ConnectError/ReadTimeout) — именно то, что бэкофф
            # призван поглощать; ретраим их наравне с 429/5xx.
            last_exc = e
            if attempt < max_retries:
                sleep(base_delay * (2**attempt))
                continue
            raise
        if resp.status_code == 429 or resp.status_code >= 500:
            last_exc = httpx.HTTPStatusError(
                f"{resp.status_code}", request=resp.request, response=resp
            )
            if attempt < max_retries:
                sleep(base_delay * (2**attempt))
                continue
            raise last_exc
        resp.raise_for_status()
        return resp.json()
    # Цикл всегда возвращает или бросает внутри; страховка на случай max_retries < 0.
    raise last_exc or RuntimeError("get_json_with_backoff: не выполнено ни одного запроса")
