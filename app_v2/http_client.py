from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


class MemoryServiceClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:9878",
        token_path: Path,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_path = token_path
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        token = self.token_path.read_text(encoding="utf-8").strip()
        return {"Authorization": f"Bearer {token}"}

    def get(self, path: str) -> dict[str, Any]:
        request = Request(f"{self.base_url}{path}", headers=self._headers())
        with urlopen(request, timeout=self.timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("memory service returned non-object JSON")
        return value

    def post(self, path: str, payload: dict[str, object]) -> dict[str, Any]:
        headers = {**self._headers(), "Content-Type": "application/json"}
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("memory service returned non-object JSON")
        return value
