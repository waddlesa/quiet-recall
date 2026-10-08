"""Thin stdio MCP adapter for the loopback QuietRecall service.

The adapter contains no retrieval or authorization policy. It forwards JSON
to the resident service and fails closed for memory while remaining fail-open
for the host conversation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from fastmcp import FastMCP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_URL = os.environ.get(
    "QUIET_RECALL_SERVICE_URL", "http://127.0.0.1:9878"
).rstrip("/")
TOKEN_PATH = Path(
    os.environ.get(
        "QUIET_RECALL_SERVICE_TOKEN_PATH",
        str(PROJECT_ROOT / "state" / "v2" / "service-token.txt"),
    )
)


def _request(path: str, payload: dict[str, object] | None = None) -> dict[str, Any]:
    try:
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {token}"}
        data = None
        method = "GET"
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = Request(
            f"{SERVICE_URL}{path}", data=data, headers=headers, method=method
        )
        with urlopen(request, timeout=20) as response:
            value = json.loads(response.read().decode("utf-8"))
        return value if isinstance(value, dict) else {
            "status": "error",
            "error": "invalid_service_response",
        }
    except HTTPError as exc:
        try:
            value = json.loads(exc.read().decode("utf-8"))
            if isinstance(value, dict):
                return value
        except (UnicodeError, json.JSONDecodeError):
            pass
        return {"status": "error", "error": "service_http_error", "http_status": exc.code}
    except (OSError, URLError, UnicodeError, json.JSONDecodeError, TimeoutError) as exc:
        return {
            "status": "unavailable",
            "error": "quiet_recall_service_unavailable",
            "detail": type(exc).__name__,
        }


mcp = FastMCP(
    "quiet-recall",
    instructions=(
        "Use only handles issued for the current turn. Search returns metadata; "
        "read opens one capability-scoped memory. Never guess or reuse handles."
    ),
)


@mcp.tool()
def quiet_recall_health() -> dict:
    """Check the resident service without reading memory content."""
    return _request("/health")


@mcp.tool()
def quiet_recall_search(query: str, turn_handle: str, scope: str = "ordinary") -> dict:
    """Search titles in an authorized scope for the current turn."""
    return _request(
        "/v1/search",
        {
            "query": query,
            "scope": scope,
            "origin": "explicit_search",
            "mode": "inherit",
            "turn_handle": turn_handle,
        },
    )


@mcp.tool()
def quiet_recall_read(turn_handle: str, capability_handle: str) -> dict:
    """Read one candidate through its current-turn capability handle."""
    return _request(
        "/v1/read",
        {"turn_handle": turn_handle, "capability_handle": capability_handle},
    )


@mcp.tool()
def quiet_recall_end_turn(turn_handle: str) -> dict:
    """End a memory turn and revoke all remaining handles."""
    return _request("/v1/end-turn", {"turn_handle": turn_handle})


if __name__ == "__main__":
    mcp.run(transport="stdio")
