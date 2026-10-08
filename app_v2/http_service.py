from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
from pathlib import Path
import secrets
from typing import Any

from .service import MemoryToolService


MAX_BODY_BYTES = 65536


def load_or_create_token(path: Path) -> str:
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(token + "\n", encoding="utf-8")
    temporary.replace(path)
    return token


class MemoryHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: MemoryToolService, token: str):
        super().__init__(address, MemoryRequestHandler)
        self.memory_service = service
        self.service_token = token


class MemoryRequestHandler(BaseHTTPRequestHandler):
    server: MemoryHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.service_token}"
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, expected)

    def _payload(self) -> dict[str, object] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        if not self._authorized():
            self._json(401, {"status": "error", "error": "unauthorized"})
            return
        if self.path == "/health":
            self._json(200, self.server.memory_service.health())
            return
        self._json(404, {"status": "error", "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        if not self._authorized():
            self._json(401, {"status": "error", "error": "unauthorized"})
            return
        payload = self._payload()
        if payload is None:
            self._json(400, {"status": "error", "error": "invalid_json"})
            return
        try:
            if self.path == "/v1/plan":
                result = self.server.memory_service.plan_turn(**payload)
            elif self.path == "/v1/search":
                result = self.server.memory_service.search(**payload)
            elif self.path == "/v1/read":
                result = self.server.memory_service.read(**payload)
            elif self.path == "/v1/end-turn":
                result = self.server.memory_service.end_turn(**payload)
            else:
                self._json(404, {"status": "error", "error": "not_found"})
                return
        except TypeError:
            self._json(400, {"status": "error", "error": "invalid_arguments"})
            return
        except Exception:
            self._json(500, {"status": "error", "error": "internal_error"})
            return
        self._json(200, result)


def serve(
    project_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 9878,
    retrieval_mode: str = "dense_bge",
) -> None:
    service = MemoryToolService.from_project(project_root, retrieval_mode=retrieval_mode)
    token = load_or_create_token(project_root / "state" / "v2" / "service-token.txt")
    server = MemoryHTTPServer((host, port), service, token)
    server.serve_forever(poll_interval=0.5)
