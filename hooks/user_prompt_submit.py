from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_v2.hook_state import HookModeStore
from app_v2.hook_context import latest_assistant_head
from app_v2.protocol import MODEL_RESPONSE_INSTRUCTION, PROTOCOL_VERSION


def build_hook_output(response: dict[str, object]) -> dict[str, object]:
    if response.get("status") != "ok":
        return {}
    advisory = response.get("advisory_text")
    if not isinstance(advisory, str) or not advisory.strip():
        return {}
    context = (
        f'<memory_protocol version="{PROTOCOL_VERSION}">'
        f"{MODEL_RESPONSE_INSTRUCTION}</memory_protocol>\n{advisory}"
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }


def main() -> int:
    try:
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="strict")

        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            print("{}")
            return 0

        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            session_id = None
        state_root = PROJECT_ROOT / "state" / "v2"
        state = HookModeStore(
            state_root / "hook-mode.sqlite3",
            state_root / "hook-mode.salt",
        ).resolve(session_id, prompt)
        if state.marker_only or not state.query.strip():
            print("{}")
            return 0

        base_url = os.environ.get(
            "QUIET_RECALL_URL", "http://127.0.0.1:9878"
        ).rstrip("/")
        token_path = Path(
            os.environ.get(
                "QUIET_RECALL_TOKEN_PATH",
                str(state_root / "service-token.txt"),
            )
        )
        token = token_path.read_text(encoding="utf-8").strip()
        body = json.dumps(
            {
                "query": state.query,
                "mode": state.mode or "missing",
                "working_context": latest_assistant_head(payload.get("transcript_path")),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{base_url}/v1/plan",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        timeout = float(os.environ.get("QUIET_RECALL_TIMEOUT", "2.5"))
        with urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read(64 * 1024).decode("utf-8"))
        output = build_hook_output(decoded if isinstance(decoded, dict) else {})
    except Exception as exc:
        if os.environ.get("QUIET_RECALL_DIAGNOSTICS") == "1":
            print(f"memory_v2_hook_error={type(exc).__name__}", file=sys.stderr)
        print("{}")
        return 0

    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
