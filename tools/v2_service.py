from __future__ import annotations

import argparse
import os
from pathlib import Path

from app_v2.http_service import serve


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Resident QuietRecall tool service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9878, type=int)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument(
        "--retrieval-mode", choices=("dense_bge", "hybrid_rrf"), default="hybrid_rrf"
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Phase 3 service must remain loopback-only")
    pid_file = args.pid_file.resolve() if args.pid_file else None
    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{os.getpid()}\n", encoding="ascii")
    try:
        serve(root, host=args.host, port=args.port, retrieval_mode=args.retrieval_mode)
    finally:
        if pid_file is not None:
            try:
                if pid_file.read_text(encoding="ascii").strip() == str(os.getpid()):
                    pid_file.unlink()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
