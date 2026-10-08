from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.cli import model_backends
from app.config import load_project_config
from app_v2.candidates import CleanCandidateEngine, ScopeCatalog, index_build_fingerprint
from app_v2.reader import TurnReadSession
from app_v2.source_segments import CanonicalSegmentStore


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Offline Phase-2 search/read probe")
    parser.add_argument("query")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--scope", default="ordinary")
    parser.add_argument("--mode", choices=("daily", "research", "missing"), default="daily")
    parser.add_argument(
        "--origin",
        choices=("automatic", "explicit_search"),
        default=None,
    )
    parser.add_argument(
        "--read-rank",
        type=int,
        action="append",
        default=[],
        help="1-based directory rank to read; may be passed twice",
    )
    parser.add_argument("--summary", action="store_true", help="omit returned memory text")
    args = parser.parse_args()

    config = load_project_config(args.project_root)
    scopes = ScopeCatalog.load(args.project_root / "config" / "v2" / "scopes.yaml")
    database_path = args.project_root / "state" / "memory-index.sqlite3"
    build_id = index_build_fingerprint(database_path)
    segment_store = CanonicalSegmentStore.ensure_current(
        config, index_path=database_path
    )
    mode = None if args.mode == "missing" else args.mode
    origin = args.origin or ("automatic" if args.scope == "ordinary" else "explicit_search")
    session = TurnReadSession(
        memory_root=config.roots["memory"],
        scopes=scopes,
        index_build_id=build_id,
        mode=mode,
    )
    authorization = session.authorize_search(scope=args.scope, origin=origin)
    if authorization["status"] != "ok":
        print(json.dumps(authorization, ensure_ascii=False, indent=2))
        return 0

    embedder, reranker = model_backends(config)
    engine = CleanCandidateEngine(
        config,
        scopes,
        embedder,
        reranker,
        database_path=database_path,
        segment_store=segment_store,
    )
    search = engine.search(args.query, scope=args.scope)
    issued = session.issue_directory(search["directory"], scope=args.scope, origin=origin)
    payload: dict[str, object] = {"authorization": authorization, "issued": issued, "reads": []}
    directory = list(issued.get("directory", []))
    for rank in args.read_rank:
        if rank < 1 or rank > len(directory):
            payload["reads"].append(
                {"status": "error", "error": "rank_out_of_range", "rank": rank}
            )
            continue
        handle = str(directory[rank - 1]["capability_handle"])
        result = session.read(handle)
        if args.summary and result.get("status") == "ok":
            parent_chars = len(str(result.pop("parent_context", "")))
            content_chars = len(str(result.pop("content", "")))
            result = {
                **result,
                "parent_context_chars": parent_chars,
                "content_chars": content_chars,
            }
        payload["reads"].append(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
