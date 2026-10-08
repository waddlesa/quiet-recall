from __future__ import annotations

import json
from pathlib import Path

from app.config import load_project_config
from app_v2.source_segments import CanonicalSegmentStore


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = load_project_config(root)
    report = CanonicalSegmentStore.build(
        config,
        index_path=root / "state" / "memory-index.sqlite3",
        output_path=root / "state" / "v2" / "canonical-segments.sqlite3",
    )
    print(
        json.dumps(
            {
                "index_build_id": report.index_build_id,
                "source_count": report.source_count,
                "chunk_count": report.chunk_count,
                "output_path": str(report.output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
