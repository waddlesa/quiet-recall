from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ConfigError, load_project_config, validate_config
from .embedding_backend import (
    BGEEmbeddingBackend,
    BGEReranker,
    HashEmbeddingBackend,
    KeywordReranker,
)
from .indexer import MemoryIndexer
from .source_registry import SourceRegistry


def model_backends(config):
    """Build the configured embedding and reranking backends.

    The deterministic pair exists for smoke tests and the bundled fictional
    example. It is deliberately not presented as a production retriever.
    """

    models = config.policies.get("models", {})
    backend = str(models.get("backend", "bge")).strip().lower()
    if backend == "hash":
        dimensions = int(models.get("dimensions", 64))
        return HashEmbeddingBackend(dimensions), KeywordReranker()
    if backend != "bge":
        raise ConfigError(f"unknown model backend: {backend}")

    embedding_path = Path(str(models.get("embedding_path", ""))).expanduser()
    reranker_path = Path(str(models.get("reranker_path", ""))).expanduser()
    if not embedding_path.is_absolute():
        embedding_path = (config.project_root / embedding_path).resolve()
    if not reranker_path.is_absolute():
        reranker_path = (config.project_root / reranker_path).resolve()
    if not embedding_path.is_dir():
        raise ConfigError(f"embedding model not found: {embedding_path}")
    if not reranker_path.is_dir():
        raise ConfigError(f"reranker model not found: {reranker_path}")
    return BGEEmbeddingBackend(embedding_path), BGEReranker(reranker_path)


def _check(project_root: Path) -> int:
    try:
        config = load_project_config(project_root)
        errors = validate_config(config)
    except ConfigError as exc:
        errors = [str(exc)]
    if errors:
        print("CONFIG CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    registry = SourceRegistry(config)
    print(f"CONFIG CHECK OK: {len(registry.records)} registered files")
    return 0


def _sync(project_root: Path, dry_run: bool) -> int:
    config = load_project_config(project_root)
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"CONFIG ERROR: {error}")
        return 1
    embedder, _ = model_backends(config)
    report = MemoryIndexer(config, embedder).sync(dry_run=dry_run)
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    return 0


def _status(project_root: Path) -> int:
    config = load_project_config(project_root)
    embedder, _ = model_backends(config)
    print(
        json.dumps(
            MemoryIndexer(config, embedder).status(),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quiet-recall")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="validate configuration and source coverage")
    sync = subparsers.add_parser("sync", help="build or update the derived index")
    sync.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("status", help="show derived-index status")
    args = parser.parse_args(argv)

    if args.command == "check":
        return _check(args.project_root)
    if args.command == "sync":
        return _sync(args.project_root, args.dry_run)
    if args.command == "status":
        return _status(args.project_root)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
