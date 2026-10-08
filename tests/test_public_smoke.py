from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

from app.config import load_project_config, validate_config
from app.embedding_backend import HashEmbeddingBackend, KeywordReranker
from app.indexer import MemoryIndexer
from app_v2.candidates import CleanCandidateEngine, ScopeCatalog
from app_v2.http_service import MemoryHTTPServer
from app_v2.service import MemoryToolService
from app_v2.source_segments import CanonicalSegmentStore
from app_v2.turn_policy import is_explicit_recall_request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PublicSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        test_temp_root = REPOSITORY_ROOT / ".tmp-tests"
        test_temp_root.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="quiet-recall-public-", dir=test_temp_root
        )
        cls.root = Path(cls.temporary.name)
        shutil.copytree(REPOSITORY_ROOT / "config", cls.root / "config")
        shutil.copytree(REPOSITORY_ROOT / "examples", cls.root / "examples")

        cls.config = load_project_config(cls.root)
        errors = validate_config(cls.config)
        if errors:
            raise AssertionError("\n".join(errors))

        cls.database_path = cls.root / "state" / "memory-index.sqlite3"
        embedder = HashEmbeddingBackend(64)
        reranker = KeywordReranker()
        MemoryIndexer(
            cls.config, embedder, database_path=cls.database_path
        ).sync(dry_run=False)
        cls.segments = CanonicalSegmentStore.ensure_current(
            cls.config, index_path=cls.database_path
        )
        cls.scopes = ScopeCatalog.load(cls.root / "config" / "v2" / "scopes.yaml")
        cls.engine = CleanCandidateEngine(
            cls.config,
            cls.scopes,
            embedder,
            reranker,
            database_path=cls.database_path,
            segment_store=cls.segments,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def service(self) -> MemoryToolService:
        return MemoryToolService(
            engine=self.engine,
            scopes=self.scopes,
            memory_root=self.config.roots["memory"],
            retrieval_mode="hybrid_rrf",
        )

    def test_example_configuration_is_complete(self) -> None:
        self.assertEqual(validate_config(self.config), [])

    def test_english_explicit_recall_is_detected(self) -> None:
        self.assertTrue(is_explicit_recall_request("Do you remember the blue door?"))
        self.assertFalse(is_explicit_recall_request("That blue door looks nice."))

    def test_ordinary_memory_can_be_discovered_and_read(self) -> None:
        service = self.service()
        plan = service.plan_turn(
            query="Do you remember the bookshop with the blue door?", mode="daily"
        )
        self.assertEqual(plan["status"], "ok")
        matches = [
            item
            for item in plan["directory"]
            if item["document_title"] == "The bookshop with the blue door"
        ]
        self.assertTrue(matches)
        result = service.read(
            turn_handle=str(plan["turn_handle"]),
            capability_handle=str(matches[0]["capability_handle"]),
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("blue door", str(result["content"]))

    def test_casual_health_mention_does_not_open_vault(self) -> None:
        plan = self.service().plan_turn(
            query="I read an interesting paper about allergies.", mode="daily"
        )
        self.assertIsNone(plan["routed_scope"])
        self.assertFalse(
            any(item["namespace"] == "private.health" for item in plan["directory"])
        )

    def test_explicit_health_recollection_routes_vault(self) -> None:
        plan = self.service().plan_turn(
            query="I told you about my allergy history. Do you remember?",
            mode="daily",
        )
        self.assertEqual(plan["routed_scope"], "private.health")
        self.assertTrue(
            any(
                item["document_title"] == "Fictional allergy history"
                for item in plan["directory"]
            )
        )

    def test_read_capability_is_single_use(self) -> None:
        service = self.service()
        plan = service.plan_turn(
            query="Do you remember the blue-door bookshop?", mode="daily"
        )
        handle = str(plan["directory"][0]["capability_handle"])
        first = service.read(turn_handle=str(plan["turn_handle"]), capability_handle=handle)
        second = service.read(turn_handle=str(plan["turn_handle"]), capability_handle=handle)
        self.assertEqual(first["status"], "ok")
        self.assertNotEqual(second["status"], "ok")

    def test_loopback_http_health(self) -> None:
        token = "public-smoke-token"
        server = MemoryHTTPServer(("127.0.0.1", 0), self.service(), token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            request = Request(
                f"http://{host}:{port}/health",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["service"], "quiet-recall")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
