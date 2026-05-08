from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_wiki.config import Config
from llm_wiki.paths import WikiPaths


def make_config(**overrides) -> Config:
    defaults = dict(
        ob_host="127.0.0.1",
        ob_port=2881,
        ob_user="root",
        ob_password="",
        ob_database="llm_wiki",
        embedding_model="text-embedding-3-small",
        embedding_dim=4,
        embedding_api_base=None,
        embedding_api_key="test-key",
        llm_model="gpt-4o-mini",
        llm_api_base=None,
        llm_api_key="test-key",
        chunk_size=100,
        chunk_overlap=10,
        wiki_dir="wiki",
        raw_dir="raw",
    )
    defaults.update(overrides)
    return Config(**defaults)


@pytest.fixture
def cfg() -> Config:
    return make_config()


@pytest.fixture
def tmp_wiki_paths(tmp_path: Path, cfg: Config) -> WikiPaths:
    paths = WikiPaths.from_config(cfg, root=tmp_path)
    paths.ensure_exists()
    return paths


@pytest.fixture
def fake_embedder():
    class FakeEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    return FakeEmbedder()


@pytest.fixture
def mock_db_conn():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchone = MagicMock(return_value=None)
    cursor.fetchall = MagicMock(return_value=[])
    conn.cursor = MagicMock(return_value=cursor)
    return conn
