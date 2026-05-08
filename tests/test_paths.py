from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from llm_wiki.paths import WikiPaths
from tests.conftest import make_config


def test_paths_derived_from_config(tmp_path: Path):
    cfg = make_config(wiki_dir="my-wiki", raw_dir="my-raw")
    paths = WikiPaths.from_config(cfg, root=tmp_path)
    assert paths.wiki == tmp_path / "my-wiki"
    assert paths.raw == tmp_path / "my-raw"
    assert paths.concepts == tmp_path / "my-wiki" / "concepts"
    assert paths.entities == tmp_path / "my-wiki" / "entities"
    assert paths.synthesis == tmp_path / "my-wiki" / "synthesis"
    assert paths.index_md == tmp_path / "my-wiki" / "index.md"
    assert paths.log_md == tmp_path / "my-wiki" / "log.md"


def test_ensure_exists_creates_directories(tmp_path: Path):
    cfg = make_config()
    paths = WikiPaths.from_config(cfg, root=tmp_path)
    paths.ensure_exists()
    assert paths.raw.is_dir()
    assert paths.concepts.is_dir()
    assert paths.entities.is_dir()
    assert paths.synthesis.is_dir()


def test_ensure_exists_is_idempotent(tmp_path: Path):
    cfg = make_config()
    paths = WikiPaths.from_config(cfg, root=tmp_path)
    paths.ensure_exists()
    paths.ensure_exists()
    assert paths.concepts.is_dir()


def test_paths_is_immutable(tmp_path: Path):
    cfg = make_config()
    paths = WikiPaths.from_config(cfg, root=tmp_path)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        paths.wiki = tmp_path / "other"  # type: ignore[misc]
