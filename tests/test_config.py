import pytest
from dataclasses import FrozenInstanceError

from llm_wiki.config import Config, ConfigError
from tests.conftest import make_config


def test_from_env_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        Config.from_env()


def test_config_is_immutable():
    cfg = make_config()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        cfg.ob_host = "other"  # type: ignore[misc]


def test_default_values():
    cfg = make_config()
    assert cfg.ob_host == "127.0.0.1"
    assert cfg.ob_port == 2881
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.chunk_size == 100
    assert cfg.wiki_dir == "wiki"
    assert cfg.raw_dir == "raw"


def test_llm_api_key_falls_back_to_embedding_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = Config.from_env()
    assert cfg.llm_api_key == "sk-test"


def test_embedding_api_base_none_when_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    cfg = Config.from_env()
    assert cfg.embedding_api_base is None


def test_custom_wiki_and_raw_dirs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("WIKI_DIR", "my-wiki")
    monkeypatch.setenv("RAW_DIR", "my-raw")
    cfg = Config.from_env()
    assert cfg.wiki_dir == "my-wiki"
    assert cfg.raw_dir == "my-raw"
