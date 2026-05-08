import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    ob_host: str
    ob_port: int
    ob_user: str
    ob_password: str
    ob_database: str
    embedding_model: str
    embedding_dim: int
    embedding_api_base: str | None
    embedding_api_key: str
    llm_model: str
    llm_api_base: str | None
    llm_api_key: str
    chunk_size: int
    chunk_overlap: int
    wiki_dir: str
    raw_dir: str

    @classmethod
    def from_env(cls) -> "Config":
        embedding_api_key = os.getenv("OPENAI_API_KEY")
        if not embedding_api_key:
            raise ConfigError("OPENAI_API_KEY is required but not set")

        llm_api_key = os.getenv("LLM_API_KEY") or embedding_api_key

        return cls(
            ob_host=os.getenv("OB_HOST", "127.0.0.1"),
            ob_port=int(os.getenv("OB_PORT", "2881")),
            ob_user=os.getenv("OB_USER", "root"),
            ob_password=os.getenv("OB_PASSWORD", ""),
            ob_database=os.getenv("OB_DATABASE", "llm_wiki"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
            embedding_api_base=os.getenv("OPENAI_API_BASE") or None,
            embedding_api_key=embedding_api_key,
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_api_base=os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE") or None,
            llm_api_key=llm_api_key,
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            wiki_dir=os.getenv("WIKI_DIR", "wiki"),
            raw_dir=os.getenv("RAW_DIR", "raw"),
        )
