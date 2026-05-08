from typing import Protocol

from openai import OpenAI

from llm_wiki.config import Config


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, cfg: Config) -> None:
        self._model = cfg.embedding_model
        self._client = OpenAI(
            api_key=cfg.embedding_api_key,
            base_url=cfg.embedding_api_base,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch_size = 100
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model)
            results.extend(data.embedding for data in response.data)
        return results
