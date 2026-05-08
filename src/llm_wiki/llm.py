import time
from typing import Literal, Protocol

from openai import OpenAI

from llm_wiki.config import Config


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        *,
        response_format: Literal["text", "json"] = "text",
    ) -> str: ...


class OpenAICompatibleLLM:
    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY = 1.0

    def __init__(self, cfg: Config) -> None:
        self._model = cfg.llm_model
        self._client = OpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_api_base,
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._RETRY_BASE_DELAY * (2**attempt))

        raise RuntimeError(f"LLM call failed after {self._MAX_RETRIES} attempts") from last_exc
