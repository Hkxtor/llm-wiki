from unittest.mock import MagicMock, patch

from llm_wiki.embeddings import OpenAIEmbedder
from tests.conftest import make_config


def _make_openai_response(vectors: list[list[float]]):
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    return response


@patch("llm_wiki.embeddings.OpenAI")
def test_embed_returns_list_of_vectors(mock_openai_cls):
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    mock_openai_cls.return_value.embeddings.create.return_value = _make_openai_response(vectors)

    embedder = OpenAIEmbedder(make_config())
    result = embedder.embed(["hello", "world"])

    assert result == vectors
    assert len(result) == 2


@patch("llm_wiki.embeddings.OpenAI")
def test_embed_batches_large_input(mock_openai_cls):
    def side_effect(input, model):
        return _make_openai_response([[0.1, 0.2]] * len(input))

    mock_openai_cls.return_value.embeddings.create.side_effect = side_effect

    embedder = OpenAIEmbedder(make_config())
    texts = [f"text {i}" for i in range(250)]
    result = embedder.embed(texts)

    # 250 texts / batch_size 100 = 3 API calls
    assert mock_openai_cls.return_value.embeddings.create.call_count == 3
    assert len(result) == 250


@patch("llm_wiki.embeddings.OpenAI")
def test_embed_single_text(mock_openai_cls):
    mock_openai_cls.return_value.embeddings.create.return_value = _make_openai_response(
        [[0.5, 0.6, 0.7]]
    )

    embedder = OpenAIEmbedder(make_config())
    result = embedder.embed(["single"])

    assert len(result) == 1
    assert result[0] == [0.5, 0.6, 0.7]
