import pytest

from oai_utils import (
    OpenAIChatCompletionRequest,
    chat_completion,
    chat_completion_batch,
    embedding,
)

pytestmark = pytest.mark.openai_api_tests


def test_chat_completion__dummy():
    r = chat_completion("What is the answer to life, the universe, and everything?")
    assert r.request.messages == [
        {
            "content": "What is the answer to life, the universe, and everything?",
            "role": "user",
        }
    ]
    assert r.request.model == "gpt-4"
    assert r.request.temperature == 0.8
    assert r.request.top_p == 1.0
    assert r.request.n == 1
    assert r.request.stop is None
    assert r.request.max_tokens is None
    assert r.request.presence_penalty == 0.0
    assert r.request.frequency_penalty == 0.0
    assert r.request.logit_bias is None
    assert r.request.metadata is None

    assert r.response.model != "gpt-4"
    assert r.response.model.startswith("gpt-4")
    assert r.response.text is not None


def test_chat_completion__request():
    r = chat_completion(
        request=OpenAIChatCompletionRequest(
            messages="Respond with a random float between 0 and 1.0 (exclusive)", n=5
        )
    )
    assert len(r.response.texts or []) == 5


def test_chat_completion__batch():
    batch = list(
        chat_completion_batch(
            (
                OpenAIChatCompletionRequest(messages=f"What is 1 + {n}")
                for n in range(5)
            ),
            "Dummy test requests up to 5",
        )
    )
    assert len(batch) == 5


def test_embeddings():
    r = embedding("Foo is bar")
    assert r.embedding is not None
    assert len(r.embedding) == 1536
