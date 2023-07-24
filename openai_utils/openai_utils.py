import copy
import logging
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from threading import Lock
from typing import Any, TypeVar, overload

import numpy as np
import openai
import tiktoken
from openai.error import RateLimitError
from tenacity import (
    DoAttempt,
    DoSleep,
    RetryCallState,
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.contrib.concurrent import thread_map

from openai_utils.result import Failure, Result, Success
from openai_utils.utils import batch, flatten, is_required, log_time

logger = logging.getLogger(__name__)

T = TypeVar("T")

OPENAI_KEY_ENV_VAR = "OPENAI_API_KEY"


def monkey_patch_openai_logging() -> None:
    """
    Monkey patch openai.util.log_info to not log rate limit errors, which are
    way too common and spammy. The log happens in the openai package, ATM:

    https://github.com/openai/openai-python/blob/b82a3f7e4c462a8a10fa445193301a3cefef9a4a/openai/api_requestor.py#L416-L423

    This could also be implemented via logging filter:

        logger = logging.getLogger("openai")
        logger.addFilter(lambda record: "rate_limit_exceeded" not in record.msg)

    `openai.util.log_info` may also log to stderr, so to be super safe, we opt for
    the monkey patching approach.
    """
    from openai import util

    old_log_info = util.log_info

    def custom_log_info(message: str, **params: Any) -> None:
        if params.get("error_code") == "rate_limit_exceeded":
            logger.debug(util.logfmt(dict(message=message, **params)))
            return
        old_log_info(message, **params)

    util.log_info = custom_log_info


monkey_patch_openai_logging()


@lru_cache(maxsize=64)
def price_per_1k_input_tokens_dollar(model: str) -> float:
    if model.startswith("gpt-4") and ("32k" not in model):
        return 0.03
    elif model.startswith("gpt-4") and ("32k" in model):
        return 0.06
    elif model.startswith("gpt-3.5-turbo") and ("16k" not in model):
        return 0.0015
    elif model.startswith("gpt-3.5-turbo") and ("16k" in model):
        return 0.003
    elif model.startswith("text-embedding-ada-002"):
        return 0.0001
    else:
        raise ValueError(f"Unknown model: {model!r}")


@lru_cache(maxsize=64)
def price_per_1k_completion_tokens_dollar(model: str) -> float:
    if model.startswith("gpt-4") and ("32k" not in model):
        return 0.06
    elif model.startswith("gpt-4") and ("32k" in model):
        return 0.12
    elif model.startswith("gpt-3.5-turbo") and ("16k" not in model):
        return 0.002
    elif model.startswith("gpt-3.5-turbo") and ("16k" in model):
        return 0.004
    elif model.startswith("text-embedding-ada-002"):
        return 0.0
    else:
        raise ValueError(f"Unknown model: {model!r}")


@lru_cache(maxsize=1)
def get_openai_api_key() -> str | None:
    """Get OpenAI API key from RS Secrets"""
    return os.environ.get(OPENAI_KEY_ENV_VAR)


@dataclass
class OpenAICompletionModelUsageStats:
    """OpenAI API usage stats"""

    requests: int = 0
    inputs_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0
    processing_ms: int = 0

    def __post_init__(self) -> None:
        self._update_lock = Lock()

    def update(
        self, result: "OpenAIResult", usage_log_level: int
    ) -> "OpenAICompletionModelUsageStats":
        with self._update_lock:
            self.requests += 1
            self.inputs_tokens += is_required(result.prompt_tokens)
            self.completion_tokens += is_required(result.completion_tokens, default=0)
            self.cost += (
                price_per_1k_input_tokens_dollar(result.model)
                * is_required(result.prompt_tokens)
                / 1000.0
            ) + (
                price_per_1k_completion_tokens_dollar(result.model)
                * is_required(result.completion_tokens, default=0)
                / 1000.0
            )
            self.processing_ms += is_required(result.processing_ms)
            logger.log(usage_log_level, result.usage_log())
            return self


global_openai_api_usage_stats = OpenAICompletionModelUsageStats()


@contextmanager
def log_openai_usage(label: str, level: int = logging.INFO) -> Iterator[None]:
    """
    This context manager logs OpenAI API usage stats. This is not thread-safe, i.e.
    if you have multiple threads (outside of this context) using the OpenAI utils,
    the stats may be wrong.

    Usage:

        with log_openai_usage("my label"):
          for p in prompts:
            chat_completion(p, ...)
    """
    with log_time(label + " OpenAI block", level):
        cur_usage = copy.copy(global_openai_api_usage_stats)
        try:
            yield
        finally:
            new_requests = global_openai_api_usage_stats.requests - cur_usage.requests
            new_inputs_tokens = (
                global_openai_api_usage_stats.inputs_tokens - cur_usage.inputs_tokens
            )
            new_completion_tokens = (
                global_openai_api_usage_stats.completion_tokens
                - cur_usage.completion_tokens
            )
            new_cost = global_openai_api_usage_stats.cost - cur_usage.cost
            new_processing_ms = (
                global_openai_api_usage_stats.processing_ms - cur_usage.processing_ms
            )
            logger.log(
                level,
                f"{label} OpenAI API usage: ${new_cost:.2f}, from "
                f"{new_requests:,} requests, "
                f"{new_inputs_tokens:,} input tokens, "
                f"{new_completion_tokens:,} completion tokens. "
                f"Total OpenAI processing time: {timedelta(milliseconds=new_processing_ms)}",
            )


@dataclass
class OpenAIResult:
    """OpenAI API response dataclass."""

    model: str
    processing_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    text: str | None  # chat completion text
    texts: list[str] | None  # chat completion texts for n > 1
    embedding: list[float] | None  # embedding
    embeddings: list[list[float]] | None  # embeddings for n > 1
    created_at: int | None  # timestamp
    response_id: str | None  # OpenAI API response ID
    original_responses: list[dict[str, Any]] | None  # original OpenAI API response

    # TODO (rav): The API has a couple of interesting field that may be worth capturing, including:
    #   x-request-id: 66991f967efb7b4eae50bbad51690989
    #   x-ratelimit-limit-requests: 200
    #   x-ratelimit-limit-tokens: 40000
    #   x-ratelimit-remaining-requests: 199
    #   x-ratelimit-remaining-tokens: 39981
    #   x-ratelimit-reset-requests: 300ms
    #   x-ratelimit-reset-tokens: 28ms

    @classmethod
    def from_api_response(cls, response: dict[str, Any]) -> "OpenAIResult":
        usage = defaultdict(lambda: None, response.get("usage", {}))

        text: str | None = None
        texts: list[str] | None = None
        emb: list[float] | None = None
        embs: list[list[float]] | None = None

        if "choices" in response:
            # completion response
            n = len(response["choices"])
            if n == 1:
                text = response["choices"][0]["message"]["content"]
                texts = [is_required(text)]
            else:
                text = None
                texts = [choice["message"]["content"] for choice in response["choices"]]
        else:
            # embedding response
            data = response["data"]

            if len(data) == 1:
                emb = data[0]["embedding"]
                embs = [is_required(emb)]
            else:
                emb = None
                embs = [d["embedding"] for d in data]
        return cls(
            model=response.get("model", "unknown"),
            # NOTE: the original name in the API header is OpenAI-Processing-Ms
            #       this gets renamed to response_ms in the python library, but
            #       processing_ms is much more intuitive.
            #       See: https://github.com/openai/openai-python/blob/b82a3f7e4c462a8a10fa445193301a3cefef9a4a/openai/openai_response.py#L29-L31
            processing_ms=getattr(response, "response_ms", None),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            text=text,
            texts=texts,
            embedding=emb,
            embeddings=embs,
            created_at=response.get("created", None),
            response_id=response.get("id", None),
            original_responses=[response],
        )

    def usage_log(self) -> str:
        return (
            f"Model: {self.model}, "
            f"processing (ms): {self.processing_ms}, "
            f"prompt tokens: {self.prompt_tokens}, "
            f"completion tokens: {self.completion_tokens}, "
            f"total tokens: {self.total_tokens}"
        )


class RetryingOpenAI(Retrying):  # type: ignore[misc]
    """
    Retrying wrapper for OpenAI API calls. Will retry rate limit errors ad infinitum.
    """

    def __call__(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        self.begin()

        retry_state = RetryCallState(retry_object=self, fn=fn, args=args, kwargs=kwargs)
        retry_rate_limit_state = RetryCallState(
            retry_object=self, fn=fn, args=args, kwargs=kwargs
        )
        # NOTE: retry_rate_limit_state is only used for rate limit retries, and basically just
        #       affects the exponential backoff in the sleep. This is working a bit out of
        #       the intended use of the Retrying class, but it's the easiest way to get the
        #       desired behavior without rewriting a bunch of stuff.
        while True:
            do = self.iter(retry_state=retry_state)
            if isinstance(do, DoAttempt):
                try:
                    result = fn(*args, **kwargs)
                except RateLimitError:
                    # NOTE: prepare_for_next_attempt increments the attempt number, which
                    #       in turn affects the exponential backoff in the sleep.
                    retry_rate_limit_state.prepare_for_next_attempt()
                    self.sleep(self.wait(retry_rate_limit_state))
                    continue
                except BaseException:  # noqa: B902
                    retry_state.set_exception(sys.exc_info())
                else:
                    retry_state.set_result(result)
            elif isinstance(do, DoSleep):
                retry_state.prepare_for_next_attempt()
                self.sleep(do)
            else:
                return do  # type: ignore[no-any-return]


def _openai_api_retry(
    fn: Callable[[], T], max_attempts: int, max_backoff_secs: float
) -> T:
    return RetryingOpenAI(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(min=1, max=max_backoff_secs),
        retry=retry_if_not_exception_type(openai.InvalidRequestError),
        reraise=True,
    )(fn)


def get_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    """
    Creates a list of messages in the chat format.

    See: https://platform.openai.com/docs/api-reference/chat/create#chat/create-messages
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


@dataclass
class OpenAIRequest:
    ...


@dataclass
class OpenAIChatCompletionRequest(OpenAIRequest):
    """
    Args:
        messages:
            The messages to generate chat completions for, in the chat format.
            See: https://platform.openai.com/docs/guides/chat/introduction)
        model:
            The model to use, atm: gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
            See: https://platform.openai.com/docs/models/model-endpoint-compatibility
        temperature:
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering
            this or top_p but not both.
        top_p:
            An alternative to sampling with temperature, called nucleus sampling, where the model considers the results
            of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability
            mass are considered. We generally recommend altering this or temperature but not both.
        n:
            How many chat completion choices to generate for each input message.
        stop:
            Up to 4 sequences where the API will stop generating further tokens.
        max_tokens:
            The maximum number of tokens to generate.
        presence_penalty:
            Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so
            far, increasing the model's likelihood to talk about new topics.
            See: https://platform.openai.com/docs/api-reference/parameter-details
        frequency_penalty:
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the
            text so far, decreasing the model's likelihood to repeat the same line verbatim.
            See: https://platform.openai.com/docs/api-reference/parameter-details
        logit_bias:
            Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps
            tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
            Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect
            will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values
            like -100 or 100 should result in a ban or exclusive selection of the relevant token.
        metadata:
            Arbitrary metadata to carry along with the request object, will not be sent to the API.
    """

    messages: list[dict[str, str]] | str
    model: str = "gpt-4"
    temperature: float = 0.8
    top_p: float = 1.0
    n: int = 1
    stop: list[str] | str | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    metadata: dict[str, str] | None = None


@dataclass
class OpenAIChatCompletionRoundTrip:
    """
    OpenAI chat completion round trip request and response.
    """

    request: OpenAIChatCompletionRequest
    response: OpenAIResult


@dataclass
class OpenAIChatCompletionRoundTripFailSafe(OpenAIChatCompletionRoundTrip):
    """
    Much like OpenAIChatCompletionRoundTrip, but with "optional" response.
    Response is of type `Result`, which may be `Success` or `Failure`.
    The use case decide what to do with the `Success` or `Failure` objects.
    """

    response: Result[OpenAIChatCompletionRequest, OpenAIResult]  # type: ignore[assignment]


@overload
def chat_completion(
    *,
    request: OpenAIChatCompletionRequest,
    usage_log_level: int = logging.DEBUG,
    retry_max_attempts: int = 10,
    retry_max_backoff_secs: int = 120,
) -> OpenAIChatCompletionRoundTrip:
    ...


@overload
def chat_completion(
    messages: list[dict[str, str]] | str,
    *,
    model: str = "gpt-4",
    temperature: float = 0.8,
    usage_log_level: int = logging.DEBUG,
    retry_max_attempts: int = 10,
    retry_max_backoff_secs: int = 120,
) -> OpenAIChatCompletionRoundTrip:
    ...


def chat_completion(
    messages: list[dict[str, str]] | str | None = None,
    *,
    request: OpenAIChatCompletionRequest | None = None,
    model: str = "gpt-4",
    temperature: float = 0.8,
    usage_log_level: int = logging.DEBUG,
    retry_max_attempts: int = 10,
    retry_max_backoff_secs: int = 120,
) -> OpenAIChatCompletionRoundTrip:
    """
    Creates a completion for the chat message.

    NOTE: this API doesn't support streaming mode.

    Args:
        request:
            The request object representing the chat completion request.
        messages:
            The messages to generate chat completions for, in the chat format.
            See: https://platform.openai.com/docs/guides/chat/introduction)
        model:
            The model to use, atm: gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
            See: https://platform.openai.com/docs/models/model-endpoint-compatibility
        temperature:
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering
            this or top_p but not both.
        usage_log_level:
            The log level to use for logging the API usage stats (default: logging.DEBUG).
        retry_max_attempts:
            The maximum number of attempts to retry the API call.
        retry_max_backoff_secs:
            The maximum number of seconds to backoff between retries.
    """
    openai.api_key = get_openai_api_key()
    if messages is not None and isinstance(messages, str):
        messages = get_messages(messages)
    if request is None:
        request = OpenAIChatCompletionRequest(
            messages=is_required(messages), model=model, temperature=temperature
        )
    else:
        if isinstance(request.messages, str):
            request.messages = get_messages(request.messages)

    # NOTE: due to lambda mypy can't infer that request is not None
    request_required = is_required(request)

    response = _openai_api_retry(
        lambda: openai.ChatCompletion.create(
            model=request_required.model,
            messages=request_required.messages,
            temperature=request_required.temperature,
            top_p=request_required.top_p,
            n=request_required.n,
            stop=request_required.stop,
            max_tokens=request_required.max_tokens,
            presence_penalty=request_required.presence_penalty,
            frequency_penalty=request_required.frequency_penalty,
            logit_bias=request_required.logit_bias or {},
        ),
        max_attempts=retry_max_attempts,
        max_backoff_secs=retry_max_backoff_secs,
    )
    result = OpenAIResult.from_api_response(response)
    global_openai_api_usage_stats.update(result, usage_log_level)
    return OpenAIChatCompletionRoundTrip(request_required, result)


def chat_completion_batch(
    completion_requests: Iterable[OpenAIChatCompletionRequest], label: str
) -> Iterator[OpenAIChatCompletionRoundTripFailSafe]:
    """
    Given an iterator of completion requests, returns an iterator of failsafe
    completion round trips. Will print final API usage stats, `label` is used
    as a prefix for the stats (human-readable label for this completion batch).
    """

    def failsafe_completion_fn(
        request: OpenAIChatCompletionRequest,
    ) -> OpenAIChatCompletionRoundTripFailSafe:
        try:
            return OpenAIChatCompletionRoundTripFailSafe(
                request, Success(chat_completion(request=request).response)
            )
        except Exception as e:
            return OpenAIChatCompletionRoundTripFailSafe(request, Failure(request, e))

    with log_openai_usage(label):
        # TODO: make this actually lazy iterable, ATM thread_map is eager
        yield from thread_map(failsafe_completion_fn, completion_requests)


def embedding(
    x: str | list[str],
    model: str = "text-embedding-ada-002",
    encoding_name: str = "cl100k_base",
    max_context_length: int = 8191,
    usage_log_level: int = logging.DEBUG,
    retry_max_attempts: int = 10,
    retry_max_backoff_secs: int = 120,
) -> OpenAIResult:
    """
    Creates an embedding vector representing the input text.

    Args:
        x:
            The text to embed.Input text to get embeddings for, encoded as a string or array of tokens. To get
            embeddings for multiple inputs in a single request, pass an array of strings or array of token arrays. Each
            input must not exceed 8192 tokens in length.
        model:
            The model to use.
        encoding_name:
            The encoding to use, this is model specific.
        max_context_length:
            The maximum number of tokens to embed, max is model specific.
        usage_log_level:
            The log level to use for logging the API usage stats (default: logging.DEBUG).
        retry_max_attempts:
            The maximum number of attempts to retry the API call.
        retry_max_backoff_secs:
            The maximum number of seconds to backoff between retries.
    """
    openai.api_key = get_openai_api_key()

    if isinstance(x, str):
        x = [x]

    def get_embedding(
        chunk: str | list[str] | list[int] | list[list[int]], model: str
    ) -> OpenAIResult:
        if isinstance(chunk, list):
            assert len(chunk) <= 2048, "The batch size should not be larger than 2048."
        return OpenAIResult.from_api_response(
            _openai_api_retry(
                lambda: openai.Embedding.create(model=model, input=chunk),
                max_attempts=retry_max_attempts,
                max_backoff_secs=retry_max_backoff_secs,
            )
        )

    encoder = tiktoken.get_encoding(encoding_name)
    x_encoded = [((e := encoder.encode(s)), len(e)) for s in x]
    num_of_inputs = len(x_encoded)
    contains_long_elements = (
        len([e for e in x_encoded if e[1] > max_context_length]) > 0
    )

    if contains_long_elements and num_of_inputs > 1:
        raise ValueError(
            f"The batch contains elements with length above context limit of {max_context_length}. "
            "This is currently unsupported. Strings longer than the max context can currently be "
            "embedded by calling this function on each string individually."
        )

    if not contains_long_elements:
        result = get_embedding([i[0] for i in x_encoded], model=model)
    else:
        assert num_of_inputs == 1
        x_tokens = x_encoded[0][0]
        chunk_results: list[OpenAIResult] = []
        chunk_embs: list[list[float]] = []
        chunk_lens: list[int] = []
        for chunk in batch(x_tokens, max_context_length):
            i = get_embedding(chunk, model=model)
            chunk_results.append(i)
            chunk_embs.append(is_required(i.embedding))
            chunk_lens.append(len(chunk))

        result_arr: Any = np.average(np.asarray(chunk_embs), axis=0, weights=chunk_lens)
        result_arr = result_arr / np.linalg.norm(result_arr)  # normalizes length to 1
        result_emb = result_arr.tolist()
        result = OpenAIResult(
            model=model,
            processing_ms=sum(is_required(i.processing_ms) for i in chunk_results),
            prompt_tokens=sum(is_required(r.prompt_tokens) for r in chunk_results),
            completion_tokens=None,
            total_tokens=sum(is_required(r.total_tokens) for r in chunk_results),
            text=None,
            texts=None,
            embedding=result_emb,
            embeddings=[result_emb],
            created_at=max(
                is_required(r.created_at, default=int(time.time()))
                for r in chunk_results
            ),
            response_id=None,
            original_responses=list(
                flatten(
                    [  # type: ignore [arg-type]
                        is_required(r.original_responses, default=[])
                        for r in chunk_results
                    ]
                )
            ),
        )

    global_openai_api_usage_stats.update(result, usage_log_level)
    return result
