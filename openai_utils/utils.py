import contextlib
import logging
import time
from collections.abc import Iterable, Iterator
from datetime import timedelta
from itertools import chain, islice
from typing import TypeVar

from tqdm import tqdm

logger = logging.getLogger(__name__)
T = TypeVar("T")


def is_required(
    o: T | None,
    name: str | None = None,
    str_empty_is_ok: bool = False,
    default: T | None = None,
    warn_default_msg: str | None = None,
) -> T:
    """
    Similar to Guava's Preconditions.checkNotNull. Check if argument was supplied.
    As a side effect makes mypy happy via return type.
    """
    msg_prefix = f"{name}" if name else "Argument"
    if o is None and default is None:
        raise ValueError(f"{msg_prefix} is required, None supplied")
    elif o is None:
        if warn_default_msg is not None:
            logger.warning(warn_default_msg)
        return default  # type:ignore[return-value]
    if isinstance(o, str):
        if not str_empty_is_ok and o == "":
            raise ValueError(f"{msg_prefix} must not be an empty string")
    return o  # type:ignore[return-value]


def flatten(i: Iterable[Iterable[T]]) -> Iterable[T]:
    """
    Flatten an iterable or iterables. This is not recursive.

    Example: [[1, 2], [1]] -> [1, 2, 1]
    """
    return chain.from_iterable(i)


@contextlib.contextmanager
def log_time(label: str, level: int = logging.INFO) -> Iterator[None]:
    """Logs execution time of the context block"""
    t_0 = time.time()
    yield
    logger.log(level, f"{label} took {timedelta(seconds=time.time() - t_0)}")


def batch(iterable: Iterable[T], size: int) -> Iterable[list[T]]:
    """
    Batching iterator over any iterable. Iterators over given iterable in a fixed size batches.
    This method is efficient for both iterables with known size and lazy generators.

    For example:
    ```
    list(batch([1, 2, 3], 2)) -> [[1, 2], [3]]
    ```
    """
    if size <= 0:
        raise ValueError("Can't iterate over batch size <= 0")
    is_required(iterable, "iterable")
    if (isinstance(iterable, tqdm) and iterable.total) or (
        not isinstance(iterable, tqdm) and hasattr(iterable, "__len__")
    ):
        # leverage len to do efficient batching
        length = len(iterable)  # type: ignore
        it = iter(iterable)
        for _i in range(0, length, size):
            yield list(islice(it, size))
    else:
        # fallback to generator based method
        yield from batch_gen(iterable, size)
        return


def batch_gen(
    generator: Iterable[T], size: int, lazy: bool = False
) -> Iterable[list[T]]:
    """
    Batching iterator over iterable with unknown size. Iterators over given iterable in a fixed size batches.

    For example:
    ```
    def foo_gen():
        for x in range(5):
            yield x

    list(batch(foo_gen(), 2)) -> [[0, 1], [2, 3], [4]]
    ```

    NOTE: default implementation is eager on the materialization of the elements, you can use lazy parameter to
    get batches of generators of elements, but keep in mind that lazy implementation is probably not thread safe, and
    batch element generator has to be exhausted before retrieving the next batch.
    """
    if size <= 0:
        raise ValueError("Can't iterate over batch size <= 0")
    is_required(generator, "generator")
    it = iter(generator)
    while True:
        batch_it = islice(it, size)
        try:
            chunk_gen = chain([next(batch_it)], batch_it)
        except StopIteration:
            return
        if lazy:
            yield chunk_gen  # type: ignore[misc]
        else:
            yield list(chunk_gen)
