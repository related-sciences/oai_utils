import logging
import time

import pytest
from _pytest.logging import LogCaptureFixture

from oai_utils.utils import batch, batch_gen, flatten, is_required, log_time


def test_is_required():
    with pytest.raises(ValueError, match="Argument is required, None supplied"):
        is_required(None)
    with pytest.raises(ValueError, match="Foo is required, None supplied"):
        is_required(None, "Foo")
    with pytest.raises(ValueError, match="Argument must not be an empty string"):
        is_required("")
    with pytest.raises(ValueError, match="Bar must not be an empty string"):
        is_required("", "Bar")
    assert is_required("", str_empty_is_ok=True) == ""
    i: int | None = None
    assert is_required(i, default=44) == 44
    assert is_required(3.3) == 3.3


def test_flatten():
    assert list(flatten([[1, 2], [1]])) == [1, 2, 1]


def test_log_time(caplog: LogCaptureFixture):
    from oai_utils.utils import logger

    logger.setLevel(logging.INFO)
    try:
        with log_time("testing foobar"):
            time.sleep(0.1)
        assert "testing foobar took 0:00:00.1" in caplog.messages[0]
    finally:
        logger.setLevel(logging.WARNING)


def dummy_gen():
    yield from range(5)


@pytest.mark.parametrize("it", [[], iter(())])
def test_batch__empty_iterable(it):
    assert list(batch(it, 2)) == []


@pytest.mark.parametrize("it", [[], dummy_gen(), None])
def test_batch__size_zero(it):
    with pytest.raises(ValueError):
        list(batch(it, 0))

    with pytest.raises(ValueError):
        list(batch(it, -1))


def test_batch__dummy():
    assert list(batch([1, 2, 3], 2)) == [[1, 2], [3]]
    assert list(batch(dummy_gen(), 2)) == [[0, 1], [2, 3], [4]]
    assert list(batch(range(10), 3)) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9],
    ]
    assert list(batch(range(3), 5)) == [[0, 1, 2]]
    assert list(batch(["foo", "bar", "baz"], 2)) == [
        ["foo", "bar"],
        ["baz"],
    ]
    assert list(batch(["foo", ["bar", "baz"], "xxx"], 2)) == [
        ["foo", ["bar", "baz"]],
        ["xxx"],
    ]


def test_batch__lazy():
    # here we get batches of generators of elements, each batch is up to the batch size limit,
    # we exhaust the generators via list
    assert list(map(list, batch_gen(dummy_gen(), 3, lazy=True))) == [[0, 1, 2], [3, 4]]
