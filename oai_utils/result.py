from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class Either(Generic[T, U], Iterable[U]):
    """Represents a value of one of two possible types (a disjoint union)"""

    _value: T | U

    @property
    def value(self) -> T | U:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 1 if isinstance(self, Right) else 0

    def __bool__(self) -> bool:
        return True if isinstance(self, Right) else False

    def __iter__(self) -> Iterator[U]:
        return iter([self.value]) if isinstance(self, Right) else iter([])


class Right(Either[Any, U]):
    def __init__(self, right: U):
        self._value = right

    @property
    def value(self) -> U:
        return self._value


class Left(Either[T, Any]):
    def __init__(self, left: T):
        self._value = left

    @property
    def value(self) -> T:
        return self._value


class Result(Either[tuple[T, Exception], U]):
    ...


class Success(Right[U], Result[Any, U]):
    ...


class Failure(Left[tuple[T, Exception]], Result[T, Any]):
    def __init__(self, input: T, exception: Exception):
        self._value = (input, exception)

    @property
    def exception(self) -> Exception:
        return self.value[1]

    @property
    def input(self) -> T:
        return self.value[0]
