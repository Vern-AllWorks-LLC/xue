"""
Result[T, E] and Option[T] types for explicit error handling.

Inspired by Rust's Result and Option types. Eliminates silent None failures
and makes error paths visible in type signatures.

Usage:
    from xue.result import Ok, Err, Result, Option, Some, Nothing

    def divide(a: float, b: float) -> Result[float, str]:
        if b == 0:
            return Err("division by zero")
        return Ok(a / b)

    match divide(10, 0):
        case Ok(value):
            print(f"Result: {value}")
        case Err(error):
            print(f"Error: {error}")

    # Option type
    def find_user(id: int) -> Option[User]:
        user = db.get(id)
        if user is None:
            return Nothing()
        return Some(user)

    # Chaining with .and_then() / .map() / .unwrap_or()
    result = (divide(10, 3)
              .map(lambda x: x * 2)
              .unwrap_or(0.0))
"""

from __future__ import annotations
import typing as _t

T = _t.TypeVar("T")
U = _t.TypeVar("U")
E = _t.TypeVar("E")
F = _t.TypeVar("F")


class Result(_t.Generic[T, E]):
    """Base class for Result[T, E] — either Ok(value) or Err(error)."""

    __slots__ = ()
    __match_args__ = ("_value",)

    def is_ok(self) -> bool:
        return isinstance(self, Ok)

    def is_err(self) -> bool:
        return isinstance(self, Err)

    def ok(self) -> Option[T]:
        if isinstance(self, Ok):
            return Some(self._value)
        return Nothing()

    def err(self) -> Option[E]:
        if isinstance(self, Err):
            return Some(self._value)
        return Nothing()

    def map(self, f: _t.Callable[[T], U]) -> Result[U, E]:
        if isinstance(self, Ok):
            return Ok(f(self._value))
        return self  # type: ignore

    def map_err(self, f: _t.Callable[[E], F]) -> Result[T, F]:
        if isinstance(self, Err):
            return Err(f(self._value))
        return self  # type: ignore

    def and_then(self, f: _t.Callable[[T], Result[U, E]]) -> Result[U, E]:
        if isinstance(self, Ok):
            return f(self._value)
        return self  # type: ignore

    def or_else(self, f: _t.Callable[[E], Result[T, F]]) -> Result[T, F]:
        if isinstance(self, Err):
            return f(self._value)
        return self  # type: ignore

    def unwrap(self) -> T:
        if isinstance(self, Ok):
            return self._value
        raise UnwrapError(f"called unwrap() on Err: {self._value!r}")

    def unwrap_or(self, default: T) -> T:
        if isinstance(self, Ok):
            return self._value
        return default

    def unwrap_or_else(self, f: _t.Callable[[E], T]) -> T:
        if isinstance(self, Ok):
            return self._value
        return f(self._value)  # type: ignore

    def expect(self, msg: str) -> T:
        if isinstance(self, Ok):
            return self._value
        raise UnwrapError(f"{msg}: {self._value!r}")

    def __bool__(self) -> bool:
        return self.is_ok()

    def __iter__(self):
        if isinstance(self, Ok):
            yield self._value


class Ok(Result[T, _t.Any]):
    """Successful result containing a value."""

    __slots__ = ("_value",)
    __match_args__ = ("_value",)

    def __init__(self, value: T) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Ok):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("Ok", self._value))


class Err(Result[_t.Any, E]):
    """Error result containing an error value."""

    __slots__ = ("_value",)
    __match_args__ = ("_value",)

    def __init__(self, error: E) -> None:
        self._value = error

    def __repr__(self) -> str:
        return f"Err({self._value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Err):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("Err", self._value))


class UnwrapError(Exception):
    """Raised when unwrap() is called on an Err or Nothing."""
    pass


# --- Option[T] ---

class Option(_t.Generic[T]):
    """Base class for Option[T] — either Some(value) or Nothing()."""

    __slots__ = ()

    def is_some(self) -> bool:
        return isinstance(self, Some)

    def is_nothing(self) -> bool:
        return isinstance(self, Nothing)

    def map(self, f: _t.Callable[[T], U]) -> Option[U]:
        if isinstance(self, Some):
            return Some(f(self._value))
        return Nothing()

    def and_then(self, f: _t.Callable[[T], Option[U]]) -> Option[U]:
        if isinstance(self, Some):
            return f(self._value)
        return Nothing()

    def or_else(self, f: _t.Callable[[], Option[T]]) -> Option[T]:
        if isinstance(self, Some):
            return self
        return f()

    def unwrap(self) -> T:
        if isinstance(self, Some):
            return self._value
        raise UnwrapError("called unwrap() on Nothing")

    def unwrap_or(self, default: T) -> T:
        if isinstance(self, Some):
            return self._value
        return default

    def unwrap_or_else(self, f: _t.Callable[[], T]) -> T:
        if isinstance(self, Some):
            return self._value
        return f()

    def ok_or(self, error: E) -> Result[T, E]:
        if isinstance(self, Some):
            return Ok(self._value)
        return Err(error)

    def ok_or_else(self, f: _t.Callable[[], E]) -> Result[T, E]:
        if isinstance(self, Some):
            return Ok(self._value)
        return Err(f())

    def filter(self, predicate: _t.Callable[[T], bool]) -> Option[T]:
        if isinstance(self, Some) and predicate(self._value):
            return self
        return Nothing()

    def __bool__(self) -> bool:
        return self.is_some()

    def __iter__(self):
        if isinstance(self, Some):
            yield self._value

    @staticmethod
    def from_nullable(value: T | None) -> Option[T]:
        if value is None:
            return Nothing()
        return Some(value)


class Some(Option[T]):
    """Option containing a value."""

    __slots__ = ("_value",)
    __match_args__ = ("_value",)

    def __init__(self, value: T) -> None:
        if value is None:
            raise ValueError("Some() cannot contain None; use Nothing() instead")
        self._value = value

    def __repr__(self) -> str:
        return f"Some({self._value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Some):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("Some", self._value))


class _NothingSingleton(type):
    _instance: Nothing | None = None

    def __call__(cls) -> Nothing:
        if cls._instance is None:
            cls._instance = super().__call__()
        return cls._instance


class Nothing(Option[_t.Any], metaclass=_NothingSingleton):
    """Option containing no value (singleton)."""

    __slots__ = ()
    __match_args__ = ()

    def __repr__(self) -> str:
        return "Nothing()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Nothing)

    def __hash__(self) -> int:
        return hash("Nothing")


# --- Utility functions ---

def catch(
    f: _t.Callable[..., T],
    *args: _t.Any,
    catch_types: tuple[type[BaseException], ...] = (Exception,),
    **kwargs: _t.Any,
) -> Result[T, BaseException]:
    """Run a function and capture exceptions as Err results.

    Usage:
        result = catch(int, "not_a_number")
        # Err(ValueError("invalid literal ..."))
    """
    try:
        return Ok(f(*args, **kwargs))
    except catch_types as e:
        return Err(e)
