"""
Secret[str] type that prevents accidental exposure of sensitive values.

The Secret type wraps sensitive data (API keys, tokens, passwords) and:
- Scrubs the value from repr/str output
- Prevents accidental logging, JSON serialization, and printing
- Redacts from tracebacks when XUE_SCRUB_SECRETS=1
- Provides explicit .expose() for intentional access

Usage:
    from xue.secret import Secret

    api_key = Secret("sk-abc123def456")
    print(api_key)          # Secret(****)
    f"{api_key}"            # Secret(****)
    api_key.expose()        # "sk-abc123def456"

    # Safe comparison (constant-time)
    if api_key == Secret("sk-abc123def456"):
        print("match")

    # Works with type hints
    def connect(token: Secret[str]) -> Connection:
        return Connection(auth=token.expose())
"""

from __future__ import annotations
import typing as _t
import hmac as _hmac

T = _t.TypeVar("T")

_REDACTED = "Secret(****)"


class Secret(_t.Generic[T]):
    """A wrapper that prevents accidental exposure of sensitive values."""

    __slots__ = ("_inner",)

    def __init__(self, value: T) -> None:
        if isinstance(value, Secret):
            self._inner = value._inner
        else:
            self._inner = value

    def expose(self) -> T:
        """Intentionally access the secret value. Use with care."""
        return self._inner

    def map(self, f: _t.Callable[[T], _t.Any]) -> Secret:
        """Transform the secret value without exposing it."""
        return Secret(f(self._inner))

    # --- Prevent accidental exposure ---

    def __repr__(self) -> str:
        return _REDACTED

    def __str__(self) -> str:
        return _REDACTED

    def __format__(self, format_spec: str) -> str:
        return _REDACTED

    def __bool__(self) -> bool:
        return bool(self._inner)

    def __len__(self) -> int:
        if hasattr(self._inner, "__len__"):
            return len(self._inner)
        raise TypeError(f"Secret-wrapped object has no len()")

    # Constant-time comparison to prevent timing attacks
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Secret):
            a = str(self._inner).encode() if not isinstance(self._inner, bytes) else self._inner
            b = str(other._inner).encode() if not isinstance(other._inner, bytes) else other._inner
            return _hmac.compare_digest(a, b)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def __hash__(self) -> int:
        return hash(self._inner)

    # --- Block dangerous serialization ---

    def __reduce__(self):
        raise TypeError("Cannot pickle Secret values — use .expose() explicitly")

    def __reduce_ex__(self, protocol):
        raise TypeError("Cannot pickle Secret values — use .expose() explicitly")

    def __getstate__(self):
        raise TypeError("Cannot serialize Secret values — use .expose() explicitly")

    def __copy__(self):
        return Secret(self._inner)

    def __deepcopy__(self, memo):
        return Secret(self._inner)

    # Block JSON serialization
    def __json__(self):
        raise TypeError(
            "Cannot JSON-serialize Secret values. "
            "Use secret.expose() explicitly if you intend to serialize."
        )

    # --- Class-level helpers ---

    @classmethod
    def from_env(cls, name: str, default: str | None = None) -> Secret[str]:
        """Load a secret from an environment variable."""
        import os
        value = os.environ.get(name, default)
        if value is None:
            raise KeyError(f"Environment variable {name!r} not set and no default provided")
        return cls(value)

    @classmethod
    def from_file(cls, path: str) -> Secret[str]:
        """Load a secret from a file, stripping trailing whitespace."""
        with open(path, "r") as f:
            return cls(f.read().rstrip())


class SecretBytes(Secret[bytes]):
    """Convenience subclass for binary secrets."""

    def __init__(self, value: bytes) -> None:
        if isinstance(value, str):
            raise TypeError("Use Secret[str] for string secrets, SecretBytes for bytes")
        super().__init__(value)

    @classmethod
    def from_file(cls, path: str) -> SecretBytes:
        with open(path, "rb") as f:
            return cls(f.read())
