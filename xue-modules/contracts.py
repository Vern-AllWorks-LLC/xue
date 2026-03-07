"""
Contract programming decorators: @requires, @ensures, @invariant.

Design-by-contract for Python — critical for scientific code where
preconditions and postconditions must hold.

Contracts are checked at runtime by default. Disable with:
    XUE_CONTRACTS=0  (env var) or  xue.contracts.set_enabled(False)

Usage:
    from xue.contracts import requires, ensures, invariant

    @requires(lambda a, b: b != 0, "divisor must not be zero")
    @ensures(lambda result, a, b: result * b == a, "result * b must equal a")
    def divide(a: float, b: float) -> float:
        return a / b

    @invariant(lambda self: self.balance >= 0, "balance must be non-negative")
    class BankAccount:
        def __init__(self, balance: float):
            self.balance = balance

        @requires(lambda self, amount: amount > 0, "amount must be positive")
        @ensures(lambda result, self, amount: self.balance >= amount,
                 "insufficient funds")
        def withdraw(self, amount: float) -> float:
            self.balance -= amount
            return amount
"""

from __future__ import annotations
import functools
import inspect
import os
import typing as _t

_enabled = os.environ.get("XUE_CONTRACTS", "1") != "0"


def set_enabled(enabled: bool) -> None:
    """Enable or disable contract checking globally."""
    global _enabled
    _enabled = enabled


def is_enabled() -> bool:
    """Return whether contract checking is currently enabled."""
    return _enabled


class ContractViolation(AssertionError):
    """Raised when a contract (precondition, postcondition, or invariant) is violated."""

    def __init__(self, kind: str, message: str, func_name: str) -> None:
        self.kind = kind
        self.func_name = func_name
        super().__init__(f"{kind} violation in {func_name}: {message}")


def _has_defaults_or_kwargs(func: _t.Callable) -> bool:
    """Check if func has defaults or keyword-only params (needs sig.bind)."""
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.default is not inspect.Parameter.empty:
            return True
        if p.kind in (p.VAR_KEYWORD, p.KEYWORD_ONLY):
            return True
    return False


def requires(
    predicate: _t.Callable[..., bool],
    message: str = "precondition failed",
) -> _t.Callable:
    """Decorator: check a precondition before function execution.

    The predicate receives the same arguments as the decorated function.
    """
    def decorator(func: _t.Callable) -> _t.Callable:
        qualname = func.__qualname__
        needs_bind = _has_defaults_or_kwargs(func)
        sig = inspect.signature(func) if needs_bind else None

        if needs_bind:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if _enabled:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    if not predicate(*bound.args, **bound.kwargs):
                        raise ContractViolation("Precondition", message, qualname)
                return func(*args, **kwargs)
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if _enabled:
                    if not predicate(*args, **kwargs):
                        raise ContractViolation("Precondition", message, qualname)
                return func(*args, **kwargs)

        if not hasattr(wrapper, "_xue_contracts"):
            wrapper._xue_contracts = []
        wrapper._xue_contracts.append(("requires", predicate, message))
        return wrapper
    return decorator


def ensures(
    predicate: _t.Callable[..., bool],
    message: str = "postcondition failed",
) -> _t.Callable:
    """Decorator: check a postcondition after function execution.

    The predicate receives (result, *original_args, **original_kwargs).
    """
    def decorator(func: _t.Callable) -> _t.Callable:
        qualname = func.__qualname__
        needs_bind = _has_defaults_or_kwargs(func)
        sig = inspect.signature(func) if needs_bind else None

        if needs_bind:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if _enabled:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    if not predicate(result, *bound.args, **bound.kwargs):
                        raise ContractViolation("Postcondition", message, qualname)
                return result
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if _enabled:
                    if not predicate(result, *args, **kwargs):
                        raise ContractViolation("Postcondition", message, qualname)
                return result

        if not hasattr(wrapper, "_xue_contracts"):
            wrapper._xue_contracts = []
        wrapper._xue_contracts.append(("ensures", predicate, message))
        return wrapper
    return decorator


def invariant(
    predicate: _t.Callable[[_t.Any], bool],
    message: str = "invariant violated",
) -> _t.Callable:
    """Class decorator: check an invariant after __init__ and every public method.

    The predicate receives (self,) and must return True for the invariant to hold.
    """
    def decorator(cls: type) -> type:
        def _check_invariant(instance: _t.Any) -> None:
            if _enabled and not predicate(instance):
                raise ContractViolation("Invariant", message, cls.__qualname__)

        # Wrap __init__
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _check_invariant(self)

        cls.__init__ = new_init

        # Wrap all public methods
        for name in list(vars(cls)):
            if name.startswith("_"):
                continue
            method = getattr(cls, name)
            if not callable(method):
                continue

            @functools.wraps(method)
            def wrapped_method(self, *args, _orig=method, **kwargs):
                result = _orig(self, *args, **kwargs)
                _check_invariant(self)
                return result

            setattr(cls, name, wrapped_method)

        # Store invariant metadata
        if not hasattr(cls, "_xue_invariants"):
            cls._xue_invariants = []
        cls._xue_invariants.append((predicate, message))
        return cls

    return decorator
