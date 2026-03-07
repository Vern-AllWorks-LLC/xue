"""
Strict typing mode for xue-python.

When enabled (via --strict flag or XUE_STRICT=1), enforces runtime type
checking on function calls based on type annotations.

This does NOT add compile-time checking (use mypy/pyright for that).
Instead, it adds runtime guards that catch type mismatches immediately
at function boundaries rather than letting them propagate as subtle bugs.

Usage:
    # Enable globally
    import xue.strict
    xue.strict.enable()

    # Or per-function with decorator
    from xue.strict import checked

    @checked
    def compute(x: float, y: float) -> float:
        return x + y

    compute(1.0, 2.0)    # OK
    compute("a", "b")    # TypeError: x: expected float, got str

    # Or per-module
    from xue.strict import strict_module
    strict_module(__name__)  # All annotated functions in this module are checked
"""

from __future__ import annotations
import functools
import inspect
import os
import sys
import types
import typing as _t


_enabled = os.environ.get("XUE_STRICT", "0") == "1"


def enable() -> None:
    """Enable strict type checking globally."""
    global _enabled
    _enabled = True


def disable() -> None:
    """Disable strict type checking globally."""
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


class StrictTypeError(TypeError):
    """Raised when a strict type check fails."""

    def __init__(self, param_name: str, expected: type, got: type,
                 func_name: str, kind: str = "parameter") -> None:
        self.param_name = param_name
        self.expected = expected
        self.got = got
        self.func_name = func_name
        expected_name = getattr(expected, '__name__', str(expected))
        got_name = getattr(got, '__name__', str(got))
        if kind == "return":
            msg = (f"Return type violation in {func_name}: "
                   f"expected {expected_name}, got {got_name}")
        else:
            msg = (f"Type violation in {func_name}: "
                   f"parameter {param_name!r} expected {expected_name}, got {got_name}")
        super().__init__(msg)


def _check_type(value, annotation, param_name: str, func_name: str,
                kind: str = "parameter") -> None:
    """Check if value matches the type annotation."""
    if annotation is inspect.Parameter.empty:
        return

    # Handle typing generics
    origin = _t.get_origin(annotation)

    # None type
    if annotation is type(None):
        if value is not None:
            raise StrictTypeError(param_name, type(None), type(value), func_name, kind)
        return

    # Union types (X | Y or Optional[X])
    if origin is _t.Union:
        args = _t.get_args(annotation)
        if any(_is_instance_of(value, a) for a in args):
            return
        raise StrictTypeError(param_name, annotation, type(value), func_name, kind)

    # Generic types (list[int], dict[str, int], etc.)
    if origin is not None:
        if not isinstance(value, origin):
            raise StrictTypeError(param_name, origin, type(value), func_name, kind)
        return

    # Plain types
    if isinstance(annotation, type):
        if not isinstance(value, annotation):
            raise StrictTypeError(param_name, annotation, type(value), func_name, kind)
        return

    # Skip annotations we can't check (string annotations, forward refs, etc.)


def _is_instance_of(value, annotation) -> bool:
    """Check if value is an instance of annotation, handling typing generics."""
    if annotation is type(None):
        return value is None
    origin = _t.get_origin(annotation)
    if origin is not None:
        return isinstance(value, origin)
    if isinstance(annotation, type):
        return isinstance(value, annotation)
    return True  # Can't check, assume OK


def checked(func: _t.Callable) -> _t.Callable:
    """Decorator: enforce type annotations at runtime.

    Checks parameter types on entry and return type on exit.
    Always active when applied directly (ignores global enable/disable).
    Use enable()/disable() only for strict_module()-applied checks.
    """
    hints = _t.get_type_hints(func)
    sig = inspect.signature(func)
    qualname = func.__qualname__
    return_hint = hints.pop('return', inspect.Parameter.empty)

    # Build an ordered list of (index, name, hint) for positional params
    param_checks: list[tuple[int, str, _t.Any]] = []
    param_names = list(sig.parameters.keys())
    has_defaults = False
    for i, name in enumerate(param_names):
        p = sig.parameters[name]
        if p.default is not inspect.Parameter.empty or p.kind in (p.VAR_KEYWORD, p.KEYWORD_ONLY):
            has_defaults = True
        if name in hints:
            param_checks.append((i, name, hints[name]))

    if not has_defaults:
        # Fast path: positional args only, index directly into args tuple
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for idx, param_name, hint in param_checks:
                if idx < len(args):
                    _check_type(args[idx], hint, param_name, qualname)

            result = func(*args, **kwargs)

            if return_hint is not inspect.Parameter.empty:
                _check_type(result, return_hint, "return", qualname, kind="return")

            return result
    else:
        # Slow path: use sig.bind for functions with defaults/kwargs
        checked_param_set = [(name, hints[name]) for name in param_names if name in hints]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, hint in checked_param_set:
                _check_type(bound.arguments[param_name], hint, param_name, qualname)

            result = func(*args, **kwargs)

            if return_hint is not inspect.Parameter.empty:
                _check_type(result, return_hint, "return", qualname, kind="return")

            return result

    wrapper._xue_checked = True
    return wrapper


def _checked_lazy(func: _t.Callable) -> _t.Callable:
    """Like @checked but respects global enable()/disable() toggle."""
    hints = _t.get_type_hints(func)
    sig = inspect.signature(func)
    qualname = func.__qualname__
    return_hint = hints.pop('return', inspect.Parameter.empty)

    param_names = list(sig.parameters.keys())
    has_defaults = any(
        sig.parameters[n].default is not inspect.Parameter.empty or
        sig.parameters[n].kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for n in param_names
    )

    if not has_defaults:
        param_checks = [(i, name, hints[name]) for i, name in enumerate(param_names) if name in hints]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return func(*args, **kwargs)

            for idx, param_name, hint in param_checks:
                if idx < len(args):
                    _check_type(args[idx], hint, param_name, qualname)

            result = func(*args, **kwargs)

            if return_hint is not inspect.Parameter.empty:
                _check_type(result, return_hint, "return", qualname, kind="return")

            return result
    else:
        checked_params = [(name, hints[name]) for name in param_names if name in hints]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return func(*args, **kwargs)

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, hint in checked_params:
                _check_type(bound.arguments[param_name], hint, param_name, qualname)

            result = func(*args, **kwargs)

            if return_hint is not inspect.Parameter.empty:
                _check_type(result, return_hint, "return", qualname, kind="return")

            return result

    wrapper._xue_checked = True
    return wrapper


def strict_module(module_name: str) -> None:
    """Apply type checking to all annotated functions in a module.

    These checks respect the global enable()/disable() toggle.
    Call as: strict_module(__name__)
    """
    module = sys.modules.get(module_name)
    if module is None:
        return

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, types.FunctionType) and _t.get_type_hints(obj):
            setattr(module, name, _checked_lazy(obj))
