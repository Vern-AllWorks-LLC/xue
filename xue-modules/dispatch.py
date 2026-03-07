"""
Multiple dispatch for Python functions.

Dispatch on the types of ALL arguments, not just self. Inspired by Julia's
dispatch system. Critical for scientific computing where operations must
specialize on operand types.

Usage:
    from xue.dispatch import multimethod

    @multimethod
    def add(a: int, b: int) -> int:
        return a + b

    @multimethod
    def add(a: float, b: float) -> float:
        return a + b

    @multimethod
    def add(a: list, b: list) -> list:
        return a + b

    add(1, 2)        # calls int, int version -> 3
    add(1.5, 2.5)    # calls float, float version -> 4.0
    add([1], [2])    # calls list, list version -> [1, 2]

    # Also supports dispatch on number of arguments:
    @multimethod
    def norm(x: float) -> float:
        return abs(x)

    @multimethod
    def norm(x: float, y: float) -> float:
        return (x**2 + y**2) ** 0.5
"""

from __future__ import annotations
import typing as _t
import inspect
import functools


class _DispatchRegistry:
    """Registry that stores and resolves multiple dispatch implementations."""

    __slots__ = ("name", "_methods", "_cache", "_doc", "_module", "_qualname")

    def __init__(self, name: str) -> None:
        self.name = name
        self._methods: list[tuple[tuple[type, ...], _t.Callable]] = []
        self._cache: dict[tuple[type, ...], _t.Callable] = {}

    def register(self, types: tuple[type, ...], func: _t.Callable) -> None:
        # Insert more specific types first (subclasses before superclasses)
        self._methods.append((types, func))
        self._cache.clear()

    def resolve(self, args: tuple) -> _t.Callable:
        arg_types = tuple(type(a) for a in args)

        # Check cache
        cached = self._cache.get(arg_types)
        if cached is not None:
            return cached

        best_func = None
        best_score = -1

        for sig_types, func in self._methods:
            if len(sig_types) != len(arg_types):
                continue

            score = 0
            match = True
            for arg_type, sig_type in zip(arg_types, sig_types):
                if arg_type is sig_type:
                    score += 2  # Exact match
                elif issubclass(arg_type, sig_type):
                    score += 1  # Subclass match
                else:
                    match = False
                    break

            if match and score > best_score:
                best_score = score
                best_func = func

        if best_func is None:
            type_names = ", ".join(t.__name__ for t in arg_types)
            raise TypeError(
                f"No matching implementation of {self.name}() "
                f"for argument types ({type_names})"
            )

        self._cache[arg_types] = best_func
        return best_func

    def __call__(self, *args, **kwargs):
        # Fast path: avoid tuple(type(a) for a in args) via direct type() calls
        n = len(args)
        if n == 1:
            key = (type(args[0]),)
        elif n == 2:
            key = (type(args[0]), type(args[1]))
        elif n == 3:
            key = (type(args[0]), type(args[1]), type(args[2]))
        else:
            key = tuple(type(a) for a in args)

        func = self._cache.get(key)
        if func is not None:
            return func(*args, **kwargs)

        func = self.resolve(args)
        return func(*args, **kwargs)

    def __repr__(self) -> str:
        n = len(self._methods)
        return f"<multimethod {self.name!r} with {n} implementation{'s' if n != 1 else ''}>"


# Global registry mapping function names to dispatch registries
_registries: dict[str, _DispatchRegistry] = {}


def _extract_types(func: _t.Callable) -> tuple[type, ...]:
    """Extract parameter types from function annotations."""
    hints = _t.get_type_hints(func)
    sig = inspect.signature(func)
    types = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        hint = hints.get(name, object)
        # Handle typing generics — fall back to origin
        origin = _t.get_origin(hint)
        if origin is not None:
            hint = origin
        if not isinstance(hint, type):
            hint = object
        types.append(hint)
    return tuple(types)


def multimethod(func: _t.Callable) -> _DispatchRegistry:
    """Decorator to create or extend a multiple-dispatch function.

    Uses type annotations to determine dispatch signature.
    Returns a callable registry that dispatches to the best-matching implementation.
    """
    qualname = func.__qualname__
    types = _extract_types(func)

    if qualname not in _registries:
        registry = _DispatchRegistry(func.__name__)
        registry._doc = func.__doc__
        registry._module = func.__module__
        registry._qualname = qualname
        _registries[qualname] = registry
    else:
        registry = _registries[qualname]

    registry.register(types, func)
    return registry


class MethodDispatch:
    """Multiple dispatch for methods within a class.

    Usage:
        class Vector:
            @method_dispatch
            def add(self, other: 'Vector') -> 'Vector':
                return Vector(self.x + other.x, self.y + other.y)

            @method_dispatch
            def add(self, scalar: float) -> 'Vector':
                return Vector(self.x + scalar, self.y + scalar)
    """

    def __init__(self, func: _t.Callable) -> None:
        self._registry = _DispatchRegistry(func.__name__)
        self._registry.__doc__ = func.__doc__
        types = _extract_types(func)
        self._registry.register(types, func)

    def register(self, func: _t.Callable) -> MethodDispatch:
        types = _extract_types(func)
        self._registry.register(types, func)
        return self

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return functools.partial(self._registry, obj)


method_dispatch = MethodDispatch
