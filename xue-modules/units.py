"""
Physical units type system for scientific and robotics code.

Prevents unit mismatch errors at runtime. Supports arithmetic composition,
conversion, and dimensionality checking.

Usage:
    from xue.units import meters, seconds, kg, newtons, Quantity

    distance = 100 * meters
    time = 9.58 * seconds
    speed = distance / time          # Quantity(10.44..., m/s)

    mass = 75 * kg
    force = mass * (speed / time)    # Quantity(..., kg*m/s^2)

    # Unit mismatch is caught:
    distance + time  # raises DimensionError

    # Conversion:
    distance_km = distance.to(kilometers)  # 0.1
"""

from __future__ import annotations
import typing as _t
import math

class Dimension:
    """Represents physical dimensions as exponents of base dimensions.

    Base dimensions (SI): length, mass, time, current, temperature, amount, luminosity
    Stored as a tuple of 7 exponents.
    """

    __slots__ = ("_exponents", "_name")

    # Indices: L, M, T, I, Θ, N, J
    _LABELS = ("m", "kg", "s", "A", "K", "mol", "cd")

    # Cache for dimension objects — avoids re-creating identical dimensions
    _cache: dict[tuple[int, ...], Dimension] = {}

    def __new__(cls, exponents: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0),
                name: str | None = None) -> Dimension:
        key = tuple(exponents)
        cached = cls._cache.get(key)
        if cached is not None and name is None:
            return cached
        obj = super().__new__(cls)
        obj._exponents = key
        obj._name = name
        if name is None:
            cls._cache[key] = obj
        return obj

    def __init__(self, exponents: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0),
                 name: str | None = None) -> None:
        # Already set in __new__
        pass

    @property
    def is_dimensionless(self) -> bool:
        return all(e == 0 for e in self._exponents)

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a + b for a, b in zip(self._exponents, other._exponents)))

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a - b for a, b in zip(self._exponents, other._exponents)))

    def __pow__(self, n: int) -> Dimension:
        return Dimension(tuple(e * n for e in self._exponents))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dimension):
            return self._exponents == other._exponents
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._exponents)

    def __repr__(self) -> str:
        if self._name:
            return self._name
        parts = []
        for label, exp in zip(self._LABELS, self._exponents):
            if exp == 1:
                parts.append(label)
            elif exp != 0:
                parts.append(f"{label}^{exp}")
        return "*".join(parts) if parts else "dimensionless"


class DimensionError(TypeError):
    """Raised when operations are attempted on incompatible dimensions."""
    pass


_unit_cache: dict[tuple, Unit] = {}


class Unit:
    """A physical unit with a dimension and a scale factor relative to SI base."""

    __slots__ = ("dimension", "scale", "name", "symbol")

    def __init__(self, dimension: Dimension, scale: float = 1.0,
                 name: str = "", symbol: str = "") -> None:
        self.dimension = dimension
        self.scale = scale
        self.name = name
        self.symbol = symbol or name

    @staticmethod
    def _cached(dimension: Dimension, scale: float, symbol: str) -> Unit:
        key = (dimension._exponents, round(scale, 10))
        cached = _unit_cache.get(key)
        if cached is not None:
            return cached
        u = Unit(dimension, scale, symbol)
        _unit_cache[key] = u
        return u

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Unit._cached(
                self.dimension * other.dimension,
                self.scale * other.scale,
                f"{self.symbol}*{other.symbol}",
            )
        if isinstance(other, (int, float)):
            return Quantity(other, self)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(other, self)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Unit):
            return Unit._cached(
                self.dimension / other.dimension,
                self.scale / other.scale,
                f"{self.symbol}/{other.symbol}",
            )
        return NotImplemented

    def __pow__(self, n: int) -> Unit:
        return Unit._cached(self.dimension ** n, self.scale ** n, f"{self.symbol}^{n}")

    def __repr__(self) -> str:
        return self.symbol or repr(self.dimension)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Unit):
            return self.dimension == other.dimension and math.isclose(self.scale, other.scale)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.dimension, round(self.scale, 10)))


class Quantity:
    """A numeric value with an associated physical unit."""

    __slots__ = ("value", "unit")

    def __init__(self, value: float, unit: Unit) -> None:
        self.value = value if type(value) is float else float(value)
        self.unit = unit

    def _check_compatible(self, other: Quantity) -> None:
        if self.unit.dimension != other.unit.dimension:
            raise DimensionError(
                f"Cannot combine {self.unit.dimension!r} with {other.unit.dimension!r}"
            )

    def to(self, target_unit: Unit) -> float:
        """Convert to a different unit of the same dimension."""
        if self.unit.dimension != target_unit.dimension:
            raise DimensionError(
                f"Cannot convert {self.unit.dimension!r} to {target_unit.dimension!r}"
            )
        return self.value * (self.unit.scale / target_unit.scale)

    def __add__(self, other):
        if isinstance(other, Quantity):
            self._check_compatible(other)
            # Convert other to self's unit scale
            converted = other.value * (other.unit.scale / self.unit.scale)
            return Quantity(self.value + converted, self.unit)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Quantity):
            self._check_compatible(other)
            converted = other.value * (other.unit.scale / self.unit.scale)
            return Quantity(self.value - converted, self.unit)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                self.value * other.value,
                self.unit * other.unit,
            )
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(other * self.value, self.unit)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                self.value / other.value,
                self.unit / other.unit,
            )
        if isinstance(other, (int, float)):
            return Quantity(self.value / other, self.unit)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            inv_dim = Dimension(tuple(-e for e in self.unit.dimension._exponents))
            inv_unit = Unit(inv_dim, 1.0 / self.unit.scale, f"1/{self.unit.symbol}")
            return Quantity(other / self.value, inv_unit)
        return NotImplemented

    def __pow__(self, n: int):
        return Quantity(self.value ** n, self.unit ** n)

    def __neg__(self):
        return Quantity(-self.value, self.unit)

    def __abs__(self):
        return Quantity(abs(self.value), self.unit)

    def __lt__(self, other):
        if isinstance(other, Quantity):
            self._check_compatible(other)
            return self.value * self.unit.scale < other.value * other.unit.scale
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Quantity):
            self._check_compatible(other)
            return self.value * self.unit.scale <= other.value * other.unit.scale
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Quantity):
            self._check_compatible(other)
            return self.value * self.unit.scale > other.value * other.unit.scale
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Quantity):
            self._check_compatible(other)
            return self.value * self.unit.scale >= other.value * other.unit.scale
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Quantity):
            if self.unit.dimension != other.unit.dimension:
                return False
            return math.isclose(
                self.value * self.unit.scale,
                other.value * other.unit.scale,
            )
        return NotImplemented

    def __repr__(self) -> str:
        return f"Quantity({self.value}, {self.unit})"

    def __format__(self, spec: str) -> str:
        if spec:
            return f"{self.value:{spec}} {self.unit}"
        return f"{self.value} {self.unit}"


# --- Base SI dimensions ---
_L = Dimension((1, 0, 0, 0, 0, 0, 0), "length")
_M = Dimension((0, 1, 0, 0, 0, 0, 0), "mass")
_T = Dimension((0, 0, 1, 0, 0, 0, 0), "time")
_I = Dimension((0, 0, 0, 1, 0, 0, 0), "current")
_Θ = Dimension((0, 0, 0, 0, 1, 0, 0), "temperature")
_N = Dimension((0, 0, 0, 0, 0, 1, 0), "amount")
_J = Dimension((0, 0, 0, 0, 0, 0, 1), "luminosity")
_DIMLESS = Dimension()

# --- Base SI units ---
meters = Unit(_L, 1.0, "meter", "m")
kilograms = Unit(_M, 1.0, "kilogram", "kg")
seconds = Unit(_T, 1.0, "second", "s")
amperes = Unit(_I, 1.0, "ampere", "A")
kelvin = Unit(_Θ, 1.0, "kelvin", "K")
moles = Unit(_N, 1.0, "mole", "mol")
candelas = Unit(_J, 1.0, "candela", "cd")

# --- Convenience aliases ---
m = meters
kg = kilograms
s = seconds

# --- Derived SI units ---
hertz = Unit(_T ** -1, 1.0, "hertz", "Hz")
newtons = Unit(_L * _M * (_T ** -2), 1.0, "newton", "N")
pascals = Unit((_L ** -1) * _M * (_T ** -2), 1.0, "pascal", "Pa")
joules = Unit(_L ** 2 * _M * (_T ** -2), 1.0, "joule", "J")
watts = Unit(_L ** 2 * _M * (_T ** -3), 1.0, "watt", "W")
volts = Unit(_L ** 2 * _M * (_T ** -3) * (_I ** -1), 1.0, "volt", "V")

# --- Common scaled units ---
kilometers = Unit(_L, 1000.0, "kilometer", "km")
centimeters = Unit(_L, 0.01, "centimeter", "cm")
millimeters = Unit(_L, 0.001, "millimeter", "mm")
grams = Unit(_M, 0.001, "gram", "g")
milliseconds = Unit(_T, 0.001, "millisecond", "ms")
microseconds = Unit(_T, 1e-6, "microsecond", "us")
nanoseconds = Unit(_T, 1e-9, "nanosecond", "ns")
minutes = Unit(_T, 60.0, "minute", "min")
hours = Unit(_T, 3600.0, "hour", "h")

# --- Robotics-specific ---
radians = Unit(_DIMLESS, 1.0, "radian", "rad")
degrees = Unit(_DIMLESS, math.pi / 180.0, "degree", "deg")
rpm = Unit(_T ** -1, 2 * math.pi / 60.0, "RPM", "rpm")

# --- Force / torque ---
newton_meters = Unit(_L ** 2 * _M * (_T ** -2), 1.0, "newton-meter", "N*m")

# ── C-accelerated override ────────────────────────────────────────
# Replace Python classes with C implementations for ~50x speedup.
# All module-level constants are recreated with C types.
try:
    from ._units_accel import (
        Dimension as Dimension,
        DimensionError as DimensionError,
        Unit as Unit,
        Quantity as Quantity,
    )
    # Rebuild base dimensions with C types
    _L = Dimension((1, 0, 0, 0, 0, 0, 0), "length")
    _M = Dimension((0, 1, 0, 0, 0, 0, 0), "mass")
    _T = Dimension((0, 0, 1, 0, 0, 0, 0), "time")
    _I = Dimension((0, 0, 0, 1, 0, 0, 0), "current")
    _THT = Dimension((0, 0, 0, 0, 1, 0, 0), "temperature")
    _N = Dimension((0, 0, 0, 0, 0, 1, 0), "amount")
    _J = Dimension((0, 0, 0, 0, 0, 0, 1), "luminosity")
    _DIMLESS = Dimension()

    meters = Unit(_L, 1.0, "meter", "m")
    kilograms = Unit(_M, 1.0, "kilogram", "kg")
    seconds = Unit(_T, 1.0, "second", "s")
    amperes = Unit(_I, 1.0, "ampere", "A")
    kelvin = Unit(_THT, 1.0, "kelvin", "K")
    moles = Unit(_N, 1.0, "mole", "mol")
    candelas = Unit(_J, 1.0, "candela", "cd")
    m = meters
    kg = kilograms
    s = seconds
    hertz = Unit(_T ** -1, 1.0, "hertz", "Hz")
    newtons = Unit(_L * _M * (_T ** -2), 1.0, "newton", "N")
    pascals = Unit((_L ** -1) * _M * (_T ** -2), 1.0, "pascal", "Pa")
    joules = Unit(_L ** 2 * _M * (_T ** -2), 1.0, "joule", "J")
    watts = Unit(_L ** 2 * _M * (_T ** -3), 1.0, "watt", "W")
    volts = Unit(_L ** 2 * _M * (_T ** -3) * (_I ** -1), 1.0, "volt", "V")
    kilometers = Unit(_L, 1000.0, "kilometer", "km")
    centimeters = Unit(_L, 0.01, "centimeter", "cm")
    millimeters = Unit(_L, 0.001, "millimeter", "mm")
    grams = Unit(_M, 0.001, "gram", "g")
    milliseconds = Unit(_T, 0.001, "millisecond", "ms")
    microseconds = Unit(_T, 1e-6, "microsecond", "us")
    nanoseconds = Unit(_T, 1e-9, "nanosecond", "ns")
    minutes = Unit(_T, 60.0, "minute", "min")
    hours = Unit(_T, 3600.0, "hour", "h")
    radians = Unit(_DIMLESS, 1.0, "radian", "rad")
    degrees = Unit(_DIMLESS, math.pi / 180.0, "degree", "deg")
    rpm = Unit(_T ** -1, 2 * math.pi / 60.0, "RPM", "rpm")
    newton_meters = Unit(_L ** 2 * _M * (_T ** -2), 1.0, "newton-meter", "N*m")
except ImportError:
    pass
