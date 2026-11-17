"""
xue - Enhanced Python standard library extensions for AI/ML/Scientific/Robotics work.

Modules:
    xue.result      - Result[T,E] and Option[T] types for explicit error handling
    xue.secret      - Secret[str] type that prevents accidental exposure
    xue.contracts   - @requires/@ensures/@invariant contract decorators
    xue.units       - Physical units type system (meters, seconds, kg, etc.)
    xue.dispatch    - Multiple dispatch for functions
    xue.autodiff    - Automatic differentiation via operator overloading
    xue.tensor      - Tensor type with runtime shape validation
    xue.sandbox     - Capability-based sandboxed imports
    xue.llmhook     - Optional LLM integration for diagnostics

Usage:
    from xue.result import Ok, Err, Option, Some, Nothing
    from xue.secret import Secret
    from xue.contracts import requires, ensures, invariant
    from xue.units import meters, seconds, kg
    from xue.dispatch import multimethod
    from xue.autodiff import Variable, grad
    from xue.tensor import Tensor, float32
    from xue.sandbox import sandboxed_import
"""

import sys as _sys

__version__ = "1.2.0"
__based_on__ = "CPython %d.%d.%d" % _sys.version_info[:3]
del _sys
__license__ = "MIT"
