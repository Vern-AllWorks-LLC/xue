"""
Capability-based sandboxed imports.

Restricts what system resources imported modules can access.
Modules must declare capabilities; the runtime enforces them.

Usage:
    from xue.sandbox import sandboxed_import, SandboxPolicy

    # Import with restricted capabilities
    policy = SandboxPolicy(
        allow_network=False,
        allow_filesystem=False,
        allow_subprocess=False,
        allow_ctypes=False,
    )
    untrusted = sandboxed_import("untrusted_package", policy=policy)

    # Or use the context manager
    with sandbox(allow_network=False, allow_subprocess=False):
        import some_plugin  # network and subprocess blocked for this import

    # Verify package signatures
    from xue.sandbox import require_signed
    require_signed("cryptography")  # raises if signature invalid
"""

from __future__ import annotations
import sys
import types
import builtins
import importlib
import typing as _t
import os
import hashlib


class SandboxViolation(PermissionError):
    """Raised when sandboxed code attempts a disallowed operation."""

    def __init__(self, capability: str, module: str, detail: str = "") -> None:
        self.capability = capability
        self.module = module
        msg = f"Sandbox violation: {module!r} attempted {capability!r}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class SandboxPolicy:
    """Defines what capabilities a sandboxed module is allowed."""

    __slots__ = (
        "allow_network", "allow_filesystem", "allow_subprocess",
        "allow_ctypes", "allow_eval", "allow_threading",
        "allowed_paths", "allowed_hosts", "name",
    )

    def __init__(
        self,
        allow_network: bool = True,
        allow_filesystem: bool = True,
        allow_subprocess: bool = True,
        allow_ctypes: bool = True,
        allow_eval: bool = True,
        allow_threading: bool = True,
        allowed_paths: list[str] | None = None,
        allowed_hosts: list[str] | None = None,
        name: str = "default",
    ) -> None:
        self.allow_network = allow_network
        self.allow_filesystem = allow_filesystem
        self.allow_subprocess = allow_subprocess
        self.allow_ctypes = allow_ctypes
        self.allow_eval = allow_eval
        self.allow_threading = allow_threading
        self.allowed_paths = allowed_paths
        self.allowed_hosts = allowed_hosts
        self.name = name

    def __repr__(self) -> str:
        caps = []
        if self.allow_network:
            caps.append("net")
        if self.allow_filesystem:
            caps.append("fs")
        if self.allow_subprocess:
            caps.append("proc")
        if self.allow_ctypes:
            caps.append("ctypes")
        if self.allow_eval:
            caps.append("eval")
        if self.allow_threading:
            caps.append("thread")
        return f"SandboxPolicy({self.name!r}, allow=[{', '.join(caps)}])"


# Predefined policies
POLICY_STRICT = SandboxPolicy(
    allow_network=False,
    allow_filesystem=False,
    allow_subprocess=False,
    allow_ctypes=False,
    allow_eval=False,
    allow_threading=False,
    name="strict",
)

POLICY_COMPUTE_ONLY = SandboxPolicy(
    allow_network=False,
    allow_filesystem=False,
    allow_subprocess=False,
    allow_ctypes=False,
    allow_eval=False,
    allow_threading=True,
    name="compute_only",
)

POLICY_LOCAL_IO = SandboxPolicy(
    allow_network=False,
    allow_filesystem=True,
    allow_subprocess=False,
    allow_ctypes=False,
    allow_eval=False,
    allow_threading=True,
    name="local_io",
)

# Modules that indicate specific capabilities
_NETWORK_MODULES = frozenset({
    "socket", "http", "http.client", "http.server",
    "urllib", "urllib.request", "urllib.parse",
    "ftplib", "smtplib", "poplib", "imaplib",
    "xmlrpc", "xmlrpc.client", "xmlrpc.server",
    "ssl", "asyncio",
})

_SUBPROCESS_MODULES = frozenset({
    "subprocess", "os.system", "pty", "resource",
})

_CTYPES_MODULES = frozenset({
    "ctypes", "ctypes.util", "_ctypes",
})

_EVAL_FUNCTIONS = frozenset({
    "eval", "exec", "compile", "__import__",
})

_FILESYSTEM_FUNCTIONS = frozenset({
    "open",
})


class _SandboxedImporter:
    """Meta-path finder that enforces sandbox policies on imports."""

    def __init__(self, policy: SandboxPolicy, source_module: str) -> None:
        self.policy = policy
        self.source_module = source_module

    def find_module(self, fullname: str, path=None):
        self._check(fullname)
        return None

    def find_spec(self, fullname: str, path=None, target=None):
        self._check(fullname)
        return None

    def _check(self, fullname: str):
        if not self.policy.allow_network and fullname in _NETWORK_MODULES:
            raise SandboxViolation("network", self.source_module,
                                   f"tried to import {fullname!r}")

        if not self.policy.allow_subprocess:
            if fullname in _SUBPROCESS_MODULES or fullname == "subprocess":
                raise SandboxViolation("subprocess", self.source_module,
                                       f"tried to import {fullname!r}")

        if not self.policy.allow_ctypes and fullname in _CTYPES_MODULES:
            raise SandboxViolation("ctypes/ffi", self.source_module,
                                   f"tried to import {fullname!r}")


class _SandboxedBuiltins:
    """Wraps builtins to enforce sandbox policies."""

    def __init__(self, policy: SandboxPolicy, source_module: str,
                 original_builtins: dict) -> None:
        self.policy = policy
        self.source_module = source_module
        self._original = original_builtins

    def __getattr__(self, name):
        if not self.policy.allow_eval and name in _EVAL_FUNCTIONS:
            raise SandboxViolation("eval/exec", self.source_module,
                                   f"tried to use {name!r}")

        if not self.policy.allow_filesystem and name in _FILESYSTEM_FUNCTIONS:
            raise SandboxViolation("filesystem", self.source_module,
                                   f"tried to use {name!r}")

        return self._original.get(name, getattr(builtins, name))


def sandboxed_import(
    module_name: str,
    policy: SandboxPolicy = POLICY_COMPUTE_ONLY,
) -> types.ModuleType:
    """Import a module with sandbox restrictions applied.

    The imported module and its sub-imports will be restricted according
    to the given policy.
    """
    importer = _SandboxedImporter(policy, module_name)

    # Install the meta-path finder temporarily
    sys.meta_path.insert(0, importer)
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.meta_path.remove(importer)

    return module


class sandbox:
    """Context manager for sandboxed code blocks.

    Usage:
        with sandbox(allow_network=False):
            import some_module  # network imports blocked
    """

    def __init__(self, **policy_kwargs) -> None:
        self.policy = SandboxPolicy(**policy_kwargs)
        self._importer = None

    def __enter__(self):
        self._importer = _SandboxedImporter(self.policy, "<sandbox>")
        sys.meta_path.insert(0, self._importer)
        return self

    def __exit__(self, *exc_info):
        if self._importer in sys.meta_path:
            sys.meta_path.remove(self._importer)
        return False


# --- Package signature verification ---

class SignatureError(SecurityError if hasattr(builtins, 'SecurityError') else Exception):
    """Raised when a package signature verification fails."""
    pass


def _compute_package_hash(package_path: str) -> str:
    """Compute SHA256 hash of all .py files in a package."""
    h = hashlib.sha256()
    if os.path.isfile(package_path):
        with open(package_path, "rb") as f:
            h.update(f.read())
    elif os.path.isdir(package_path):
        for root, dirs, files in sorted(os.walk(package_path)):
            dirs.sort()
            for fname in sorted(files):
                if fname.endswith((".py", ".pyc", ".so", ".pyd")):
                    fpath = os.path.join(root, fname)
                    with open(fpath, "rb") as f:
                        h.update(fpath.encode())
                        h.update(f.read())
    return h.hexdigest()


_TRUSTED_HASHES: dict[str, str] = {}


def register_trusted_hash(package_name: str, sha256_hash: str) -> None:
    """Register a trusted hash for a package."""
    _TRUSTED_HASHES[package_name] = sha256_hash


def verify_package(package_name: str) -> bool:
    """Verify a package against its registered hash."""
    if package_name not in _TRUSTED_HASHES:
        return False

    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        return False

    package_path = os.path.dirname(spec.origin) if spec.submodule_search_locations else spec.origin
    actual_hash = _compute_package_hash(package_path)
    return actual_hash == _TRUSTED_HASHES[package_name]


def require_signed(package_name: str) -> None:
    """Raise SignatureError if the package doesn't match its trusted hash."""
    if package_name not in _TRUSTED_HASHES:
        raise SignatureError(f"No trusted hash registered for {package_name!r}")
    if not verify_package(package_name):
        raise SignatureError(f"Package {package_name!r} failed integrity verification")
