"""
Optional LLM integration for enhanced Python diagnostics.

Activated with: python --llm  or  XUE_LLM_HOOK=1

Provides:
- Smart error diagnostics with fix suggestions
- Enhanced interactive help()
- Natural language contract translation

The LLM hook communicates via Unix socket or HTTP API. It does NOT
run in the interpreter's hot path and adds zero overhead when disabled.

Usage:
    from xue.llmhook import explain, ask, configure

    # Configure the LLM backend
    configure(
        backend="http",
        url="http://localhost:11434/api/generate",  # Ollama
        model="llama3",
    )
    # Or:
    configure(backend="unix", socket_path="/run/llmd/llmd.sock")
    # Or:
    configure(backend="openai", api_key="sk-...", model="gpt-4o")

    # Explain an error
    try:
        result = some_function()
    except Exception as e:
        explanation = explain(e)
        print(explanation)

    # Ask a question about the current environment
    answer = ask("How do I reshape tensor A to match tensor B?",
                 context={"A": tensor_a, "B": tensor_b})

    # Install as the default exception hook
    from xue.llmhook import install_exception_hook
    install_exception_hook()
    # Now all unhandled exceptions get LLM-powered explanations
"""

from __future__ import annotations
import sys
import os
import json
import traceback
import typing as _t


# --- Configuration ---

class LLMConfig:
    """Configuration for the LLM backend connection."""

    __slots__ = ("backend", "url", "socket_path", "api_key", "model",
                 "timeout", "max_tokens", "temperature", "enabled")

    def __init__(self) -> None:
        self.backend: str = "none"
        self.url: str = ""
        self.socket_path: str = ""
        self.api_key: str = ""
        self.model: str = ""
        self.timeout: float = 10.0
        self.max_tokens: int = 1024
        self.temperature: float = 0.3
        self.enabled: bool = os.environ.get("XUE_LLM_HOOK", "0") == "1"


_config = LLMConfig()


def configure(
    backend: str = "http",
    url: str = "",
    socket_path: str = "",
    api_key: str = "",
    model: str = "",
    timeout: float = 10.0,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> None:
    """Configure the LLM backend.

    Backends:
        "http"   - Generic HTTP API (Ollama, vLLM, etc.)
        "openai" - OpenAI-compatible API
        "unix"   - Unix domain socket (llmd daemon)
        "none"   - Disabled
    """
    _config.backend = backend
    _config.url = url
    _config.socket_path = socket_path
    _config.api_key = api_key
    _config.model = model
    _config.timeout = timeout
    _config.max_tokens = max_tokens
    _config.temperature = temperature
    _config.enabled = backend != "none"


def is_enabled() -> bool:
    return _config.enabled


# --- LLM Communication ---

def _query_llm(prompt: str, system: str = "") -> str:
    """Send a query to the configured LLM backend."""
    if not _config.enabled:
        return "(LLM hook not enabled. Set XUE_LLM_HOOK=1 or call xue.llmhook.configure())"

    if _config.backend == "http":
        return _query_http(prompt, system)
    elif _config.backend == "openai":
        return _query_openai(prompt, system)
    elif _config.backend == "unix":
        return _query_unix(prompt, system)
    else:
        return f"(Unknown LLM backend: {_config.backend!r})"


def _query_http(prompt: str, system: str) -> str:
    """Query an HTTP-based LLM API (Ollama, vLLM, etc.)."""
    import urllib.request
    import urllib.error

    payload = {
        "model": _config.model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": _config.temperature,
            "num_predict": _config.max_tokens,
        },
    }

    req = urllib.request.Request(
        _config.url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_config.timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", data.get("text", str(data)))
    except (urllib.error.URLError, TimeoutError) as e:
        return f"(LLM query failed: {e})"


def _query_openai(prompt: str, system: str) -> str:
    """Query an OpenAI-compatible API."""
    import urllib.request
    import urllib.error

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": _config.model,
        "messages": messages,
        "max_tokens": _config.max_tokens,
        "temperature": _config.temperature,
    }

    req = urllib.request.Request(
        _config.url or "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_config.api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_config.timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except (urllib.error.URLError, TimeoutError, KeyError) as e:
        return f"(LLM query failed: {e})"


def _query_unix(prompt: str, system: str) -> str:
    """Query via Unix domain socket (llmd daemon)."""
    import socket

    payload = json.dumps({
        "prompt": prompt,
        "system": system,
        "max_tokens": _config.max_tokens,
        "temperature": _config.temperature,
    }).encode()

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_config.timeout)
        sock.connect(_config.socket_path)
        sock.sendall(len(payload).to_bytes(4, "big") + payload)

        # Read response length then response
        length_bytes = sock.recv(4)
        length = int.from_bytes(length_bytes, "big")
        chunks = []
        received = 0
        while received < length:
            chunk = sock.recv(min(4096, length - received))
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
        sock.close()

        response = json.loads(b"".join(chunks))
        return response.get("text", str(response))
    except (OSError, TimeoutError, json.JSONDecodeError) as e:
        return f"(LLM query failed: {e})"


# --- Public API ---

_EXPLAIN_SYSTEM = """You are a Python debugging assistant integrated into the xue-python interpreter.
Given an exception traceback and context, provide:
1. A clear explanation of what went wrong
2. The likely root cause
3. A concrete fix with code example
Be concise and precise. Focus on actionable fixes."""


def explain(
    exception: BaseException | None = None,
    include_locals: bool = False,
) -> str:
    """Get an LLM-powered explanation of an exception.

    If no exception is provided, uses the current exception (sys.exc_info()).
    """
    if exception is None:
        exc_info = sys.exc_info()
        if exc_info[1] is None:
            return "(No active exception to explain)"
        exception = exc_info[1]

    # Format the traceback
    tb_lines = traceback.format_exception(type(exception), exception,
                                          exception.__traceback__)
    tb_text = "".join(tb_lines)

    prompt = f"Explain this Python error and suggest a fix:\n\n```\n{tb_text}\n```"

    if include_locals and exception.__traceback__:
        frame = exception.__traceback__.tb_frame
        local_vars = {
            k: _safe_repr(v) for k, v in frame.f_locals.items()
            if not k.startswith("_")
        }
        if local_vars:
            prompt += f"\n\nLocal variables:\n{json.dumps(local_vars, indent=2)}"

    return _query_llm(prompt, _EXPLAIN_SYSTEM)


def ask(question: str, context: dict | None = None) -> str:
    """Ask the LLM a question about Python or the current environment."""
    prompt = question
    if context:
        ctx_str = {k: _safe_repr(v) for k, v in context.items()}
        prompt += f"\n\nContext:\n{json.dumps(ctx_str, indent=2)}"

    return _query_llm(prompt, "You are a Python expert assistant. Be concise.")


def suggest_fix(code: str, error_msg: str) -> str:
    """Given code and an error message, suggest a fix."""
    prompt = (
        f"This Python code:\n```python\n{code}\n```\n\n"
        f"Produces this error:\n```\n{error_msg}\n```\n\n"
        f"Suggest the minimal fix."
    )
    return _query_llm(prompt, _EXPLAIN_SYSTEM)


# --- Exception hook ---

_original_excepthook = sys.excepthook


def _llm_excepthook(exc_type, exc_value, exc_tb):
    """Custom exception hook that adds LLM explanations to unhandled exceptions."""
    # Print the original traceback first
    _original_excepthook(exc_type, exc_value, exc_tb)

    if not _config.enabled:
        return

    # Then add LLM explanation
    try:
        explanation = explain(exc_value)
        print(f"\n{'=' * 60}")
        print("xue-python LLM diagnostic:")
        print(f"{'=' * 60}")
        print(explanation)
        print(f"{'=' * 60}\n")
    except Exception:
        pass  # Never let the hook itself crash the interpreter


def install_exception_hook() -> None:
    """Install the LLM-powered exception hook for unhandled exceptions."""
    sys.excepthook = _llm_excepthook


def uninstall_exception_hook() -> None:
    """Restore the original exception hook."""
    sys.excepthook = _original_excepthook


# --- Helpers ---

def _safe_repr(obj, max_len: int = 200) -> str:
    """Get a truncated repr of an object, safe for LLM context."""
    try:
        r = repr(obj)
        if len(r) > max_len:
            r = r[:max_len] + "..."
        return r
    except Exception:
        return f"<{type(obj).__name__}: repr failed>"
