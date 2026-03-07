"""
xue-python site customization.

This module is imported at interpreter startup when xue-python is installed.
It handles:
- Registering the xue-unicode codec for Unicode operator support
- Enabling strict mode if --strict or XUE_STRICT=1
- Installing the LLM exception hook if --llm or XUE_LLM_HOOK=1
- Registering Secret scrubbing in tracebacks if XUE_SCRUB_SECRETS=1

To install, add to sitecustomize.py:
    import xue.sitecustomize_xue
"""

import os
import sys


def _setup_xue():
    """Initialize xue-python extensions."""

    # Register Unicode operator codec
    try:
        from xue.unicode_ops import register
        register()
    except ImportError:
        pass

    # Enable strict mode from env or flag
    if os.environ.get("XUE_STRICT", "0") == "1" or "--strict" in sys.argv:
        try:
            from xue.strict import enable
            enable()
            # Remove --strict from argv so it doesn't confuse user scripts
            if "--strict" in sys.argv:
                sys.argv.remove("--strict")
        except ImportError:
            pass

    # Install LLM exception hook
    if os.environ.get("XUE_LLM_HOOK", "0") == "1" or "--llm" in sys.argv:
        try:
            from xue.llmhook import install_exception_hook, configure
            # Auto-configure from env vars
            backend = os.environ.get("XUE_LLM_BACKEND", "none")
            if backend != "none":
                configure(
                    backend=backend,
                    url=os.environ.get("XUE_LLM_URL", ""),
                    api_key=os.environ.get("XUE_LLM_API_KEY", ""),
                    model=os.environ.get("XUE_LLM_MODEL", ""),
                    socket_path=os.environ.get("XUE_LLM_SOCKET", ""),
                )
                install_exception_hook()
            if "--llm" in sys.argv:
                sys.argv.remove("--llm")
        except ImportError:
            pass


_setup_xue()
