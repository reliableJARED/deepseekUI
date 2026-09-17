"""Backend for the local DeepSeek chat UI.

Layering, from the outside in::

    run.py            process entrypoint, binds the socket
    server.routes     HTTP surface (Starlette routes + SSE)
    server.llm        one completion: build -> stream -> persist
    server.tools_builtin / server.mcp
                      what the model can call
    server.rehydrate  stored messages -> API-ready messages
    server.media      bytes on disk <-> URLs
    server.store      conversation.json
    server.settings   environment -> configuration

Everything above ``deepseek_client`` is deployment policy; ``deepseek_client`` is
protocol. Nothing in that package imports from this one.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
