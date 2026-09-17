"""A small, modular async client for OpenAI ``chat-completions`` endpoints.

Built for DeepSeek but not DeepSeek-specific: point ``base_url`` at anything that
speaks the same protocol.  Four layers, usable independently:

``config``      ``.env`` and ``providers.json`` loading with ``${VAR}`` resolution
``messages``    content/message builders and history sanitisation
``streaming``   SSE decoding and streamed-response reassembly
``tools``       tool schemas, a registry, and the agent loop

Quick start::

    import asyncio
    from deepseek_client import DeepSeekClient, user

    async def main():
        async with DeepSeekClient.from_providers() as client:
            print(await client.complete("Explain SSE in one sentence."))

    asyncio.run(main())
"""

from __future__ import annotations

from .client import DEFAULT_STREAM_TIMEOUT, DEFAULT_TIMEOUT, DeepSeekClient, raise_for_status
from .config import (
    DEFAULT_API_PATH,
    DEFAULT_API_TYPE,
    DEFAULT_BASE_URL,
    ModelSpec,
    Provider,
    ProvidersConfig,
    expand_env_refs,
    find_config_file,
    load_env_file,
)
from .errors import (
    AuthenticationError,
    BadRequestError,
    ConfigError,
    ConnectionFailure,
    ContextLengthError,
    DeepSeekError,
    NotFoundError,
    PaymentRequiredError,
    RateLimitError,
    RetryExhausted,
    ServerError,
    StreamError,
    ToolError,
)
from .messages import (
    CHARS_PER_TOKEN,
    assistant,
    content,
    estimate_tokens,
    image_bytes,
    image_data,
    image_file,
    image_url,
    merge_text,
    sanitize_messages,
    system,
    text,
    tool,
    user,
)
from .streaming import ChatStream, DeltaAccumulator, SSEDecoder
from .tools import Tool, ToolRegistry, ToolResult, run_tool_loop, tool_schema
from .types import ChatMessage, ChatResponse, StreamEvent, ToolCall, Usage

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # client
    "DeepSeekClient",
    "raise_for_status",
    "DEFAULT_TIMEOUT",
    "DEFAULT_STREAM_TIMEOUT",
    # config
    "ProvidersConfig",
    "Provider",
    "ModelSpec",
    "load_env_file",
    "expand_env_refs",
    "find_config_file",
    "DEFAULT_BASE_URL",
    "DEFAULT_API_PATH",
    "DEFAULT_API_TYPE",
    # messages
    "text",
    "image_url",
    "image_data",
    "image_bytes",
    "image_file",
    "content",
    "system",
    "user",
    "assistant",
    "tool",
    "merge_text",
    "sanitize_messages",
    "estimate_tokens",
    "CHARS_PER_TOKEN",
    # streaming
    "ChatStream",
    "SSEDecoder",
    "DeltaAccumulator",
    # tools
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "tool_schema",
    "run_tool_loop",
    # types
    "ChatResponse",
    "ChatMessage",
    "ToolCall",
    "Usage",
    "StreamEvent",
    # errors
    "DeepSeekError",
    "ConfigError",
    "AuthenticationError",
    "PaymentRequiredError",
    "RateLimitError",
    "BadRequestError",
    "ContextLengthError",
    "NotFoundError",
    "ServerError",
    "ConnectionFailure",
    "StreamError",
    "ToolError",
    "RetryExhausted",
]
