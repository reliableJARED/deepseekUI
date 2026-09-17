"""Typed exceptions raised by :mod:`deepseek_client`.

Every failure mode the wrapper can hit maps to exactly one class here, so
callers can ``except RateLimitError`` instead of sniffing status codes.  All of
them derive from :class:`DeepSeekError`, so a single ``except`` still catches
everything the package throws.
"""

from __future__ import annotations

from typing import Any

__all__ = [
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


class DeepSeekError(Exception):
    """Base class for every error raised by this package."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: Any = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        #: Raw decoded response body, when there was one.  Useful for logging.
        self.body = body
        #: Whether the client would normally retry this failure.
        self.retryable = retryable

    def __str__(self) -> str:
        if self.status is None:
            return self.message
        return f"HTTP {self.status}: {self.message}"


class ConfigError(DeepSeekError):
    """Provider/model configuration is missing, malformed, or unresolved."""


class AuthenticationError(DeepSeekError):
    """401/403 — the API key is absent, wrong, or lacks access to the model."""


class PaymentRequiredError(DeepSeekError):
    """402 — the account is out of credit."""


class RateLimitError(DeepSeekError):
    """429 — too many requests, or the endpoint is throttling us."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, retryable=True, **kw)
        #: Seconds requested by a ``Retry-After`` header, if present.
        self.retry_after = retry_after


class BadRequestError(DeepSeekError):
    """400 — the payload was rejected.

    Overwhelmingly this means a malformed message history rather than a bug in
    the wrapper: an orphaned ``tool`` message, an assistant turn with empty
    content, or unparseable tool-call arguments.  See
    :func:`deepseek_client.messages.sanitize_messages`.
    """


class ContextLengthError(BadRequestError):
    """The conversation is larger than the model's input window."""


class NotFoundError(DeepSeekError):
    """404 — wrong base URL or unknown model id."""


class ServerError(DeepSeekError):
    """5xx — the upstream endpoint failed."""

    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, retryable=True, **kw)


class ConnectionFailure(DeepSeekError):
    """The endpoint could not be reached: DNS, TLS, proxy, offline, timeout."""

    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, retryable=True, **kw)


class StreamError(DeepSeekError):
    """The SSE stream ended or delivered something we cannot interpret."""


class ToolError(DeepSeekError):
    """A tool handler raised, or the model emitted unparseable arguments."""


class RetryExhausted(DeepSeekError):
    """The retry budget ran out; ``cause`` holds the final underlying error."""

    def __init__(self, message: str, *, cause: Exception | None = None, attempts: int = 0) -> None:
        super().__init__(message)
        self.cause = cause
        self.attempts = attempts
