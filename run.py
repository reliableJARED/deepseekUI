"""Start the local server.

    py run.py                # 127.0.0.1:5000
    py run.py --port 5050
    py run.py --lan          # only if you really mean to expose it

The bind address is deliberately loopback-only. A non-loopback bind means the API
key behind this server is reachable by anything on the network, so it requires the
explicit ``--lan`` flag *and* ``ALLOW_LAN=true``.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.app import configure_logging, create_app          # noqa: E402
from server.settings import load_settings                     # noqa: E402

logger = logging.getLogger("deepseek_ui.run")


def is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deepseekUI server.")
    parser.add_argument("--host", default=None, help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=None, help="Port (default 5000).")
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Allow binding to a non-loopback address. Exposes your API key to the network.",
    )
    parser.add_argument("--reload", action="store_true", help="Reload on source changes (development).")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    overrides: dict[str, object] = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port

    settings = load_settings(**overrides)

    if not is_loopback(settings.host) and not (args.lan and settings.allow_lan):
        logger.error(
            "refusing to bind to %s: it is not a loopback address.\n"
            "  Anyone who can reach this port can spend your API key.\n"
            "  If that is genuinely what you want, set ALLOW_LAN=true in .env and pass --lan.",
            settings.host,
        )
        return 2

    try:
        app = create_app(settings)
    except Exception as exc:
        logger.error("could not start: %s: %s", type(exc).__name__, exc)
        logger.error(
            "Check that .env has DEEPSEEK_API_KEY set and that providers.json parses."
        )
        return 1

    import uvicorn

    display = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
    logger.info("starting on http://%s:%s", display, settings.port)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=args.log_level.lower(),
        reload=args.reload,
        # Long-lived SSE streams must not be reaped mid-answer.
        timeout_keep_alive=75,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
