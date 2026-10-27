"""Run the API. Proxy headers stay off so a forwarded address cannot look local."""

from __future__ import annotations

import os

import uvicorn


def serve_options() -> dict:
    port_raw = os.environ.get("PORT", "8080")
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise SystemExit("PORT must be an integer") from exc
    if port < 1 or port > 65535:
        raise SystemExit("PORT is out of range")
    return {
        "app": "scanlab.server.app:app",
        "host": os.environ.get("SCANLAB_HOST", "0.0.0.0"),
        "port": port,
        "proxy_headers": False,
        "forwarded_allow_ips": "",
        "access_log": False,
        "server_header": False,
        "h11_max_incomplete_event_size": 16 * 1024,
        "log_level": "info",
    }


def main() -> None:
    uvicorn.run(**serve_options())


if __name__ == "__main__":
    main()
