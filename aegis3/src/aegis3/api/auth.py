"""Local API auth + loopback enforcement.

The API binds to 127.0.0.1 by default; this middleware enforces it again at
the application layer to mitigate DNS rebinding when a browser is involved.
HMAC tokens are read from the OS keychain at startup, never persisted to disk.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class LoopbackOnlyMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Host header is not loopback (DNS-rebinding defense).

    The primary control against remote access is binding uvicorn to 127.0.0.1;
    this Host-header check is the defense-in-depth layer that also stops a
    browser tab from reaching the API via a rebound DNS name. We intentionally
    do not inspect ``request.client.host`` — the ASGI transport reports a
    synthetic peer, and the bind address already constrains real connections.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        host_header = (request.headers.get("host") or "").split(":")[0].lower()
        if host_header and host_header not in ALLOWED_HOSTS:
            return JSONResponse({"error": "loopback_only"}, status_code=status.HTTP_403_FORBIDDEN)
        return await call_next(request)


def _expected_token() -> str:
    """Resolve the API token. Tries keychain first, then env for dev only."""
    try:
        import keyring

        token = keyring.get_password("aegis3", "api_token")
        if token:
            return token
    except Exception:  # noqa: BLE001 — keyring backends vary
        pass
    return os.environ.get("AEGIS_API_TOKEN", "")


def hmac_auth(authorization: str | None = Header(default=None)) -> None:
    expected = _expected_token()
    if not expected:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "api token not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    presented = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad token")
