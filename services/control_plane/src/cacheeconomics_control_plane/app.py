from __future__ import annotations

import ipaddress
import re
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import Engine
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .api import router
from .db import create_database_engine, create_session_factory
from .observability import MetricsRegistry, event_logger, log_event
from .security import OidcTokenVerifier, TokenVerifier
from .settings import Settings


SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class LocalLivenessMiddleware:
    """Let the container probe liveness without weakening normal host checks."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        client = scope.get("client")
        is_local_health = (
            scope["type"] == "http"
            and scope.get("path") == "/healthz"
            and client is not None
            and _is_loopback(client[0])
        )
        if is_local_health:
            await JSONResponse({"status": "ok"})(scope, receive, send)
            return
        await self.app(scope, receive, send)


class IngestBodyLimitMiddleware:
    """Buffer at most the configured ingest limit, including chunked bodies."""

    def __init__(self, app: ASGIApp, *, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/v1/ingest/events":
            await self.app(scope, receive, send)
            return

        for name, raw_value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                declared = int(raw_value)
            except ValueError:
                await JSONResponse(
                    {"detail": "Content-Length must be an integer"},
                    status_code=400,
                )(scope, receive, send)
                return
            if declared < 0:
                await JSONResponse(
                    {"detail": "Content-Length must not be negative"},
                    status_code=400,
                )(scope, receive, send)
                return
            if declared > self.max_bytes:
                await self._too_large(scope, receive, send)
                return

        messages = []
        received = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                break
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                await self._too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> dict:
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse(
            {"detail": "ingest body exceeds the configured size limit"},
            status_code=413,
        )(scope, receive, send)


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(
    settings: Optional[Settings] = None,
    *,
    engine: Optional[Engine] = None,
    token_verifier: Optional[TokenVerifier] = None,
) -> FastAPI:
    settings = settings or Settings()
    engine = engine or create_database_engine(settings)

    application = FastAPI(
        title="cacheeconomics control plane",
        version=__version__,
        docs_url=None if settings.environment == "production" else "/docs",
        redoc_url=None if settings.environment == "production" else "/redoc",
        openapi_url=None if settings.environment == "production" else "/openapi.json",
    )
    application.state.settings = settings
    application.state.engine = engine
    application.state.session_factory = create_session_factory(engine)
    application.state.token_verifier = token_verifier or OidcTokenVerifier(settings)
    application.state.metrics = MetricsRegistry()
    application.state.access_logger = event_logger(
        "cacheeconomics.control_plane.access",
        json_logs=settings.json_logs,
    )
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )
    # Docker probes 127.0.0.1, whose Host header should not have to appear in a
    # production public-host allow-list. This outer middleware bypasses only
    # the fixed liveness response and only for a loopback client. All other
    # paths, including readiness and authenticated APIs, still reach the host
    # validator.
    application.add_middleware(LocalLivenessMiddleware)
    application.add_middleware(
        IngestBodyLimitMiddleware,
        max_bytes=settings.ingest_max_body_bytes,
    )

    @application.middleware("http")
    async def security_headers(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if SAFE_REQUEST_ID.fullmatch(supplied) else str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.monotonic()
        observe = request.url.path != "/metrics"
        if observe:
            application.state.metrics.request_started()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Cache-Control"] = "no-store"
            return response
        finally:
            if observe:
                duration = max(0.0, time.monotonic() - started)
                route_object = request.scope.get("route")
                route = getattr(route_object, "path", "unmatched")
                application.state.metrics.request_finished(
                    method=request.method,
                    route=route,
                    status_code=status_code,
                    duration_seconds=duration,
                )
                log_event(
                    application.state.access_logger,
                    "http.request",
                    request_id=request_id,
                    method=request.method,
                    route=route,
                    status_code=status_code,
                    duration_ms=round(duration * 1_000, 3),
                )

    application.include_router(router)
    return application


app = create_app()
