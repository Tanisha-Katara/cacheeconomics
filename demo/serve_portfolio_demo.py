#!/usr/bin/env python3
"""Serve the checked-in synthetic packet through the real dashboard UI."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / "demo/fixtures/portfolio-demo-v1.json"
DASHBOARD = ROOT / "apps/dashboard"
DEMO_TOKEN = "synthetic-demo-only"
STATIC_FILES = {
    "/": DASHBOARD / "index.html",
    "/index.html": DASHBOARD / "index.html",
    "/styles.css": DASHBOARD / "styles.css",
    "/app.js": DASHBOARD / "app.js",
    "/favicon.svg": DASHBOARD / "favicon.svg",
    "/media/recommendations-tour.webm": DASHBOARD / "media/recommendations-tour.webm",
    "/media/recommendations-tour-poster.jpg": DASHBOARD
    / "media/recommendations-tour-poster.jpg",
    "/media/operations-tour.webm": DASHBOARD / "media/operations-tour.webm",
    "/media/operations-tour-poster.jpg": DASHBOARD
    / "media/operations-tour-poster.jpg",
    "/media/social-preview.png": DASHBOARD / "media/social-preview.png",
}


def _api_response(
    packet: dict[str, Any],
    method: str,
    raw_path: str,
    headers: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    parsed = urlsplit(raw_path)
    path = parsed.path
    responses = packet["responses"]
    normalized_headers = {key.lower(): value for key, value in headers.items()}
    if method != "GET":
        return 405, {"detail": "The portfolio packet is read-only."}
    if path == "/api/v1/dashboard/config":
        return 200, responses["config"]
    if normalized_headers.get("authorization") != f"Bearer {DEMO_TOKEN}":
        return 401, {"detail": "Use the printed synthetic demo token."}
    if path == "/api/v1/me":
        return 200, responses["me"]

    expected_org = packet["demo_access"]["organization_id"]
    if normalized_headers.get("x-organization-id") != expected_org:
        return 403, {"detail": "The synthetic organization context is required."}
    source_id = packet["demo_access"]["source_id"]
    if path == "/api/v1/sources":
        return 200, responses["sources"]
    if path == "/api/v1/jobs":
        return 200, responses["jobs"]
    if path == f"/api/v1/sources/{source_id}/health":
        return 200, responses["health"]
    if path == f"/api/v1/sources/{source_id}/analyses/latest":
        return 200, responses["analysis"]
    if path == f"/api/v1/sources/{source_id}/operations":
        window = parse_qs(parsed.query).get("window_hours", ["168"])[0]
        if window not in responses["operations"]:
            return 422, {"detail": "The replay supports 24, 168, or 720 hours."}
        return 200, responses["operations"][window]
    return 404, {"detail": "This route is not present in the demo packet."}


class _Handler(BaseHTTPRequestHandler):
    packet: dict[str, Any]

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            status, body = _api_response(
                self.packet,
                "GET",
                self.path,
                {key: value for key, value in self.headers.items()},
            )
            self._send_json(status, body)
            return
        asset = STATIC_FILES.get(path)
        if asset is None:
            self._send_json(404, {"detail": "asset not found"})
            return
        content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        payload = asset.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        status, body = _api_response(
            self.packet,
            "POST",
            self.path,
            {key: value for key, value in self.headers.items()},
        )
        self._send_json(status, body)

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "frame-ancestors 'none'; base-uri 'none'",
        )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"demo {self.client_address[0]} {format % args}")


def load_packet(path: Path = PACKET_PATH) -> dict[str, Any]:
    packet = json.loads(path.read_text())
    if (
        packet.get("schema") != "cacheeconomics.portfolio-demo-packet"
        or packet.get("schema_version") != 1
    ):
        raise ValueError("not a cacheeconomics portfolio demo packet")
    for evidence_name in ("trace", "golden_result"):
        evidence = packet["provenance"][evidence_name]
        source = ROOT / evidence["path"]
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != evidence["sha256"]:
            raise ValueError(
                f"the {evidence_name.replace('_', ' ')} no longer matches "
                "the demo packet"
            )
    if packet["demo_access"] != {
        "token": DEMO_TOKEN,
        "organization_id": "00000000-0000-0000-0000-000000000502",
        "source_id": "00000000-0000-0000-0000-000000000503",
        "read_only": True,
    }:
        raise ValueError("the demo access contract changed")
    return packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the read-only synthetic portfolio demo")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "::1"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")

    packet = load_packet()
    handler = type("PortfolioDemoHandler", (_Handler,), {"packet": packet})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    host, port = server.server_address[:2]
    print("Synthetic, read-only portfolio demo")
    print(f"Open:  http://{host}:{port}/")
    print(f"Token: {DEMO_TOKEN}")
    print("No prompt or completion bodies are present in this packet.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
