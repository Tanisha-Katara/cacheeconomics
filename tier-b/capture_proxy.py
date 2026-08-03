"""A forwarding proxy that writes a bodies export as it goes.

Point a real agent at this instead of `api.anthropic.com`, run it on a real
task, and you get a `--from bodies` export of exactly what the agent sent and
exactly what came back. Nothing is simulated and nothing is reconstructed: the
rows are the wire.

Why this and not the recorder. The recorder needs a code change inside somebody
else's agent, which is fine for a client who has agreed to instrument and
useless for measuring a project you do not control. A base URL is usually a
constructor argument or an environment variable, so this reaches workloads the
recorder cannot.

What it deliberately does not do: mutate. It forwards the request byte for byte.
Anything this measures is the agent's own behaviour, not the behaviour of an
agent with our plugin in the path, and a case study that confused those two
would be worthless.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 tier-b/capture_proxy.py --out run.jsonl --port 8787
    # then point the agent at http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# Overridable for the same reason the counter's endpoint is: a client behind a
# gateway cannot reach this host, and the proxy exists precisely for clients who
# cannot change their export pipeline.
DEFAULT_UPSTREAM = "https://api.anthropic.com"

_HOP_BY_HOP = frozenset((
    "content-length", "transfer-encoding", "connection", "content-encoding",
    "keep-alive", "te", "trailer", "upgrade"))


def _wants_stream(raw: bytes) -> bool:
    """Did the caller ask for a streamed response?"""
    try:
        return bool(json.loads(raw or b"{}").get("stream"))
    except Exception:                                          # noqa: BLE001
        return False


_lock = threading.Lock()
_state = {"n": 0, "out": None, "errors": 0, "upstream": DEFAULT_UPSTREAM}


class _BadChunkedBody(Exception):
    """The client's chunked framing could not be parsed, so the body is not
    knowable. A 400 to the caller and nothing forwarded upstream."""


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):            # noqa: A003
        pass                              # the run's own output is the log

    def _read_request_body(self) -> bytes:
        """The request body, however the client framed it.

        This read `content-length` alone, so an HTTP/1.1 client using
        `Transfer-Encoding: chunked` -- which is legal and which some SDKs use
        for large or streamed uploads -- had its body read as zero bytes and
        forwarded empty. The provider then answered a request the caller never
        made, and the capture recorded that as what was sent.

        Three things went wrong in the first version of the chunked branch, and
        all three are about telling one kind of nothing from another.

        `readline()` returns `b""` at end of file and `b"\\r\\n"` on a blank
        line, and both are falsy after `.strip()`. The loop did `continue` on
        falsy, so a client that hung up mid-upload -- an abort, a crash, a
        timeout -- span that handler thread at 100% CPU for the life of the
        process. Every aborted upload leaked another one.

        A malformed length `break`s out and returned the bytes read so far. That
        forwards a plausible-looking truncated request, which is harder to
        notice than the empty body this function was written to fix, and the
        capture records it as what the caller sent. Unparseable framing means
        the body is not knowable, so nothing is forwarded.

        Trailers after the final chunk were left unread in `rfile`. On a
        keep-alive connection the next request then starts parsing at
        `X-Checksum: abc`, so one capture with trailers corrupts every request
        behind it on that socket.
        """
        if "chunked" in (self.headers.get("transfer-encoding") or "").lower():
            out = []
            while True:
                raw = self.rfile.readline()
                if not raw:
                    raise _BadChunkedBody("connection closed mid-body")
                line = raw.strip()
                if not line:
                    continue                       # a genuine blank line
                try:
                    size = int(line.split(b";")[0], 16)
                except ValueError:
                    raise _BadChunkedBody(f"unparseable chunk length {line[:32]!r}")
                if size < 0:
                    raise _BadChunkedBody("negative chunk length")
                if size == 0:
                    # Drain trailers to the genuinely empty line that ends them.
                    while True:
                        raw = self.rfile.readline()
                        if not raw:
                            raise _BadChunkedBody("connection closed in trailers")
                        if not raw.strip():
                            break
                    break
                chunk = self.rfile.read(size)
                if len(chunk) != size:
                    raise _BadChunkedBody(
                        f"chunk declared {size} bytes and {len(chunk)} arrived")
                out.append(chunk)
                self.rfile.readline()              # CRLF after each chunk
            return b"".join(out)
        return self.rfile.read(int(self.headers.get("content-length") or 0))

    def do_POST(self):                    # noqa: N802
        try:
            raw = self._read_request_body()
        except _BadChunkedBody as e:
            # Nothing goes upstream and nothing is recorded. A truncated body
            # forwarded as if whole is a capture of a request nobody made.
            #
            # And the connection dies with it. This raises partway through a
            # body, so an unknown number of bytes are still unread on the
            # socket -- and `protocol_version = "HTTP/1.1"` means keep-alive, so
            # the next read starts inside the abandoned body and parses it as a
            # request line. That is the same corruption draining the trailers
            # was written to prevent, reintroduced on the error path: one
            # malformed upload taking the following good request with it, and
            # silently dropping it from the capture. The remainder cannot be
            # drained because its framing is what failed, so the only correct
            # answer is to close.
            self.close_connection = True
            body = json.dumps({"error": {
                "type": "cacheeconomics_proxy_bad_request",
                "message": f"could not read the request body: {e}"}}).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return
        sent_at = datetime.now(timezone.utc)

        # Refused, not silently reframed. `urllib` buffers the upstream
        # response, so an event stream reaches the client only once the model
        # has finished -- or hits the 300s timeout and becomes a synthetic 502.
        # A capture that changes the behaviour it exists to observe is worse
        # than no capture, and a proxy that claims to forward byte for byte
        # should say when it cannot.
        #
        # Relaying the stream chunk by chunk was tried and measured: first byte
        # still arrived at 1.22s on a stream spanning 1.2s, so the buffering is
        # upstream of the relay. Fixing it properly means replacing urllib with
        # a socket-level client, which is a different tool than this one.
        if _wants_stream(raw):
            msg = {"error": {
                "type": "cacheeconomics_proxy_unsupported",
                "message": (
                    "capture_proxy does not forward streaming responses "
                    "faithfully and will not pretend to. Re-run the capture "
                    "with stream disabled, or point the agent straight at the "
                    "provider and export bodies from your gateway instead.")}}
            payload = json.dumps(msg).encode()
            with _lock:
                _state["refused"] = _state.get("refused", 0) + 1
            self.send_response(501)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        upstream = _state["upstream"] + self.path
        # `accept-encoding` is dropped so upstream replies uncompressed. urllib
        # does not decompress, and forwarding gzipped bytes with the
        # `content-encoding` header stripped hands the client a body it decodes
        # as text: measured as "'utf-8' codec can't decode byte 0x8b", which is
        # the gzip magic number, six steps into a real agent run.
        # Hop-by-hop headers describe this connection, not the request, and
        # forwarding them upstream is how a proxy corrupts framing.
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection",
                                        "accept-encoding", "transfer-encoding",
                                        "keep-alive", "te", "trailer",
                                        "upgrade", "proxy-authorization")}
        headers["accept-encoding"] = "identity"
        headers.setdefault("x-api-key", os.environ.get("ANTHROPIC_API_KEY", ""))
        headers.setdefault("anthropic-version", "2023-06-01")

        req = urllib.request.Request(upstream, data=raw, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out_headers = dict(r.headers)
                status = r.status
                body = r.read()
        except urllib.error.HTTPError as e:
            body, status = e.read(), e.code
            out_headers = dict(e.headers)
            with _lock:
                _state["errors"] += 1
        except Exception as e:                                  # noqa: BLE001
            body, status = json.dumps({"error": str(e)}).encode(), 502
            out_headers = {"content-type": "application/json"}
            with _lock:
                _state["errors"] += 1

        self._record(raw, body, status, sent_at)

        payload = body
        self.send_response(status)
        for k, v in out_headers.items():
            if k.lower() in _HOP_BY_HOP:
                continue
            self.send_header(k, v)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record(self, raw: bytes, body: bytes, status: int, sent_at) -> None:
        # Only messages traffic is a request in the sense this analysis means.
        # Token counting and model listing are not billed prompts.
        if "/messages" not in self.path or self.path.endswith("count_tokens"):
            return
        try:
            request = json.loads(raw)
        except ValueError:
            return
        try:
            response = json.loads(body)
        except ValueError:
            response = {}
        with _lock:
            _state["n"] += 1
            # Session identity, which the analysis needs and the wire does not
            # carry. Without it every request lands in one reuse chain, and a
            # workload that interleaves call types -- an agent loop beside a
            # judgement call, say -- reads as one conversation whose tools keep
            # changing. Measured on a real browser-use capture: VOL-1 reported
            # the tool definition as volatile across 3 values, which were three
            # different *kinds* of request rather than one drifting prefix.
            #
            # Keyed on the shape of the stable prefix, which is the closest
            # thing to a conversation the wire offers: same system prompt and
            # same tools means the same cache pool, which is the grouping every
            # finding here actually reasons about.
            tools = json.dumps(request.get("tools") or [], sort_keys=True)
            system = json.dumps(request.get("system") or "", sort_keys=True)
            session = hashlib.sha256((tools + system).encode()).hexdigest()[:16]
            row = {"request_id": response.get("id") or f"row-{_state['n']}",
                   "sent_at": sent_at.isoformat(),
                   "session": session,
                   "agent": request.get("model", "unknown"),
                   "status": status,
                   "request": request,
                   "response": response}
            _state["out"].write(json.dumps(row) + "\n")
            _state["out"].flush()
            n, err = _state["n"], _state["errors"]
        print(f"  captured {n:>3}  status={status}  "
              f"model={request.get('model', '?')}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="where to write the bodies export")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--upstream", default=DEFAULT_UPSTREAM,
                   help=f"where to forward to (default: {DEFAULT_UPSTREAM})")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    _state["upstream"] = args.upstream.rstrip("/")
    _state["out"] = open(args.out, "a")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"  forwarding 127.0.0.1:{args.port} -> {_state['upstream']}", file=sys.stderr)
    print(f"  writing {args.out}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _state["out"].close()
        print(f"\n  {_state['n']} requests captured, {_state['errors']} errors",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
