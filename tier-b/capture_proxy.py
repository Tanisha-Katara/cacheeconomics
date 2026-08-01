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

UPSTREAM = "https://api.anthropic.com"

_lock = threading.Lock()
_state = {"n": 0, "out": None, "errors": 0}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):            # noqa: A003
        pass                              # the run's own output is the log

    def do_POST(self):                    # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length)
        sent_at = datetime.now(timezone.utc)

        upstream = UPSTREAM + self.path
        # `accept-encoding` is dropped so upstream replies uncompressed. urllib
        # does not decompress, and forwarding gzipped bytes with the
        # `content-encoding` header stripped hands the client a body it decodes
        # as text: measured as "'utf-8' codec can't decode byte 0x8b", which is
        # the gzip magic number, six steps into a real agent run.
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection",
                                        "accept-encoding")}
        headers["accept-encoding"] = "identity"
        headers.setdefault("x-api-key", os.environ.get("ANTHROPIC_API_KEY", ""))
        headers.setdefault("anthropic-version", "2023-06-01")

        req = urllib.request.Request(upstream, data=raw, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body, status = r.read(), r.status
                out_headers = dict(r.headers)
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
            if k.lower() in ("content-length", "transfer-encoding", "connection",
                             "content-encoding"):
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
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    _state["out"] = open(args.out, "a")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"  forwarding 127.0.0.1:{args.port} -> {UPSTREAM}", file=sys.stderr)
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
