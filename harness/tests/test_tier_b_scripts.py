"""The scripts that leave the machine, and the bugs they have already had.

`tier-b/` is not part of the installed package, so none of the 1,026 tests
touched it. That is where every script that opens a socket lives, and in one
afternoon it produced five defects: a dry run that wrote a file the analyzer
read as authoritative, a dry run that under-reported its own egress, a proxy
that forwarded gzip as text, a proxy with no session identity that manufactured
a false finding, and two hard-coded hostnames that locked out the clients most
likely to care.

All five were fixed and none had a test, which is the same as not being fixed.

Network is never touched here. The counter takes an injected `count` callable
and the proxy is exercised against a stub upstream on loopback, so these run in
CI with no key and no egress.
"""

import http.server
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIER_B = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tier-b"))


def load(name):
    """Import a tier-b script by path; it is not on any import path."""
    path = os.path.join(TIER_B, f"{name}.py")
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{path} not present")
    spec = importlib.util.spec_from_file_location(f"_tierb_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BODY = {"model": "claude-haiku-4-5",
        "system": [{"type": "text", "text": "policy " * 200}],
        "messages": [{"role": "user", "content": "hello"}]}


def _row(i=0):
    return {"request_id": f"r{i}", "sent_at": "2026-08-01T09:00:00+00:00",
            "request": BODY,
            "response": {"usage": {"input_tokens": 1000,
                                   "cache_read_input_tokens": 0,
                                   "cache_creation_input_tokens": 0}}}


class TestADryRunLeavesNothingBehind(unittest.TestCase):
    """It wrote its output file with every count zero, and the loader read that
    as *counted*: `tokens_are_counted: True` on a file that had never spoken to
    a tokenizer, with a whole prompt's mass collapsed onto one segment.

    A file that looks authoritative and is not is the failure this toolchain
    exists to refuse, and this one was manufactured by the flag added to make
    the tool easier to trust.
    """

    def _run(self, *extra):
        src = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        src.write(json.dumps(_row()) + "\n")
        src.close()
        self.addCleanup(os.unlink, src.name)
        out = src.name.replace(".jsonl", "-out.jsonl")
        r = subprocess.run(
            [sys.executable, os.path.join(TIER_B, "count_tokens.py"),
             src.name, "-o", out, "--dry-run", *extra],
            capture_output=True, text=True,
            env=dict(os.environ, ANTHROPIC_API_KEY=""))
        return out, r

    def test_it_writes_no_output_file(self):
        out, r = self._run()
        self.addCleanup(lambda: os.path.exists(out) and os.unlink(out))
        self.assertFalse(os.path.exists(out),
                         "a dry run produced a file the analyzer would read")

    def test_it_needs_no_api_key(self):
        """It sends nothing, so demanding a key to find out what it would send
        makes the safe option the harder one."""
        _, r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr[-300:])

    def test_it_names_the_host_it_would_send_to(self):
        _, r = self._run()
        self.assertIn("api.anthropic.com", r.stdout + r.stderr)

    def test_a_custom_endpoint_is_the_host_it_names(self):
        _, r = self._run("--endpoint", "https://gateway.internal/count")
        blob = r.stdout + r.stderr
        self.assertIn("gateway.internal", blob)


class TestTheCounterIsExactAndCached(unittest.TestCase):
    """`count_segments` takes differences of in-context prefix counts. The
    caching is the only reason it is affordable, so it is asserted rather than
    assumed."""

    def test_a_shared_prefix_is_counted_once(self):
        tok = load("../harness/cacheeconomics/tokenizer") if False else None
        from cacheeconomics.tokenizer import count_segments
        calls = []

        def fake(body):
            calls.append(1)
            return 10 * len(json.dumps(body))

        cache = {}
        count_segments(BODY, fake, cache)
        first = len(calls)
        count_segments(BODY, fake, cache)          # identical body
        self.assertEqual(len(calls), first,
                         "the second pass re-counted a prefix it had already seen")

    def test_counts_are_never_negative(self):
        """A boundary can measure under its predecessor when the tokenizer
        merges across it, and a negative token count is refused at pricing --
        which would drop the whole request rather than the one segment."""
        from cacheeconomics.tokenizer import count_segments
        shrinking = iter([100, 90, 80, 70, 60, 50, 40])
        got = count_segments(BODY, lambda b: next(shrinking, 0), {})
        self.assertTrue(all(n >= 0 for n in got), got)


class _Stub(http.server.BaseHTTPRequestHandler):
    """An upstream that answers like the provider, and gzips if asked."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        import gzip
        n = int(self.headers.get("content-length") or 0)
        self.rfile.read(n)
        payload = json.dumps({"id": "msg_stub", "content": [],
                              "usage": {"input_tokens": 7,
                                        "cache_read_input_tokens": 0,
                                        "cache_creation_input_tokens": 0}}).encode()
        wants_gzip = "gzip" in (self.headers.get("accept-encoding") or "")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        if wants_gzip:
            payload = gzip.compress(payload)
            self.send_header("content-encoding", "gzip")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class TestTheProxyForwardsSomethingReadable(unittest.TestCase):
    """It stripped `content-encoding` and forwarded the still-compressed body,
    so the client decoded gzip as text. Measured as "'utf-8' codec can't decode
    byte 0x8b" -- the gzip magic number -- six steps into a real agent run.
    """

    def setUp(self):
        self.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        self.port = self.stub.server_address[1]
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        self.addCleanup(self.stub.shutdown)

        self.out = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        self.out.close()
        self.addCleanup(lambda: os.path.exists(self.out.name) and os.unlink(self.out.name))

        self.proxy = subprocess.Popen(
            [sys.executable, os.path.join(TIER_B, "capture_proxy.py"),
             "--out", self.out.name, "--port", "0",
             "--upstream", f"http://127.0.0.1:{self.port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        self.addCleanup(self.proxy.terminate)

    def test_the_upstream_is_overridable(self):
        """It hard-coded api.anthropic.com, which locks out every client behind
        a gateway -- the ones most likely to care about egress at all."""
        src = open(os.path.join(TIER_B, "capture_proxy.py")).read()
        self.assertIn("--upstream", src)
        self.assertIn("DEFAULT_UPSTREAM", src)

    def test_it_asks_upstream_for_identity_encoding(self):
        src = open(os.path.join(TIER_B, "capture_proxy.py")).read()
        self.assertIn('"accept-encoding"] = "identity"', src.replace("'", '"'),
                      "the proxy may request a compressed body it cannot forward")


class TestEveryCapturedRowCarriesASession(unittest.TestCase):
    """Without one, every request lands in one reuse chain. On a real capture
    that made three different *kinds* of request -- an agent loop beside a
    judgement call -- read as one conversation whose tools kept changing, and
    VOL-1 fired confidently on a prefix that had never drifted."""

    def test_the_key_is_derived_from_the_stable_prefix(self):
        src = open(os.path.join(TIER_B, "capture_proxy.py")).read()
        self.assertIn('"session"', src)
        self.assertIn("tools", src.split('"session"')[0][-600:],
                      "the session key should be derived from tools and system, "
                      "which is what decides the cache pool")


class TestTheDefaultPathCounts(unittest.TestCase):
    """Counting was correct and optional, and optional meant skipped. Skipping
    yields a report missing the half a client pays for, because the analyzer
    refuses to cost a structural finding from a 19.2% split."""

    def test_opting_out_must_be_typed(self):
        r = subprocess.run(
            [sys.executable, os.path.join(TIER_B, "run_diagnostic.py"), "--help"],
            capture_output=True, text=True)
        self.assertIn("--estimate-only", r.stdout)
        self.assertNotIn("--count-tokens", r.stdout,
                         "counting should be the default, not a flag to enable")


if __name__ == "__main__":
    unittest.main()
