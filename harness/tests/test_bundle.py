"""The browser bundle must not drift from the package it was generated from.

The page runs the real analyzer compiled to WASM rather than a JavaScript
restatement of it, because the hand-written checker already on that page shipped
a bug the Python side never had. Two implementations of one idea drift the
moment either is touched.

Compiling the canonical code removes the second implementation, but it
introduces a copy — and a copy has exactly the same failure mode unless
something checks it. This is that check: it recomputes the digest from
`harness/` and compares it to the one baked into the bundle, so a stale bundle
is a failing test rather than a page quietly serving last week's logic.

If this fails, run `python3 web/build_bundle.py`.
"""

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WEB = os.path.join(ROOT, "web")
HARNESS = os.path.join(ROOT, "harness")
BUNDLE = os.path.join(WEB, "harness-bundle.js")

sys.path.insert(0, WEB)
sys.path.insert(0, HARNESS)


@unittest.skipUnless(os.path.exists(BUNDLE), "bundle not built")
class TestBundleIsCurrent(unittest.TestCase):

    def _built(self):
        import build_bundle
        return build_bundle

    def _baked_digest(self):
        with open(BUNDLE) as f:
            head = f.read(4096)
        m = re.search(r'BUNDLE_DIGEST = "([0-9a-f]{64})"', head)
        self.assertIsNotNone(m, "bundle carries no digest")
        return m.group(1)

    def test_the_bundle_matches_the_package_it_came_from(self):
        b = self._built()
        self.assertEqual(
            b.digest(b.sources()), self._baked_digest(),
            "web/harness-bundle.js is stale. Run: python3 web/build_bundle.py")

    def test_the_registry_is_bundled_too(self):
        """Capabilities and pricing are data, and the browser needs both or it
        prices against nothing."""
        b = self._built()
        names = set(b.sources())
        self.assertIn("cacheeconomics/data/providers.json", names)
        self.assertIn("cacheeconomics/data/pricing.json", names)

    def test_the_recorder_is_not_bundled(self):
        """It opens files, fsyncs and takes locks. Shipping it would advertise
        a capability the browser does not have."""
        self.assertNotIn("cacheeconomics/recorder.py", set(self._built().sources()))

    def test_the_analysis_path_is_complete(self):
        b = self._built()
        names = set(self._built().sources())
        for mod in ("trace", "analyzer", "report", "cost", "money", "registry",
                    "checks", "allocate", "relocate", "simulate"):
            self.assertIn(f"cacheeconomics/{mod}.py", names, mod)

    def test_no_bundled_module_needs_a_native_dependency(self):
        """The no-drift guarantee holds only while the package stays
        pure-Python. One native dependency and the browser has to fall back to
        a reimplementation, which is the situation this exists to avoid."""
        allowed = {
            "__future__", "collections", "copy", "dataclasses", "datetime",
            "enum", "statistics", "hashlib", "hmac", "html", "inspect", "json",
            "os", "re", "threading", "typing", "math", "itertools", "functools",
        }
        for path, src in self._built().sources().items():
            if not path.endswith(".py"):
                continue
            # Anchored so prose in a docstring ("inferred from one body") is
            # not read as an import statement.
            found = (re.findall(r"^\s*import\s+([a-zA-Z_][\w.]*)", src, re.MULTILINE)
                     + re.findall(r"^\s*from\s+([a-zA-Z_.][\w.]*)\s+import\s",
                                  src, re.MULTILINE))
            for raw in found:
                if raw.startswith("."):
                    continue            # relative: inside the package by definition
                mod = raw.split(".")[0]
                if mod in ("cacheeconomics",) or mod.startswith("_"):
                    continue
                self.assertIn(mod, allowed,
                              f"{path} imports {mod!r}, which may not exist in Pyodide")




class TestTheShippedSampleReportIsCurrent(unittest.TestCase):
    """`web/sample-report.html` is on the public page, beside a claim that the
    Pyodide demo reproduces it exactly. It had drifted: the file said "Findings
    - 2" while the analyzer produced four, so a visitor comparing the two would
    have found the tool disagreeing with its own published output.

    Nothing regenerated it and nothing checked it, which is the same shape as
    the bundle staleness this file already guards -- a committed artifact
    derived from code, with no test tying the two together.
    """

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FIXTURE = os.path.join(ROOT, "fixtures", "demo-report.html")
    PUBLISHED = os.path.normpath(os.path.join(ROOT, "..", "web", "sample-report.html"))

    # The report stamps the day it was generated, which is not part of what
    # these tests assert. Left in, they failed every time the date rolled over,
    # for a reason that had nothing to do with the code -- and a test that cries
    # wolf daily is a test somebody eventually deletes. The claim is that the
    # published sample is what this code produces; the date it was produced on
    # is not that claim.
    _GENERATED = re.compile(r"(<span>Generated</span><span>)\d{4}-\d{2}-\d{2}(</span>)")

    def _undated(self, html):
        return self._GENERATED.sub(r"\1DATE\2", html)

    def _current(self):
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.report import render_html
        from cacheeconomics.trace import load_jsonl
        traces = os.path.join(self.ROOT, "fixtures", "demo-traces.jsonl")
        return self._undated(
            render_html(analyze(load_jsonl(traces), invoice_usd=17.45)))

    def test_the_fixture_matches_what_the_code_renders(self):
        with open(self.FIXTURE) as f:
            self.assertEqual(self._undated(f.read()), self._current(),
                             "harness/fixtures/demo-report.html is stale. "
                             "Regenerate it and web/sample-report.html together.")

    @unittest.skipUnless(os.path.exists(PUBLISHED), "web/sample-report.html not found")
    def test_the_published_copy_matches_it_too(self):
        with open(self.PUBLISHED) as f:
            self.assertEqual(self._undated(f.read()), self._current(),
                             "web/sample-report.html is stale. It is on the public "
                             "page next to a claim that the demo reproduces it.")


if __name__ == "__main__":
    unittest.main()
