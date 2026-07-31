"""The browser prompt-checker, exercised headless against the shipped default.

This is the page's only interactive proof, and it once contradicted the article
it sits inside: it tracked the *last* volatile line, so a late session id hid an
early timestamp and the shipped sample reported "the stable prefix is
protected" while a timestamp sat above three static sections.

The logic lives in web/index.html because the page must stay self-contained, so
the test extracts it and runs it under node rather than duplicating it in
Python. Duplicating it would test the copy, not the thing that ships.

Skips if node is unavailable. A skipped test is honest; a Python reimplementation
that passes while the page is broken is not.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.normpath(os.path.join(HERE, "..", "..", "web", "index.html"))

# Exactly the contents of the textarea that ships in the page.
SHIPPED_SAMPLE = """CURRENT TIME: 2026-07-28T18:03:41Z
TOOL DEFINITIONS (6000 tokens)
SYSTEM INSTRUCTIONS (8000 tokens)
COMPANY POLICIES (20000 tokens)
session_id: 4f2c-90ab
conversation history
user request"""


def _harness(case_js: str) -> str:
    """Lift the checker out of the page and stub the three DOM touches."""
    with open(PAGE) as f:
        s = f.read()
    body = s[s.index("const VOLATILE"):s.index("$('reset').onclick")]
    body = (body
            .replace("$('run').onclick = () => {", "function check(text) {")
            .replace("const lines = $('sections').value.split", "const lines = text.split")
            .replace("$('results').innerHTML = out.map(([c,t,b]) =>",
                     "return out; (([c,t,b]) =>")
            .replace('      <span><b>${t}</b><span class="t">${b}</span></span></div>`)'
                     ".join('');\n};",
                     '      <span></span></div>`);\n}'))
    return body + "\n" + case_js


def _run(case_js: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(_harness(case_js))
        path = f.name
    try:
        out = subprocess.run([shutil.which("node"), path], capture_output=True,
                             text=True, timeout=30)
        if out.returncode != 0:
            raise AssertionError(f"checker threw: {out.stderr.strip()}")
        return out.stdout.strip()
    finally:
        os.unlink(path)


@unittest.skipUnless(shutil.which("node"), "node not available")
@unittest.skipUnless(os.path.exists(PAGE), "web/index.html not found")
class TestWebPromptChecker(unittest.TestCase):

    def test_the_shipped_sample_is_reported_as_blocked(self):
        """The regression that mattered: the page's own example is the failure
        case the page exists to describe, and it used to pass it."""
        out = _run(f"console.log(check(`{SHIPPED_SAMPLE}`)[0][1]);")
        self.assertEqual(out, "A moving line is blocking the cache.")

    def test_it_names_the_first_blocker_not_the_last_volatile_line(self):
        out = _run(f"console.log(check(`{SHIPPED_SAMPLE}`)[0][2]);")
        self.assertIn("Line 1", out, "the timestamp at line 1 is the blocker")
        self.assertNotIn("Line 5", out, "the later session id must not mask it")
        self.assertIn("3 static sections", out)

    def test_volatile_content_below_everything_static_is_fine(self):
        out = _run("console.log(check(`TOOL DEFINITIONS\\n"
                   "SYSTEM INSTRUCTIONS\\nsession_id: abc`)[0][1]);")
        self.assertEqual(out, "The moving parts are low enough.")

    def test_nothing_volatile_reads_as_nothing_volatile(self):
        out = _run("console.log(check(`TOOL DEFINITIONS\\nSYSTEM INSTRUCTIONS`)[0][1]);")
        self.assertEqual(out, "Nothing obviously volatile near the top.")

    def test_a_single_volatile_line_above_one_stable_line_is_caught(self):
        out = _run("console.log(check(`CURRENT TIME: x\\nSYSTEM INSTRUCTIONS`)[0][2]);")
        self.assertIn("1 static section sits behind it", out)




WORKER = os.path.normpath(os.path.join(HERE, "..", "..", "web", "analyzer-worker.js"))


@unittest.skipUnless(os.path.exists(PAGE), "web/index.html not found")
class TestThirdPartyCodeStaysOffThePage(unittest.TestCase):
    """The page invites people to paste a real prompt into a textarea. Nothing
    third-party may execute in the document that holds it.

    Python is fetched from a public CDN, which is fine in a worker -- a worker
    has no document, no window and no way to reach an input element -- and not
    fine on the page, where a compromised CDN module would run with the page's
    privileges and could read what was typed. Verified in Chrome on 29 Jul 2026
    that the document cannot fetch or import the CDN at all while the worker
    still loads Python.
    """

    def setUp(self):
        with open(PAGE) as f:
            self.html = f.read()

    def test_the_document_never_names_a_third_party_origin_as_a_url(self):
        """The only mention left is a comment explaining why. A `https://cdn...`
        anywhere else is a script tag, an import, or a fetch."""
        for line in self.html.splitlines():
            if "cdn.jsdelivr.net" not in line:
                continue
            self.assertTrue(line.lstrip().startswith("//"),
                            f"live reference to the CDN on the page: {line.strip()!r}")

    def test_the_csp_permits_no_third_party_script_or_connection(self):
        csp = next(line for line in self.html.splitlines()
                   if "Content-Security-Policy" in line)
        for directive in ("script-src", "connect-src"):
            clause = csp.split(directive + " ")[1].split(";")[0]
            self.assertNotIn("http", clause,
                             f"{directive} admits a third-party origin: {clause!r}")

    def test_workers_are_same_origin_only(self):
        csp = next(line for line in self.html.splitlines()
                   if "Content-Security-Policy" in line)
        self.assertIn("worker-src 'self'", csp)

    def test_the_page_still_runs_the_real_analyzer(self):
        """Moving Python off the page must not quietly become dropping it."""
        self.assertIn("analyzer-worker.js", self.html)
        self.assertTrue(os.path.exists(WORKER))

    def test_the_worker_is_where_the_cdn_import_lives(self):
        with open(WORKER) as f:
            worker = f.read()
        self.assertIn("cdn.jsdelivr.net", worker)
        self.assertIn("runPythonAsync", worker)

    def test_the_privacy_copy_matches_where_the_code_actually_runs(self):
        """An earlier version of this paragraph asked readers to trust the CDN,
        which was the honest description of the old design and is now wrong in
        the other direction."""
        self.assertIn("runs in a worker rather than on this page", self.html)
        self.assertNotIn("third-party code running with this page", self.html)


if __name__ == "__main__":
    unittest.main()
