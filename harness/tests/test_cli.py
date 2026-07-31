"""The CLI is the only part of this package most people will ever touch.

Two things it is responsible for beyond running the right function: it must not
put the HMAC key anywhere it can leak, and it must not become the place where
the withheld-figures rule quietly stops applying. Both are asserted here rather
than trusted.
"""

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import cli                                          # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "fixtures", "demo-traces.jsonl")


def run(*argv):
    """Return (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as e:                       # argparse usage errors
            code = e.code if isinstance(e.code, int) else 2
    return code, out.getvalue(), err.getvalue()


class TestTheKeyNeverTouchesArgv(unittest.TestCase):
    """`ps` shows argv to every user on the box, and the shell writes it to
    history. A key that has been on a command line is a key that has leaked, so
    the flag does not exist -- and this asserts it stays that way."""

    def _option_strings(self, parser):
        """Every flag argparse will actually accept, subcommands included.

        Read off the parser rather than grepped out of `--help`: the help text
        explains *why* there is no --key flag, so a substring search found the
        sentence warning against it and called that a violation.
        """
        out = set()
        for action in parser._actions:
            out.update(action.option_strings)
            # `choices` is None on most actions, a plain tuple on ones like
            # --format, and a name->parser dict only on the subparser action.
            # Only the dict is worth descending into.
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                for sub in choices.values():
                    if hasattr(sub, "_actions"):
                        out |= self._option_strings(sub)
        return out

    def test_there_is_no_key_flag(self):
        flags = self._option_strings(cli.build_parser())
        self.assertNotIn("--key", flags,
                         "argv is visible in ps and saved to shell history")
        self.assertIn("--key-file", flags)

    def test_a_world_readable_key_file_is_refused(self):
        fd, path = tempfile.mkstemp()
        os.write(fd, b"k" * 32)
        os.close(fd)
        try:
            os.chmod(path, 0o644)
            code, _out, err = run("analyze", FIXTURE, "--key-file", path)
            self.assertEqual(code, 1)
            self.assertIn("readable by other users", err)
            self.assertIn("chmod 600", err)
        finally:
            os.unlink(path)

    def test_a_private_key_file_is_accepted(self):
        fd, path = tempfile.mkstemp()
        os.write(fd, b"k" * 32)
        os.close(fd)
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            code, _out, err = run("analyze", FIXTURE, "--key-file", path,
                                  "--allow-unreconciled")
            self.assertEqual(code, 0, err)
        finally:
            os.unlink(path)

    def test_a_short_key_is_refused_with_the_reason(self):
        fd, path = tempfile.mkstemp()
        os.write(fd, b"short")
        os.close(fd)
        try:
            os.chmod(path, 0o600)
            code, _out, err = run("analyze", FIXTURE, "--key-file", path)
            self.assertEqual(code, 1)
            self.assertIn("minimum", err)
            self.assertIn("openssl rand", err, "say how to make a good one")
        finally:
            os.unlink(path)

    def test_the_body_adapter_will_not_run_unkeyed(self):
        """It derives ids by hashing prompt content, and an unkeyed digest of a
        prompt is confirmable by anyone holding a guess."""
        env = os.environ.pop(cli.KEY_ENV, None)
        try:
            code, _out, err = run("analyze", FIXTURE, "--from", "bodies")
            self.assertEqual(code, 1)
            self.assertIn("needs an HMAC key", err)
        finally:
            if env is not None:
                os.environ[cli.KEY_ENV] = env


class TestFiguresStayWithheldByDefault(unittest.TestCase):
    """The rule the whole project is sold on. A CLI is an easy place for it to
    stop being true, because the convenient default is to print the number."""

    def test_no_invoice_means_no_dollars(self):
        code, out, _err = run("analyze", FIXTURE)
        self.assertEqual(code, 0)
        self.assertIn("FIGURES WITHHELD", out)
        self.assertIn("[figure withheld]", out)
        self.assertNotRegex(out, r"~\$[\d,]+/mo")

    def test_the_draft_flag_is_what_releases_them(self):
        code, out, _err = run("analyze", FIXTURE, "--allow-unreconciled")
        self.assertEqual(code, 0)
        self.assertNotIn("[figure withheld]", out)
        self.assertRegex(out, r"~\$[\d,.]+/mo")

    def test_json_output_serialises_a_withheld_figure_as_text(self):
        """Not as a number. A script reading this must not be handed a float it
        would treat as spend -- `raw()` is deliberately unreachable from here."""
        code, out, _err = run("analyze", FIXTURE, "--format", "json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        for value in payload["spend"].values():
            self.assertIsInstance(value, str)
        self.assertTrue(any("withheld" in v for v in payload["spend"].values()))


class TestTheSubcommandsRun(unittest.TestCase):

    def test_analyze_text(self):
        code, out, _err = run("analyze", FIXTURE, "--allow-unreconciled")
        self.assertEqual(code, 0)
        self.assertIn("ingest tier", out)
        # The header is padded to its longest label; it used to render
        # "prefix efficiency17%" with no gap.
        self.assertNotRegex(out, r"prefix efficiency\d")

    def test_analyze_html_to_a_file(self):
        fd, path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        try:
            code, _out, err = run("analyze", FIXTURE, "--format", "html",
                                  "--out", path, "--allow-unreconciled")
            self.assertEqual(code, 0, err)
            with open(path) as f:
                self.assertIn("<html", f.read().lower())
        finally:
            os.unlink(path)

    def test_bakeoff(self):
        code, out, _err = run("bakeoff", FIXTURE)
        self.assertEqual(code, 0)
        self.assertIn("litellm-auto", out)

    def test_registry_reports_pricing_coverage(self):
        """A surface the registry describes but cannot price is the difference
        between "we support that" and "we can answer a question about it"."""
        code, out, _err = run("registry")
        self.assertEqual(code, 0)
        self.assertIn("anthropic/direct", out)
        self.assertIn("priced_models=", out)

    def test_checks_exit_code_carries_the_verdict(self):
        bad, _o, _e = run("checks", "--prefix-tokens", "300",
                          "--model", "claude-opus-5", "--breakpoints", "5")
        good, _o, _e = run("checks", "--prefix-tokens", "9000",
                           "--model", "claude-opus-5", "--breakpoints", "2")
        self.assertEqual(bad, 2, "a failing configuration must be detectable")
        self.assertEqual(good, 0)

    def test_a_missing_file_is_a_message_not_a_traceback(self):
        code, _out, err = run("analyze", "/definitely/not/here.jsonl")
        self.assertEqual(code, 1)
        self.assertIn("no such file", err)
        self.assertNotIn("Traceback", err)

    def test_an_unknown_model_abstains_rather_than_passing(self):
        """The registry refuses to guess a minimum, because they are
        non-monotonic across generations. That has to reach the exit code: a
        first mapping sent ABSTAIN to 0, so a pipeline got a green build in which
        the check that catches silently-ignored markers never ran."""
        code, out, err = run("checks", "--prefix-tokens", "9000",
                             "--model", "not-a-real-model", "--breakpoints", "1")
        self.assertEqual(code, 3, "abstain must not read as success")
        self.assertIn("ABSTAIN", out)
        self.assertNotIn("Traceback", err)


class TestNothingHereReachesTheNetwork(unittest.TestCase):
    """Local-first with zero egress is the claim the analysis is sold on, and a
    CLI is where it would quietly stop being true.

    Asserted structurally rather than by example: an import is the only way a
    stdlib-only package gets a socket, and the defect appears with the call that
    has not been written yet.
    """

    NETWORK = {"socket", "http", "urllib", "requests", "httpx", "ftplib",
               "smtplib", "telnetlib", "asyncio.streams", "ssl", "xmlrpc"}

    def test_no_module_in_the_analysis_path_imports_a_network_library(self):
        import ast
        import pathlib

        import cacheeconomics
        pkg = pathlib.Path(cacheeconomics.__file__).parent
        offenders = {}
        for path in sorted(pkg.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    if n.split(".")[0] in self.NETWORK:
                        offenders.setdefault(path.name, []).append(n)
        self.assertFalse(
            offenders,
            f"network imports in the package: {offenders}. Analysis is "
            f"local-first with zero egress; that is the claim clients are asked "
            f"to trust when they hand over a trace.")


if __name__ == "__main__":
    unittest.main()
