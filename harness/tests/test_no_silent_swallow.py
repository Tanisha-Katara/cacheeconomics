"""Broad exception handlers, enumerated and justified.

Three review rounds produced the same defect: a guard that catches `Exception`
on a path where the thing worth catching is not one. `asyncio.CancelledError`,
`KeyboardInterrupt` and `SystemExit` all inherit from `BaseException`, and every
one of them means a request was abandoned mid-flight -- which is precisely the
traffic the recorder exists to keep. Catching `Exception` there drops the
degraded calls and flatters every ratio built on what is left.

Fixed in round 9 for two sites, round 12 for a third, and a fourth was still
there afterwards. Finding them one at a time by review is the process this file
replaces: it enumerates every broad handler in the package and fails unless the
handler is on a list somebody wrote a reason next to.

Adding a new `except Exception` is not forbidden. Adding one silently is.
"""

import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "cacheeconomics")

# (module, enclosing function) -> why catching Exception is right here.
#
# The bar: the handler is not on a path that records whether a request
# happened, and the exceptions that are not Exceptions would be wrong to
# swallow.
JUSTIFIED = {
    ("plugin.py", "_custom_logger_base"):
        "optional third-party import. ImportError is an Exception, and a "
        "KeyboardInterrupt during import should propagate rather than silently "
        "fall back to a stub base class.",
}


def _broad_handlers():
    """Every `except Exception` and bare `except:` in the package."""
    out = []
    for root, _, files in os.walk(PKG):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, PKG)
            with open(path) as f:
                tree = ast.parse(f.read(), filename=rel)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for inner in ast.walk(node):
                    if not isinstance(inner, ast.ExceptHandler):
                        continue
                    kind = inner.type
                    broad = kind is None or (
                        isinstance(kind, ast.Name) and kind.id == "Exception")
                    if broad:
                        out.append((rel, node.name, inner.lineno))
    return out


class TestEveryBroadHandlerIsJustified(unittest.TestCase):

    def test_no_unjustified_broad_handler(self):
        unjustified = [(m, fn, line) for m, fn, line in _broad_handlers()
                       if (m, fn) not in JUSTIFIED]
        self.assertEqual(
            unjustified, [],
            "broad exception handler with no recorded justification. If it is "
            "on a path that records whether a request happened, it should catch "
            "BaseException -- a cancelled or interrupted call is exactly the "
            "traffic worth keeping. If it is genuinely fine, add it to "
            "JUSTIFIED with the reason.")

    def test_the_justification_list_has_no_stale_entries(self):
        """A justified handler that no longer exists is a comment pretending to
        be a control."""
        live = {(m, fn) for m, fn, _ in _broad_handlers()}
        self.assertEqual([k for k in JUSTIFIED if k not in live], [])

    def test_the_recording_paths_all_catch_baseexception(self):
        """The specific class of site that produced three findings.

        Structure, not substring: a first attempt grepped the source text and
        failed on a docstring that *discusses* the pattern, which this codebase
        does on purpose. A test that matches prose measures prose.
        """
        offenders = [(fn, line) for m, fn, line in _broad_handlers()
                     if m == "recorder.py"]
        self.assertEqual(
            offenders, [],
            "recorder.py records whether a request happened, so every guard on "
            "that path must catch BaseException -- a cancelled or interrupted "
            "call is the traffic most worth keeping")


if __name__ == "__main__":
    unittest.main()
