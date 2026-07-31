// The analyzer runs here, not on the page.
//
// Python comes from a public CDN, which is third-party code executing with
// whatever privileges its host gives it. On the page that host was a document
// containing a textarea people are invited to paste their own prompt into, and
// a Content-Security-Policy that permits connections back to that same CDN.
// The page's promise about that textarea -- "this runs in your browser and
// sends nothing anywhere" -- was true of our code and unenforceable against
// anyone else's.
//
// A worker has no DOM. Not restricted access, none: no document, no window, no
// way to read an input element. Moving Python here makes the promise structural
// instead of a matter of trusting a CDN, and costs nothing, because the only
// data this ever sees is a sample trace we ship ourselves.
//
// Everything below talks to the page by message. Nothing is imported from here
// into the page and nothing from the page is evaluated here.
//
// The document's own Content-Security-Policy no longer mentions the CDN at all:
// verified in Chrome on 29 Jul 2026 that a `fetch` and a dynamic `import` of the
// CDN from the page are both refused, while this worker still loads Python. That
// works because a same-origin worker script served without CSP headers does not
// inherit the document's policy. It is not guaranteed across engines. If some
// browser does inherit it, the import below fails and the page reports that
// Python could not start -- the feature goes away, which is the right direction
// for a failure of this kind to go.

const CDN = 'https://cdn.jsdelivr.net/pyodide/v314.0.3/full/';

const say = text => self.postMessage({ type: 'status', text });
const fail = (text, detail) => self.postMessage({ type: 'error', text, detail });

const PY_ANALYSE = `
import json, os
from cacheeconomics.trace import load_jsonl
from cacheeconomics.adapters.bodies import load_bodies
from cacheeconomics.analyzer import analyze
from cacheeconomics.report import render_text

# A key that exists only for this run, in this worker, and is discarded with it.
# Segment ids are keyed so a short segment cannot be recovered by dictionary
# attack; generating it here means the ids mean nothing outside this page.
_KEY = os.urandom(32)

def _has_body(path, scan=500):
    """Does any row carry a request body a gateway logged?"""
    seen = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if seen > scan:
                break
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            for k in ('request', 'request_body', 'input', 'body', 'kwargs'):
                v = row.get(k)
                if isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except ValueError:
                        v = None
                if isinstance(v, dict) and ('messages' in v or 'system' in v):
                    return True
            if ('messages' in row) or ('system' in row):
                return True
    return False

try:
    ts = (load_bodies('/trace.jsonl', key=_KEY) if _has_body('/trace.jsonl')
          else load_jsonl('/trace.jsonl', key=_KEY))
except ValueError as e:
    result = "This trace could not be read safely:\\n\\n" + str(e)
except Exception as e:
    result = "This file did not parse as a trace:\\n\\n" + type(e).__name__ + ": " + str(e)
else:
    # No allow_unreconciled. That flag is the internal draft escape hatch, and
    # using it here would render released dollar figures with no invoice behind
    # them -- bypassing the exact gate this page argues for.
    result = render_text(analyze(ts))
result
`;

async function run() {
  say('fetching Python…');
  const { loadPyodide } = await import(CDN + 'pyodide.mjs');
  const pyodide = await loadPyodide({ indexURL: CDN });

  say('installing the analyzer…');
  const { BUNDLE, BUNDLE_DIGEST } = await import('./harness-bundle.js');
  // Pyodide's in-memory filesystem. Nothing touches anybody's disk.
  // Directories come from the paths themselves rather than a hard-coded list:
  // hard-coding them meant adding the adapters package silently broke the load
  // with a non-Error nobody could read.
  const mk = d => { try { pyodide.FS.mkdir(d); } catch (e) { /* exists */ } };
  mk('/harness');
  for (const path of Object.keys(BUNDLE)) {
    const parts = ('/harness/' + path).split('/');
    for (let i = 2; i < parts.length; i++) mk(parts.slice(0, i).join('/'));
  }
  for (const [path, src] of Object.entries(BUNDLE)) {
    pyodide.FS.writeFile('/harness/' + path, src);
  }
  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, '/harness')
import cacheeconomics.analyzer  # fail loudly here rather than on first use
`);
  self.postMessage({ type: 'ready', build: BUNDLE_DIGEST.slice(0, 12) });

  say('reading the sample trace…');
  const text = await (await fetch('./sample-trace.jsonl')).text();
  pyodide.FS.writeFile('/trace.jsonl', text);
  say('analysing ' + (text.length / 1024).toFixed(0) + ' KB…');
  const report = await pyodide.runPythonAsync(PY_ANALYSE);
  self.postMessage({ type: 'report', report });
}

self.onmessage = event => {
  if (event.data !== 'start') return;
  run().catch(err => fail('Could not start Python.',
                          (err && (err.message || String(err))) +
                          '\n\nThis needs WebAssembly and a reachable CDN.'));
};
