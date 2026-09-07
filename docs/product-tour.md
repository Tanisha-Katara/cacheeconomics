# Product tour

This tour uses the checked-in **synthetic, read-only portfolio demo**. It is not
a production deployment and contains no customer traffic. Start it with:

```sh
python demo/serve_portfolio_demo.py
```

Open the printed loopback URL and enter the printed synthetic token. The token
is a fixed demo string, not a credential for any external system.

## 1. Sign-in boundary

The first screen explains the boundary before asking for a token: the hosted
application receives prompt-free operational metadata, while the installed
Python package remains local and socket-free. The deployable dashboard uses
OIDC Authorization Code with PKCE when configured; the token field in this
demo exists only in development mode.

## 2. Organization and source context

The left rail shows `Synthetic Demo Organization` and
`Synthetic Normalized Trace`. Every ordinary API request carries both the
in-memory identity token and selected organization ID. The packet was produced
through the real API against portable SQLite. In a PostgreSQL deployment, the
API and row policies—not the browser—enforce access.

## 3. Overview

The overview connects three questions:

- Is ingestion healthy?
- How much of the input came from cache?
- Are monetary figures safe to show?

The demo has 286 accepted events, one idempotent duplicate, and one rejected
prompt-bearing event. No invoice is supplied, so every calculated dollar value
remains visibly withheld and has no numeric amount in the response packet.

## 4. Recommendations

The recommendation view preserves the analyzer's order and evidence labels.
The fixture returns four findings: `TTL-1`, `EFF-1`, `CAC-1`, and `MIN-1`.
Each card carries the concrete action, confidence, evidence class, affected
request count, and quality risk. The browser never calculates savings.

## 5. Operations

The operations view shows bounded, prompt-free aggregates: request volume,
outcomes, latency, time to first token, and normalized error categories. The
demo timing values are deliberately assigned synthetic scenario inputs to
exercise the charts; they are not API performance measurements or objectives.

## 6. Jobs and health

The final view links the successful leased worker job to source health,
attempt count, duplicate/rejection counters, and the most recent analysis.
The replay is intentionally read-only, so it demonstrates viewer access and
does not pretend to execute mutations against a live deployment.

## Screenshot status

The repository includes 1280×720 Recommendations and Operations posters plus
the corresponding short WebM tours. They were generated from the real dashboard
and checked packet on 7 September 2026. `apps/dashboard/media/manifest.json`
records the packet and artifact hashes, viewport, view, and synthetic label.

The public hero contains a compact illustrative preview built in HTML and CSS.
The labelled film posters and videos are the reproducible evidence captures.
