# Short demo recording guide

Record only the synthetic local demo or a clearly labelled staging deployment.
Never place a real token, organization name, source identifier, or provider
account data in the recording.

## Published short films

The public site contains two generated films:

- Recommendations: secure entry, Overview, ranked findings, evidence labels,
  quality risk, and withheld monetary impact.
- Operations: secure entry, request aggregates, a 24-hour window, job state,
  and ingestion health.

Rebuild both from the checked synthetic packet:

```sh
uv run --isolated --no-project --with playwright \
  python demo/record_dashboard_videos.py
```

## Narrated portfolio walkthrough

1. Show the sign-in page and say: “The hosted layer is separate; the installed
   Python package still opens no sockets.”
2. Enter the fixed synthetic token and show the organization/source selectors.
3. On Overview, point to 286 accepted, one duplicate, one rejected event, and
   the visibly withheld money card.
4. On Recommendations, open with `TTL-1`, then show the evidence and quality
   labels on the remaining three findings.
5. On Operations, state that timing inputs are synthetic UI scenario data—not
   service performance—and show the prompt-free aggregate chart.
6. On Jobs & health, show the successful leased job and viewer-only controls.
7. End on the packet provenance and the command that rebuilds it.

## Recording checklist

- Keep `Synthetic Demo Organization` visible.
- Say “synthetic” before quoting any number.
- Do not call the local replay a deployment.
- Do not state savings; the demo intentionally withholds all money.
- Record the commit SHA, packet SHA-256, capture date, and viewport beside the
  final video artifact.
- Review every frame for tokens, browser history, notifications, and unrelated
  account information before publishing.

The generated product films are silent and browser-chrome-free. A narrated
version should follow the same disclosure rules and may use this sequence as
its script.
