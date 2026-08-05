# Contributing

Thanks for taking the time to improve cacheeconomics. The project is designed
to be conservative: it should abstain when provider facts, prices, or trace
evidence are missing.

## Development Setup

```bash
git clone https://github.com/Tanisha-Katara/cacheeconomics.git
cd cacheeconomics
pip install .
python3 -m pytest -q
```

The installed package intentionally has no runtime dependencies. Tests require
`pytest`; the live LiteLLM plugin is optional and uses the `litellm` extra.

## Before Opening A Pull Request

- Run `python3 -m pytest -q`.
- If you changed anything under `harness/cacheeconomics`, run
  `python3 web/build_bundle.py` and include any resulting
  `web/harness-bundle.js` change.
- Keep registry changes sourced. New provider facts need dated provenance and
  tests. Do not add plausible values to silence an abstention.
- Keep private traces, counted exports, key files, and generated prompt-body
  artifacts out of git.

## Registry Changes

Provider behavior is part of the product contract. A registry row should say
what surface it describes, where the fact came from, when it was checked, and
whether the row is contested. If the source is missing or ambiguous, prefer an
abstention and a clear `contested_reason`.

## Privacy Expectations

The package should remain local-first and socket-free. Scripts under `tier-b/`
may perform explicit network work for measurement or token counting, but that
egress must stay visible in the command and documentation.
