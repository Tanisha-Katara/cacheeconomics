# cacheeconomics collector

This is the network-enabled edge application. It is deliberately separate from
the installed `cacheeconomics` analysis package, which continues to open no
sockets.

The implemented integration is a LiteLLM proxy callback built on the existing
`cacheeconomics.plugin.litellm_handler`. It observes by default and uploads only
timestamps, token counters, normalized status, model/surface labels, and locally
keyed segment fingerprints. Raw prompts, completions, messages, tool calls, and
raw error messages are rejected before they can enter its SQLite outbox. This
is a complete field allow-list, not only a list of common sensitive names.

The outbox makes delivery durable. Accepted and duplicate events are removed;
temporary failures use bounded exponential retry; permanent rejections and
exhausted deliveries move to a local dead-letter table for operator review.
Authenticated uploads do not follow HTTP redirects, preventing a collector
credential from being forwarded to a different origin. Configure the final
HTTPS endpoint directly.

Existing LiteLLM JSONL logs and normalized cacheeconomics traces can also be
uploaded explicitly:

```console
cacheeconomics-collector trace.jsonl \
  --format litellm \
  --endpoint https://cacheeconomics.example.com
```

For normalized traces whose segments do not already carry keyed `hmac:` ids,
pass a local hexadecimal key through `--hmac-key-file`. The key must contain at
least 16 bytes (32 hexadecimal characters); a newly generated 32-byte key is
recommended. If a provider surface is absent, pass `--target-id` or the
analysis will correctly withhold provider-priced figures rather than guess.
Use `CACHEECONOMICS_COLLECTOR_TOKEN` or `--token-file`; putting a credential
directly on a command line can expose it in process listings. A scheduler may
invoke this one-shot command repeatedly; source/event idempotency makes replay
safe.

For a live LiteLLM proxy, expose a callback object from a small local module:

```python
import os

from cacheeconomics.plugin import CachePlugin
from cacheeconomics_collector.client import DurableUploader, HttpTransport
from cacheeconomics_collector.litellm import live_litellm_handler

uploader = DurableUploader(
    "/var/lib/cacheeconomics/outbox.sqlite3",
    HttpTransport(
        os.environ["CACHEECONOMICS_ENDPOINT"],
        os.environ["CACHEECONOMICS_COLLECTOR_TOKEN"],
    ),
)
handler = live_litellm_handler(
    CachePlugin(key=bytes.fromhex(os.environ["CACHEECONOMICS_HMAC_KEY"])),
    uploader,
    mutate=False,
    target_id=os.environ["CACHEECONOMICS_TARGET_ID"],
)
```

`mutate=False` observes only. Changing it to `True` opts into the existing
marker-placement behavior and should be evaluated against the workload first.
