# Data contracts

These schemas are the narrow bridge between the local `cacheeconomics` engine
and a separate hosted application.

- `ingest-event-v1.schema.json` accepts token counts, timestamps, normalized
  status, and keyed segment fingerprints. It has no field for prompts,
  completions, messages, tool arguments, or raw error text.
- `ingest-batch-v1.schema.json` is the API envelope. The deployment applies a
  configurable event-count and byte-size bound before processing it.
- `analysis-result-v1.schema.json` describes complete analyzer output for a
  dashboard. Withheld monetary figures have `amount_usd: null`; the service must
  never recover their internal value.

The ingest request body does not choose its organization. The future service
will get organization and source identity from the authenticated collector
credential, then attach both on the server. This prevents one customer from
writing data into another customer's organization by changing JSON.

Schema changes that remove or reinterpret a field require a new version. New
optional fields may be added compatibly after both collector and service tests
cover them.
