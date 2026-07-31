# Tier A evidence — source provenance

Third-party source analysed for `FINDINGS.md`. **Not vendored** — each file is
pinned to an immutable upstream commit and a full SHA-256, so every claim and
line number in the findings is recoverable even after upstream `main` moves.

Verify with:

```bash
python3 tier-a/verify_sources.py --refresh
```

| Local file | Upstream (pinned) | SHA-256 |
|---|---|---|
| `openhands/llm_message.py` | [`OpenHands/software-agent-sdk@1f9f0b1aa` · `openhands-sdk/openhands/sdk/llm/message.py`](https://github.com/OpenHands/software-agent-sdk/blob/1f9f0b1aa0356e082d971e8a5cf82256d67fe576/openhands-sdk/openhands/sdk/llm/message.py) | `aa18869b482f2eea942c4b444ab84ff7aaf6434dd043cfbea68b406fd435c457` |
| `openhands/llm_llm.py` | [`OpenHands/software-agent-sdk@1f9f0b1aa` · `openhands-sdk/openhands/sdk/llm/llm.py`](https://github.com/OpenHands/software-agent-sdk/blob/1f9f0b1aa0356e082d971e8a5cf82256d67fe576/openhands-sdk/openhands/sdk/llm/llm.py) | `45f483ab6c3277fb842fb8b0f2e63c6c3806d70fba96ef483f629e438fa45773` |
| `openhands/utils_model_features.py` | [`OpenHands/software-agent-sdk@1f9f0b1aa` · `openhands-sdk/openhands/sdk/llm/utils/model_features.py`](https://github.com/OpenHands/software-agent-sdk/blob/1f9f0b1aa0356e082d971e8a5cf82256d67fe576/openhands-sdk/openhands/sdk/llm/utils/model_features.py) | `802bd6befb124da7c10b7baae2f847addac0ed7d8bfd04250895bbec1c298afa` |
| `openhands/llm_convertible_system.py` | [`OpenHands/software-agent-sdk@1f9f0b1aa` · `openhands-sdk/openhands/sdk/event/llm_convertible/system.py`](https://github.com/OpenHands/software-agent-sdk/blob/1f9f0b1aa0356e082d971e8a5cf82256d67fe576/openhands-sdk/openhands/sdk/event/llm_convertible/system.py) | `d28a1c618e3e0cd659cb68551e5b593f0d49d40f08e2b8975bce2f3c1b34d44d` |
| `swe-agent/models.py` | [`SWE-agent/SWE-agent@3ea751c08` · `sweagent/agent/models.py`](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/models.py) | `91a8cb62703d7db656b0e615811ff0b9eedd51cd3ea89a6eeb2393f1d35c6cce` |
| `swe-agent/history_processors.py` | [`SWE-agent/SWE-agent@3ea751c08` · `sweagent/agent/history_processors.py`](https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/history_processors.py) | `8e3621a9b7d94761d184c57dc2ab1028f7eb04b01fdfcbe8ba122977d19e477a` |
| `browser-use/chat.py` | [`browser-use/browser-use@f0aa3a8bb` · `browser_use/llm/anthropic/chat.py`](https://github.com/browser-use/browser-use/blob/f0aa3a8bb03779c71a5aa262d389e3bfe6b77cdc/browser_use/llm/anthropic/chat.py) | `cdd9d9b91e2d2865e3d3eae857142da1510a60f936cb579a2846349a362adb6d` |
| `browser-use/serializer.py` | [`browser-use/browser-use@f0aa3a8bb` · `browser_use/llm/anthropic/serializer.py`](https://github.com/browser-use/browser-use/blob/f0aa3a8bb03779c71a5aa262d389e3bfe6b77cdc/browser_use/llm/anthropic/serializer.py) | `50b0621b4bbf245b977a5fe378f6bc3e75011f143e319fcef65cf2138002ee61` |

Pinned commits — `OpenHands/software-agent-sdk` @ `1f9f0b1aa0356e082d971e8a5cf82256d67fe576`

Fetched and pinned 2026-07-28.
