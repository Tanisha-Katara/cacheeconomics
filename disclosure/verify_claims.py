"""Check every code claim in the disclosure drafts against upstream `main`.

All four issues were filed on 2026-07-28 and are recorded at the top of each
draft. Do not file again -- run `--filed` to see their current state. This script
now serves the follow-up, not the filing: a maintainer who looks at a
three-week-old issue and finds drifted line numbers stops reading.

Each draft quotes file paths and line numbers, and all three repositories are
actively developed. A claim whose line numbers have drifted is worse than none:
the maintainer looks, sees something else there, and stops reading.

    python3 disclosure/verify_claims.py

Exits non-zero if any claim no longer holds. Network only; reads nothing local
except the claim table below.
"""

import sys
import urllib.error
import urllib.request

RAW = "https://raw.githubusercontent.com/{repo}/main/{path}"

# (draft, repo, path, [(line, substring that must appear on it)], [absent substrings])
CLAIMS = [
    (
        "1-openhands",
        "OpenHands/software-agent-sdk",
        "openhands-sdk/openhands/sdk/llm/message.py",
        [(200, '"cache_control"'), (214, '"cache_control"'), (384, '"cache_control"')],
        ['"ttl"'],
    ),
    (
        "2b-swe-agent",
        "SWE-agent/SWE-agent",
        "sweagent/agent/history_processors.py",
        [(59, '"cache_control"'), (63, "cache_control"), (67, "cache_control")],
        ['"ttl"'],
    ),
    (
        "2a-swe-agent",
        "SWE-agent/SWE-agent",
        "sweagent/agent/models.py",
        [],
        # The claim is an absence: this file never reads the response's usage,
        # so cache reads and writes are invisible to its cost accounting.
        ["response.usage", "cache_read_input", "cache_creation_input",
         "prompt_tokens_details"],
    ),
    (
        "4-aider",
        "Aider-AI/aider",
        "aider/coders/base_coder.py",
        [(2092, "prompt_tokens - input_cost_per_token_cache_hit"),
         (2095, "cache_write_tokens * input_cost_per_token * 1.25"),
         (2097, "cost += prompt_tokens * input_cost_per_token")],
        [],
    ),
    (
        "5-crewai",
        "crewAIInc/crewAI",
        "lib/crewai/src/crewai/llms/providers/anthropic/completion.py",
        [(1977, "input_tokens + output_tokens")],
        [],
    ),
    (
        "3-browser-use",
        "browser-use/browser-use",
        "browser_use/llm/anthropic/chat.py",
        # Filed against 195/196 on 2026-07-28; upstream moved them to 200/201
        # by 2026-08-03. The claim is unchanged -- the file still reads both
        # counters and still never emits a 1h marker -- but the issue text
        # points at line numbers that are now wrong, which is exactly the
        # "maintainer looks, sees something else, stops reading" failure this
        # script exists to catch. Worth a correcting comment on #5321.
        [(200, "ephemeral_5m_input_tokens"), (201, "ephemeral_1h_input_tokens")],
        [],
    ),
]


def fetch(repo, path):
    try:
        with urllib.request.urlopen(RAW.format(repo=repo, path=path),
                                    timeout=60) as r:
            return r.read().decode("utf-8", "replace").splitlines()
    except urllib.error.HTTPError as e:
        return None if e.code == 404 else []


FILED = [
    ("1-openhands", "OpenHands/software-agent-sdk", 4292),
    ("2a-swe-agent", "SWE-agent/SWE-agent", 1481),
    ("2b-swe-agent", "SWE-agent/SWE-agent", 1482),
    ("3-browser-use", "browser-use/browser-use", 5321),
]


def filed_state():
    """Where each disclosure went, and whether anyone has replied.

    Exists because the drafts did not record this and their absence was read as
    "not filed yet" -- twice, over several days, until a duplicate check caught
    it one command before four duplicate issues went out.
    """
    import subprocess
    for draft, repo, number in FILED:
        try:
            out = subprocess.run(
                ["gh", "issue", "view", str(number), "--repo", repo, "--json",
                 "url,state,comments",
                 "--jq", '"[\(.state)] comments=\(.comments|length) \(.url)"'],
                capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:                                   # noqa: BLE001
            out = f"could not read: {e}"
        print(f"  {draft:<16} {out or 'no response from gh'}")
    return 0


def main():
    if "--filed" in sys.argv:
        return filed_state()
    ok = True
    for draft, repo, path, lines, absent in CLAIMS:
        print(f"\n{draft}  ({repo}/{path})")
        src = fetch(repo, path)
        if src is None:
            print("   FILE IS GONE — the code moved or was renamed. Do not file.")
            ok = False
            continue
        if not src:
            print("   could not fetch; check the network and re-run")
            ok = False
            continue
        print(f"   {len(src)} lines")

        for lineno, needle in lines:
            got = src[lineno - 1] if 0 < lineno <= len(src) else ""
            hit = needle in got
            ok = ok and hit
            print(f"   line {lineno:<4} {'OK  ' if hit else 'DRIFTED'} "
                  f"expected {needle!r}")
            if not hit:
                # Where it went, so the draft can be corrected rather than binned.
                moved = [i + 1 for i, l in enumerate(src) if needle in l]
                print(f"              now at {moved[:6] or 'nowhere in this file'}")

        for needle in absent:
            hits = [i + 1 for i, l in enumerate(src) if needle in l]
            if lines:
                # An "absent" alongside line claims means the *shape* is absent:
                # e.g. no `ttl` key anywhere, which is the whole finding.
                label = f"{needle!r} still absent"
            else:
                label = f"{needle!r} still unread"
            if hits:
                ok = False
                print(f"   {'FIXED?':<8} {label} — now appears at {hits[:6]}")
            else:
                print(f"   {'OK':<8} {label}")

    print("\nAll claims still hold. Safe to file." if ok else
          "\nSomething changed. Re-read the draft before filing.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
