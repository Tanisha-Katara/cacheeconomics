#!/usr/bin/env python3
"""
Verify that the third-party source analysed in FINDINGS.md still matches the
pinned upstream revision.

Every source is pinned to an immutable commit SHA, not a mutable branch, and
recorded with a full 64-character SHA-256. This script re-fetches each pinned
URL and fails loudly on any mismatch, so FINDINGS.md can never silently drift
away from the code its line numbers refer to.

Exit codes:
    0  every pinned file matches
    1  drift detected, or a file could not be fetched
    2  manifest missing / malformed

Usage:
    python3 verify_sources.py                # verify against local copies
    python3 verify_sources.py --refresh      # re-download pinned content first
    python3 verify_sources.py --write-manifest   # regenerate SOURCES.md
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "sources.json")


def raw_url(entry):
    return (f"https://raw.githubusercontent.com/{entry['repo']}/"
            f"{entry['commit']}/{entry['path']}")


def blob_url(entry):
    return (f"https://github.com/{entry['repo']}/blob/"
            f"{entry['commit']}/{entry['path']}")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "cache-economics-verify"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def load_manifest():
    if not os.path.exists(MANIFEST):
        sys.exit(f"[2] manifest not found: {MANIFEST}")
    try:
        return json.load(open(MANIFEST))
    except Exception as e:
        sys.exit(f"[2] manifest unreadable: {e}")


def write_markdown(entries):
    out = [
        "# Tier A evidence — source provenance",
        "",
        "Third-party source analysed for `FINDINGS.md`. **Not vendored** — each file is",
        "pinned to an immutable upstream commit and a full SHA-256, so every claim and",
        "line number in the findings is recoverable even after upstream `main` moves.",
        "",
        "Verify with:",
        "",
        "```bash",
        "python3 tier-a/verify_sources.py --refresh",
        "```",
        "",
        "| Local file | Upstream (pinned) | SHA-256 |",
        "|---|---|---|",
    ]
    for e in entries:
        out.append(
            f"| `{e['local']}` | [`{e['repo']}@{e['commit'][:9]}` · `{e['path']}`]"
            f"({blob_url(e)}) | `{e['sha256']}` |"
        )
    out += [
        "",
        f"Pinned commits — `{entries[0]['repo']}` @ `{entries[0]['commit']}`" if entries else "",
        "",
        "Fetched and pinned 2026-07-28.",
    ]
    path = os.path.join(HERE, "SOURCES.md")
    open(path, "w").write("\n".join(out) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true",
                    help="also re-fetch each pinned URL and confirm it still "
                         "serves the recorded bytes (requires network)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download pinned content into the local copies "
                         "(implies --remote)")
    ap.add_argument("--allow-offline", action="store_true",
                    help="with --remote, treat an unreachable upstream as a "
                         "skip rather than a failure")
    ap.add_argument("--write-manifest", action="store_true",
                    help="regenerate SOURCES.md from sources.json")
    args = ap.parse_args()

    entries = load_manifest()
    if args.write_manifest:
        write_markdown(entries)
        return

    # Local verification is the default and needs no network. The manifest
    # records a full SHA-256 per file, so a maintainer or a CI job behind a
    # firewall can still confirm that the analysed bytes are the bytes cited
    # in FINDINGS.md. Remote checking answers a different question — whether
    # the pinned commit still serves those bytes — and is opt-in.
    do_remote = args.remote or args.refresh
    allow_offline = args.allow_offline
    failures, remote_skipped = [], 0

    print(f"mode: local hash check{' + remote pin check' if do_remote else ''}"
          f"{'' if do_remote else '   (offline; use --remote to also verify upstream)'}")
    print(f"\n{'local file':<38} {'manifest sha256':<16} {'local':<16} {'remote':<10} result")
    print("-" * 96)

    for e in entries:
        local_path = os.path.join(HERE, e["local"])
        remote_col, remote_bytes = "skipped", None

        if do_remote:
            try:
                remote_bytes = fetch(raw_url(e))
                remote_h = sha256_bytes(remote_bytes)
                if remote_h != e["sha256"]:
                    failures.append((e["local"],
                                     f"pinned URL now serves different bytes "
                                     f"(manifest {e['sha256'][:12]} != fetched "
                                     f"{remote_h[:12]}); a pinned commit should be "
                                     f"immutable, so investigate"))
                    remote_col = "MISMATCH"
                else:
                    remote_col = "match"
            except Exception as ex:
                remote_col = "unreachable"
                remote_skipped += 1
                if allow_offline:
                    print(f"  note: {e['local']}: remote check skipped "
                          f"({type(ex).__name__}), --allow-offline set")
                else:
                    # The caller explicitly asked for a remote check. Exiting 0
                    # here would let CI report "upstream provenance verified"
                    # when nothing upstream was ever reached.
                    failures.append((e["local"],
                                     f"remote check requested but unreachable "
                                     f"({type(ex).__name__}); pass --allow-offline "
                                     f"to treat this as skippable"))

        if args.refresh and remote_bytes is not None:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            open(local_path, "wb").write(remote_bytes)

        if not os.path.exists(local_path):
            failures.append((e["local"],
                             "local copy missing (run with --refresh to fetch it)"))
            print(f"{e['local']:<38} {e['sha256'][:12]:<16} {'missing':<16} "
                  f"{remote_col:<10} MISSING")
            continue

        local_h = sha256_bytes(open(local_path, "rb").read())
        ok = local_h == e["sha256"]
        if not ok:
            failures.append((e["local"],
                             f"local copy differs from the analysed revision "
                             f"({local_h[:12]} != {e['sha256'][:12]}); "
                             f"re-run with --refresh"))
        print(f"{e['local']:<38} {e['sha256'][:12]:<16} {local_h[:12]:<16} "
              f"{remote_col:<10} {'OK' if ok else 'DRIFT'}")

    print("-" * 96)
    if failures:
        print(f"\nFAILED — {len(failures)} problem(s):")
        for name, why in failures:
            print(f"  {name}: {why}")
        print("\nFINDINGS.md line references may no longer be valid.")
        sys.exit(1)

    msg = f"\nOK — all {len(entries)} sources match the analysed revision."
    if not do_remote:
        msg += " Upstream not checked (--remote)."
    elif remote_skipped:
        msg += f" {remote_skipped} remote check(s) unreachable; local evidence still valid."
    print(msg)


if __name__ == "__main__":
    main()
