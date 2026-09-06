"""Upload an existing local trace through the separate networked collector."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cacheeconomics.adapters.litellm import load_litellm
from cacheeconomics.cli import MIN_KEY_BYTES
from cacheeconomics.trace import load_jsonl

from .client import DurableUploader, HttpTransport
from .events import events_from_requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload prompt-free cacheeconomics events")
    parser.add_argument("path", help="local JSONL trace")
    parser.add_argument("--format", choices=("litellm", "normalized_trace"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token", default=os.environ.get("CACHEECONOMICS_COLLECTOR_TOKEN"))
    parser.add_argument("--token-file")
    parser.add_argument("--outbox", default=".cacheeconomics-collector.sqlite3")
    parser.add_argument("--hmac-key-file")
    parser.add_argument("--target-id")
    args = parser.parse_args(argv)
    if args.token_file:
        if args.token:
            parser.error("use only one of --token, --token-file, or the token environment variable")
        try:
            args.token = Path(args.token_file).read_text().strip()
        except OSError:
            parser.error("could not read --token-file")
    if not args.token:
        parser.error("a token file or CACHEECONOMICS_COLLECTOR_TOKEN is required")

    if args.format == "litellm":
        trace = load_litellm(args.path, default_target=args.target_id)
    else:
        key = None
        if args.hmac_key_file:
            try:
                key = bytes.fromhex(Path(args.hmac_key_file).read_text().strip())
            except (OSError, ValueError):
                parser.error("--hmac-key-file must contain a hexadecimal key")
            if len(key) < MIN_KEY_BYTES:
                parser.error(
                    "--hmac-key-file must contain at least "
                    f"{MIN_KEY_BYTES} bytes ({MIN_KEY_BYTES * 2} hexadecimal characters)"
                )
        trace = load_jsonl(args.path, key=key, default_target=args.target_id or "unknown/unattributed")

    uploader = DurableUploader(
        Path(args.outbox),
        HttpTransport(args.endpoint, args.token),
    )
    skipped = 0
    for request in trace.requests:
        try:
            event = events_from_requests([request], source_type=args.format)[0]
        except ValueError:
            skipped += 1
            continue
        uploader.enqueue(event)
    totals = {"sent": 0, "retried": 0, "dead_lettered": 0}
    while uploader.counts()["pending"]:
        result = uploader.flush()
        for key in totals:
            totals[key] += result[key]
        if result["sent"] == 0:
            break
    counts = uploader.counts()
    print(
        f"sent={totals['sent']} pending={counts['pending']} "
        f"dead_letters={counts['dead_letters']} skipped={skipped}"
    )
    if counts["dead_letters"]:
        return 2
    return 1 if counts["pending"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
