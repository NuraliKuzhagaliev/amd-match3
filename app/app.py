#!/usr/bin/env python3
"""The exact evaluator entrypoint. The model lives in mc3.worker."""
from pathlib import Path
import argparse
import logging
import re
import sys

from mc3.config import Config
from mc3.io import atomic_json, request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--query-id")
    parser.add_argument("--query")
    args = parser.parse_args()
    cfg = Config()
    if args.index is not None:
        if any(v is not None for v in (args.corpus, args.query_id, args.query)):
            parser.error("--index cannot be combined with query arguments")
        request(cfg.socket, {"op": "index", "corpus": str(args.index.resolve())},
                cfg.startup_seconds, wait_ready=True)
        return 0
    if args.corpus is None or args.query is None or not args.query_id:
        parser.error("Specify --index OR --corpus, --query-id and --query")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", args.query_id):
        parser.error("--query-id must be a safe filename stem")
    out = cfg.output / f"{args.query_id}_output.json"
    # Remove an output for this ID left by an earlier failed invocation.
    out.unlink(missing_ok=True)
    try:
        result = request(cfg.socket, {
            "op": "answer", "corpus": str(args.corpus.resolve()), "query": args.query,
        }, cfg.query_seconds + 1)
        if (not isinstance(result.get("answer"), str)
                or not isinstance(result.get("citations"), list)
                or not all(isinstance(x, str) for x in result["citations"])):
            raise ValueError("Worker returned an invalid response")
        atomic_json(out, result)
        return 0
    except Exception:
        # Infrastructure failure is a failed invocation, never a claimed refusal.
        logging.exception("Question failed; no valid output was produced")
        return 1


if __name__ == "__main__":
    sys.exit(main())
