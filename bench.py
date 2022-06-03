#!/usr/bin/env python3
import argparse
from typing import Any, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from _bench import MODULES, TESTS, Namespace


def parse_args() -> Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("module", choices=MODULES)
    ap.add_argument("command", choices=TESTS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--use-async", default=False, action="store_true")
    ap.add_argument("--keys", type=int, default=10)
    ap.add_argument("--key-size", type=int, default=100)
    ap.add_argument("--ensemble-size", type=int, default=100)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--suffix", type=str, default="")

    namespace = Namespace()
    ap.parse_args(namespace=namespace)
    return namespace


def main() -> None:
    args = parse_args()

    keep = args.command.startswith("load")
    storage = MODULES[args.module](args, keep)

    kwargs: Dict[str, Any] = {}
    if args.threads > 1:
        command = f"test_{args.command}_mt"
        if storage.__use_threads__:
            kwargs["executor"] = ThreadPoolExecutor(max_workers=args.threads)
        else:
            kwargs["executor"] = ProcessPoolExecutor(max_workers=args.threads)
    elif args.use_async:
        command = f"test_{args.command}_async"
    else:
        command = f"test_{args.command}"

    getattr(storage, command)(**kwargs)


if __name__ == "__main__":
    main()
