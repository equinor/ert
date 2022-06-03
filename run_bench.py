#!/usr/bin/env python3
from datetime import datetime
from subprocess import check_output
from typing import Generator, Union, List, Dict
import hashlib
import pandas as pd
import scipy.stats
import sys


TRIALS = 5


MODULES = [
#    "EnkfFs",
    "EnkfFsMt",
#   "ErtStorage",
#   "PdHdf5",
    "PdHdf5Open",
#   "Sqlite",
#   "XArrayNetCDF",
]

RESULTS: List[Dict[str, Union[str, float, int, bool]]] = []


def bench(
    module: str,
    command: str,
    *,
    ensemble_size: int = 100,
    keys: int = 10,
    key_size: int = 100,
    threads: int = 1,
    use_async: bool = False,
) -> None:
    p_args = ", ".join(f"{k}={v}" for k, v in locals().items() if k not in ("module", "command"))
    print(f"Running ({module} {command} :: {p_args})")

    extra_args = []
    if use_async:
        extra_args.append("--use-async")

    output = check_output(
        [
            sys.executable,
            "bench.py",
            module,
            command,
            "--ensemble-size",
            str(ensemble_size),
            "--keys",
            str(keys),
            "--key-size",
            str(key_size),
            "--threads",
            str(threads),
            "--trials",
            str(TRIALS),
            "--suffix",
            hashlib.md5(p_args.encode()).hexdigest(),
            *extra_args,
        ]
    )

    time_mean = 0.0
    time_std = -1.0
    try:
        if output.startswith(b"skip"):
            text = "SKIP"
        else:
            times = [float(line) for line in output.decode().splitlines()]
            time_mean, time_std = scipy.stats.norm.fit(times)
            text = f"μ={time_mean}, σ={time_std}"
    except:
        text = "<nan>"
        raise

    print(f"\r\033[ARan     ({module} {command} :: {p_args}): {text}")

    if time_std >= 0.0:
        RESULTS.append({
            "module": module,
            "command": command,
            "ensemble_size": ensemble_size,
            "key_size": key_size,
            "keys": keys,
            "threads": threads,
            "use_async": use_async,
            "trials": TRIALS,
            "time_mean": time_mean,
            "time_std": time_std,
        })


def vary_threads(command: str) -> None:
    for module in ("EnkfFs", "EnkfFsMt", "PdHdf5", "PdHdf5Open", "XArrayNetCDF"):
        for threads in 1, 2, 4, 8:
            bench(module, command, threads=threads)


def vary_keys(command: str) -> None:
    for module in MODULES:
        for keys in 1, 5, 10, 15, 20, 25:
            bench(module, command, keys=keys)


def main() -> None:
    # vary_threads("save_parameter")
    # vary_threads("load_response")
    # vary_threads("save_response")
    # vary_threads("load_response")

    vary_keys("save_response")
    vary_keys("load_response")
    # bench("EnkfFsMt", "save_response", keys=25)
    # bench("EnkfFsMt", "load_response", keys=25)

    pd.DataFrame(RESULTS).to_csv(f"bench_results-{datetime.now().isoformat()}.csv")


if __name__ == "__main__":
    main()
