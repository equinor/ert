#!/usr/bin/env python3
import sys
from subprocess import check_output
import hashlib


MODULES = [
#    "EnkfFs",
    "EnkfFsMt",
#   "ErtStorage",
#   "PdHdf5",
    "PdHdf5Open",
#   "Sqlite",
#   "XArrayNetCDF",
]


def bench(
    module: str,
    command: str,
    *,
    ensemble_size: int = 100,
    keys: int = 10,
    threads: int = 1,
    use_async: bool = False,
) -> None:
    p_args = ", ".join(f"{k}={v}" for k, v in locals().items() if k not in ("module", "command"))
    print(f"Running ({module} {command} :: {p_args})")

    extra_args = []
    if use_async:
        extra_args.append("--use-async")

    time = check_output(
        [
            sys.executable,
            "bench.py",
            module,
            command,
            "--ensemble-size",
            str(ensemble_size),
            "--keys",
            str(keys),
            "--threads",
            str(threads),
            "--trials",
            "1",
            "--suffix",
            hashlib.md5(p_args.encode()).hexdigest(),
            *extra_args,
        ]
    )
    try:
        text = time.decode()
        if text.startswith("skip"):
            text = "SKIP"
        else:
            text = str(float(text))
    except:
        text = "<nan>"

    print(f"\r\033[ARan     ({module} {command} :: {p_args}): {text}")


def vary_threads(command: str) -> None:
    for module in ("EnkfFs", "EnkfFsMt", "PdHdf5", "PdHdf5Open", "XArrayNetCDF"):
        for threads in 1, 2, 4, 8:
            bench(module, command, threads=threads)


def vary_keys(command: str) -> None:
    for module in MODULES:
        for keys in 1, 5, 25:
            bench(module, command, keys=keys)


def main() -> None:
    # vary_threads("save_parameter")
    # vary_threads("load_response")
    # vary_threads("save_response")
    # vary_threads("load_response")

    # vary_keys("save_response")
    # vary_keys("load_response")
    # bench("EnkfFsMt", "save_response", keys=25)
    bench("EnkfFsMt", "load_response", keys=25)


if __name__ == "__main__":
    main()
