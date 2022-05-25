#!/usr/bin/env python3
import sys
from subprocess import check_output


MODULES = [
    "EnkfFs",
    "EnkfFsMt",
    "PdHdf5",
    "PdHdf5Open",
    # "XrCdf",
]


def bench(module: str, command: str, *args) -> None:
    print(f"Running {module}, {command} {args}")
    time = check_output([sys.executable, "bench.py", module, command, *args])
    try:
        text = time.decode()
        if text.startswith("skip"):
            text = "SKIP"
        else:
            text = str(float(text))
    except:
        text = "<nan>"

    print(f"\r\033[ARan     {module}, {command} {args}: {text}")


def vary_threads(command: str):
    for module in MODULES:
        for threads in 1, 2, 4, 8:
            bench(module, command, f"--threads={threads}")


def main() -> None:
   # vary_threads("save_parameter")
    vary_threads("save_response")


if __name__ == "__main__":
    main()
