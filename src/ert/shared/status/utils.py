import os
import resource
import sys
from pathlib import Path

import pandas as pd


def byte_with_unit(byte_count: float) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    power = float(10**3)

    i = 0
    while byte_count >= power and i < len(suffixes) - 1:
        byte_count /= power
        i += 1

    return f"{byte_count:.2f} {suffixes[i]}"


def file_has_content(file_path: str) -> bool:
    file_path_exists = os.path.isfile(str(file_path))
    if file_path_exists:
        return os.path.getsize(file_path) > 0
    return False


def get_ert_memory_usage() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_scale = 1
    if sys.platform == "darwin":
        # macOS apparently outputs the maxrss value as bytes
        # rather than kilobytes as on Linux.
        # https://stackoverflow.com/questions/59913657/strange-values-of-get-rusage-maxrss-on-macos-and-linux
        rss_scale = 1000

    return usage.ru_maxrss // rss_scale


def get_mount_directory(runpath: Path) -> Path:
    path = Path(runpath).absolute()

    while not path.is_mount():
        path = path.parent

    return path


def convert_to_numeric(x: str | pd.Series) -> str | float | pd.Series:
    try:
        return pd.to_numeric(x)
    except ValueError:
        return x
