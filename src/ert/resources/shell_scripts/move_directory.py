#!/usr/bin/env python
import shutil
import sys
from pathlib import Path


def move_directory(src_dir: str, target: str) -> None:
    """Will raise IOError if src_dir is not a folder."""
    if Path(src_dir).is_dir():
        if Path(target).exists():
            shutil.rmtree(target)
        shutil.move(src_dir, target)
    else:
        raise OSError(f"Input argument {src_dir} is not an existing directory")


if __name__ == "__main__":
    src = sys.argv[1]
    target = sys.argv[2]
    try:
        move_directory(src, target)
    except OSError as e:
        sys.exit(f"MOVE_DIRECTORY failed with the following error: {e}")
