#!/usr/bin/env python
import shutil
import sys
from pathlib import Path


def move_file(src_file: str | Path, target: str | Path) -> None:
    """Will raise IOError if src_file is not a file."""
    src_file = Path(src_file)
    target = Path(target)
    if src_file.is_file():
        # shutil.move works best (as unix mv) when target is a file.
        if target.is_dir():
            target /= src_file.name
        shutil.move(src_file, target)
    else:
        raise OSError(f"Input argument {src_file} is not an existing file")


if __name__ == "__main__":
    src = sys.argv[1]
    target = sys.argv[2]
    try:
        move_file(src, target)
    except OSError as e:
        sys.exit(f"MOVE_FILE failed with the following error: {e}")
