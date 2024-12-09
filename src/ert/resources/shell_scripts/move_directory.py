#!/usr/bin/env python
import os
import shutil
import sys


def move_directory(src_dir: str, target: str) -> None:
    """
    Will raise IOError if src_dir is not a folder.

    """
    if os.path.isdir(src_dir):
        if os.path.exists(target):
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
