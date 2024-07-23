#!/usr/bin/env python
import os
import shutil
import sys


def move_file(src_file, target):
    """
    Will raise IOError if src_file is not a file.

    """
    if os.path.isfile(src_file):
        # shutil.move works best (as unix mv) when target is a file.
        if os.path.isdir(target):
            target = os.path.join(target, os.path.basename(src_file))
        shutil.move(src_file, target)
    else:
        raise IOError(f"Input argument {src_file} is not an existing file")


if __name__ == "__main__":
    src = sys.argv[1]
    target = sys.argv[2]
    try:
        move_file(src, target)
    except IOError as e:
        sys.exit(f"MOVE_FILE failed with the following error: {e}")
