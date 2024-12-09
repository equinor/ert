#!/usr/bin/env python
import os
import shutil
import sys


def copy_file(src: str, target: str | None = None) -> None:
    if os.path.isfile(src):
        if target is None:
            target = os.path.basename(src)

        if os.path.isdir(target):
            target_file = os.path.join(target, os.path.basename(src))
            shutil.copyfile(src, target_file)
            print(f"Copying file '{src}' -> '{target_file}'")
        else:
            target_path = os.path.dirname(target)
            if target_path and not os.path.isdir(target_path):
                os.makedirs(target_path)
                print(f"Creating directory '{target_path}' ")
            if os.path.isdir(target):
                target_file = os.path.join(target, os.path.basename(src))
            else:
                target_file = target

            print(f"Copying file '{src}' -> '{target_file}'")
            shutil.copyfile(src, target_file)
    else:
        raise OSError(f"Input argument:'{src}' does not correspond to an existing file")


if __name__ == "__main__":
    src = sys.argv[1]
    try:
        if len(sys.argv) > 2:
            target = sys.argv[2]
            copy_file(src, target)
        else:
            copy_file(src)
    except OSError as e:
        sys.exit(f"COPY_FILE failed with the following error: {e}")
