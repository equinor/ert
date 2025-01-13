#!/usr/bin/env python
import os
import shutil
import sys

from make_directory import mkdir  # type: ignore


def copy_directory(src_path: str, target_path: str) -> None:
    if os.path.isdir(src_path):
        src_basename = os.path.basename(src_path)
        target_root, _ = os.path.split(target_path)

        if target_root and not os.path.isdir(target_root):
            print(f"Creating empty folder structure {target_root}")
            mkdir(target_root)

        print(f"Copying directory structure {src_path} -> {target_path}")
        if os.path.isdir(target_path):
            target_path = os.path.join(target_path, src_basename)
        try:
            shutil.copytree(src_path, target_path, dirs_exist_ok=True)
        except shutil.Error as err:
            # Check for shutil bug in Python <3.14:
            if len(err.args[0]) > 10 and {
                len(somestring) for somestring in err.args[0]
            } == {1}:
                # https://github.com/python/cpython/issues/102931
                # This can only occur when the shutil.Error is a single error
                raise OSError("".join(err.args[0])) from err
            else:
                raise OSError(
                    ", ".join([err_arg[2] for err_arg in err.args[0]])
                ) from err
    else:
        raise OSError(
            f"Input argument: '{src_path}' does not correspond to an existing directory"
        )


if __name__ == "__main__":
    src_path = sys.argv[1]
    target_path = sys.argv[2]
    try:
        copy_directory(src_path, target_path)
    except OSError as oserror:
        sys.exit(f"COPY_DIRECTORY failed with the following error(s): {oserror}")
