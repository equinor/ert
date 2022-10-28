#!/usr/bin/env python
import distutils.dir_util
import os
import sys

from make_directory import mkdir


def copy_directory(src_path, target_path):
    if os.path.isdir(src_path):
        src_basename = os.path.basename(src_path)
        target_root, _ = os.path.split(target_path)

        if target_root:
            if not os.path.isdir(target_root):
                print(f"Creating empty folder structure {target_root}")
                mkdir(target_root)

        print(f"Copying directory structure {src_path} -> {target_path}")
        if os.path.isdir(target_path):
            target_path = os.path.join(target_path, src_basename)
        distutils.dir_util.copy_tree(src_path, target_path, preserve_times=0)
    else:
        raise IOError(
            f"Input argument:'{src_path}' "
            "does not correspond to an existing directory"
        )


if __name__ == "__main__":
    src_path = sys.argv[1]
    target_path = sys.argv[2]
    try:
        copy_directory(src_path, target_path)
    except IOError as e:
        sys.exit(f"COPY_DIRECTORY failed with the following error: {e}")
