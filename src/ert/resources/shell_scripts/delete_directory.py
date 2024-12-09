#!/usr/bin/env python
import os
import sys


def delete_file(filename: str) -> None:
    stat_info = os.lstat(filename)
    uid = stat_info.st_uid
    if uid == os.getuid():
        os.unlink(filename)
        print(f"Removing file:'{filename}'")
    else:
        sys.stderr.write(f"Sorry you are not owner of file:{filename} - not deleted\n")


def delete_empty_directory(dirname: str) -> None:
    stat_info = os.stat(dirname)
    uid = stat_info.st_uid
    if uid == os.getuid():
        if os.path.islink(dirname):
            os.remove(dirname)
            print(f"Removing symbolic link:'{dirname}'")
        else:
            try:
                os.rmdir(dirname)
                print(f"Removing directory:'{dirname}'")
            except OSError as error:
                if error.errno == 39:
                    sys.stderr.write(
                        f"Failed to remove directory:{dirname} - not empty\n"
                    )
                else:
                    raise
    else:
        sys.stderr.write(
            f"Sorry you are not owner of directory:{dirname} - not deleted\n"
        )


def delete_directory(path: str) -> None:
    """
    Will ignore if you are not owner.
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False, followlinks=False):
                if not os.path.islink(root):
                    for file in files:
                        delete_file(os.path.join(root, file))

                    for _dir in dirs:
                        delete_empty_directory(os.path.join(root, _dir))

        else:
            raise OSError(f"Entry:'{path}' is not a directory")

        delete_empty_directory(path)
    else:
        sys.stderr.write(f"Directory:'{path}' does not exist - delete ignored\n")


if __name__ == "__main__":
    try:
        for d in sys.argv[1:]:
            delete_directory(d)
    except OSError as e:
        sys.exit(f"DELETE_DIRECTORY failed with the following error: {e}")
