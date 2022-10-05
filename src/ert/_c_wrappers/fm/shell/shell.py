import distutils.dir_util
import os
import os.path
import shutil
import sys


class Shell:
    """
    Utility class to simplify common shell operations.
    """

    @staticmethod
    def symlink(target, link_name):
        """Will create a symbol link 'link_name -> target'.

        If the @link_name already exists as a symbolic link it will be
        removed first; if the @link_name exists and is *not* a
        symbolic link OSError will be raised. If the @target does not
        exists IOError will be raised.
        """
        link_path, link_base = os.path.split(link_name)
        if len(link_path) == 0:
            target_check = target
        else:
            if not os.path.isdir(link_path):
                print(f"Creating directory for link: {link_path}")
                os.makedirs(link_path)
            target_check = os.path.join(link_path, target)

        if not os.path.exists(target_check):
            raise IOError(
                f"{target} (target) and {link_name} (link_name) requested, "
                f"which implies that {target_check} must exist, but it does not."
            )

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(target, link_name)
        print(f"Linking '{link_name}' -> '{target}' [ cwd:{os.getcwd()} ]")

    @staticmethod
    def mkdir(path):
        if os.path.isdir(path):
            print(f"OK - directory: '{path}' already exists")
        else:
            try:
                os.makedirs(path)
                print(f"Created directory: '{path}'")
            except OSError as error:
                # Seems in many cases the directory just suddenly appears;
                # synchronization issues?
                if not os.path.isdir(path):
                    msg = f'ERROR: Failed to create directory "{path}": {error}.'
                    raise OSError(msg)

    @staticmethod
    def moveFile(src_file, target):
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

    @staticmethod
    def __deleteFile(filename):
        stat_info = os.stat(filename)
        uid = stat_info.st_uid
        if uid == os.getuid():
            os.unlink(filename)
            print(f"Removing file:'{filename}'")
        else:
            sys.stderr.write(
                f"Sorry you are not owner of file:{filename} - not deleted\n"
            )

    @staticmethod
    def __deleteDirectory(dirname):
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

    @staticmethod
    def deleteFile(filename):
        if os.path.exists(filename):
            if os.path.isfile(filename):
                Shell.__deleteFile(filename)
            else:
                raise IOError(f"Entry:'{filename}' is not a regular file")
        else:
            if os.path.islink(filename):
                os.remove(filename)
            else:
                sys.stderr.write(
                    f"File: '{filename}' does not exist - delete ignored\n"
                )

    @staticmethod
    def deleteDirectory(path):
        """
        Will ignore if you are not owner.
        """
        if os.path.exists(path):
            if os.path.isdir(path):
                for root, dirs, files in os.walk(
                    path, topdown=False, followlinks=False
                ):
                    if not os.path.islink(root):
                        for file in files:
                            Shell.__deleteFile(os.path.join(root, file))

                        for _dir in dirs:
                            Shell.__deleteDirectory(os.path.join(root, _dir))

            else:
                raise IOError(f"Entry:'{path}' is not a directory")

            Shell.__deleteDirectory(path)
        else:
            sys.stderr.write(f"Directory:'{path}' not exist - delete ignored\n")

    @staticmethod
    def copyDirectory(src_path, target_path):
        if os.path.isdir(src_path):
            src_basename = os.path.basename(src_path)
            target_root, target_basename = os.path.split(target_path)

            if target_root:
                if not os.path.isdir(target_root):
                    print(f"Creating empty folder structure {target_root}")
                    Shell.mkdir(target_root)

            print(f"Copying directory structure {src_path} -> {target_path}")
            if os.path.isdir(target_path):
                target_path = os.path.join(target_path, src_basename)
            distutils.dir_util.copy_tree(src_path, target_path, preserve_times=0)
        else:
            raise IOError(
                f"Input argument:'{src_path}' "
                "does not correspond to an existing directory"
            )

    @staticmethod
    def copyFile(src, target=None):
        if os.path.isfile(src):
            if target is None:
                target = os.path.basename(src)

            if os.path.isdir(target):
                target_file = os.path.join(target, os.path.basename(src))
                shutil.copyfile(src, target_file)
                print(f"Copying file '{src}' -> '{target_file}'")
            else:
                target_path = os.path.dirname(target)
                if target_path:
                    if not os.path.isdir(target_path):
                        os.makedirs(target_path)
                        print(f"Creating directory '{target_path}' ")
                if os.path.isdir(target):
                    target_file = os.path.join(target, os.path.basename(src))
                else:
                    target_file = target

                print(f"Copying file '{src}' -> '{target_file}'")
                shutil.copyfile(src, target_file)
        else:
            raise IOError(
                f"Input argument:'{src}' does not correspond to an existing file"
            )

    @staticmethod
    def carefulCopyFile(src, target=None):
        if os.path.exists(target):
            print(f"File: {target} already present - not updated")
            return
        Shell.copyFile(src, target)
