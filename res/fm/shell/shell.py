import os
import os.path
import shutil
import sys
import distutils.dir_util


class Shell(object):
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
                print("Creating directory for link: %s" % link_path)
                os.makedirs(link_path)
            target_check = os.path.join(link_path, target)

        if not os.path.exists(target_check):
            raise IOError(
                "{} (target) and {} (link_name) requested, which implies that {} must exist, but it does not.".format(
                    target, link_name, target_check
                )
            )

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(target, link_name)
        print("Linking '%s' -> '%s' [ cwd:%s ]" % (link_name, target, os.getcwd()))

    @staticmethod
    def mkdir(path):
        if os.path.isdir(path):
            print("OK - directory: '%s' already exists" % path)
        else:
            try:
                os.makedirs(path)
                print("Created directory: '%s'" % path)
            except OSError as e:
                # Seems in many cases the directory just suddenly appears; syncronization
                # issues?
                if not os.path.isdir(path):
                    msg = 'ERROR: Failed to create directory "%s": %s.' % (path, e)
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
            raise IOError("Input argument %s is not an existing file" % src_file)

    @staticmethod
    def __deleteFile(filename):
        stat_info = os.stat(filename)
        uid = stat_info.st_uid
        if uid == os.getuid():
            os.unlink(filename)
            print("Removing file:'%s'" % filename)
        else:
            sys.stderr.write(
                "Sorry you are not owner of file:%s - not deleted\n" % filename
            )

    @staticmethod
    def __deleteDirectory(dirname):
        stat_info = os.stat(dirname)
        uid = stat_info.st_uid
        if uid == os.getuid():
            if os.path.islink(dirname):
                os.remove(dirname)
                print("Removing symbolic link:'%s'" % dirname)
            else:
                try:
                    os.rmdir(dirname)
                    print("Removing directory:'%s'" % dirname)
                except OSError as e:
                    if e.errno == 39:
                        sys.stderr.write(
                            "Failed to remove directory:%s - not empty\n" % dirname
                        )
                    else:
                        raise
        else:
            sys.stderr.write(
                "Sorry you are not owner of directory:%s - not deleted\n" % dirname
            )

    @staticmethod
    def deleteFile(filename):
        if os.path.exists(filename):
            if os.path.isfile(filename):
                Shell.__deleteFile(filename)
            else:
                raise IOError("Entry:'%s' is not a regular file" % filename)
        else:
            if os.path.islink(filename):
                os.remove(filename)
            else:
                sys.stderr.write("File: '%s' not exist - delete ignored\n" % filename)

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

                        for dir in dirs:
                            Shell.__deleteDirectory(os.path.join(root, dir))

            else:
                raise IOError("Entry:'%s' is not a directory" % path)

            Shell.__deleteDirectory(path)
        else:
            sys.stderr.write("Directory:'%s' not exist - delete ignored\n" % path)

    @staticmethod
    def copyDirectory(src_path, target_path):
        if os.path.isdir(src_path):
            src_basename = os.path.basename(src_path)
            target_root, target_basename = os.path.split(target_path)
            full_target = os.path.join(target_path, src_basename)

            if target_root:
                if not os.path.isdir(target_root):
                    print("Creating empty folder structure %s" % target_root)
                    Shell.mkdir(target_root)

            print("Copying directory structure %s -> %s" % (src_path, target_path))
            if os.path.isdir(target_path):
                target_path = os.path.join(target_path, src_basename)
            distutils.dir_util.copy_tree(src_path, target_path, preserve_times=0)
        else:
            raise IOError(
                "Input argument:'%s' does not correspond to an existing directory"
                % src_path
            )

    @staticmethod
    def copyFile(src, target=None):
        if os.path.isfile(src):
            if target is None:
                target = os.path.basename(src)

            if os.path.isdir(target):
                target_file = os.path.join(target, os.path.basename(src))
                shutil.copyfile(src, target_file)
                print("Copying file '%s' -> '%s'" % (src, target_file))
            else:
                target_path = os.path.dirname(target)
                if target_path:
                    if not os.path.isdir(target_path):
                        os.makedirs(target_path)
                        print("Creating directory '%s' " % target_path)
                if os.path.isdir(target):
                    target_file = os.path.join(target, os.path.basename(src))
                else:
                    target_file = target

                print("Copying file '%s' -> '%s'" % (src, target_file))
                shutil.copyfile(src, target_file)
        else:
            raise IOError(
                "Input argument:'%s' does not correspond to an existing file" % src
            )

    @staticmethod
    def carefulCopyFile(src, target=None):
        if os.path.exists(target):
            print("File: {} already present - not updated".format(target))
            return
        Shell.copyFile(src, target)
