from .shell import Shell


def symlink(target, link_name):
    Shell.symlink(target, link_name)


def mkdir(path):
    Shell.mkdir(path)


def move_file(src_file, target):
    Shell.moveFile(src_file, target)


def delete_file(filename):
    Shell.deleteFile(filename)


def delete_directory(path):
    Shell.deleteDirectory(path)


def copy_directory(src_path, target_path):
    Shell.copyDirectory(src_path, target_path)


def copy_file(src, target=None):
    Shell.copyFile(src, target)


def careful_copy_file(src, target=None):
    Shell.carefulCopyFile(src, target)
