import contextlib
import os
import os.path
import subprocess
import sys
from pathlib import Path

import pytest

from ert.config import ErtConfig


@contextlib.contextmanager
def pushd(path):
    cwd0 = os.getcwd()
    os.chdir(path)

    yield

    os.chdir(cwd0)


class Shell:
    """
    A test utility that runs the forward-model scripts and returns the error
    output (if any)
    """

    def __init__(self, source_root):
        self.script_dir = os.path.join(
            source_root,
            "src",
            "ert",
            "shared",
            "share",
            "ert",
            "shell_scripts",
        )

    def _call_script(self, name, args):
        return subprocess.run(
            [sys.executable, os.path.join(self.script_dir, name)] + list(args),
            check=False,
            capture_output=True,
        )

    def symlink(self, *args):
        return self._call_script("symlink.py", args)

    def mkdir(self, *args):
        return self._call_script("make_directory.py", args)

    def careful_copy_file(self, *args):
        return self._call_script("careful_copy_file.py", args)

    def copy_directory(self, *args):
        return self._call_script("copy_directory.py", args)

    def copy_file(self, *args):
        return self._call_script("copy_file.py", args)

    def delete_directory(self, *args):
        return self._call_script("delete_directory.py", args)

    def delete_file(self, *args):
        return self._call_script("delete_file.py", args)

    def move_file(self, *args):
        return self._call_script("move_file.py", args)


@pytest.fixture
def shell(source_root):
    return Shell(source_root)


@pytest.mark.usefixtures("use_tmpdir")
def test_symlink(shell):
    assert b"must exist" in shell.symlink("target/does/not/exist", "link").stderr
    with open("target", "w", encoding="utf-8") as fileH:
        fileH.write("target ...")

    shell.symlink("target", "link")
    assert os.path.islink("link")
    assert os.readlink("link") == "target"

    with open("target2", "w", encoding="utf-8") as fileH:
        fileH.write("target ...")

    assert b"File exists" in shell.symlink("target2", "target").stderr

    shell.symlink("target2", "link")
    assert os.path.islink("link")
    assert os.readlink("link") == "target2"

    os.makedirs("root1/sub1/sub2")
    os.makedirs("root2/sub1/sub2")
    os.makedirs("run")

    shell.symlink("../target", "linkpath/link")
    assert os.path.isdir("linkpath")
    assert os.path.islink("linkpath/link")

    shell.symlink("../target", "linkpath/link")
    assert os.path.isdir("linkpath")
    assert os.path.islink("linkpath/link")


@pytest.mark.usefixtures("use_tmpdir")
def test_symlink2(shell):
    os.makedirs("path")
    with open("path/target", "w", encoding="utf-8") as f:
        f.write("1234")

    shell.symlink("path/target", "link")
    assert os.path.islink("link")
    assert os.path.isfile("path/target")

    shell.symlink("path/target", "link")
    assert os.path.islink("link")
    assert os.path.isfile("path/target")
    with open("link", encoding="utf-8") as f:
        s = f.read()
        assert s == "1234"


@pytest.mark.usefixtures("use_tmpdir")
def test_mkdir(shell):
    with open("file", "w", encoding="utf-8") as f:
        f.write("Hei")

    assert b"File exists" in shell.mkdir("file").stderr

    shell.mkdir("path")
    assert os.path.isdir("path")
    shell.mkdir("path")

    shell.mkdir("path/subpath")
    assert os.path.isdir("path/subpath")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_file(shell):
    with open("file", "w", encoding="utf-8") as f:
        f.write("Hei")

    shell.move_file("file", "file2")
    assert os.path.isfile("file2")
    assert not os.path.isfile("file")

    assert b"No such file or directory" in shell.move_file("file2", "path/file2").stderr

    shell.mkdir("path")
    shell.move_file("file2", "path/file2")
    assert os.path.isfile("path/file2")
    assert not os.path.isfile("file2")

    assert b"not an existing file" in shell.move_file("path", "path2").stderr

    assert b"not an existing file" in shell.move_file("not_existing", "target").stderr

    with open("file2", "w", encoding="utf-8") as f:
        f.write("123")

    shell.move_file("file2", "path/file2")
    assert os.path.isfile("path/file2")
    assert not os.path.isfile("file2")

    shell.mkdir("rms/ipl")
    with open("global_variables.ipl", "w", encoding="utf-8") as f:
        f.write("123")

    shell.move_file("global_variables.ipl", "rms/ipl/global_variables.ipl")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_file_into_folder_file_exists(shell):
    shell.mkdir("dst_folder")
    with open("dst_folder/file", "w", encoding="utf-8") as f:
        f.write("old")

    with open("file", "w", encoding="utf-8") as f:
        f.write("new")

    with open("dst_folder/file", "r", encoding="utf-8") as f:
        content = f.read()
        assert content == "old"

    shell.move_file("file", "dst_folder")
    with open("dst_folder/file", "r", encoding="utf-8") as f:
        content = f.read()
        assert content == "new"

    assert not os.path.exists("file")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_pathfile_into_folder(shell):
    shell.mkdir("dst_folder")
    shell.mkdir("source1/source2/")
    with open("source1/source2/file", "w", encoding="utf-8") as f:
        f.write("stuff")

    shell.move_file("source1/source2/file", "dst_folder")
    with open("dst_folder/file", "r", encoding="utf-8") as f:
        content = f.read()
        assert content == "stuff"

    assert not os.path.exists("source1/source2/file")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_pathfile_into_folder_file_exists(shell):
    shell.mkdir("dst_folder")
    shell.mkdir("source1/source2/")
    with open("source1/source2/file", "w", encoding="utf-8") as f:
        f.write("stuff")

    with open("dst_folder/file", "w", encoding="utf-8") as f:
        f.write("garbage")

    shell.move_file("source1/source2/file", "dst_folder")
    with open("dst_folder/file", "r", encoding="utf-8") as f:
        content = f.read()
        assert content == "stuff"

    assert not os.path.exists("source1/source2/file")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_file_cannot_delete_directories(shell):
    shell.mkdir("pathx")
    assert b"not a regular file" in shell.delete_file("pathx").stderr


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_file_ignores_non_existing_files(shell):
    assert b"ignored" in shell.delete_file("does/not/exist").stderr


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_file_deletes_broken_symlinks(shell):
    Path("file").write_text("hei", encoding="utf-8")
    shell.symlink("file", "link")
    assert Path("link").is_symlink()

    shell.delete_file("file")
    assert not Path("file").exists()
    assert Path("link").is_symlink() and not Path("link").exists()
    shell.delete_file("link")
    assert not Path("link").is_symlink() and not Path("link").exists()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_non_existing_directory_is_silently_ignored(shell):
    assert b"delete ignored" in shell.delete_directory("does/not/exist").stderr


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_on_regular_file_fails(shell):
    with open("file", mode="w", encoding="utf-8") as f:
        f.write("hei")

    assert b"not a directory" in shell.delete_directory("file").stderr


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_does_not_follow_symlinks(shell):
    shell.mkdir("link_target/subpath")
    with open("link_target/link_file", "w", encoding="utf-8") as f:
        f.write("hei")

    shell.mkdir("path/subpath")
    with open("path/file", "w", encoding="utf-8") as f:
        f.write("hei")

    with open("path/subpath/file", "w", encoding="utf-8") as f:
        f.write("hei")

    shell.symlink("../link_target", "path/link")
    shell.delete_directory("path")
    assert not os.path.exists("path")
    assert os.path.exists("link_target/link_file")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_can_delete_directories_with_internal_symlinks(shell):
    shell.mkdir("to_be_deleted")
    Path("to_be_deleted/link_target.txt").write_text("hei", encoding="utf-8")

    os.chdir("to_be_deleted")  # shell.symlink() requires this
    shell.symlink("link_target.txt", "link")
    os.chdir("..")
    assert Path("to_be_deleted/link").exists()

    shell.delete_directory("to_be_deleted")
    assert not Path("to_be_deleted").exists()


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_directory_error(shell):
    assert (
        b"existing directory" in shell.copy_directory("does/not/exist", "target").stderr
    )

    with open("file", "w", encoding="utf-8") as f:
        f.write("hei")

    assert b"existing directory" in shell.copy_directory("hei", "target").stderr


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_file(shell):
    assert b"existing file" in shell.copy_file("does/not/exist", "target").stderr

    shell.mkdir("path")
    assert b"existing file" in shell.copy_file("path", "target").stderr

    with open("file1", "w", encoding="utf-8") as f:
        f.write("hei")

    shell.copy_file("file1", "file2")
    assert os.path.isfile("file2")

    shell.copy_file("file1", "path")
    assert os.path.isfile("path/file1")

    shell.copy_file("file1", "path2/file1")
    assert os.path.isfile("path2/file1")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_file2(shell):
    shell.mkdir("root/sub/path")

    with open("file", "w", encoding="utf-8") as f:
        f.write("Hei ...")

    shell.copy_file("file", "root/sub/path/file")
    assert os.path.isfile("root/sub/path/file")

    with open("file2", "w", encoding="utf-8") as f:
        f.write("Hei ...")

    with pushd("root/sub/path"):
        shell.copy_file("../../../file2")
        assert os.path.isfile("file2")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_file3(shell):
    shell.mkdir("rms/output")

    with open("file.txt", "w", encoding="utf-8") as f:
        f.write("Hei")

    shell.copy_file("file.txt", "rms/output/")
    assert os.path.isfile("rms/output/file.txt")


@pytest.mark.usefixtures("use_tmpdir")
def test_careful_copy_file(shell):
    with open("file1", "w", encoding="utf-8") as f:
        f.write("hei")
    with open("file2", "w", encoding="utf-8") as f:
        f.write("hallo")

    shell.careful_copy_file("file1", "file2")
    with open("file2", "r", encoding="utf-8") as f:
        assert f.readline() == "hallo"

    print(shell.careful_copy_file("file1", "file3"))
    assert os.path.isfile("file3")


@pytest.fixture
def minimal_case(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fout:
            fout.write("NUM_REALIZATIONS 1")
        yield


def test_shell_script_jobs_availability(minimal_case):
    ert_config = ErtConfig.from_file("config.ert")
    fm_shell_jobs = {}
    for job in ert_config.installed_jobs.values():
        exe = job.executable
        if "shell_scripts" in exe:
            fm_shell_jobs[job.name.upper()] = Path(exe).resolve()

    wf_shell_jobs = {}
    for wf_name, wf in ert_config.workflow_jobs.items():
        if wf.executable is not None and "shell_scripts" in wf.executable:
            wf_shell_jobs[wf_name] = Path(wf.executable).resolve()

    assert fm_shell_jobs == wf_shell_jobs


def test_shell_script_jobs_names(minimal_case):
    shell_job_names = [
        "DELETE_FILE",
        "DELETE_DIRECTORY",
        "COPY_DIRECTORY",
        "MAKE_SYMLINK",
        "MOVE_FILE",
        "MAKE_DIRECTORY",
        "CAREFUL_COPY_FILE",
        "SYMLINK",
        "COPY_FILE",
    ]

    ert_config = ErtConfig.from_file("config.ert")
    found_jobs = set()
    for wf_name, wf in ert_config.workflow_jobs.items():
        if wf.executable is not None and "shell_scripts" in wf.executable:
            assert wf_name in shell_job_names
            found_jobs.add(wf_name)

    assert len(shell_job_names) == len(found_jobs)
