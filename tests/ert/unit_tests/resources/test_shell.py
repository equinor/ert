import contextlib
import os
import os.path
import shutil
import sys
from contextlib import suppress
from pathlib import Path

import pytest

from ert.config import ErtConfig
from tests.ert.utils import SOURCE_DIR

from ._import_from_location import import_from_location


@contextlib.contextmanager
def pushd(path):
    cwd0 = os.getcwd()
    os.chdir(path)

    yield

    os.chdir(cwd0)


symlink = import_from_location(
    "symlink",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/symlink.py"),
).symlink

careful_copy_file = import_from_location(
    "careful_copy",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/careful_copy_file.py"),
).careful_copy_file
mkdir = import_from_location(
    "make_directory",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/make_directory.py"),
).mkdir
careful_copy_file = import_from_location(
    "careful_copy",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/careful_copy_file.py"),
).careful_copy_file
copy_directory = import_from_location(
    "careful_copy",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/copy_directory.py"),
).copy_directory
copy_file = import_from_location(
    "copy_file",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/copy_file.py"),
).copy_file
delete_directory = import_from_location(
    "delete_directory",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/delete_directory.py"),
).delete_directory
delete_file = import_from_location(
    "delete_file",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/delete_file.py"),
).delete_file
move_directory = import_from_location(
    "move_directory",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/move_directory.py"),
).move_directory
move_file = import_from_location(
    "move_file",
    os.path.join(SOURCE_DIR, "src/ert/resources/shell_scripts/move_file.py"),
).move_file


@pytest.mark.usefixtures("use_tmpdir")
def test_symlink():
    with pytest.raises(IOError, match="must exist"):
        symlink("target/does/not/exist", "link")

    with open("target", "w", encoding="utf-8") as fileH:
        fileH.write("target ...")

    symlink("target", "link")
    assert os.path.islink("link")
    assert os.readlink("link") == "target"

    with open("target2", "w", encoding="utf-8") as fileH:
        fileH.write("target ...")

    with pytest.raises(IOError, match="File exists"):
        symlink("target2", "target")

    symlink("target2", "link")
    assert os.path.islink("link")
    assert os.readlink("link") == "target2"

    os.makedirs("root1/sub1/sub2")
    os.makedirs("root2/sub1/sub2")
    os.makedirs("run")

    symlink("../target", "linkpath/link")
    assert os.path.isdir("linkpath")
    assert os.path.islink("linkpath/link")

    symlink("../target", "linkpath/link")
    assert os.path.isdir("linkpath")
    assert os.path.islink("linkpath/link")


@pytest.mark.usefixtures("use_tmpdir")
def test_symlink2():
    os.makedirs("path")
    with open("path/target", "w", encoding="utf-8") as f:
        f.write("1234")

    symlink("path/target", "link")
    assert os.path.islink("link")
    assert os.path.isfile("path/target")

    symlink("path/target", "link")
    assert os.path.islink("link")
    assert os.path.isfile("path/target")
    with open("link", encoding="utf-8") as f:
        s = f.read()
        assert s == "1234"


@pytest.mark.usefixtures("use_tmpdir")
def test_mkdir():
    with open("file", "w", encoding="utf-8") as f:
        f.write("Hei")

    with pytest.raises(OSError, match="File exists"):
        mkdir("file")

    mkdir("path")
    assert os.path.isdir("path")
    mkdir("path")

    mkdir("path/subpath")
    assert os.path.isdir("path/subpath")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_file():
    with open("file", "w", encoding="utf-8") as f:
        f.write("Hei")

    move_file("file", "file2")
    assert os.path.isfile("file2")
    assert not os.path.isfile("file")

    with pytest.raises(OSError, match="No such file or directory"):
        move_file("file2", "path/file2")

    mkdir("path")
    move_file("file2", "path/file2")
    assert os.path.isfile("path/file2")
    assert not os.path.isfile("file2")

    with pytest.raises(OSError, match="not an existing file"):
        move_file("path", "path2")

    with pytest.raises(OSError, match="not an existing file"):
        move_file("not_existing", "target")

    with open("file2", "w", encoding="utf-8") as f:
        f.write("123")

    move_file("file2", "path/file2")
    assert os.path.isfile("path/file2")
    assert not os.path.isfile("file2")

    mkdir("rms/ipl")
    with open("global_variables.ipl", "w", encoding="utf-8") as f:
        f.write("123")

    move_file("global_variables.ipl", "rms/ipl/global_variables.ipl")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_directory():
    # Test moving directory that does not exist
    with pytest.raises(
        OSError, match="Input argument dir1 is not an existing directory"
    ):
        move_directory("dir1", "path/file2")

    # Test happy path
    mkdir("dir1")
    Path("dir1/file").write_text("Hei!", encoding="utf-8")
    move_directory("dir1", "dir2")
    assert os.path.exists("dir2")
    assert os.path.exists("dir2/file")
    assert not os.path.exists("dir1")

    # Test overwriting directory
    mkdir("dir1")
    Path("dir1/file2").write_text("Hei!", encoding="utf-8")
    move_directory("dir1", "dir2")
    assert os.path.exists("dir2")
    assert os.path.exists("dir2/file2")
    assert not os.path.exists("dir2/file")
    assert not os.path.exists("dir1")

    # Test moving directory inside already existing direcotry
    mkdir("dir1")
    Path("dir1/file3").write_text("Hei!", encoding="utf-8")
    move_directory("dir1", "dir2/dir1")
    assert os.path.exists("dir2/dir1")
    assert os.path.exists("dir2/file2")
    assert os.path.exists("dir2/dir1/file3")
    assert not os.path.exists("dir1")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_file_into_folder_file_exists():
    mkdir("dst_folder")
    with open("dst_folder/file", "w", encoding="utf-8") as f:
        f.write("old")

    with open("file", "w", encoding="utf-8") as f:
        f.write("new")

    with open("dst_folder/file", encoding="utf-8") as f:
        content = f.read()
        assert content == "old"

    move_file("file", "dst_folder")
    with open("dst_folder/file", encoding="utf-8") as f:
        content = f.read()
        assert content == "new"

    assert not os.path.exists("file")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_pathfile_into_folder():
    mkdir("dst_folder")
    mkdir("source1/source2/")
    with open("source1/source2/file", "w", encoding="utf-8") as f:
        f.write("stuff")

    move_file("source1/source2/file", "dst_folder")
    with open("dst_folder/file", encoding="utf-8") as f:
        content = f.read()
        assert content == "stuff"

    assert not os.path.exists("source1/source2/file")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_pathfile_into_folder_file_exists():
    mkdir("dst_folder")
    mkdir("source1/source2/")
    with open("source1/source2/file", "w", encoding="utf-8") as f:
        f.write("stuff")

    with open("dst_folder/file", "w", encoding="utf-8") as f:
        f.write("garbage")

    move_file("source1/source2/file", "dst_folder")
    with open("dst_folder/file", encoding="utf-8") as f:
        content = f.read()
        assert content == "stuff"

    assert not os.path.exists("source1/source2/file")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_file_cannot_delete_directories():
    mkdir("pathx")
    with pytest.raises(OSError, match="not a regular file"):
        delete_file("pathx")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_file_ignores_non_existing_files(capsys):
    delete_file("does/not/exist")
    assert "ignored" in capsys.readouterr().err


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_file_deletes_broken_symlinks():
    Path("file").write_text("hei", encoding="utf-8")
    symlink("file", "link")
    assert Path("link").is_symlink()

    delete_file("file")
    assert not Path("file").exists()
    assert Path("link").is_symlink() and not Path("link").exists()
    delete_file("link")
    assert not Path("link").is_symlink() and not Path("link").exists()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_non_existing_directory_is_silently_ignored(capsys):
    delete_directory("does/not/exist")
    assert "ignored" in capsys.readouterr().err


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_on_regular_file_fails():
    with open("file", mode="w", encoding="utf-8") as f:
        f.write("hei")

    with pytest.raises(OSError, match="not a directory"):
        delete_directory("file")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_does_not_follow_symlinks():
    mkdir("link_target/subpath")
    with open("link_target/link_file", "w", encoding="utf-8") as f:
        f.write("hei")

    mkdir("path/subpath")
    with open("path/file", "w", encoding="utf-8") as f:
        f.write("hei")

    with open("path/subpath/file", "w", encoding="utf-8") as f:
        f.write("hei")

    symlink("../link_target", "path/link")
    delete_directory("path")
    assert not os.path.exists("path")
    assert os.path.exists("link_target/link_file")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_on_a_symlink_to_file_fails():
    with open("link_target", "w", encoding="utf-8") as f:
        f.write("hei")
    symlink("link_target", "link")
    with pytest.raises(IOError, match="is not a directory"):
        delete_directory("link")
    assert os.path.exists("link_target")
    assert os.path.exists("link")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_on_a_symlink_to_a_directory_only_deletes_link():
    mkdir("link_target")
    with open("link_target/file", "w", encoding="utf-8") as f:
        f.write("hei")
    symlink("link_target", "link")
    delete_directory("link")
    assert os.path.exists("link_target/file")
    assert not os.path.exists("link")


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("trailing", ["", "/"])
def test_that_delete_directory_on_a_symlink_to_a_directory_is_conditionally_ignored(
    trailing,
):
    """The documentation states a warning for the DELETE_DIRECTORY job:

    "If the directory to delete is a symlink to a directory, it will only delete
    the link and not the directory. However, if you add a trailing slash to the
    directory name (the symlink), then the link itself is kept, but the directory
    it links to will be removed."

    This is true for Linux, but there is an oddity for Mac in which this is slighly
    altered as documented by this test.
    """
    mkdir("link_target")
    with open("link_target/file", "w", encoding="utf-8") as f:
        f.write("hei")
    symlink("link_target", "link")

    with suppress(NotADirectoryError):
        delete_directory(f"link{trailing}")

    if trailing:
        assert not os.path.exists("link_target/file")

        if sys.platform.startswith("darwin"):
            # Mac will also delete the symlink, while Linux will not
            assert not os.path.exists("link")
            assert not os.path.exists("link_target")
        else:
            assert os.path.exists("link")
            assert os.path.exists("link_target")
    else:
        assert os.path.exists("link_target/file")
        assert os.path.exists("link_target")
        assert not os.path.exists("link")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_can_delete_directories_with_internal_symlinks():
    mkdir("to_be_deleted")
    Path("to_be_deleted/link_target.txt").write_text("hei", encoding="utf-8")

    os.chdir("to_be_deleted")  # symlink() requires this
    symlink("link_target.txt", "link")
    os.chdir("..")
    assert Path("to_be_deleted/link").exists()

    delete_directory("to_be_deleted")
    assert not Path("to_be_deleted").exists()


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_directory_error():
    with pytest.raises(OSError, match="existing directory"):
        copy_directory("does/not/exist", "target")

    with open("file", "w", encoding="utf-8") as f:
        f.write("hei")

    with pytest.raises(OSError, match="existing directory"):
        copy_directory("hei", "target")

    empty_dir = "emptytestdir"
    if not os.path.exists(empty_dir):
        os.makedirs(empty_dir)

    file_path = os.path.join(empty_dir, "file")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("some_text")

    with pytest.raises(OSError):
        copy_directory(empty_dir, ".")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_file():
    with pytest.raises(OSError, match="existing file"):
        copy_file("does/not/exist", "target")

    mkdir("path")
    with pytest.raises(OSError, match="existing file"):
        copy_file("path", "target")

    with open("file1", "w", encoding="utf-8") as f:
        f.write("hei")

    copy_file("file1", "file2")
    assert os.path.isfile("file2")

    copy_file("file1", "path")
    assert os.path.isfile("path/file1")

    copy_file("file1", "path2/file1")
    assert os.path.isfile("path2/file1")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_file2():
    mkdir("root/sub/path")

    with open("file", "w", encoding="utf-8") as f:
        f.write("Hei ...")

    copy_file("file", "root/sub/path/file")
    assert os.path.isfile("root/sub/path/file")

    with open("file2", "w", encoding="utf-8") as f:
        f.write("Hei ...")

    with pushd("root/sub/path"):
        copy_file("../../../file2")
        assert os.path.isfile("file2")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_file3():
    mkdir("rms/output")

    with open("file.txt", "w", encoding="utf-8") as f:
        f.write("Hei")

    copy_file("file.txt", "rms/output/")
    assert os.path.isfile("rms/output/file.txt")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_when_target_is_none():
    Path("somedir").mkdir()
    Path("somedir/file.txt").write_text("Hei", encoding="utf-8")

    copy_file("somedir/file.txt", None)
    assert Path("file.txt").read_text(encoding="utf-8") == "Hei"


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_when_target_is_none_in_same_directory():
    Path("file.txt").write_text("Hei", encoding="utf-8")
    with pytest.raises(shutil.SameFileError):
        copy_file("file.txt", None)


@pytest.mark.usefixtures("use_tmpdir")
def test_careful_copy_file():
    with open("file1", "w", encoding="utf-8") as f:
        f.write("hei")
    with open("file2", "w", encoding="utf-8") as f:
        f.write("hallo")

    careful_copy_file("file1", "file2")
    with open("file2", encoding="utf-8") as f:
        assert f.readline() == "hallo"

    print(careful_copy_file("file1", "file3"))
    assert os.path.isfile("file3")


@pytest.mark.usefixtures("use_tmpdir")
def test_careful_copy_file_when_target_is_none():
    Path("somedir").mkdir()
    Path("somedir/file.txt").write_text("Hei", encoding="utf-8")

    careful_copy_file("somedir/file.txt", None)
    assert Path("file.txt").read_text(encoding="utf-8") == "Hei"


@pytest.mark.usefixtures("use_tmpdir")
def test_careful_copy_when_target_is_none_in_same_directory_is_noop():
    Path("file.txt").write_text("Hei", encoding="utf-8")
    careful_copy_file("file.txt", None)  # File will not be touched
    assert Path("file.txt").read_text(encoding="utf-8") == "Hei"


@pytest.mark.usefixtures("use_tmpdir")
def test_careful_copy_when_target_is_none_does_not_touch_existing():
    Path("somedir").mkdir()
    Path("somedir/file.txt").write_text("Hei", encoding="utf-8")
    Path("file.txt").write_text("I will survive", encoding="utf-8")
    careful_copy_file("somedir/file.txt", None)
    assert Path("file.txt").read_text(encoding="utf-8") == "I will survive"


@pytest.fixture
def minimal_case(tmpdir):
    with tmpdir.as_cwd():
        with open("config.ert", "w", encoding="utf-8") as fout:
            fout.write("NUM_REALIZATIONS 1")
        yield


def test_shell_script_fmstep_availability(minimal_case):
    ert_config = ErtConfig.with_plugins().from_file("config.ert")
    fm_shell_jobs = {}
    for fm_step in ert_config.installed_forward_model_steps.values():
        exe = fm_step.executable
        if "shell_scripts" in exe:
            fm_shell_jobs[fm_step.name.upper()] = Path(exe).resolve()

    wf_shell_jobs = {}
    for wf_name, wf in ert_config.workflow_jobs.items():
        if wf.executable is not None and "shell_scripts" in wf.executable:
            wf_shell_jobs[wf_name] = Path(wf.executable).resolve()

    assert fm_shell_jobs == wf_shell_jobs


def test_shell_script_fmstep_names(minimal_case):
    shell_job_names = [
        "DELETE_FILE",
        "DELETE_DIRECTORY",
        "COPY_DIRECTORY",
        "MAKE_SYMLINK",
        "MOVE_FILE",
        "MOVE_DIRECTORY",
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
