import contextlib
import os
import os.path
import shutil
import sys
from contextlib import suppress
from pathlib import Path

import pytest

from ert.config import ErtConfig
from ert.config.workflow_job import ExecutableWorkflow
from ert.plugins import ErtPluginContext
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

    Path("target").write_text("target ...", encoding="utf-8")

    symlink("target", "link")
    assert os.path.islink("link")
    assert os.readlink("link") == "target"

    Path("target2").write_text("target ...", encoding="utf-8")

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
    Path("path/target").write_text("1234", encoding="utf-8")

    symlink("path/target", "link")
    assert os.path.islink("link")
    assert Path("path/target").is_file()

    symlink("path/target", "link")
    assert os.path.islink("link")
    assert Path("path/target").is_file()

    assert Path("link").read_text(encoding="utf-8") == "1234"


@pytest.mark.usefixtures("use_tmpdir")
def test_mkdir():
    Path("file").write_text("Hei", encoding="utf-8")

    with pytest.raises(OSError, match="File exists"):
        mkdir("file")

    mkdir("path")
    assert os.path.isdir("path")
    mkdir("path")

    mkdir("path/subpath")
    assert os.path.isdir("path/subpath")


@pytest.mark.usefixtures("use_tmpdir")
def test_move_file():
    Path("file").write_text("hei", encoding="utf-8")

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

    Path("file2").write_text("123", encoding="utf-8")

    move_file("file2", "path/file2")
    assert os.path.isfile("path/file2")
    assert not os.path.isfile("file2")

    mkdir("rms/ipl")
    Path("global_variables.ipl").write_text("123", encoding="utf-8")

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

    Path("dst_folder/file").write_text("old", encoding="utf-8")
    Path("file").write_text("new", encoding="utf-8")

    move_file("file", "dst_folder")
    assert Path("dst_folder/file").read_text(encoding="utf-8") == "new"

    assert not Path("file").exists()


@pytest.mark.usefixtures("use_tmpdir")
def test_move_pathfile_into_folder():
    mkdir("dst_folder")
    mkdir("source1/source2/")
    orig_file = Path("source1/source2/file")
    orig_file.write_text("stuff", encoding="utf-8")

    move_file("source1/source2/file", "dst_folder")
    assert Path("dst_folder/file").read_text(encoding="utf-8") == "stuff"

    assert not orig_file.exists()


@pytest.mark.usefixtures("use_tmpdir")
def test_move_pathfile_into_folder_file_exists():
    mkdir("dst_folder")
    mkdir("source1/source2/")
    orig_file = Path("source1/source2/file")

    orig_file.write_text("stuff", encoding="utf-8")

    Path("dst_folder/file").write_text("garbage", encoding="utf-8")

    move_file("source1/source2/file", "dst_folder")
    assert Path("dst_folder/file").read_text(encoding="utf-8") == "stuff"

    assert not orig_file.exists()


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
    assert Path("link").is_symlink()
    assert not Path("link").exists()
    delete_file("link")
    assert not Path("link").is_symlink()
    assert not Path("link").exists()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_non_existing_directory_is_silently_ignored(capsys):
    delete_directory("does/not/exist")
    assert "ignored" in capsys.readouterr().err


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_on_regular_file_fails():
    Path("file").write_text("hei", encoding="utf-8")

    with pytest.raises(OSError, match="not a directory"):
        delete_directory("file")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_does_not_follow_symlinks():
    mkdir("link_target/subpath")
    Path("link_target/link_file").write_text("hei", encoding="utf-8")

    mkdir("path/subpath")
    Path("path/file").write_text("hei", encoding="utf-8")
    Path("path/subpath/file").write_text("hei", encoding="utf-8")

    symlink("../link_target", "path/link")
    delete_directory("path")
    assert not os.path.exists("path")
    assert os.path.exists("link_target/link_file")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_on_a_symlink_to_file_fails():
    Path("link_target").write_text("hei", encoding="utf-8")
    symlink("link_target", "link")
    with pytest.raises(IOError, match="is not a directory"):
        delete_directory("link")
    assert os.path.exists("link_target")
    assert os.path.exists("link")


@pytest.mark.usefixtures("use_tmpdir")
def test_that_delete_directory_on_a_symlink_to_a_directory_only_deletes_link():
    mkdir("link_target")
    Path("link_target/file").write_text("hei", encoding="utf-8")
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
    Path("link_target/file").write_text("hei", encoding="utf-8")
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
def test_copy_directory_errors_when_source_directory_does_not_exist():
    not_existing = "does/not/exist"
    with pytest.raises(
        OSError,
        match=(
            f"Input argument: '{not_existing}' does not correspond "
            "to an existing directory"
        ),
    ):
        copy_directory(not_existing, "target")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_directory_errors_when_source_directory_is_file():
    textfilename = "hei"
    Path("file").write_text(textfilename, encoding="utf-8")
    with pytest.raises(
        OSError,
        match=(
            f"Input argument: '{textfilename}' does not correspond "
            "to an existing directory"
        ),
    ):
        copy_directory(textfilename, "target")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_directory_errors_when_copying_dir_to_itself():
    empty_dir = Path("testdir")
    empty_dir.mkdir()

    (empty_dir / "file").write_text("some_text", encoding="utf-8")

    with pytest.raises(OSError, match="are the same file"):
        copy_directory(empty_dir.name, ".")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_directory_errors_when_symlinks_point_nowhere():
    somedir = "somedir"
    some_symlink = f"{somedir}/some_symlink"
    Path(somedir).mkdir()
    os.symlink("/not_existing", some_symlink)
    with pytest.raises(OSError, match=f"No such file or directory: '{some_symlink}'"):
        copy_directory(somedir, "copydir")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_directory_reports_multiple_errors():
    somedir = "somedir"
    some_symlink = f"{somedir}/some_symlink"
    some_other_symlink = f"{somedir}/some_other_symlink"

    Path(somedir).mkdir()
    os.symlink("/not_existing", some_symlink)
    os.symlink("/not_existing", some_other_symlink)
    try:
        copy_directory(somedir, "copydir")
        raise AssertionError("copy_directory should raise")
    except OSError as err:
        # (The order of occurence of the filenames in the string is non-deterministic)
        assert some_symlink in str(err)
        assert some_other_symlink in str(err)


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_file():
    with pytest.raises(OSError, match="existing file"):
        copy_file("does/not/exist", "target")

    mkdir("path")
    with pytest.raises(OSError, match="existing file"):
        copy_file("path", "target")

    Path("file1").write_text("hei", encoding="utf-8")

    copy_file("file1", "file2")
    assert os.path.isfile("file2")

    copy_file("file1", "path")
    assert os.path.isfile("path/file1")

    copy_file("file1", "path2/file1")
    assert os.path.isfile("path2/file1")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_file2():
    mkdir("root/sub/path")

    Path("file").write_text("Hei ...", encoding="utf-8")

    copy_file("file", "root/sub/path/file")
    assert Path("root/sub/path/file").read_text(encoding="utf-8") == "Hei ..."

    Path("file2").write_text("Hei ...", encoding="utf-8")

    with pushd("root/sub/path"):
        copy_file("../../../file2")
        assert os.path.isfile("file2")


@pytest.mark.usefixtures("use_tmpdir")
def test_copy_file3():
    mkdir("rms/output")

    Path("file.txt").write_text("Hei", encoding="utf-8")
    copy_file("file.txt", "rms/output/")
    assert Path("rms/output/file.txt").read_text(encoding="utf-8") == "Hei"


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
    Path("file1").write_text("hei", encoding="utf-8")
    Path("file2").write_text("hallo", encoding="utf-8")
    careful_copy_file("file1", "file2")
    assert Path("file2").read_text(encoding="utf-8") == "hallo"

    print(careful_copy_file("file1", "file3"))
    assert Path("file3").is_file()


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
        Path("config.ert").write_text("NUM_REALIZATIONS 1", encoding="utf-8")
        yield


def test_shell_script_fmstep_availability(minimal_case):
    with ErtPluginContext() as ctx:
        ert_config = ErtConfig.with_plugins(ctx).from_file("config.ert")
    fm_shell_jobs = {}
    for fm_step in ert_config.installed_forward_model_steps.values():
        exe = fm_step.executable
        if "shell_scripts" in exe:
            fm_shell_jobs[fm_step.name.upper()] = Path(exe).resolve()

    wf_shell_jobs = {}
    for wf_name, wf in ert_config.workflow_jobs.items():
        if isinstance(wf, ExecutableWorkflow) and "shell_scripts" in wf.executable:
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
    with ErtPluginContext() as ctx:
        ert_config = ErtConfig.with_plugins(ctx).from_file("config.ert")
    found_jobs = set()
    for wf_name, wf in ert_config.workflow_jobs.items():
        if isinstance(wf, ExecutableWorkflow) and "shell_scripts" in wf.executable:
            assert wf_name in shell_job_names
            found_jobs.add(wf_name)

    assert len(shell_job_names) == len(found_jobs)
