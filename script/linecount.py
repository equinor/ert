#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from os import path
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Optional, Tuple

import pytest

projectdir = Path(__file__).parent.parent


# done at 691316d2f
introduction_of_clang_format = datetime(2021, 8, 28, 0, 0, 0, tzinfo=timezone.utc)
# last py2 file was removed by 722092742
end_of_py2 = datetime(2019, 7, 16, 0, 0, 0, tzinfo=timezone.utc)


def get_revision_datetime(hash: str) -> datetime:
    return datetime.strptime(
        subprocess.check_output(
            ["git", "show", "-s", "--format='%ci'", hash],
            cwd=projectdir,
        ).decode(),
        "'%Y-%m-%d %H:%M:%S %z'\n",
    )


def get_revision_range(start: str, stop: str) -> Iterator[str]:
    return map(
        lambda rev: rev[1:-1],
        subprocess.check_output(
            ["git", "log", "--format='%H'", f"{start}..{stop}"],
            cwd=projectdir,
        )
        .decode()
        .split(),
    )


def reset_changes():
    subprocess.run(
        ["git", "reset", "-q", "--hard"],
        check=True,
        cwd=projectdir,
    )


def content_matches(file: str, pattern: str) -> bool:
    try:
        content = Path(file).read_text()
        return re.search(pattern, content) is not None
    except Exception:
        return False


def all_files() -> Iterator[str]:
    return map(
        lambda p: path.normpath(path.join(projectdir, p)) if not path.isabs(p) else p,
        subprocess.check_output(
            [
                "find",
                ".",
                "-type",
                "f",
                "-not",
                "-path",
                '"./.git/*"',
                "-not",
                "-name",
                '"linecount.py"',
            ],
            cwd=projectdir,
        )
        .decode()
        .split(),
    )


def files_to_rename_by_extension(extension, pattern) -> Iterator[str]:
    return filter(
        lambda file: os.path.isfile(file)
        and not file.endswith(extension)
        and content_matches(file, pattern),
        all_files(),
    )


def find_and_rename_by_extension(extension, pattern):
    for file in files_to_rename_by_extension(extension, pattern):
        subprocess.check_output(["git", "mv", file, file + extension], cwd=projectdir)


def checkout_revision(revision: str) -> None:
    subprocess.run(["git", "checkout", "-q", revision], check=True, cwd=projectdir)


def run_black():
    subprocess.run(["black", "-q", "."], cwd=projectdir)


def run_isort():
    subprocess.run(["isort", "*", "-q", "--profile=black"], cwd=projectdir)


def potential_python2_files() -> Iterator[str]:
    return filter(
        lambda file: os.path.isfile(file)
        and file.endswith(".py")
        and content_matches(file, "print ")
        and content_matches(file, "0[1-9]"),
        all_files(),
    )


def run_2_to_3():
    for file in potential_python2_files():
        subprocess.check_output(["2to3", "--nobackups", "-w", file], cwd=projectdir)


def run_clang_format():
    if not (projectdir / ".clang-format").exists():
        (projectdir / ".clang-format").write_text(
            """# clang-format
---
Language: Cpp
BasedOnStyle: LLVM
AccessModifierOffset: -4
IndentWidth: 4

# Some of our comments include insightful insight
ReflowComments: false
...
"""
        )
    subprocess.run(
        "find . -iname *.h -o -iname *.hpp -o -iname *.cpp -o -iname *.c"
        " | xargs timeout 5 clang-format -i",
        shell=True,
        cwd=projectdir,
        capture_output=True,
    )


def ensure_consistent_formatting(revision: str) -> None:
    find_and_rename_by_extension(".py", r"^#\!\s*/usr/bin/env\s*python")
    find_and_rename_by_extension(".sh", r"(^#\!\s*/bin/bash)|(^#\!/usr/bin/env\s*bash)")
    if get_revision_datetime(revision) <= end_of_py2:
        run_2_to_3()
    if get_revision_datetime(revision) < introduction_of_clang_format:
        run_clang_format()
    run_black()
    run_isort()


def run_pygount() -> Dict[str, Any]:
    return json.loads(
        subprocess.check_output(["pygount", "--format", "json"], cwd=projectdir)
    )


def collect_line_count(
    revisions: Iterator[str],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    for revision in revisions:
        checkout_revision(revision)
        ensure_consistent_formatting(revision)
        yield (revision, run_pygount())
        reset_changes()


class Categorizer:
    def __init__(self):
        self.files = {}

    def _get_pre_libres_merge_name(self, file: str, date: datetime) -> str:
        if date < introduction_of_clang_format:
            return file.replace("src/libres/", "")
        return file

    def _update_added_and_removed_files(self, date: datetime, hash: str) -> None:
        """
        Some files are counted twice as the result of splitting and merging
        of ert into libres/libecl etc. To fix this, we assume that
        files with the same path when removing "src/libres/" are the same files
        and use diff-tree to keep track of added and removed files.
        """
        file_changes = [
            l.split("\t")
            for l in subprocess.run(
                [
                    "git",
                    "diff-tree",
                    "--no-commit-id",
                    "--name-status",
                    "-r",
                    hash,
                ],
                cwd=projectdir,
                capture_output=True,
            )
            .stdout.decode()
            .split("\n")
            if l != ""
        ]
        for file in [f[1] for f in file_changes if f[0] == "A"]:
            file = self._get_pre_libres_merge_name(file, date)
            if file in self.files:
                del self.files[file]
            elif file + ".py" in self.files:
                del self.files[file + ".py"]

    # directories that are in the repository in some old commits,
    # but are now part of other projects
    SKIPPED_PATHS = [
        "devel/libecl",
        "devel/python/test_ecl",
        "devel/python/test/ert_tests/ecl",
        "libenkf/src/sqlite3",
        "script/linecount.py",  # avoid being self referential
    ]

    def __call__(self, hash: str, data: Dict[str, Any]) -> Tuple[Any, ...]:
        date = get_revision_datetime(hash)
        for file in data["files"]:
            path = file["path"]
            if any(skipped_dir in path for skipped_dir in self.SKIPPED_PATHS):
                continue
            path = self._get_pre_libres_merge_name(path, date)
            self.files[path] = file
        py_code = sum(
            f["sourceCount"]
            for f in self.files.values()
            if f["language"] == "Python"
            and "ert3" not in f["path"]
            and "test" not in f["path"]
        )
        ert3_code = sum(
            f["sourceCount"]
            for f in self.files.values()
            if f["language"] == "Python"
            and "ert3" in f["path"]
            and "test" not in f["path"]
        )
        c_code = sum(
            f["sourceCount"]
            for f in self.files.values()
            if f["language"] in ["C", "C++"] and "test" not in f["path"]
        )
        py_tests = sum(
            f["sourceCount"]
            for f in self.files.values()
            if f["language"] == "Python"
            and "test" in f["path"]
            and "ert3" not in f["path"]
        )
        ert3_tests = sum(
            f["sourceCount"]
            for f in self.files.values()
            if f["language"] == "Python" and "test" in f["path"] and "ert3" in f["path"]
        )
        c_tests = sum(
            f["sourceCount"]
            for f in self.files.values()
            if f["language"] in ["C", "C++"] and "test" in f["path"]
        )
        self._update_added_and_removed_files(date, hash)
        return date, c_code, c_tests, py_code, py_tests, ert3_code, ert3_tests


def run(arguments: argparse.Namespace) -> int:
    check_dependencies(
        date=min(
            [
                get_revision_datetime(arguments.from_revision),
                get_revision_datetime(arguments.to_revision),
            ]
        )
    )

    print("date, c code, c tests, py code, py tests, ert3 code, ert3 tests")
    categorizer = Categorizer()
    for code_lines in collect_line_count(
        get_revision_range(arguments.from_revision, arguments.to_revision)
    ):
        print(", ".join(str(count) for count in categorizer(*code_lines)))
    return 0


def is_valid_revision(argument: str) -> str:
    try:
        subprocess.check_output(f"git show {argument}", cwd=projectdir, shell=True)
    except subprocess.CalledProcessError as err:
        raise SystemExit(f"argument {argument!r} is not a valid revision") from err
    return argument


first_commit = "a1deb7f7aa88db30fc0f090ff7bd048ae9a53ab6"


def make_parser(prog="open_petro_elastic") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Counts the lines in the predefined categories python code, python tests, c code, c tests, ert3",
    )
    ap.add_argument(
        "--from-revision",
        type=is_valid_revision,
        default=first_commit,
        dest="from_revision",
        help="Optional additional csv file for data to be inserted into config.",
    )
    ap.add_argument(
        "--to-revision",
        type=is_valid_revision,
        default="HEAD",
        dest="to_revision",
        help="Optional additional csv file for data to be inserted into config.",
    )
    ap.add_argument("--version", action="version", version="1.0.0")
    return ap


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = make_parser(argv[0])
    return parser.parse_args(argv[1:])


def check_dependencies(
    depends_commands: Collection[Tuple[Optional[datetime], str]] = (
        (None, "git --version"),
        (None, "black --version"),
        (None, "isort --version"),
        (None, "pygount --version"),
        (introduction_of_clang_format, "clang-format --version"),
        (introduction_of_clang_format, "2to3 --help"),
    ),
    date: Optional[datetime] = None,
) -> None:
    try:
        for command_date, command in depends_commands:
            if command_date is None:
                subprocess.check_output(command, cwd=projectdir, shell=True)
            elif date and date < command_date:
                subprocess.check_output(command, cwd=projectdir, shell=True)
    except subprocess.CalledProcessError as err:
        raise SystemExit(
            f"linecount could not find the command {err.cmd.split()[0]!r}"
        ) from err


def check_for_clean_repo() -> None:
    changed = subprocess.run(
        "git diff --stat", cwd=projectdir, shell=True, capture_output=True
    ).stdout
    if changed != b"":
        raise SystemExit(f"linecount can only start from a clean repository")


def main() -> None:
    check_for_clean_repo()
    sys.exit(run(parse_arguments(sys.argv)))


if __name__ == "__main__":
    main()
else:
    check_for_clean_repo()

    @pytest.fixture
    def parser():
        return make_parser()

    def test_dependencies_check():
        non_existent_command = "HOPEFULLY_NO_SYSTEM_HAS_THIS_AS_A_COMMAND"
        with pytest.raises(
            SystemExit,
            match=f"linecount could not find the command {non_existent_command!r}",
        ):
            check_dependencies(((None, f"{non_existent_command} --version"),))
        check_dependencies()

    def test_get_revision_range():
        revisions = get_revision_range("HEAD~10", "HEAD")
        assert len(list(revisions)) == 10
        assert all(is_valid_revision(r) for r in revisions)

    def test_rename_python_files():
        checkout_revision("6d25eddd62aba562f20dd0dd5abaf43f68109286")
        rename_files = list(
            files_to_rename_by_extension(".py", r"^#!\s*/usr/bin/env\s*python")
        )
        assert [path.basename(f) for f in rename_files] == [
            "copy_ext_param_script",
            "mock_bsub",
            "mock_bjobs",
            "rms",
            "ecl_run_fail",
            "flow",
            "ecl100",
            "rms",
            "ecl300",
            "template_render",
            "copy_directory",
            "move_file",
            "copy_file",
            "careful_copy_file",
            "make_directory",
            "delete_directory",
            "symlink",
            "delete_file",
            "cmake-format",
            "clang-format",
        ]
        checkout_revision("main")

    def test_rename_bash_files():
        checkout_revision("6d25eddd62aba562f20dd0dd5abaf43f68109286")
        rename_files = list(
            files_to_rename_by_extension(
                ".sh", r"(^#\!\s*/bin/bash)|(^#\!/usr/bin/env\s*bash)"
            )
        )
        assert [path.basename(f) for f in rename_files] == ["run_demo"] * 2
        checkout_revision("main")

    def test_python2_files():
        checkout_revision("a7ced5dacfd13af230dd0df37f56d7040ee2ec07")
        rename_files = list(potential_python2_files())
        assert [path.basename(f) for f in rename_files] == [
            "sensitivity_study.py",
            "util.py",
            "analysis_module.py",
            "cwrap.py",
            "cenum.py",
            "install.py",
            "job_dispatch.py",
            "job_dispatch.py",
            "job_dispatch.py",
        ]

        checkout_revision("main")

    def test_revision_validation(parser):
        with pytest.raises(SystemExit, match="not a valid revision"):
            _ = parser.parse_args(["--from-revision", "hello"])
        with pytest.raises(SystemExit, match="not a valid revision"):
            _ = parser.parse_args(["--to-revision", "hello"])
        default_namespace = parser.parse_args([])
        assert default_namespace.from_revision == first_commit
        assert default_namespace.to_revision == "HEAD"
        namespace = parser.parse_args(
            ["--from-revision", "HEAD~1", "--to-revision", "HEAD"]
        )
        assert namespace.from_revision == "HEAD~1"
        assert namespace.to_revision == "HEAD"

    def test_get_revision_datetime():
        assert get_revision_datetime(first_commit) == datetime(
            2006, 10, 2, 11, 18, 37, tzinfo=timezone.utc
        )
