import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Literal

from ert import (
    ForwardModelStepDocumentation,
    ForwardModelStepJSON,
    ForwardModelStepPlugin,
    ForwardModelStepValidationError,
    ForwardModelStepWarning,
    plugin,
)


class CarefulCopyFile(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="CAREFUL_COPY_FILE",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/shell_scripts/careful_copy_file.py"
                    ).resolve()
                ),
                "<FROM>",
                "<TO>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            description=dedent(
                """
                :code:`CAREFUL_COPY_FILE` will copy a file. If the :code:`<TO>`
                argument has a directory component, that directory will be created.
                This is an extension of the normal :code:`cp` command
                which will *not* create directories in this way.
                This supersedes an older version called :code:`CAREFULL_COPY`
                and should be used instead.
                """
            ),
            examples="""
.. code-block:: bash

    FORWARD_MODEL CAREFUL_COPY_FILE(<FROM>=file1, <TO>=path/to/directory/file1)
""",
            category="utility.file_system",
        )


class CopyDirectory(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="COPY_DIRECTORY",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/shell_scripts/copy_directory.py"
                    ).resolve()
                ),
                "<FROM>",
                "<TO>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
Copies the directory :code:`<FROM>` to the target :code:`<TO>`. If
:code:`<TO>` points to a non-existing directory structure, it will be
created first. If the :code:`<TO>` folder already exist it creates a new
directory within the existing one. E.g. :code:`COPY_DIRECTORY (<FROM>=foo, <TO>=bar)`
creates :code:`bar/foo` if the directory :code:`bar` already exists. If :code:`bar`
does not exist it becomes a copy of :code:`foo`.
            """,
            examples="""
.. code-block:: bash

    FORWARD_MODEL COPY_DIRECTORY(<FROM>=dir1, <TO>=path/to/directory/dir1)
""",
        )


class CopyFile(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="COPY_FILE",
            command=[
                str(
                    (
                        Path(__file__) / "../../../resources/shell_scripts/copy_file.py"
                    ).resolve()
                ),
                "<FROM>",
                "<TO>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
Copies file from :code:`<FROM>` to :code:`<TO>`. If no directory is specified
in :code:`<TO>`, the file will be copied to :code:`RUNPATH`. If the :code:`<TO>`
argument includes a directory component, that directory will be created. Unlike
the standard :code:`cp` command, this will automatically create any missing
directories in the destination path.
""",
            examples="""
.. code-block:: bash

    FORWARD_MODEL COPY_FILE(<FROM>=file1, <TO>=path/to/directory/file1)
""",
        )


class DeleteDirectory(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="DELETE_DIRECTORY",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/shell_scripts/delete_directory.py"
                    ).resolve()
                ),
                "<DIRECTORY>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
:code:`DELETE_DIRECTORY` will recursively remove a directory
and all the files in the directory. Like :code:`DELETE_FILE`
it will *only* delete files and directories which are owned by the
current user. If one delete operation fails it will continue, but
unless all delete calls succeed (parts of) the directory structure
will remain.

.. warning::
  If the directory to delete is a symlink to a directory, it will only delete
  the link and not the directory. However, if you add a trailing slash to the
  directory name (the symlink), then the link itself is kept, but the directory
  it links to will be removed.

""",
            examples="""
.. code-block:: bash

    FORWARD_MODEL DELETE_DIRECTORY(<DIRECTORY>=path/to/directory)
""",
        )


class DeleteFile(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="DELETE_FILE",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/shell_scripts/delete_file.py"
                    ).resolve()
                ),
                "<FILES>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
:code:`DELETE_FILE` will *only* remove files which are owned
by the current user, *even if* file system permissions would have
allowed the delete operation to proceed. The :code:`DELETE_FILE` will
*not* delete a directory, and if presented with a symbolic link it
will *only* delete the link, and not the target.
""",
            examples="""
            .. code-block:: bash

                FORWARD_MODEL DELETE_FILE(<FILES>="path/file path/file2 path/fileN")
""",
        )


class Eclipse100(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="ECLIPSE100",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/forward_models/run_reservoirsimulator.py"
                    ).resolve()
                ),
                "eclipse",
                "<ECLBASE>",
                "--version",
                "<VERSION>",
                "-n",
                "<NUM_CPU>",
                "<OPTS>",
            ],
            default_mapping={"<NUM_CPU>": 1, "<OPTS>": ""},
        )

    def validate_pre_experiment(self, fm_json: ForwardModelStepJSON) -> None:
        if "<VERSION>" not in self.private_args:
            raise ForwardModelStepValidationError(
                "Forward model step ECLIPSE100 must be given a VERSION argument"
            )
        version = self.private_args["<VERSION>"]
        available_versions = _available_eclrun_versions(
            simulator="eclipse", env_vars=fm_json["environment"]
        )

        if available_versions and version not in available_versions:
            raise ForwardModelStepValidationError(
                f"Unavailable ECLIPSE100 version {version}. "
                f"Available versions: {available_versions}"
            )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            description="The Eclipse 100 black-oil reservoir simulator from SLB",
            category="simulators.reservoir",
            examples="""
The version, number of cpu, and whether or not to ignore errors and whether
or not to produce `YOUR_CASE_NAME.h5` output files can be configured in the
configuration file when adding the job, as such:


.. code-block:: bash

    FORWARD_MODEL ECLIPSE100(<ECLBASE>, <VERSION>=xxxx, \
        <OPTS>={"--ignore-errors", "--summary-conversion"})

The :code:`OPTS` argument is optional and can be removed, fully or partially.
In absence of :code:`--ignore-errors` eclipse will fail on errors.
Adding the flag :code:`--ignore-errors` will result in eclipse ignoring errors.

And in absence of :code:`--summary-conversions` eclipse will run without producing
`YOUR_CASE_NAME.h5` output files. Add flag :code:`--summary-conversions` to produce
`YOUR_CASE_NAME.h5` output files.
""",
        )


class Eclipse300(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="ECLIPSE300",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/forward_models/run_reservoirsimulator.py"
                    ).resolve()
                ),
                "e300",
                "<ECLBASE>",
                "--version",
                "<VERSION>",
                "-n",
                "<NUM_CPU>",
                "<OPTS>",
            ],
            default_mapping={"<NUM_CPU>": 1, "<OPTS>": "", "<VERSION>": "version"},
        )

    def validate_pre_experiment(
        self,
        fm_step_json: ForwardModelStepJSON,
    ) -> None:
        if "<VERSION>" not in self.private_args:
            raise ForwardModelStepValidationError(
                "Forward model step ECLIPSE300 must be given a VERSION argument"
            )
        version = self.private_args["<VERSION>"]
        available_versions = _available_eclrun_versions(
            simulator="e300", env_vars=fm_step_json["environment"]
        )
        if available_versions and version not in available_versions:
            raise ForwardModelStepValidationError(
                f"Unavailable ECLIPSE300 version {version}. "
                f"Available versions: {available_versions}"
            )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            description="The Eclipse 300 compositional reservoir simulator from SLB",
            category="simulators.reservoir",
            examples="""
The version, number of cpu and whether or not to ignore errors can
be configured in the configuration file when adding the job, as such:

.. code-block:: bash

    FORWARD_MODEL ECLIPSE300(<ECLBASE>, <VERSION>=xxxx, <OPTS>="--ignore-errors")

The :code:`OPTS` argument is optional and can be removed, thus running eclipse
without ignoring errors
""",
        )


class Flow(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="FLOW",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/forward_models/run_reservoirsimulator.py"
                    ).resolve()
                ),
                "flow",
                "<ECLBASE>",
                "--version",
                "<VERSION>",
                "-n",
                "<NUM_CPU>",
                "<OPTS>",
            ],
            default_mapping={
                "<VERSION>": "default",
                "<NUM_CPU>": 1,
                "<OPTS>": "",
            },
        )

    def validate_pre_experiment(self, fm_json: ForwardModelStepJSON) -> None:
        allowed_args = {"<VERSION>", "<NUM_CPU>", "<OPTS>", "<ECLBASE>"}
        if unknowns := set(self.private_args) - allowed_args:
            raise ForwardModelStepWarning(
                f"Unknown option(s) supplied to Flow: {sorted(unknowns)}"
            )
        available_versions = _available_flow_versions(env_vars=fm_json["environment"])
        version = fm_json["argList"][fm_json["argList"].index("--version") + 1]
        if version not in available_versions:
            raise ForwardModelStepValidationError(
                f"Unavailable Flow version {version}. "
                f"Available versions: {available_versions}"
            )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="simulators.reservoir",
            examples="""
.. code-block:: bash

    FORWARD_MODEL FLOW(<ECLBASE>, <VERSION>=xxx, <OPTS>="--ignore-errors")

The :code:`OPTS` argument is optional and can be removed. :code:`ECLBASE` can
also be defaulted. Multiple options can be supplied by separating them
with a space.

ERT will be able to run the flow simulator if there is a binary named
:code:`flow` or the wrapper :code:`flowrun` in the users :code:`$PATH`
environment variable.

If :code:`flowrun` is found, it will take precedence, and then it will be
possible to select the version of flow to use by setting :code:`<VERSION>`.
Available versions are verified towards what :code:`flowrun --report-versions`
returns.
""",
            description="""Forward model for OPM Flow simulator""",
        )


class MakeDirectory(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="MAKE_DIRECTORY",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/shell_scripts/make_directory.py"
                    ).resolve()
                ),
                "<DIRECTORY>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
Will create the directory :code:`<DIRECTORY>`, with all sub
directories.
""",
            examples="""
            .. code-block:: bash

                FORWARD_MODEL MAKE_DIRECTORY(<DIRECTORY>=path/to/new_directory)
""",
        )


class MakeSymlink(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="MAKE_SYMLINK",
            command=[
                str(
                    (
                        Path(__file__) / "../../../resources/shell_scripts/symlink.py"
                    ).resolve()
                ),
                "<TARGET>",
                "<LINKNAME>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
Will create a symbolic link with name :code:`<LINKNAME>` which points to
:code:`<TARGET>`. If :code:`<LINKNAME>` already exists, it will be updated.
""",
            examples="""
            .. code-block:: bash

                FORWARD_MODEL MAKE_SYMLINK(<TARGET>=path/to/target,<LINKNAME>=linkname)
""",
        )


class MoveFile(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="MOVE_FILE",
            command=[
                str(
                    (
                        Path(__file__) / "../../../resources/shell_scripts/move_file.py"
                    ).resolve()
                ),
                "<FROM>",
                "<TO>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
:code:`MOVE_FILE` will move file to target directory.
If file already exists, it will move the file to the target directory
and then replace the existing file.
""",
            examples="""
            .. code-block:: bash

                FORWARD_MODEL MOVE_FILE(<FROM>=file/to/move,<TO>=to/new/path)
""",
        )


class MoveDirectory(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="MOVE_DIRECTORY",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/shell_scripts/move_directory.py"
                    ).resolve()
                ),
                "<FROM>",
                "<TO>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
:code:`MOVE_DIRECTORY` will move a directory.
If the target directory already exists, it will be replaced.
""",
            examples="""
            .. code-block:: bash

                FORWARD_MODEL MOVE_DIRECTORY(<FROM>=dir/to/move,<TO>=to/new/path)
""",
        )


class Symlink(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="SYMLINK",
            command=[
                str(
                    (
                        Path(__file__) / "../../../resources/shell_scripts/symlink.py"
                    ).resolve()
                ),
                "<TARGET>",
                "<LINKNAME>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
Will create a symbolic link with name :code:`<LINKNAME>` which points to
:code:`<TARGET>`. If :code:`<LINKNAME>` already exists, it will be updated.
""",
            examples="""
            .. code-block:: bash

FORWARD_MODEL SYMLINK(<TARGET>=path/to/target,<LINKNAME>=linkname)
""",
        )


class TemplateRender(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="TEMPLATE_RENDER",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../../resources/forward_models/template_render.py"
                    ).resolve()
                ),
                "-i",
                "<INPUT_FILES>",
                "-o",
                "<OUTPUT_FILE>",
                "-t",
                "<TEMPLATE_FILE>",
            ],
            default_mapping={
                "<VERSION>": "default",
                "<NUM_CPU>": 1,
                "<OPTS>": "",
            },
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
Loads the data from each file (:code:`some/path/filename.xxx`) in :code:`INPUT_FILES`
and exposes it as the variable :code:`filename`. It then loads the Jinja2
template :code:`TEMPLATE_FILE` and dumps the rendered result to :code:`OUTPUT`.
""",
            examples="""
Given an input file :code:`my_input.json`:

.. code-block:: json

    {
        "my_variable": "my_value"
    }

And a template file :code:`tmpl.jinja`:

.. code-block:: bash

    This is written in my file together with {{my_input.my_variable}}

This job will produce an output file:

.. code-block:: bash

    This is written in my file together with my_value

By invoking the :code:`FORWARD_MODEL` as such:

.. code-block:: bash

    FORWARD_MODEL TEMPLATE_RENDER(<INPUT_FILES>=my_input.json, \
        <TEMPLATE_FILE>=tmpl.jinja, <OUTPUT_FILE>=output_file)
""",
        )


_UpperCaseFMSteps: list[type[ForwardModelStepPlugin]] = [
    CarefulCopyFile,
    CopyDirectory,
    CopyFile,
    DeleteDirectory,
    DeleteFile,
    Eclipse100,
    Eclipse300,
    Flow,
    MakeDirectory,
    MakeSymlink,
    MoveFile,
    MoveDirectory,
    Symlink,
    TemplateRender,
]


# Legacy:
# Mimicking old style lower case forward models, which pointed only to
# executables with no validation.
def _create_lowercase_fm_step_cls_with_only_executable(
    fm_step_name: str, executable: str
) -> type[ForwardModelStepPlugin]:
    class _LowerCaseFMStep(ForwardModelStepPlugin):
        def __init__(self) -> None:
            super().__init__(name=fm_step_name, command=[executable])

        @staticmethod
        def documentation() -> ForwardModelStepDocumentation | None:
            return None

    return _LowerCaseFMStep


_LowerCaseFMSteps: list[type[ForwardModelStepPlugin]] = []
for fm_step_subclass in _UpperCaseFMSteps:
    assert issubclass(fm_step_subclass, ForwardModelStepPlugin)
    inst = fm_step_subclass()  # type: ignore
    _LowerCaseFMSteps.append(
        _create_lowercase_fm_step_cls_with_only_executable(
            inst.name.lower(), inst.executable
        )
    )


@plugin(name="ert")
def installable_forward_model_steps() -> list[type[ForwardModelStepPlugin]]:
    return [*_UpperCaseFMSteps, *_LowerCaseFMSteps]


def _available_flow_versions(env_vars: dict[str, str]) -> list[str]:
    default_versions: list[str] = ["default"]
    flowrun_path: str = env_vars.get("FLOWRUN_PATH", "")
    runner_abspath = shutil.which(Path(flowrun_path) / "flowrun")
    if runner_abspath is None:
        return default_versions
    try:
        versionlines = (
            subprocess.check_output(
                [
                    runner_abspath,
                    "--report-versions",
                ],
            )
            .decode("utf-8")
            .splitlines()
        )
        return sorted([line.split(":")[0].strip() for line in versionlines])
    except subprocess.CalledProcessError:
        return default_versions


def _available_eclrun_versions(
    simulator: Literal["eclipse", "e300"], env_vars: dict[str, str]
) -> list[str]:
    eclrun_path = env_vars.get("ECLRUN_PATH", "")
    try:
        eclrun_abspath = shutil.which(Path(eclrun_path) / "eclrun")
        if eclrun_abspath is None:
            return []
        return (
            subprocess.check_output(
                [
                    eclrun_abspath,
                    simulator,
                    "--report-versions",
                ],
                env=env_vars,
            )
            .decode("utf-8")
            .strip()
            .split(" ")
        )
    except subprocess.CalledProcessError:
        return []
