import os.path
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Type

from ert import (
    ForwardModelStepJSON,
    ForwardModelStepPlugin,
    ForwardModelStepValidationError,
)
from ert.config.forward_model_step import ForwardModelStepDocumentation
from ert.shared.plugins.plugin_manager import hook_implementation
from ert.shared.plugins.plugin_response import plugin_response


class CarefulCopyFile(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="CAREFUL_COPY_FILE",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../share/ert/shell_scripts/careful_copy_file.py"
                    ).resolve()
                ),
                "<FROM>",
                "<TO>",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            description=dedent(
                """
                The :code:`CAREFUL_COPY_FILE` job will copy a file. If the :code:`<TO>`
                argument has a directory component, that directory will be created.
                This is an extension of the normal :code:`cp` command
                which will *not* create directories in this way.
                This job superseded an older version called :code:`CAREFULL_COPY`
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
                        / "../../share/ert/shell_scripts/copy_directory.py"
                    ).resolve()
                ),
                "<FROM>",
                "<TO>",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
The job copies the directory :code:`<FROM>` to the target :code:`<TO>`. If
:code:`<TO>` points to a non-existing directory structure, it will be
created first. If the :code:`<TO>` folder already exist it creates a new directory within the existing one.
E.g. :code:`COPY_DIRECTORY (<FROM>=foo, <TO>=bar)` creates :code:`bar/foo` if the directory
:code:`bar` already exists. If :code:`bar` does not exist it becomes a copy of :code:`foo`.
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
                        Path(__file__) / "../../share/ert/shell_scripts/copy_file.py"
                    ).resolve()
                ),
                "<FROM>",
                "<TO>",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
The :code:`COPY_FILE` job will copy a file. If the :code:`<TO>`
argument has a directory component, that directory will be created.
This is an extension of the normal :code:`cp` command
which will *not* create directories in this way.
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
                        / "../../share/ert/shell_scripts/delete_directory.py"
                    ).resolve()
                ),
                "<DIRECTORY>",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
The :code:`DELETE_DIRECTORY` job will recursively remove a directory
and all the files in the directory. Like the :code:`DELETE_FILE` job
it will *only* delete files and directories which are owned by the
current user. If one delete operation fails the job will continue, but
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
                        Path(__file__) / "../../share/ert/shell_scripts/delete_file.py"
                    ).resolve()
                ),
                "<FILES>",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
The :code:`DELETE_FILE` job will *only* remove files which are owned
by the current user, *even if* file system permissions would have
allowed the delete operation to proceed. The :code:`DELETE_FILE` will
*not* delete a directory, and if presented with a symbolic link it
will *only* delete the link, and not the target.
""",
            examples="""
            .. code-block:: bash

                FORWARD_MODEL DELETE_FILE(<FILES>=path/file path/file2 path/fileN)
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
                        / "../../share/ert/forward-models/res/script/ecl100.py"
                    ).resolve()
                ),
                "<ECLBASE>",
                "-v",
                "<VERSION>",
                "-n",
                "<NUM_CPU>",
                "<OPTS>",
            ],
            default_mapping={"<NUM_CPU>": 1, "<OPTS>": ""},
        )

    def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
        if "<VERSION>" not in self.private_args:
            raise ForwardModelStepValidationError(
                "ECLIPSE100 job must be given a VERSION argument"
            )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="simulators.reservoir",
            examples="""
The version, number of cpu, and whether or not to ignore errors and whether or not to produce `YOUR_CASE_NAME.h5` output files can
be configured in the configuration file when adding the job, as such:


.. code-block:: bash

    FORWARD_MODEL ECLIPSE100(<ECLBASE>, <VERSION>=xxxx, <OPTS>={"--ignore-errors", "--summary-conversion"})

The :code:`OPTS` argument is optional and can be removed, fully or partially.
In absence of :code:`--ignore-errors` eclipse will fail on errors.
Adding the flag :code:`--ignore-errors` will result in eclipse ignoring errors.

And in absence of :code:`--summary-conversions` eclipse will run without producing `YOUR_CASE_NAME.h5` output files.
Add flag :code:`--summary-conversions` to produce `YOUR_CASE_NAME.h5` output files.
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
                        / "../../share/ert/forward-models/res/script/ecl300.py"
                    ).resolve()
                ),
                "<ECLBASE>",
                "-v",
                "<VERSION>",
                "-n",
                "<NUM_CPU>",
                "<OPTS>",
            ],
            default_mapping={"<NUM_CPU>": 1, "<OPTS>": "", "<VERSION>": "version"},
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
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
                        / "../../share/ert/forward-models/res/script/flow.py"
                    ).resolve()
                ),
                "<ECLBASE>",
                "-v",
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

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="simulators.reservoir",
            examples="""
The version, number of cpu and whether or not to ignore errors can
be configured in the configuration file when adding the job, as such:

.. code-block:: bash

    FORWARD_MODEL FLOW(<ECLBASE>, <VERSION>=xxxx, <OPTS>="--ignore-errors")

The :code:`OPTS` argument is optional and can be removed, thus running flow
without ignoring errors.
""",
            description="""
Forward model for OPM Flow
""",
        )


class MakeDirectory(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="MAKE_DIRECTORY",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../share/ert/shell_scripts/make_directory.py"
                    ).resolve()
                ),
                "<DIRECTORY>",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
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
                        Path(__file__) / "../../share/ert/shell_scripts/symlink.py"
                    ).resolve()
                ),
                "<TARGET>",
                "<LINKNAME>",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
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
                        Path(__file__) / "../../share/ert/shell_scripts/move_file.py"
                    ).resolve()
                ),
                "<FROM>",
                "<TO>",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            description="""
The :code:`MOVE_FILE` job will move file to target directory.
If file already exists, this job will move file to the target directory
and then replace the existing file.
""",
            examples="""
            .. code-block:: bash

                FORWARD_MODEL MOVE_FILE(<FROM>=file/to/move,<TO>=to/new/path)
""",
        )


class Rms(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="RMS",
            command=[
                str(
                    (
                        Path(__file__)
                        / "../../share/ert/forward-models/res/script/rms.py"
                    ).resolve()
                ),
                "<IENS>",
                "<RMS_PROJECT>",
                "<RMS_WORKFLOW>",
                "-r",
                "<RMS_RUNPATH>",
                "-t",
                "<RMS_TARGET_FILE>",
                "-i",
                "<RMS_IMPORT_PATH>",
                "-v",
                "<RMS_VERSION>",
                "-e",
                "<RMS_EXPORT_PATH>",
                "<RMS_OPTS>",
            ],
            target_file="<RMS_TARGET_FILE>",
            default_mapping={
                "<RMS_IMPORT_PATH>": "./",
                "<RMS_EXPORT_PATH>": "./",
                "<RMS_RUNPATH>": "rms/model",
                "<RMS_OPTS>": "",
            },
            exec_env={
                "PYTHONPATH": "<RMS_PYTHONPATH>",
                "PATH_PREFIX": "<RMS_PATH_PREFIX>",
            },
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="modelling.reservoir",
            description="""
Forward model for running a given workflow in an existing RMS-project.

The forward model requires explicit knowledge of the version used to
produce the RMS project loaded. The Python environment is adapted to
work with Python inside RMS.
""",
            examples="""
Running the forward model
#########################

RMS is usually incorporated in ERT configurations using statements like

.. code-block:: bash

    DEFINE  <RMS_PROJECT>         reek.rms11.0.1
    DEFINE  <RMS_VERSION>         11.0.1
    DEFINE  <RMS_WORKFLOW_NAME>   MAIN_WORKFLOW
    FORWARD_MODEL RMS(<IENS>=<IENS>, <RMS_VERSION>=<RMS_VERSION>, <RMS_PROJECT>=<CONFIG_PATH>/../../rms/model/<RMS_NAME>)


RMS script documentation
########################

The script must be invoked with minimum three positional arguments:

Positional arguments:
  IENS
        Realization number
  RMS_PROJECT
        The RMS project we are running
  RMS_WORKFLOW
        The rms workflow we intend to run

Optional arguments:
  -r, --run-path RUN_PATH
                        The directory which will be used as cwd when running
                        rms
  -t, --target-file TARGET_FILE
                        name of file which should be created/updated by rms
  -i, --import-path IMPORT_PATH
                        the prefix of all relative paths when rms is importing
  -e, --export-path EXPORT_PATH
                        the prefix of all relative paths when rms is exporting
  -v, --version VERSION
                        The version of rms to use
  -a, --allow-no-env
                        Boolean flag to allow RMS to run without a site configured
                        environment. WARNING: This is typically a sign of using an
                        unsupported version of RMS and the configuration of the
                        environment is now entirely on the shoulders of the user.
                        Consider contacting your system administrators.
""",
        )


class Symlink(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="SYMLINK",
            command=[
                str(
                    (
                        Path(__file__) / "../../share/ert/shell_scripts/symlink.py"
                    ).resolve()
                ),
                "<TARGET>",
                "<LINKNAME>",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
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
                os.path.abspath(
                    "../../share/ert/forward-models/templating/script/template_render.py"
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
    def documentation() -> Optional[ForwardModelStepDocumentation]:
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

    FORWARD_MODEL TEMPLATE_RENDER(<INPUT_FILES>=my_input.json, <TEMPLATE_FILE>=tmpl.jinja, <OUTPUT_FILE>=output_file)
""",
        )


@hook_implementation
@plugin_response(plugin_name="ert")
def installable_forward_model_steps() -> List[Type[ForwardModelStepPlugin]]:
    return [
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
        Rms,
        Symlink,
        TemplateRender,
    ]
