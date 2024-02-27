from __future__ import annotations

import codecs
import os
import stat
import sys
import tempfile
from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any, Optional

from .ert_script import ErtScript

if TYPE_CHECKING:
    from ert.enkf_main import EnKFMain
    from ert.storage import Storage


class ExternalErtScript(ErtScript):
    def __init__(self, ert: EnKFMain, storage: Storage, executable: str):
        super().__init__(ert, storage, None)

        self.__executable = executable
        self.__job: Optional[Popen[bytes]] = None

    def run(self, *args: Any) -> None:
        script_content = (
            Path(self.__executable).read_text(encoding="utf-8").splitlines()
        )
        tmp_script_path: Optional[str] = None
        if self.__executable[-2:] == "sh":
            script_content.insert(1, "set -e")
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", encoding="utf-8", delete=False
            ) as tmp_file:
                tmp_file.write("\n".join(script_content))
                tmp_file.flush()
                tmp_script_path = tmp_file.name
            os.chmod(tmp_script_path, os.stat(tmp_script_path).st_mode | stat.S_IEXEC)
            command = [tmp_script_path]
            command.extend([str(arg) for arg in args])

            # we take care to terminate the process in cancel()
            self.__job = Popen(command, stdout=PIPE, stderr=PIPE)

            # The job will complete before stdout and stderr is returned
            stdoutdata, stderrdata = self.__job.communicate()

            self._stdoutdata = codecs.decode(stdoutdata, "utf8", "replace")
            self._stderrdata = codecs.decode(stderrdata, "utf8", "replace")

            sys.stdout.write(self._stdoutdata)

            if self.__job.returncode != 0:
                raise RuntimeError(self._stderrdata)
        finally:
            if tmp_script_path is not None:
                os.remove(tmp_script_path)

    def cancel(self) -> Any:
        super().cancel()
        if self.__job is not None:
            self.__job.terminate()

            self.__job.kill()
