from __future__ import annotations

import codecs
import sys
from subprocess import PIPE, Popen
from typing import Any

from .ert_script import ErtScript, ExternalScriptError


class ExternalErtScript(ErtScript):
    def __init__(self, executable: str) -> None:
        super().__init__()

        self.__executable = executable
        self.__job: Popen[bytes] | None = None

    def run(self, *args: Any) -> None:
        command = [self.__executable]
        command.extend([str(arg) for arg in args])

        # we take care to terminate the process in cancel()
        self.__job = Popen(command, stdout=PIPE, stderr=PIPE)

        # The job will complete before stdout and stderr is returned
        stdoutdata, stderrdata = self.__job.communicate()

        # Written to the current stdout/stderr, which ErtScript captures into
        # self.stdoutdata/self.stderrdata while the script is running.
        sys.stdout.write(codecs.decode(stdoutdata, "utf8", "replace"))
        sys.stderr.write(codecs.decode(stderrdata, "utf8", "replace"))

        if self.__job.returncode != 0:
            raise ExternalScriptError(
                f"{self.__executable} failed with exit code {self.__job.returncode}"
            )

    def cancel(self) -> Any:
        super().cancel()
        if self.__job is not None:
            self.__job.terminate()

            self.__job.kill()
