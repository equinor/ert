import codecs
import sys
from subprocess import Popen, PIPE
from res.job_queue import ErtScript


class ExternalErtScript(ErtScript):
    def __init__(self, ert, executable):
        super(ExternalErtScript, self).__init__(ert)

        self.__executable = executable
        self.__job = None

    def run(self, *args):
        command = [self.__executable]
        command.extend([str(arg) for arg in args])

        self.__job = Popen(command, stdout=PIPE, stderr=PIPE)

        # The job will complete before stdout and stderr is returned
        self._stdoutdata, self._stderrdata = self.__job.communicate()

        self._stdoutdata = codecs.decode(self._stdoutdata, "utf8", "replace")
        self._stderrdata = codecs.decode(self._stderrdata, "utf8", "replace")

        sys.stdout.write(self._stdoutdata)

        if self.__job.returncode != 0:
            raise Exception(self._stderrdata)

        return None

    def cancel(self):
        super(ExternalErtScript, self).cancel()
        if self.__job is not None:
            self.__job.terminate()

            self.__job.kill()
