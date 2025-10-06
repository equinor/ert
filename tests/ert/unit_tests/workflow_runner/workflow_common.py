import os
import stat
from pathlib import Path


class WorkflowCommon:
    @staticmethod
    def createExternalDumpJob():
        Path("dump_job").write_text(
            "EXECUTABLE dump.py\nMIN_ARG 2\nMAX_ARG 2\nARG_TYPE 0 STRING\n",
            encoding="utf-8",
        )

        Path("dump_failing_job").write_text(
            "EXECUTABLE dump_failing.py\n", encoding="utf-8"
        )

        Path("dump.py").write_text(
            "#!/usr/bin/env python\n"
            "import sys\n"
            "f = open('%s' % sys.argv[1], 'w')\n"
            "f.write('%s' % sys.argv[2])\n"
            "f.close()\n"
            'print("Hello World")',
            encoding="utf-8",
        )

        Path("dump_failing.py").write_text(
            '#!/usr/bin/env python\nprint("Hello Failing")\nraise Exception',
            encoding="utf-8",
        )
        st = os.stat("dump.py")
        os.chmod(
            "dump.py", st.st_mode | stat.S_IEXEC
        )  # | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        st = os.stat("dump_failing.py")
        os.chmod("dump_failing.py", st.st_mode | stat.S_IEXEC)

        Path("dump_workflow").write_text(
            "DUMP dump1 dump_text_1\nDUMP dump2 dump_<PARAM>_2\n", encoding="utf-8"
        )

    @staticmethod
    def createErtScriptsJob():
        Path("subtract_script.py").write_text(
            "from ert import ErtScript\n"
            "class SubtractScript(ErtScript):\n"
            "    def run(self, *argv):\n"
            "        return argv[0] - argv[1]\n",
            encoding="utf-8",
        )

        Path("subtract_script_job").write_text(
            "INTERNAL True\n"
            "SCRIPT subtract_script.py\n"
            "MIN_ARG 2\n"
            "MAX_ARG 2\n"
            "ARG_TYPE 0 FLOAT\n"
            "ARG_TYPE 1 FLOAT\n",
            encoding="utf-8",
        )

    @staticmethod
    def createWaitJob():
        Path("wait_job.py").write_text(
            "import time\n"
            "from pathlib import Path\n"
            "\n"
            "from ert import ErtScript\n"
            "\n"
            "class WaitScript(ErtScript):\n"
            "    def dump(self, filename, content):\n"
            "        Path(filename).write_text(content, encoding='utf-8')\n"
            "\n"
            "    def run(self, *argv):\n"
            "        number, wait_time = argv\n"
            "        self.dump('wait_started_%d' % number, 'text')\n"
            "        start = time.time()\n"
            "        diff = 0\n"
            "        while not self.isCancelled() and diff < wait_time: \n"
            "           time.sleep(0.2)\n"
            "           diff = time.time() - start\n"
            "        if self.isCancelled():\n"
            "            self.dump('wait_cancelled_%d' % number, 'text')\n"
            "        else:\n"
            "            self.dump('wait_finished_%d' % number, 'text')\n"
            "        return None\n",
            encoding="utf-8",
        )

        Path("external_wait_job.sh").write_text(
            "#!/usr/bin/env bash\n"
            'echo "text" > wait_started_$1\n'
            "sleep $2\n"
            'echo "text" > wait_finished_$1\n',
            encoding="utf-8",
        )

        st = os.stat("external_wait_job.sh")
        os.chmod(
            "external_wait_job.sh", st.st_mode | stat.S_IEXEC
        )  # | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

        Path("wait_job").write_text(
            "INTERNAL True\n"
            "SCRIPT wait_job.py\n"
            "MIN_ARG 2\n"
            "MAX_ARG 2\n"
            "ARG_TYPE 0 INT\n"
            "ARG_TYPE 1 INT\n",
            encoding="utf-8",
        )

        Path("external_wait_job").write_text(
            "EXECUTABLE external_wait_job.sh\n"
            "MIN_ARG 2\n"
            "MAX_ARG 2\n"
            "ARG_TYPE 0 INT\n"
            "ARG_TYPE 1 INT\n",
            encoding="utf-8",
        )

        Path("wait_workflow").write_text(
            "WAIT 0 1\nWAIT 1 10\nWAIT 2 1\n", encoding="utf-8"
        )
        Path("fast_wait_workflow").write_text(
            "WAIT 0 1\nEXTERNAL_WAIT 1 1\n", encoding="utf-8"
        )
