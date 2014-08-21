import os
import stat

class WorkflowCommon(object):

    @staticmethod
    def createExternalDumpJob():
        with open("dump_job", "w") as f:
            f.write("INTERNAL FALSE\n")
            f.write("EXECUTABLE dump.py\n")
            f.write("MIN_ARG 2\n")
            f.write("MAX_ARG 2\n")
            f.write("ARG_TYPE 0 STRING\n")
            f.write("ARG_TYPE 1 STRING\n")


        with open("dump.py", "w") as f:
            f.write("#!/usr/bin/env python\n")
            f.write("import sys\n")
            f.write("f = open('%s' % sys.argv[1], 'w')\n")
            f.write("f.write('%s' % sys.argv[2])\n")
            f.write("f.close()\n")

        st = os.stat("dump.py")
        os.chmod("dump.py", st.st_mode | stat.S_IEXEC) # | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

        with open("dump_workflow", "w") as f:
            f.write("DUMP dump1 dump_text_1\n")
            f.write("DUMP dump2 dump_<PARAM>_2\n")


    @staticmethod
    def createInternalScriptJob():

        # with open("ert_info.py", "w") as f:
        #     f.write("import ert.enkf.")

        with open("script_job", "w") as f:
            f.write("INTERNAL True\n")
            f.write("SCRIPT /path/to/script\n")
            f.write("MIN_ARG 1\n")
            f.write("MAX_ARG 1\n")
            f.write("ARG_TYPE 0 STRING\n")
