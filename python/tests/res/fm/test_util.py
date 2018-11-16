
from ecl.util.test import TestAreaContext
from tests import ResTest
from res.fm.util import *

import subprocess
import os
import sys

class UtilTest(ResTest):
    def test_template_parser(self):
        with TestAreaContext("util/template") as tac:
            with open("template", 'w') as template_file:
                template_file.write(
                    "FILENAME\n"+
                    "F1 {{parameters.key1.subkey1}}\n"+
                    "FX {{parameters.key1.subkey2}}\n"+
                    "F1_T {{parameters.key2.subkey1}}")

            with open("parameters.json", 'w') as json_file:
                json_file.write('{\n'+\
                    '   "key1": {\n'+\
                    '       "subkey1": 1999.22,\n'+\
                    '       "subkey2": 200 \n' +\
                    '   },\n'+\
                    '   "key2": {\n'+\
                    '       "subkey1": 300\n'+\
                    '   },'+\
                    '}')

            render_template('parameters.json','template','parameter_file')

            with open("parameter_file", 'r') as parameter_file:
                expected_output = "FILENAME\n"+\
                    "F1 1999.22\n"+\
                    "FX 200\n"+\
                    "F1_T 300"

                self.assertEqual(parameter_file.read(), expected_output)

    def test_template_executable(self):
        with TestAreaContext("util/template") as tac:
            with open("template", 'w') as template_file:
                template_file.write(
                    "FILENAME\n"+
                    "F1 {{parameters.key1.subkey1}}\n"+
                    "FX {{parameters.key1.subkey2}}\n"+
                    "F1_T {{parameters.key2.subkey1}}")

            with open("parameters.json", 'w') as json_file:
                json_file.write('{\n'+ \
                                '   "key1": {\n'+ \
                                '       "subkey1": 1999.22,\n'+ \
                                '       "subkey2": 200 \n' + \
                                '   },\n'+ \
                                '   "key2": {\n'+ \
                                '       "subkey1": 300\n'+ \
                                '   },'+ \
                                '}')

           
            params = ' --output_file out_file --template_file template --input_files parameters.json'
            template_render_exec = os.path.join(self.SOURCE_ROOT, 'share/ert/forward-models/util/script/template_render')

            subprocess.call(template_render_exec +  params, shell=True,stdout=subprocess.PIPE)

            with open("out_file", 'r') as parameter_file:
                expected_output = "FILENAME\n"+ \
                                  "F1 1999.22\n"+ \
                                  "FX 200\n"+ \
                                  "F1_T 300"

                self.assertEqual(parameter_file.read(), expected_output)