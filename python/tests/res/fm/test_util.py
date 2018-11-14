
from ecl.util.test import TestAreaContext
from tests import ResTest
from res.fm.util import *

class UtilTest(ResTest):
    def test_template_parser(self):
        with TestAreaContext("util/template") as tac:
            with open("template", 'w') as template_file:
                template_file.write(
                    "FILENAME\n"+
                    "F1 {{key1.subkey1}}\n"+
                    "FX {{key1:subkey2}}\n"+
                    "F1_T {{key2.subkey1}}")

            with open("parameters.json", 'w') as json_file:
                json_file.write('{\n'+\
                    '"key1": {\n'+\
                    '   "subkey1": 1999.22,\n'+\
                    '   "subkey2": 200 \n' +\
                    '},\n'+\
                    '"key2": {\n'+\
                    '   "subkey1": 300\n'+\
                    '},\n' +\
                    '"key1:subkey1":1999.22\n,'+\
                    '"key1:subkey2":200\n,' +\
                    '"key2:subkey1":300\n' +\
                    '}')

            build_from_template('template')

            with open("parameter_file", 'r') as parameter_file:
                print(parameter_file.readlines())
