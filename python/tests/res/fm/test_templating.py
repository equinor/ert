
from ecl.util.test import TestAreaContext
from tests import ResTest
from tests.utils import tmpdir
from res.fm.templating import *

import jinja2

import subprocess
import os
import json
import yaml


class TemplatingTest(ResTest):
    well_drill_tmpl ='PROD1 takes value {{ well_drill.PROD1 }}, implying {{ "on" if well_drill.PROD1 >= 0.5 else "off" }}\n' \
                         'PROD2 takes value {{ well_drill.PROD2 }}, implying {{ "on" if well_drill.PROD2 >= 0.5 else "off" }}\n' \
                         '---------------------------------- \n' \
                         '{%- for well in well_drill.INJ %}\n' \
                         '{{ well.name }} takes value {{  well.value|round(1) }}, implying {{ "on" if  well.value >= 0.5 else "off"}}\n' \
                         '{%- endfor %}'

    optimal_template = '{{well_drill.values() | sum()}}'
    dual_input = '{{ well_drill_north.PROD1 }} vs {{ well_drill_south.PROD1 }}'

    mulitple_input_template = "FILENAME\n"+\
                    "F1 {{parameters.key1.subkey1}}\n"+\
                    "OTH {{second.key1.subkey2}}\n"+\
                    "OTH_TEST {{third.key1.subkey1}}"

    default_parameters = {
        "key1":{
            "subkey1":1999.22,
            "subkey2":200
        },
        "key2":{
            "subkey1":300
        }
    }

    @tmpdir()
    def test_render_invalid(self):
        with TestAreaContext("templating") as tac:

            prod_wells = { 'PROD%d' % idx: 0.3*idx for idx in range(4) }
            prod_in = 'well_drill.json'
            with open(prod_in, 'w') as fout:
                json.dump(prod_wells, fout)
            with open('parameters.json', 'w') as fout:
                json.dump(self.default_parameters, fout)
            with open('template_file','w') as fout:
                fout.write(self.well_drill_tmpl)

            wells_out = 'wells.out'

            #undefined template elements
            with self.assertRaises(jinja2.exceptions.UndefinedError):
                render_template(None, 'template_file', wells_out)

            #file not found
            with self.assertRaises(ValueError):
                render_template(2*prod_in, 'template_file', wells_out)

            #no template file
            with self.assertRaises(TypeError):
                render_template(prod_in, None, wells_out)

            #templatefile not found
            with self.assertRaises(ValueError):
                render_template(prod_in, 'template_file'+'nogo', wells_out)

            #no output file
            with self.assertRaises(TypeError):
                render_template(prod_in, 'template_file', None)

    @tmpdir()
    def test_render(self):

        wells = { 'PROD%d' % idx: 0.2*idx for idx in range(1, 5) }
        wells.update( {'INJ': [ {'name':'INJ{0}'.format(idx),'value': 1-0.2*idx} for idx in range(1, 5) ]} )
        wells_in = 'well_drill.json'
        wells_tmpl = 'well_drill_tmpl'
        wells_out = 'wells.out'

        with open(wells_in, 'w') as fout:
            json.dump(wells, fout)
        with open('parameters.json', 'w') as fout:
            json.dump(self.default_parameters, fout)
        with open(wells_tmpl, 'w') as fout:
            fout.write(self.well_drill_tmpl)

        render_template(wells_in, wells_tmpl, wells_out)
        self.maxDiff = None
        expected_template_out = [
            'PROD1 takes value 0.2, implying off\n',
            'PROD2 takes value 0.4, implying off\n',
            '----------------------------------\n',
            'INJ1 takes value 0.8, implying on\n',
            'INJ2 takes value 0.6, implying on\n',
            'INJ3 takes value 0.4, implying off\n',
            'INJ4 takes value 0.2, implying off'
        ]

        with open(wells_out) as fin:
            output = fin.readlines()

        self.assertEqual(expected_template_out, output)

    @tmpdir()
    def test_template_multiple_input(self):
        with open("template", 'w') as template_file:
            template_file.write(self.mulitple_input_template)

        with open("parameters.json", 'w') as json_file:
            json_file.write(json.dumps(self.default_parameters))

        with open("second.json", 'w') as json_file:
            parameters = {
                "key1":{
                    "subkey2":1400
                }
            }
            json.dump(parameters, json_file)
        with open("third.json", 'w') as json_file:
            parameters = {
                "key1":{
                    "subkey1":3000.22,
                }
            }
            json.dump(parameters, json_file)

        render_template(['second.json', 'third.json'], 'template', 'out_file')

        with open("out_file", 'r') as parameter_file:
            expected_output = "FILENAME\n"+ \
                              "F1 1999.22\n"+ \
                              "OTH 1400\n"+ \
                              "OTH_TEST 3000.22"

            self.assertEqual(parameter_file.read(), expected_output)

    @tmpdir()
    def test_template_executable(self):
        with TestAreaContext("templating") as tac:
            with open("template", 'w') as template_file:
                template_file.write("FILENAME\n"+\
                                    "F1 {{parameters.key1.subkey1}}\n" +\
                                    "F2 {{other.key1.subkey1}}")

            with open("parameters.json", 'w') as json_file:
                json_file.write(json.dumps(self.default_parameters))

            with open("other.json", 'w') as json_file:
                parameters = {
                    "key1":{
                        "subkey1":200,
                    }
                }
                json_file.write(json.dumps(parameters))

            params = ' --output_file out_file --template_file template --input_files other.json'
            template_render_exec = os.path.join(self.SOURCE_ROOT, 'share/ert/forward-models/templating/script/template_render')

            subprocess.call(template_render_exec +  params, shell=True,stdout=subprocess.PIPE)

            with open("out_file", 'r') as parameter_file:
                expected_output = "FILENAME\n"+ \
                                  "F1 1999.22\n"+ \
                                  "F2 200"
                self.assertEqual(parameter_file.read(), expected_output)

    @tmpdir()
    def test_load_parameters(self):

        with TestAreaContext("templating") as tac:
            with open("parameters.json", 'w') as json_file:
                json_file.write(json.dumps(self.default_parameters))

            input_parameters = load_parameters()

            self.assertEqual(input_parameters, self.default_parameters)
