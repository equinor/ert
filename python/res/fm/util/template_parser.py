gen_kw_export_file = "parameters";

import re
import json
import os

class TemplateParser:

    @staticmethod
    def fill_in_template(template, reg_ex, json_file):
        new_lines = []
        with open(json_file) as f:
            json_data = json.load(f)

        with open(template, 'r') as f:
            for line in f.readlines():
                m = re.search(reg_ex, line)
                if m:
                    if len(m.group(1).split('.')) > 1:
                        key, subkey = m.group(1).split('.')
                        if key in json_data and subkey in json_data[key]:
                            new_lines.append(re.sub(reg_ex, str(json_data[key][subkey]), line))
                    else:
                        key = m.group(1)
                        new_lines.append(re.sub(reg_ex, str(json_data[key]), line))
                else:
                    new_lines.append(line)
        return new_lines

    @staticmethod
    def write_to_file(lines, filename):
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def build_from_template(template_file,output_file="parameter_file"):
        filled_in_lines = TemplateParser.fill_in_template(template=template_file, reg_ex='{{(.*)}}',json_file=gen_kw_export_file+'.json')
        path = os.path.dirname(template_file)
        TemplateParser.write_to_file(filled_in_lines, os.path.join(path,output_file))
