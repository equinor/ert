import ert3

import json


def load_record(workspace, record_name, record_file):
    record = json.load(record_file)
    record_file.close()

    ert3.storage.add_variables(
        workspace,
        record_name,
        record,
    )
