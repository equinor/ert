from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from ert.storage.local_storage import local_storage_get_ert_config

if TYPE_CHECKING:
    from pathlib import Path


def migrate(path: Path) -> None:
    ert_config = local_storage_get_ert_config()
    ens_config = ert_config.ensemble_config
    for experiment in path.glob("experiments/*"):
        gen_kw_file = experiment / "gen-kw-priors.json"
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)
        if gen_kw_file.exists():
            with open(gen_kw_file, encoding="utf-8") as fin:
                parameter_info = json.load(fin)
            for parameter in parameter_info:
                if parameter in ens_config.parameter_configs:
                    parameter_config = ens_config.parameter_configs[parameter]
                    parameters_json[parameter] = parameter_config.to_dict()
                    parameter_config.save_experiment_data(experiment)

            os.remove(gen_kw_file)
            with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
                fout.write(json.dumps(parameters_json))
        else:
            key_to_remove = "template_file_path"
            for param_group in parameters_json:
                if key_to_remove in parameters_json[param_group]:
                    del parameters_json[param_group][key_to_remove]
            with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
                fout.write(json.dumps(parameters_json))
