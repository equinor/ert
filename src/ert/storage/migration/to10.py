import json
import shutil
from pathlib import Path

from ert.storage.local_storage import local_storage_get_ert_config

info = "Internalizes run templates in to the storage and removes genkw template"


def migrate(path: Path) -> None:
    ert_config = local_storage_get_ert_config()
    for experiment in path.glob("experiments/*"):
        templates_abs: list[tuple[str, str]] = []
        templates_path = experiment / "templates"
        templates_path.mkdir(parents=True, exist_ok=True)
        for idx, (src, dst) in enumerate(ert_config.ert_templates):
            incoming_template = Path(src)
            template_file_path = (
                templates_path
                / f"{incoming_template.stem}_{idx}{incoming_template.suffix}"
            )
            shutil.copyfile(incoming_template, template_file_path)
            templates_abs.append((str(template_file_path.relative_to(experiment)), dst))

        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)

        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            for param in parameters_json.values():
                if param["_ert_kind"] == "GenKwConfig":
                    param.pop("forward_init_file", None)
                    param.pop("template_file", None)
                    param.pop("output_file", None)
            fout.write(json.dumps(parameters_json, indent=4))

        with open(experiment / "templates.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(templates_abs))
