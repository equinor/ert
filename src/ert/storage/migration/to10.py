import json
import shutil
from pathlib import Path

from ert.storage.local_storage import local_storage_get_ert_config

info = "Internalizes run templates in to the storage"


def migrate(path: Path) -> None:
    ert_config = local_storage_get_ert_config()
    if ert_config.ert_templates:
        for experiment in path.glob("experiments/*"):
            templates_abs: list[tuple[str, str]] = []
            templates_path = experiment / "templates"
            templates_path.mkdir(parents=True, exist_ok=True)
            for src, dst in ert_config.ert_templates:
                incoming_template_file_path = Path(src)
                template_file_path = Path(
                    templates_path / incoming_template_file_path.name
                )
                shutil.copyfile(incoming_template_file_path, template_file_path)
                templates_abs.append((str(template_file_path.resolve()), dst))
            with open(experiment / "templates.json", "w", encoding="utf-8") as fout:
                fout.write(json.dumps(templates_abs))
