import importlib
from pathlib import Path

template_names = (
    "schmerge.tmpl",
    "slot_delay.tmpl",
    "well_delay.tmpl",
    "well_drill.tmpl",
    "well_order.tmpl",
)


def fetch_template(template_name: str) -> str:
    return str(
        Path(importlib.util.find_spec("everest.templates").origin).parent
        / template_name
    )
