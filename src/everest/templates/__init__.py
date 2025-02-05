from importlib import util
from pathlib import Path

template_names = (
    "schmerge.tmpl",
    "slot_delay.tmpl",
    "well_delay.tmpl",
    "well_drill.tmpl",
    "well_order.tmpl",
)


def fetch_template(template_name: str) -> str:
    module_spec = util.find_spec("everest.templates")
    assert module_spec, "everest.templates not found"
    assert module_spec.origin, "everest.templates has no origin"

    return str(Path(module_spec.origin).parent / template_name)
