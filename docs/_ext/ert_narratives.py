import json
import logging
import sys
from docutils import nodes, statemachine
from docutils.parsers.rst import Directive
from os.path import basename
from pathlib import Path
from sphinx.application import Sphinx

from ert.shared.ensemble_evaluator import narratives

TMPL = """.. datatemplate:json:: {source_path}
    :template: ert_narratives.tmpl"""


class ErtNarratives(Directive):
    """Find narratives and render them readable."""

    has_content = False

    def run(self):
        tab_width = self.options.get(
            "tab-width", self.state.document.settings.tab_width
        )
        source = self.state_machine.input_lines.source(
            self.lineno - self.state_machine.input_offset - 1
        )
        try:
            lines = []
            dest_dir = Path(source).parent
            for name in narratives.__all__:
                narrative_function = getattr(narratives, name)
                narrative = narrative_function()
                file_name = f"{narrative.name.replace(' ', '_')}.json"
                with open(dest_dir / file_name, "w") as n_file:
                    n_file.write(json.dumps(narrative.json(), indent=4, sort_keys=True))
                lines.extend(
                    statemachine.string2lines(
                        TMPL.format(source_path=file_name),
                        tab_width,
                        convert_whitespace=True,
                    )
                )
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            logging.exception(
                f"Failed to produce ert_narratives in {basename(source)}:{self.lineno}:"
            )
            return [
                nodes.error(
                    None,
                    nodes.paragraph(
                        text=(
                            "Failed to produce ert_narratives in "
                            f"{basename(source)}:{self.lineno}:"
                        )
                    ),
                    nodes.paragraph(text=str(sys.exc_info()[1])),
                )
            ]


def setup(app: Sphinx):
    app.setup_extension("sphinxcontrib.datatemplates")
    app.add_css_file("css/narratives.css")
    app.add_directive("ert_narratives", ErtNarratives)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
