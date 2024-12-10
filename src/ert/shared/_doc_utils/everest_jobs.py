from __future__ import annotations

from typing import Any, ClassVar

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

from everest.plugins.everest_plugin_manager import EverestPluginManager

from . import _escape_id, _parse_raw_rst


class _EverestDocumentation(SphinxDirective):
    """
    This is an abstract class that should never be used directly. A class should be set
    up that inherits from this class. The child class must implement a run function.
    """

    has_content = True

    _CATEGORY_DEFAULT = "other"
    _SOURCE_PACKAGE_DEFAULT = "PACKAGE NOT PROVIDED"
    _DESCRIPTION_DEFAULT = ""
    _CONFIG_FILE_DEFAULT = "No config file provided"
    _EXAMPLES_DEFAULT = ""
    _PARSER_DEFAULT = None

    def _generate_job_documentation(
        self,
        docs: dict[str, Any],
    ) -> list[nodes.section]:
        if not docs:
            node = nodes.section(ids=[_escape_id("no-forward-models-category")])
            node.append(nodes.literal_block(text="No forward model jobs installed"))
            return [node]

        node_list = []
        for job_name, job_doc in docs.items():
            full_job_name = job_doc["full_job_name"]
            cmd_name = job_doc["cmd_name"]

            node = nodes.section(ids=[_escape_id(job_name + "-category")])
            node.append(nodes.title(text=full_job_name))

            argparse_template = f"""
.. argparse::
    :module: everest_models.jobs.{cmd_name}.parser
    :func: build_argument_parser
    :prog: {cmd_name}
"""
            _parse_raw_rst(argparse_template, node, self.state)

            if job_doc["examples"]:
                _parse_raw_rst(job_doc["examples"], node, self.state)

            node_list.append(node)

        return node_list


class EverestForwardModelDocumentation(_EverestDocumentation):
    pm = EverestPluginManager()
    _JOBS: ClassVar[dict[str, Any]] = {**pm.get_documentation()}

    def run(self) -> list[nodes.section]:
        return self._generate_job_documentation(
            EverestForwardModelDocumentation._JOBS,
        )
