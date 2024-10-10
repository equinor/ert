from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

from everest.plugins.everest_plugin_manager import EverestPluginManager

from . import _create_section_with_title


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

    def _generate_job_documentation_without_title(
        self,
        docs: Dict[str, Any],
    ) -> List[nodes.section]:
        node_list = []
        for job_name, job_doc in dict(sorted(docs.items())).items():
            category_section_node = _create_section_with_title(
                section_id=job_name + "-category", title=job_doc["full_job_name"]
            )
            node_list.append(category_section_node)
            n = nodes.literal_block(text=job_doc["help"])
            n["xml:space"] = "preserve"
            node_list.append(n)

        return node_list


class EverestForwardModelDocumentation(_EverestDocumentation):
    pm = EverestPluginManager()
    _JOBS: ClassVar[dict[str, Any]] = {**pm.get_documentation()}

    def run(self) -> List[nodes.section]:
        return self._generate_job_documentation_without_title(
            EverestForwardModelDocumentation._JOBS,
        )
