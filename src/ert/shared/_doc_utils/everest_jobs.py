from __future__ import annotations

from collections import defaultdict
from typing import Any,  ClassVar, Dict, List,  Union

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

from ert.config.forward_model_step import ForwardModelStepDocumentation
from ert.plugins import ErtPluginManager, JobDoc
from ert.shared._doc_utils.forward_model_documentation import _ForwardModelDocumentation


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

    @staticmethod
    def _divide_into_categories(
        jobs: Union[
            Dict[str, JobDoc], Dict[str, Union[ForwardModelStepDocumentation, JobDoc]]
        ],
    ) -> Dict[str, Dict[str, List[_ForwardModelDocumentation]]]:
        categories: Dict[str, Dict[str, List[_ForwardModelDocumentation]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for job_name, docs in jobs.items():
            # Job names in ERT traditionally used upper case letters
            # for the names of the job. However, at some point duplicate
            # jobs where added with lower case letters for some of the scripts.
            # To avoid duplicate entries in the documentation, lower case
            # job names are skipped.
            if job_name.islower():
                continue

            if isinstance(docs, ForwardModelStepDocumentation):
                docs = {
                    "description": docs.description,
                    "examples": docs.examples,
                    "config_file": docs.config_file,
                    "parser": None,
                    "source_package": docs.source_package,
                    "category": docs.category,
                }

            category = docs.get(
                "category",
                _EverestDocumentation._CATEGORY_DEFAULT,
            )

            split_categories = category.split(".")
            if len(split_categories) > 1:
                main_category, sub_category = split_categories[0:2]
            elif len(split_categories) == 1:
                main_category, sub_category = split_categories[0], "other"
            else:
                main_category, sub_category = "other", "other"

            categories[main_category][sub_category].append(
                _ForwardModelDocumentation(
                    name=job_name,
                    category=category,
                    job_source=docs.get(
                        "source_package",
                        _EverestDocumentation._SOURCE_PACKAGE_DEFAULT,
                    ),
                    description=docs.get(
                        "description",
                        _EverestDocumentation._DESCRIPTION_DEFAULT,
                    ),
                    job_config_file=docs.get(
                        "config_file",
                        _EverestDocumentation._CONFIG_FILE_DEFAULT,
                    ),
                    examples=docs.get(
                        "examples",
                        _EverestDocumentation._EXAMPLES_DEFAULT,
                    ),
                    parser=docs.get("parser", _EverestDocumentation._PARSER_DEFAULT),
                )
            )

        return {k: dict(v) for k, v in categories.items()}
    
    def _generate_job_documentation_without_title(
        self,
        jobs: Union[
            Dict[str, JobDoc], Dict[str, Union[ForwardModelStepDocumentation, JobDoc]]
        ],
    ) -> List[nodes.section]:
        job_categories = _EverestDocumentation._divide_into_categories(jobs)
        node_list = []

        for category_index, category in enumerate(sorted(job_categories.keys())):
            category_section_node = _create_section_with_title(
                section_id=category + "-category", title=category.capitalize()
            )
            sub_jobs_map = job_categories[category]
            for sub_i, sub in enumerate(sorted(sub_jobs_map.keys())):
                sub_section_node = _create_section_with_title(
                    section_id=category + "-" + sub + "-subcategory",
                    title=sub.capitalize(),
                )

                for job in sub_jobs_map[sub]:
                    job_section_node = job.create_node(self.state)
                    sub_section_node.append(job_section_node)

                # A section is not allowed to end with a transition,
                # so we don't add after the last sub-category
                if sub_i < len(sub_jobs_map) - 1:
                    sub_section_node.append(nodes.transition())

                category_section_node.append(sub_section_node)

            node_list.append(category_section_node)

            # A section is not allowed to end with a transition,
            # so we don't add after the last category
            if category_index < len(job_categories) - 1:
                category_section_node.append(nodes.transition())
        return node_list



class EverestForwardModelDocumentation(_EverestDocumentation):
    pm = ErtPluginManager()
    _JOBS: ClassVar[dict[str, Any]] = {
        **pm.get_documentation_for_jobs(),
        **pm.get_documentation_for_forward_model_steps(),
    }

    def run(self) -> List[nodes.section]:
        return self._generate_job_documentation_without_title(
            EverestForwardModelDocumentation._JOBS,
        )


