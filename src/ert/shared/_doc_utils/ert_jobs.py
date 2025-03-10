from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

from ert.config import ForwardModelStepDocumentation
from ert.plugins import ErtPluginManager, JobDoc
from ert.shared._doc_utils.forward_model_documentation import _ForwardModelDocumentation

from . import _create_section_with_title, _parse_string_list


class _ErtDocumentation(SphinxDirective):
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
        jobs: dict[str, JobDoc] | dict[str, ForwardModelStepDocumentation | JobDoc],
    ) -> dict[str, dict[str, list[_ForwardModelDocumentation]]]:
        categories: dict[str, dict[str, list[_ForwardModelDocumentation]]] = (
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
                _ErtDocumentation._CATEGORY_DEFAULT,
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
                        _ErtDocumentation._SOURCE_PACKAGE_DEFAULT,
                    ),
                    description=docs.get(
                        "description",
                        _ErtDocumentation._DESCRIPTION_DEFAULT,
                    ),
                    job_config_file=docs.get(
                        "config_file",
                        _ErtDocumentation._CONFIG_FILE_DEFAULT,
                    ),
                    examples=docs.get(
                        "examples",
                        _ErtDocumentation._EXAMPLES_DEFAULT,
                    ),
                    parser=docs.get("parser", _ErtDocumentation._PARSER_DEFAULT),
                )
            )

        return {k: dict(v) for k, v in categories.items()}

    def _create_forward_model_section_node(
        self, section_id: str, title: str
    ) -> nodes.section:
        node = _create_section_with_title(section_id=section_id, title=title)
        _parse_string_list(self.content, node, self.state)
        return node

    def _generate_job_documentation(
        self,
        jobs: dict[str, JobDoc] | dict[str, ForwardModelStepDocumentation | JobDoc],
        section_id: str,
        title: str,
    ) -> list[nodes.section]:
        job_categories = _ErtDocumentation._divide_into_categories(jobs)

        main_node = self._create_forward_model_section_node(section_id, title)

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

            main_node.append(category_section_node)

            # A section is not allowed to end with a transition,
            # so we don't add after the last category
            if category_index < len(job_categories) - 1:
                category_section_node.append(nodes.transition())
        return [main_node]

    def _generate_job_documentation_without_title(
        self,
        jobs: dict[str, JobDoc] | dict[str, ForwardModelStepDocumentation | JobDoc],
    ) -> list[nodes.section]:
        job_categories = _ErtDocumentation._divide_into_categories(jobs)
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


class ErtForwardModelDocumentation(_ErtDocumentation):
    pm = ErtPluginManager()
    _JOBS: ClassVar[dict[str, Any]] = {
        **pm.get_documentation_for_jobs(),
        **pm.get_documentation_for_forward_model_steps(),
    }

    def run(self) -> list[nodes.section]:
        return self._generate_job_documentation_without_title(
            ErtForwardModelDocumentation._JOBS,
        )


class ErtWorkflowDocumentation(_ErtDocumentation):
    pm = ErtPluginManager()
    _JOBS = pm.get_documentation_for_workflows()
    _TITLE = "Workflow jobs"
    _SECTION_ID = "ert-workflow-jobs"

    def run(self) -> list[nodes.section]:
        section_id = ErtWorkflowDocumentation._SECTION_ID
        title = ErtWorkflowDocumentation._TITLE
        return self._generate_job_documentation(
            ErtWorkflowDocumentation._JOBS, section_id, title
        )
