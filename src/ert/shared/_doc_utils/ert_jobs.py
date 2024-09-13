from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

import docutils.statemachine
from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles

from ert.config.forward_model_step import ForwardModelStepDocumentation
from ert.plugins import ErtPluginManager, JobDoc


class _ForwardModelDocumentation:
    def __init__(
        self,
        name: str,
        category: str,
        job_source: str,
        description: str,
        job_config_file: Optional[str],
        parser: Optional[Callable[[], ArgumentParser]],
        examples: Optional[str] = "",
    ) -> None:
        self.name = name
        self.job_source = job_source
        self.category = category
        self.description = description
        self.job_config_file = job_config_file
        self.examples = examples
        self.parser = parser

    def _create_job_config_section(self) -> nodes.section:
        config_section_node = _create_section_with_title(
            section_id=self.name + "-job-config", title="Job configuration used by ERT"
        )

        if self.job_config_file:
            with open(self.job_config_file, encoding="utf-8") as fh:
                job_config_text = fh.read()
            job_config_text_node = nodes.literal_block(text=job_config_text)
            config_section_node.append(job_config_text_node)

        return config_section_node

    def _create_job_details(self) -> nodes.definition_list:
        job_category_node = nodes.definition_list_item(
            "",
            nodes.term("", "", nodes.strong(text="Category")),
            nodes.definition("", nodes.strong(text=self.category)),
        )
        job_package_node = nodes.definition_list_item(
            "",
            nodes.term("", "", nodes.strong(text="Source package")),
            nodes.definition("", nodes.strong(text=self.job_source)),
        )
        job_details_node = nodes.definition_list(
            "", job_category_node, job_package_node
        )
        return job_details_node

    def _create_example_section(self, state: Any) -> nodes.section:
        assert self.examples is not None
        example_section_node = _create_section_with_title(
            section_id=self.name + "-examples", title="Examples"
        )
        _parse_raw_rst(self.examples, example_section_node, state)
        return example_section_node

    def _create_argparse_section(self) -> nodes.section:
        config_section_node = _create_section_with_title(
            section_id=self.name + "-job-config", title="Job arguments"
        )
        assert self.parser is not None
        parser = self.parser()
        text = parser.format_help()
        text = text.replace(parser.prog, self.name)
        job_config_text_node = nodes.literal_block(text=text)
        config_section_node.append(job_config_text_node)
        return config_section_node

    def create_node(self, state: Any) -> nodes.section:
        # Create main node section
        node = _create_section_with_title(section_id=self.name, title=self.name)

        # Add description and forward model details
        _parse_raw_rst(self.description, node, state)
        job_details_node = self._create_job_details()
        node.append(job_details_node)

        # Add examples
        if self.examples:
            example_section_node = self._create_example_section(state)
            node.append(example_section_node)

        # Add parser
        if self.parser:
            parser_section_node = self._create_argparse_section()
            node.append(parser_section_node)

        # Add forward model config file
        if self.job_config_file:
            config_section_node = self._create_job_config_section()
            node.append(config_section_node)

        return node


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
        jobs: Union[
            Dict[str, JobDoc], Dict[str, Union[ForwardModelStepDocumentation, JobDoc]]
        ],
        section_id: str,
        title: str,
    ) -> List[nodes.section]:
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
        jobs: Union[
            Dict[str, JobDoc], Dict[str, Union[ForwardModelStepDocumentation, JobDoc]]
        ],
    ) -> List[nodes.section]:
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


def _parse_raw_rst(rst_string: str, node: nodes.Node, state: Any) -> None:
    string_list = docutils.statemachine.StringList(list(rst_string.split("\n")))
    _parse_string_list(string_list, node, state)


def _parse_string_list(string_list: StringList, node: nodes.Node, state: Any) -> None:
    nested_parse_with_titles(state, string_list, node)


def _escape_id(id_string: str) -> str:
    return id_string.replace(" ", "-")


def _create_section_with_title(section_id: str, title: str) -> nodes.section:
    node = nodes.section(ids=[_escape_id(section_id)])
    jobs_title = nodes.title(text=title)
    node.append(jobs_title)
    return node


class ErtForwardModelDocumentation(_ErtDocumentation):
    pm = ErtPluginManager()
    _JOBS: ClassVar[dict[str, Any]] = {
        **pm.get_documentation_for_jobs(),
        **pm.get_documentation_for_forward_model_steps(),
    }

    def run(self) -> List[nodes.section]:
        return self._generate_job_documentation_without_title(
            ErtForwardModelDocumentation._JOBS,
        )


class ErtWorkflowDocumentation(_ErtDocumentation):
    pm = ErtPluginManager()
    _JOBS = pm.get_documentation_for_workflows()
    _TITLE = "Workflow jobs"
    _SECTION_ID = "ert-workflow-jobs"

    def run(self) -> List[nodes.section]:
        section_id = ErtWorkflowDocumentation._SECTION_ID
        title = ErtWorkflowDocumentation._TITLE
        return self._generate_job_documentation(
            ErtWorkflowDocumentation._JOBS, section_id, title
        )
