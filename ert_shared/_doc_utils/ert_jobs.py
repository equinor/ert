import sys
from collections import defaultdict

import docutils.statemachine
from docutils import nodes
from sphinx.util import nested_parse_with_titles
from sphinx.util.docutils import SphinxDirective

from ert_shared.plugins import ErtPluginManager


class _ForwardModelDocumentation:
    def __init__(
        self,
        name,
        category,
        job_source,
        description,
        job_config_file,
        examples="",
        parser=None,
    ):
        self.name = name
        self.job_source = job_source
        self.category = category
        self.description = description
        self.job_config_file = job_config_file
        self.examples = examples
        self.parser = parser

    def _create_job_config_section(self):
        config_section_node = _create_section_with_title(
            section_id=self.name + "-job-config", title="Job configuration used by ERT"
        )

        with open(self.job_config_file) as fh:
            job_config_text = fh.read()
        job_config_text_node = nodes.literal_block(text=job_config_text)
        config_section_node.append(job_config_text_node)
        return config_section_node

    def _create_job_details(self):
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

    def _create_example_section(self, state):
        example_section_node = _create_section_with_title(
            section_id=self.name + "-examples", title="Examples"
        )
        _parse_raw_rst(self.examples, example_section_node, state)
        return example_section_node

    def _create_argparse_section(self):
        config_section_node = _create_section_with_title(
            section_id=self.name + "-job-config", title="Job arguments"
        )
        parser = self.parser()
        text = parser.format_help()
        text = text.replace(parser.prog, self.name)
        job_config_text_node = nodes.literal_block(text=text)
        config_section_node.append(job_config_text_node)
        return config_section_node

    def create_node(self, state):
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
        config_section_node = self._create_job_config_section()
        node.append(config_section_node)

        return node


class _ErtDocumentation(SphinxDirective):
    """
    This is an abstract class that should never be used directly. A class should be set
    up that inherits from this class. The child class must implement a run function.
    """

    has_content = True

    _CATEGORY_KEY = "category"
    _SOURCE_PACKAGE_KEY = "source_package"
    _DESCRIPTION_KEY = "description"
    _CONFIG_FILE_KEY = "config_file"
    _EXAMPLES_KEY = "examples"
    _PARSER_KEY = "parser"

    _CATEGORY_DEFAULT = "other"
    _SOURCE_PACKAGE_DEFAULT = "PACKAGE NOT PROVIDED"
    _DESCRIPTION_DEFAULT = ""
    _CONFIG_FILE_DEFAULT = "No config file provided"
    _EXAMPLES_DEFAULT = ""
    _PARSER_DEFAULT = None

    @staticmethod
    def _divide_into_categories(jobs):

        categories = defaultdict(lambda: defaultdict(list))
        for job_name, docs in jobs.items():
            # Job names in ERT traditionally used upper case letters
            # for the names of the job. However, at some point duplicate
            # jobs where added with lower case letters for some of the scripts.
            # To avoid duplicate entries in the documentation, lower case
            # job names are skipped.
            if job_name.islower():
                continue

            category = docs.get(
                _ErtDocumentation._CATEGORY_KEY,
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
                        _ErtDocumentation._SOURCE_PACKAGE_KEY,
                        _ErtDocumentation._SOURCE_PACKAGE_DEFAULT,
                    ),
                    description=docs.get(
                        _ErtDocumentation._DESCRIPTION_KEY,
                        _ErtDocumentation._DESCRIPTION_DEFAULT,
                    ),
                    job_config_file=docs.get(
                        _ErtDocumentation._CONFIG_FILE_KEY,
                        _ErtDocumentation._CONFIG_FILE_DEFAULT,
                    ),
                    examples=docs.get(
                        _ErtDocumentation._EXAMPLES_KEY,
                        _ErtDocumentation._EXAMPLES_DEFAULT,
                    ),
                    parser=docs.get(
                        _ErtDocumentation._PARSER_KEY, _ErtDocumentation._PARSER_DEFAULT
                    ),
                )
            )

        return dict({k: dict(v) for k, v in categories.items()})

    def _create_forward_model_section_node(self, section_id, title):
        node = _create_section_with_title(section_id=section_id, title=title)
        _parse_string_list(self.content, node, self.state)
        return node

    def _generate_job_documentation(self, jobs, section_id, title):

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


def _parse_raw_rst(rst_string, node, state):
    string_list = docutils.statemachine.StringList(list(rst_string.split("\n")))
    _parse_string_list(string_list, node, state)


def _parse_string_list(string_list, node, state):
    nested_parse_with_titles(state, string_list, node)


def _escape_id(id_string):
    return id_string.replace(" ", "-")


def _create_section_with_title(section_id, title):
    node = nodes.section(ids=[_escape_id(section_id)])
    jobs_title = nodes.title(text=title)
    node.append(jobs_title)
    return node


class ErtForwardModelDocumentation(_ErtDocumentation):
    pm = ErtPluginManager()
    _JOBS = pm.get_documentation_for_jobs()
    _TITLE = "Forward models"
    _SECTION_ID = "ert-forward-models"

    def run(self):
        section_id = ErtForwardModelDocumentation._SECTION_ID
        title = ErtForwardModelDocumentation._TITLE
        return self._generate_job_documentation(
            ErtForwardModelDocumentation._JOBS, section_id, title
        )


class ErtWorkflowDocumentation(_ErtDocumentation):
    pm = ErtPluginManager()
    _JOBS = pm.get_documentation_for_workflows()
    _TITLE = "Workflow jobs"
    _SECTION_ID = "ert-workflow-jobs"

    def run(self):
        section_id = ErtWorkflowDocumentation._SECTION_ID
        title = ErtWorkflowDocumentation._TITLE
        return self._generate_job_documentation(
            ErtWorkflowDocumentation._JOBS, section_id, title
        )
