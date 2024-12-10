from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable
from typing import Any

from docutils import nodes

from . import _create_section_with_title, _parse_raw_rst


class _ForwardModelDocumentation:
    def __init__(
        self,
        name: str,
        category: str,
        job_source: str,
        description: str,
        job_config_file: str | None,
        parser: Callable[[], ArgumentParser] | None,
        examples: str | None = "",
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
