from __future__ import annotations

from typing import Any

import docutils.statemachine
from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.nodes import nested_parse_with_titles


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


__all__ = [
    "_create_section_with_title",
    "_escape_id",
    "_parse_raw_rst",
    "_parse_string_list",
]
