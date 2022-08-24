import inspect
import logging
import sys
from os.path import basename

from docutils import nodes, statemachine
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx

from ert.ert3.config import ConfigPluginRegistry
from ert.ert3.config.plugins import TransformationConfigBase
from ert.ert3.plugins import ErtPluginManager

HEADER_TMPL = """{name_capitalized}s
{title_line}-

The default type for this category is: **{default_value}**

The following shows the available {name} configurations:

"""

TMPL = """.. autoclass:: {path}::{config_cls}
    :members:
    :undoc-members:
    :inherited-members: {base_cls}

"""


def _load_all_plugins_for_category(category: str):
    plugin_registry = ConfigPluginRegistry()
    plugin_registry.register_category(
        category=category,
        base_config=TransformationConfigBase,
        optional=True,
    )
    plugin_manager = ErtPluginManager()
    plugin_manager.collect(registry=plugin_registry)
    return plugin_registry


class ERT3Plugins(Directive):
    """Load all plugin for the given category (required argument)
    and render configs for all of them, as well as  the default config for
    each category.
    """

    has_content = False
    required_arguments = 1

    def run(self):
        category = self.arguments[0]
        tab_width = self.options.get(
            "tab-width", self.state.document.settings.tab_width
        )

        try:
            plugin_registry = _load_all_plugins_for_category(category)
            config_base_class = plugin_registry.get_base_config(category=category)
            category_default = plugin_registry.get_default_for_category(
                category=category
            )

            # Find parent of config base class, as we want to document all
            # inherited members from all base classes up to this class (but NOT
            # including).
            base_cls = (
                ""
                if len(config_base_class.__bases__) == 0
                else config_base_class.__bases__[0].__name__
            )

            lines = self._insert_category_header(tab_width, category, category_default)
            for config_cls in plugin_registry.get_original_configs(category).values():
                lines.extend(self._insert_configs(tab_width, base_cls, config_cls))

            source = self.state_machine.input_lines.source(
                self.lineno - self.state_machine.input_offset - 1
            )
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            logging.exception(
                "Failed to produce plugin documentation for "
                f"category {category} on {basename(source)}:{self.lineno}:"
            )
            return [
                nodes.error(
                    None,
                    nodes.paragraph(
                        text="Failed to produce plugin documentation for category "
                        f"{category} on {basename(source)}:{self.lineno}:"
                    ),
                    nodes.paragraph(text=str(sys.exc_info()[1])),
                )
            ]

    def _insert_configs(self, tab_width, base_cls, config_cls):
        return statemachine.string2lines(
            TMPL.format(
                path=inspect.getmodule(config_cls).__name__,
                config_cls=config_cls.__name__,
                base_cls=base_cls,
            ),
            tab_width,
            convert_whitespace=True,
        )

    def _insert_category_header(self, tab_width, category, default_value):
        return statemachine.string2lines(
            HEADER_TMPL.format(
                name_capitalized=category.capitalize(),
                title_line="-" * len(category),
                default_value=default_value,
                name=category,
            ),
            tab_width,
            convert_whitespace=True,
        )


def setup(app: Sphinx):
    app.add_directive("ert3_plugins", ERT3Plugins)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
