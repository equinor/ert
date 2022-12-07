from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ert._c_wrappers.config import ConfigParser


class DeprecationMigrationSuggester:
    def __init__(self, parser: "ConfigParser"):
        self._parser = parser
        self._add_deprecated_keywords_to_parser()

    def _add_deprecated_keywords_to_parser(self):
        self._parser.add("UMASK")

    def suggest_migrations(self, filename: str):
        suggestions = []

        content = self._parser.parse(filename)
        if content.hasKey("UMASK"):
            suggestions.append(
                "The UMASK keyword has been removed "
                "and has no effect, it can safely be removed"
            )

        return suggestions
