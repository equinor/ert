from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ert._c_wrappers.config import ConfigParser


class DeprecationMigrationSuggester:
    def __init__(self, parser: "ConfigParser"):
        self._parser = parser
        self._add_deprecated_keywords_to_parser()

    REPLACE_WITH_GEN_KW = [
        "RELPERM",
        "MULTZ",
        "EQUIL",
        "GEN_PARAM",
        "SCHEDULE_PREDICTION_FILE",
    ]
    JUST_REMOVE_KEYWORDS = ["UMASK", "LOG_FILE", "LOG_LEVEL"]

    def _add_deprecated_keywords_to_parser(self):
        for kw in self.REPLACE_WITH_GEN_KW:
            self._parser.add(kw)
        for kw in self.JUST_REMOVE_KEYWORDS:
            self._parser.add(kw)
        self._parser.add("HAVANA_FAULT")
        self._parser.add("MULTFLT")
        self._parser.add("REFCASE_LIST")
        self._parser.add("RFTPATH")
        self._parser.add("END_DATE")
        self._parser.add("CASE_TABLE")
        self._parser.add("RERUN_START")

    def suggest_migrations(self, filename: str):
        suggestions = []

        content = self._parser.parse(filename)
        for kw in self.REPLACE_WITH_GEN_KW:
            if content.hasKey(kw):
                suggestions.append(
                    "The keyword {kw} was deprecated in 2009 in favor of using"
                    " GEN_KW and FORWARD_MODEL."
                )
        for kw in self.JUST_REMOVE_KEYWORDS:
            if content.hasKey(kw):
                suggestions.append(
                    f"The {kw} keyword no longer has any effect "
                    "and can safely be removed from the config file."
                )
        if content.hasKey("HAVANA_FAULT"):
            suggestions.append(
                "Direct interoperability with havana was removed from ert in 2009."
                " The behavior of HAVANA_FAULT can be reproduced using"
                " GEN_KW and FORWARD_MODEL."
            )
        if content.hasKey("MULTFLT"):
            suggestions.append(
                "The MULTFLT keyword was replaced by the GENKW keyword in 2009."
                "Please see https://ert.readthedocs.io/en/latest/"
                "reference/configuration/keywords.html#gen-kw"
                "to see how to migrate from MULTFLT to GEN_KW."
            )
        if content.hasKey("REFCASE_LIST"):
            suggestions.append(
                "The REFCASE_LIST keyword was used to give a .DATA file "
                "to be used for plotting. The corresponding plotting functionality "
                "was removed in 2015, and this keyword can be safely removed from "
                "the config file."
            )
        if content.hasKey("RFTPATH"):
            suggestions.append(
                "The RFTPATH keyword was used to give a path to well observations "
                "to be used for plotting. The corresponding plotting functionality "
                "was removed in 2015, and the RFTPATH keyword can safely be removed "
                "from the config file."
            )
        if content.hasKey("END_DATE"):
            suggestions.append(
                "The END_DATE keyword was used to check that a the dates in a summary "
                "file would go beyond a certaind date. This would only display a "
                "warning in case of problems. The keyword has since been deprecated, "
                "and can be safely removed from the config file."
            )
        if content.hasKey("CASE_TABLE"):
            suggestions.append(
                "The CASE_TABLE keyword was used with a deprecated sensitivity "
                "analysis feature to give descriptive names to cases. It no longer has "
                " any effect and can safely be removed from the config file."
            )
        if content.hasKey("RERUN_START"):
            suggestions.append(
                "The RERUN_START keyword was used for the deprecated run mode "
                "ENKF_ASSIMILATION which was removed in 2016. It does not have "
                "any effect on run modes currently supported by ERT, and can "
                "be safely removed."
            )

        return suggestions
