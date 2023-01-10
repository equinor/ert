import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ert._c_wrappers.config import ConfigParser

logger = logging.getLogger(__name__)


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
        "MULTFLT",
    ]
    JUST_REMOVE_KEYWORDS = ["UMASK", "LOG_FILE", "LOG_LEVEL", "ENKF_RERUN"]
    RSH_KEYWORDS = ["RSH_HOST", "RHS_COMMAND"]

    def _add_deprecated_keywords_to_parser(self):
        for kw in self.REPLACE_WITH_GEN_KW:
            self._parser.add(kw)
        for kw in self.JUST_REMOVE_KEYWORDS:
            self._parser.add(kw)
        for kw in self.RSH_KEYWORDS:
            self._parser.add(kw)
        self._parser.add("HAVANA_FAULT")
        self._parser.add("REFCASE_LIST")
        self._parser.add("RFTPATH")
        self._parser.add("END_DATE")
        self._parser.add("CASE_TABLE")
        self._parser.add("RERUN_START")
        self._parser.add("DELETE_RUNPATH")
        self._parser.add("RUNPATH")

    def suggest_migrations(self, filename: str):
        suggestions = []
        content = self._parser.parse(filename)

        def add_suggestion(kw, suggestion):
            if content.hasKey(kw):
                logger.info("Deprecated keyword %s", kw)
                suggestions.append(suggestion)

        if content.hasKey("RUNPATH") and "%d" in content.getValue("RUNPATH"):
            add_suggestion(
                "RUNPATH",
                "The use of %d in RUNPATH has been deprecated. "
                "Instead use the <IENS>, <ITER> keywords, "
                "e.g.: realization-<IENS>/iter-<ITER>",
            )

        for kw in self.REPLACE_WITH_GEN_KW:
            add_suggestion(
                kw,
                f"The {kw} keyword was replaced by the GEN_KW keyword."
                "Please see https://ert.readthedocs.io/en/latest/"
                "reference/configuration/keywords.html#gen-kw"
                "to see how to migrate from MULTFLT to GEN_KW.",
            )
        for kw in self.JUST_REMOVE_KEYWORDS:
            add_suggestion(
                kw,
                f"The keyword {kw} no longer has any effect, and can "
                "be safely removed.",
            )
        for kw in self.RSH_KEYWORDS:
            add_suggestion(
                kw,
                f"The {kw} was used for the deprecated and removed "
                "support for RHS queues. It no longer has any effect "
                "and can safely be removed from the config file.",
            )
        add_suggestion(
            "HAVANA_FAULT",
            "Direct interoperability with havana was removed from ert in 2009."
            " The behavior of HAVANA_FAULT can be reproduced using"
            " GEN_KW and FORWARD_MODEL.",
        )
        add_suggestion(
            "REFCASE_LIST",
            "The REFCASE_LIST keyword was used to give a .DATA file "
            "to be used for plotting. The corresponding plotting functionality "
            "was removed in 2015, and this keyword can be safely removed from "
            "the config file.",
        )
        add_suggestion(
            "RFTPATH",
            "The RFTPATH keyword was used to give a path to well observations "
            "to be used for plotting. The corresponding plotting functionality "
            "was removed in 2015, and the RFTPATH keyword can safely be removed "
            "from the config file.",
        )
        add_suggestion(
            "END_DATE",
            "The END_DATE keyword was used to check that a the dates in a summary "
            "file would go beyond a certain date. This would only display a "
            "warning in case of problems. The keyword has since been deprecated, "
            "and can be safely removed from the config file.",
        )
        add_suggestion(
            "CASE_TABLE",
            "The CASE_TABLE keyword was used with a deprecated sensitivity "
            "analysis feature to give descriptive names to cases. It no longer has "
            " any effect and can safely be removed from the config file.",
        )
        add_suggestion(
            "RERUN_START",
            "The RERUN_START keyword was used for the deprecated run mode "
            "ENKF_ASSIMILATION which was removed in 2016. It does not have "
            "any effect on run modes currently supported by ERT, and can "
            "be safely removed.",
        )
        add_suggestion(
            "DELETE_RUNPATH",
            "The DELETE_RUNPATH keyword would clear the runpath directories "
            "between runs. It was removed in 2017 and no longer has any effect.",
        )

        return suggestions
