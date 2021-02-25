from ert_shared.ide.keywords.definitions import (
    KeywordDefinition,
    ConfigurationLineDefinition,
    StringArgument,
)


class ParametrizationKeywords(object):
    def __init__(self, ert_keywords):
        super(ParametrizationKeywords, self).__init__()
        self.group = "Parametrization"

        ert_keywords.addKeyword(self.addField())
        ert_keywords.addKeyword(self.addGenData())
        ert_keywords.addKeyword(self.addGenKw())
        ert_keywords.addKeyword(self.addSummary())
        ert_keywords.addKeyword(self.addGenParam())
        ert_keywords.addKeyword(self.addGenKwTagFormat())
        ert_keywords.addKeyword(self.addSurface())

    def addField(self):
        field = ConfigurationLineDefinition(
            keyword=KeywordDefinition("FIELD"),
            arguments=[
                StringArgument(),
                StringArgument(),
                StringArgument(rest_of_line=True, allow_space=True),
            ],
            documentation_link="keywords/field",
            required=False,
            group=self.group,
        )
        return field

    def addGenData(self):
        gen_data = ConfigurationLineDefinition(
            keyword=KeywordDefinition("GEN_DATA"),
            arguments=[
                StringArgument(),
                StringArgument(),
                StringArgument(rest_of_line=True, allow_space=True),
            ],
            documentation_link="keywords/gen_data",
            required=False,
            group=self.group,
        )
        return gen_data

    def addGenKw(self):
        gen_kw = ConfigurationLineDefinition(
            keyword=KeywordDefinition("GEN_KW"),
            arguments=[
                StringArgument(),
                StringArgument(),
                StringArgument(rest_of_line=True, allow_space=True),
            ],
            documentation_link="keywords/gen_kw",
            required=False,
            group=self.group,
        )
        return gen_kw

    def addGenKwTagFormat(self):
        gen_kw_tag_format = ConfigurationLineDefinition(
            keyword=KeywordDefinition("GEN_KW_TAG_FORMAT"),
            arguments=[StringArgument(built_in=True, allow_space=True)],
            documentation_link="keywords/gen_kw_tag_format",
            required=False,
            group=self.group,
        )
        return gen_kw_tag_format

    def addGenParam(self):
        gen_param = ConfigurationLineDefinition(
            keyword=KeywordDefinition("GEN_PARAM"),
            arguments=[
                StringArgument(),
                StringArgument(),
                StringArgument(built_in=True, allow_space=True),
            ],
            documentation_link="keywords/gen_param",
            required=False,
            group=self.group,
        )
        return gen_param

    def addSummary(self):
        summary = ConfigurationLineDefinition(
            keyword=KeywordDefinition("SUMMARY"),
            arguments=[StringArgument(rest_of_line=True, allow_space=True)],
            documentation_link="keywords/summary",
            required=False,
            group=self.group,
        )
        return summary

    def addSurface(self):
        surface = ConfigurationLineDefinition(
            keyword=KeywordDefinition("SURFACE"),
            arguments=[StringArgument(rest_of_line=True, allow_space=True)],
            documentation_link="keywords/surface",
            required=False,
            group=self.group,
        )
        return surface
