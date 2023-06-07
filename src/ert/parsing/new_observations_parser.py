"""
This is the new parser based on lark for observationfiles.
Not in use yet.
"""
from enum import Enum, auto

from lark import Lark, Transformer

observations_grammar = r"""
start: observation*
?observation: type STRING value? ";"
type: "HISTORY_OBSERVATION" -> history
    | "SUMMARY_OBSERVATION" -> summary
    | "GENERAL_OBSERVATION" -> general
?value: object
      | STRING


CHAR: /[^; \t\n{}=]/
STRING : CHAR+
object : "{" [(declaration";")*] "}"
?declaration: "SEGMENT" STRING object -> segment
            | pair
pair   : STRING "=" value


%import common.WS
%ignore WS

COMMENT: /--[^\n]*/
%ignore COMMENT
"""


class ObservationType(Enum):
    HISTORY = auto()
    SUMMARY = auto()
    GENERAL = auto()

    @classmethod
    def from_rule(cls, rule: str) -> "ObservationType":
        if rule == "summary":
            return cls.SUMMARY
        if rule == "general":
            return cls.GENERAL
        if rule == "history":
            return cls.HISTORY
        raise ValueError(f"Unexpected observation type {rule}")


class TreeToObservations(Transformer):
    start = list

    def observation(self, tree):
        return tuple([ObservationType.from_rule(tree[0].data), *tree[1:]])

    segment = tuple
    object = dict
    pair = tuple


observations_parser = Lark(
    observations_grammar,
)


def parse(content: str):
    return TreeToObservations().transform(observations_parser.parse(content))
