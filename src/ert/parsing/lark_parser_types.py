from typing import List, Tuple, Union

from ert.parsing.lark_parser_file_context_token import FileContextToken
from ert.parsing.lark_parser_primitive_tokens import PrimitiveWithContext

# The type of the leaf nodes in the Tree after transformation is done
Instruction = List[
    List[Union[FileContextToken, List[Tuple[FileContextToken, FileContextToken]]]]
]

Defines = List[List[str]]

Primitives = Union[float, bool, str, int]

# Old config parser gives primitives (which are without any context)
# while the new config parser gives primitives WITH context.
# Thus, we need a type to represent a union of these two as both
# parsers are in use.
MaybeWithContext = Union[PrimitiveWithContext, Primitives, FileContextToken]
