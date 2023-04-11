import datetime
import logging
import os
import os.path
import warnings
from typing import Any, Dict, List, Mapping, Optional, Union

from lark import Discard, Lark, Token, Transformer, Tree, UnexpectedCharacters

from ert.parsing import (
    ConfigValidationError,
    ConfigWarning,
    Defines,
    FileContextToken,
    Instruction,
)
from ert.parsing.config_keywords import (
    SchemaItem,
    check_required,
    define_keyword,
    init_site_config,
    init_user_config,
)

grammar = r"""
WHITESPACE: (" "|"\t")+
%ignore WHITESPACE

%import common.CNAME
%import common.SIGNED_NUMBER    -> NUMBER
%import common.NEWLINE          -> NEWLINE


_STRING_INNER: /.+?/
_STRING_ESC_INNER: _STRING_INNER /(?<!\\)(\\\\)*?/

STRING: "\"" _STRING_ESC_INNER "\"" | "'" _STRING_ESC_INNER "'"

DIGIT: "0".."9"
LCASE_LETTER: "a".."z"
UCASE_LETTER: "A".."Z"

LETTER: UCASE_LETTER | LCASE_LETTER
WORD: LETTER+

CHAR: /[&$\[\]=,.\*a-zæøåA-ZÆØÅ10-9_%:\<\>\/-]/
UNQUOTED: CHAR+

CHAR_NO_EQ: /[+.\*a-zæøåA-ZÆØÅ10-9_%:\<\>\/-]/
UNQUOTED_NO_EQ: /(?!([ ]))/ CHAR_NO_EQ+

CHAR_KW: /[a-zæøåA-ZÆØÅ10-9_:-]/
UNQUOTED_KW: CHAR_KW+
ENV_STRING: "$" UNQUOTED_NO_EQ

arg: STRING | UNQUOTED

kw_list: "(" [ kw_pair ("," kw_pair)*] ")"
kw_val: UNQUOTED_NO_EQ | STRING | ENV_STRING
kw_pair: KW_NAME "=" kw_val
KW_NAME_STRICT: "<" UNQUOTED_KW ">"
KW_NAME: UNQUOTED_KW | "<" KEYWORD_NAME ">"

COMMENT: /--.*/ NEWLINE

KEYWORD_NAME: LETTER (LETTER | DIGIT | "_" | "-" | "<" | ">" )*

start: instruction+

FORWARD_MODEL: "FORWARD_MODEL"
DEFINE: "DEFINE"
DATA_KW: "DATA_KW"

inst: FORWARD_MODEL UNQUOTED kw_list -> job_instruction
    | KEYWORD_NAME arg* -> regular_instruction

instruction: inst COMMENT | COMMENT | inst NEWLINE | NEWLINE
"""  # noqa: E501


class StringQuotationTransformer(Transformer):
    """Strips quotation marks from strings"""

    def STRING(self, token: Token) -> Token:
        return Token(
            token.type,
            token.value[1 : len(token.value) - 1],
            token.start_pos,
            token.line,
            token.column,
            token.end_line,
            token.end_column,
            token.end_pos,
        )


class ArgumentToStringTransformer(Transformer):
    """Flattens all argument types to just tokens or
    relevant python datastructures"""

    def arg(self, rule: List) -> Token:
        return rule[0]

    def kw_val(self, rule: List):
        return rule[0]

    def kw_list(self, kw_list):
        args = []
        for kw_pair in kw_list:
            if kw_pair is not None:
                key, val = kw_pair.children
                args.append((key, val))
        return args


class FileContextTransformer(Transformer):
    """Adds filename to each token,
    to ensure we have enough context for error messages"""

    def __init__(self, filename):
        self.filename = filename
        super().__init__(visit_tokens=True)

    def __default_token__(self, token) -> FileContextToken:
        return FileContextToken(token, self.filename)


class InstructionTransformer(Transformer):
    """Removes all unneccessary levels from the tree,
    resulting in a Tree where each child is one
    instruction from the file, as a list of tokens or
    in the case of job arguments, a list of tuples of
    tokens"""

    def instruction(self, children):
        return children[0] if len(children) > 0 else Discard

    def regular_instruction(self, children):
        return children

    def job_instruction(self, children):
        return children

    def COMMENT(self, _token):
        return Discard

    def NEWLINE(self, _token):
        return Discard


_parser = Lark(grammar, propagate_positions=True)

logger = logging.getLogger(__name__)


def _substitute(
    defines: Defines,
    token: FileContextToken,
    expand_env: bool = True,
) -> str:
    current: FileContextToken = token

    # replace from env
    if expand_env:
        for key, val in os.environ.items():
            current = current.replace(f"${key}", val)
    if not defines:
        return current

    # replace from defines
    prev = None
    n = 0
    while prev != current and n < 100:
        n = n + 1
        for key, val in defines:
            prev = current
            current = current.replace(key, str(val))

    if n >= 100:
        logger.warning(
            "reached max iterations while"
            " trying to resolve defines in file '%s'. Matched to '%s'",
            token.filename,
            token,
        )

    return current


def _tree_to_dict(
    config_file: str,
    pre_defines: Defines,
    tree: Tree[Instruction],
    site_config: Optional[Dict[str, Any]] = None,
) -> Mapping[str, Instruction]:
    schema: Mapping[str, SchemaItem] = (
        init_user_config() if site_config is not None else init_site_config()
    )

    config_dict = site_config if site_config else {}
    defines = pre_defines.copy()
    config_dict["DEFINE"] = defines

    for node in tree.children:
        try:
            kw, *args = node
            if kw not in schema:
                warnings.warn(f"Unknown keyword {kw!r}", category=ConfigWarning)
                continue
            constraints = schema[kw]

            args = constraints.join_args(args)
            args = _substitute_args(args, constraints, defines)
            args = constraints.apply_constraints(args)

            if constraints.multi_occurrence:
                arglist = config_dict.get(kw, [])
                arglist.append(args)
                config_dict[kw] = arglist
            else:
                config_dict[kw] = args
        except ConfigValidationError as e:
            token: Token = kw
            raise ConfigValidationError(
                f"{e.errors}\nWas used in {token.value} at line {token.line}",
                config_file=token.filename,
            ) from e

    check_required(schema, config_dict, filename=config_file)

    return config_dict


def _substitute_args(
    args: List[Any], constraints: SchemaItem, defines: Defines
) -> List[Any]:
    if constraints.substitute_from < 1:
        return args
    return [
        _substitute(defines, x, constraints.expand_envvar)
        if i + 1 >= constraints.substitute_from and isinstance(x, FileContextToken)
        else x
        for i, x in enumerate(args)
    ]


def _handle_includes(
    tree: Tree[Instruction],
    defines: Defines,
    config_file: str,
    already_included_files: List[str] = None,
):
    if already_included_files is None:
        already_included_files = [config_file]

    config_dir = os.path.dirname(config_file)
    to_include = []
    for i, node in enumerate(tree.children):
        kw, *args = node
        if kw == "DEFINE":
            constraints = define_keyword()
            args = constraints.join_args(args)
            args = _substitute_args(args, constraints, defines)
            defines.append(args)
        if kw == "INCLUDE":
            if len(args) > 1:
                raise ConfigValidationError(
                    "Keyword:INCLUDE must have exactly one argument",
                    config_file=node[0].filename,
                )
            file_to_include = _substitute(defines, args[0])
            if not os.path.isabs(file_to_include):
                file_to_include = os.path.normpath(
                    os.path.join(config_dir, file_to_include)
                )

            if file_to_include in already_included_files:
                raise ConfigValidationError(
                    f"Cyclical import detected, {file_to_include} is already included"
                )

            sub_tree = _parse_file(file_to_include, "INCLUDE")

            _handle_includes(
                sub_tree,
                defines,
                file_to_include,
                [*already_included_files, file_to_include],
            )

            to_include.append((sub_tree, i))

    for sub_tree, i in reversed(to_include):
        tree.children.pop(i)
        tree.children[i:i] = sub_tree.children


def _parse_file(
    file: Union[str, bytes, os.PathLike], error_context_string: str = ""
) -> Tree[Instruction]:
    try:
        with open(file, encoding="utf-8") as f:
            content = f.read()
        tree = _parser.parse(content + "\n")
        return (
            StringQuotationTransformer()
            * FileContextTransformer(file)
            * ArgumentToStringTransformer()
            * InstructionTransformer()
        ).transform(tree)
    except FileNotFoundError:
        if error_context_string == "INCLUDE":
            raise ConfigValidationError(
                f"{error_context_string} file: {file} not found"
            )
        raise IOError(f"{error_context_string} file: {file} not found")
    except UnexpectedCharacters as e:
        raise ConfigValidationError(str(e), config_file=file) from e
    except UnicodeDecodeError as e:
        error_words = str(e).split(" ")
        hex_str = error_words[error_words.index("byte") + 1]
        try:
            unknown_char = chr(int(hex_str, 16))
        except ValueError:
            unknown_char = f"hex:{hex_str}"
        raise ConfigValidationError(
            f"Unsupported non UTF-8 character {unknown_char!r} "
            f"found in file: {file!r}",
            config_file=file,
        )


def parse(
    file: str,
    site_config: Optional[Mapping[str, Instruction]] = None,
) -> Mapping[str, Instruction]:
    filepath = os.path.normpath(os.path.abspath(file))
    tree = _parse_file(filepath)
    config_dir = os.path.dirname(filepath)
    config_file_name = os.path.basename(file)
    config_file_base = config_file_name.split(".")[0]
    pre_defines = [
        ["<CONFIG_PATH>", config_dir],
        ["<CONFIG_FILE_BASE>", config_file_base],
        ["<DATE>", datetime.date.today().isoformat()],
        ["<CWD>", config_dir],
        ["<CONFIG_FILE>", os.path.basename(file)],
    ]
    # need to copy pre_defines because _handle_includes will
    # add to this list
    _handle_includes(tree, pre_defines.copy(), filepath)

    return _tree_to_dict(
        config_file=file,
        pre_defines=pre_defines,
        tree=tree,
        site_config=site_config,
    )
