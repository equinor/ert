# mypy: ignore-errors
import datetime
import os
import os.path
from typing import Self

from lark import (
    Discard,
    Lark,
    Token,
    Transformer,
    Tree,
    UnexpectedCharacters,
    UnexpectedToken,
)

from ._read_file import read_file
from .config_dict import ConfigDict
from .config_errors import ConfigValidationError, ConfigWarning
from .config_schema import SchemaItem, define_keyword
from .error_info import ErrorInfo
from .schema_dict import SchemaItemDict
from .types import Defines, FileContextToken, Instruction, MaybeWithContext

grammar = r"""
%import common.CNAME
%import common.SIGNED_NUMBER    -> NUMBER
%import common.NEWLINE          -> NEWLINE

WHITESPACE: (" "|"\t"|"\\"NEWLINE)+
%ignore WHITESPACE

_STRING_INNER: /.+?/
_STRING_ESC_INNER: _STRING_INNER /(?<!\\)(\\\\)*?/

STRING: "\"" _STRING_ESC_INNER "\"" | "'" _STRING_ESC_INNER "'" | "\"\""

DIGIT: "0".."9"
LCASE_LETTER: "a".."z"
UCASE_LETTER: "A".."Z"

LETTER: UCASE_LETTER | LCASE_LETTER

COMMENT.9: "--" /[^\n]*/
%ignore COMMENT

UNQUOTED: (/[^\" \t\n]/)+
UNQUOTED_ARGUMENT: (/[^\" \t\n\(\),=]/)+

forward_model_arguments: "(" [ argument ("," argument)*] ")"
argument_value: (STRING | UNQUOTED_ARGUMENT)+
argument: UNQUOTED_ARGUMENT "=" argument_value

KEYWORD_NAME: /(?!FORWARD_MODEL\b)/ LETTER (LETTER | DIGIT | "_" | "-" | "<" | ">" )*

start: instruction+

FORWARD_MODEL: "FORWARD_MODEL"
DEFINE: "DEFINE"

FORWARD_MODEL_NAME: UNQUOTED_ARGUMENT

arg: STRING | UNQUOTED
inst: FORWARD_MODEL FORWARD_MODEL_NAME forward_model_arguments? -> job_instruction
    | KEYWORD_NAME arg* -> regular_instruction

instruction: inst NEWLINE | NEWLINE
"""


class StringQuotationTransformer(Transformer):
    """Strips quotation marks from strings"""

    @staticmethod
    def STRING(token: Token) -> Token:
        token.value = token.value[1 : len(token.value) - 1]
        return token


class ArgumentToStringTransformer(Transformer):
    """Flattens all argument types to just tokens or
    relevant python datastructures"""

    @staticmethod
    def arg(rule: list[FileContextToken]) -> FileContextToken:
        return rule[0]

    @staticmethod
    def argument_value(rule: list[FileContextToken]) -> FileContextToken:
        return FileContextToken.join_tokens(rule, separator="")

    @staticmethod
    def forward_model_arguments(
        kw_list,
    ) -> list[tuple[FileContextToken, FileContextToken]]:
        args = []
        for kw_pair in kw_list:
            if kw_pair is not None:
                key, val = kw_pair.children
                args.append((key, val))
        return args


class FileContextTransformer(Transformer[Token, Tree[FileContextToken]]):
    """Adds filename to each token,
    to ensure we have enough context for error messages"""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(visit_tokens=True)

    def __default_token__(self, token) -> FileContextToken:
        return FileContextToken(token, self.filename)


class InstructionTransformer(Transformer):
    """Removes all unnecessary levels from the tree,
    resulting in a Tree where each child is one
    instruction from the file, as a list of tokens or
    in the case of job arguments, a list of tuples of
    tokens"""

    @staticmethod
    def instruction(children):
        return children[0] if len(children) > 0 else Discard

    @staticmethod
    def regular_instruction(children):
        return children

    @staticmethod
    def job_instruction(children):
        return children

    @staticmethod
    def NEWLINE(_token):
        return Discard


_parser = Lark(grammar, propagate_positions=True, parser="lalr")


def _substitute_token(
    defines: Defines,
    token: FileContextToken,
    expand_env: bool = True,
) -> FileContextToken:
    current: FileContextToken = token

    # replace from env
    if expand_env and "$" in current:
        for key, val in os.environ.items():
            current = current.replace_value(f"${key}", val)
    if not defines:
        return current

    # replace from defines
    prev = None
    n = 0
    while prev != current and n < 100:
        n += 1
        for key, val in defines:
            prev = current
            current = current.replace_value(key, str(val))

    for key, val in defines:
        if key in current:
            ConfigWarning.warn(
                f"Gave up replacing in {token}.\n"
                f"After replacing the value is now: {current}.\n"
                f"This still contains the replacement value: {key}, "
                f"which would be replaced by {val}. "
                "Probably this causes a loop.",
                token,
            )

    return current


def _tree_to_dict(
    config_file: str,
    pre_defines: Defines,
    tree: Tree[Instruction],
    schema: SchemaItemDict,
) -> ConfigDict:
    config_dict = {}
    defines = pre_defines.copy()
    config_dict["DEFINE"] = defines  # type: ignore

    errors = []
    cwd = os.path.dirname(os.path.abspath(config_file))

    for node in tree.children:
        args: list[FileContextToken]
        kw: FileContextToken
        kw, *args = node  # type: ignore
        if kw not in schema:
            ConfigWarning.warn(f"Unknown keyword {kw!r}", kw)
            continue

        constraints = schema[kw]
        if kw != constraints.kw:  # is an alias
            kw.value = constraints.kw

        try:
            args = constraints.join_args(args)
            args = _substitute_args(args, constraints, defines)
            option_args = list(constraints.parse_options(args))
            value_list = constraints.apply_constraints(option_args, kw, cwd)

            arglist = config_dict.get(kw, [])
            if kw == "DEFINE":
                define_key, *define_args = value_list
                existing_define = next(
                    (define for define in arglist if define[0] == define_key), None
                )
                if existing_define:
                    existing_define[1:] = define_args
                else:
                    arglist.append(value_list)
            elif constraints.multi_occurrence:
                arglist.append(value_list)

                config_dict[kw] = arglist
            else:
                config_dict[kw] = value_list
        except ConfigValidationError as e:
            if not constraints.multi_occurrence:
                config_dict[kw] = None

            errors.append(e)

    try:
        schema.check_required(config_dict, filename=config_file)
    except ConfigValidationError as e:
        errors.append(e)

    if len(errors) > 0:
        raise ConfigValidationError.from_collected(errors)

    return config_dict


ArgPairList = list[tuple[FileContextToken]]
ParsedArgList = list[FileContextToken | ArgPairList]


def _substitute_args(
    args: ParsedArgList,
    constraints: SchemaItem,
    defines: Defines,
) -> ParsedArgList:
    def substitute_arglist_tuple(
        tup: tuple[FileContextToken, FileContextToken],
    ) -> tuple[FileContextToken, FileContextToken]:
        key, value = tup
        substituted_value = _substitute_token(defines, value, constraints.expand_envvar)

        return (key, substituted_value)

    def substitute_arg(
        arg: FileContextToken | list[tuple[FileContextToken]],
    ) -> FileContextToken | list[tuple[FileContextToken]]:
        if isinstance(arg, FileContextToken):
            return _substitute_token(defines, arg, constraints.expand_envvar)

        if isinstance(arg, list):
            # It is a list of keyword tuples
            return [substitute_arglist_tuple(x) for x in arg]

        raise ValueError(
            "Expected "
            "FileContextToken | list[tuple[FileContextToken]], "
            f"got {type(arg)}"
        )

    return [
        substitute_arg(arg) if (i + 1) >= constraints.substitute_from else arg
        for i, arg in enumerate(args)
    ]


class IncludedFile:
    def __init__(self, included_from: "IncludedFile", filename: str) -> None:
        self.included_from = included_from
        self.filename = filename
        self.context = None

    def __contains__(self, filename: str) -> bool:
        if filename == self.filename:
            return True

        if self.included_from is None:
            return False

        return filename in self.included_from

    @property
    def root(self):
        if self.included_from is None:
            return self

        return self.included_from.root

    @property
    def path_from_root(self):
        return reversed(self.path_to_root)

    @property
    def path_to_root(self):
        if self.included_from is None:
            return [self.filename]

        return [self.filename, *self.included_from.path_to_root]

    def set_context(self, context: MaybeWithContext) -> Self:
        self.context = context
        return self


def _handle_includes(
    tree: Tree[Instruction],
    defines: Defines,
    config_file: str,
    current_included_file: IncludedFile | None = None,
):
    if current_included_file is None:
        current_included_file = IncludedFile(included_from=None, filename=config_file)

    config_dir = os.path.dirname(config_file)
    to_include = []

    errors = []
    for i, node in enumerate(tree.children):
        args: list[FileContextToken]
        kw: FileContextToken
        kw, *args = node  # type: ignore
        if kw == "DEFINE":
            constraints = define_keyword()
            args = constraints.join_args(args)
            args = _substitute_args(args, constraints, defines)
            args = constraints.apply_constraints(args, kw, os.getcwd())
            defines.append(args)  # type: ignore
        if kw == "INCLUDE":
            if len(args) > 1:
                superfluous_tokens: list[FileContextToken] = args[1:]
                errors.append(
                    ErrorInfo(
                        message="Keyword:INCLUDE must have exactly one argument",
                    ).set_context_list(superfluous_tokens)
                )

                args = args[0:1]
            file_to_include = _substitute_token(defines, args[0])

            if not os.path.isabs(file_to_include):
                file_to_include = os.path.normpath(
                    os.path.join(config_dir, file_to_include)
                )

            if file_to_include in current_included_file:
                # Note that: The "original" file is in current_included_file[0]
                # This is where the error will be shown/linted, so this is also
                # the filename even though "technically" it originates from elsewhere
                master_ert_file = current_included_file.root.filename

                # The cycle comes from the "current file" config_file, trying to
                # include file_to_include, this is the info user needs to know
                # We need the chain of imports, fak

                import_trace = [
                    os.path.basename(f)
                    for f in [*current_included_file.path_from_root, file_to_include]
                ]

                errors.append(
                    ErrorInfo(
                        message=f"Cyclical import detected, {'->'.join(import_trace)}",
                        filename=os.path.basename(master_ert_file),
                    ).set_context(current_included_file.context)
                )
                continue

            try:
                sub_tree = _parse_file(file_to_include)
            except ConfigValidationError as err:
                for info in err.errors:
                    info.message = (
                        f"Error in {file_to_include} included from {config_file}: "
                        + info.message
                    )
                    if info.filename is None:
                        info.set_context_list(args)
                    errors.append(info)
                continue

            child_included_file = IncludedFile(
                included_from=current_included_file, filename=file_to_include
            ).set_context(current_included_file.context or args[0])

            try:
                _handle_includes(
                    sub_tree,
                    defines,
                    file_to_include,
                    current_included_file=child_included_file,
                )

                to_include.append((sub_tree, i))
            except ConfigValidationError as err:
                errors += err.errors

    for sub_tree, i in reversed(to_include):
        tree.children.pop(i)
        tree.children[i:i] = sub_tree.children

    if len(errors) > 0:
        raise ConfigValidationError.from_collected(errors)


def _parse_contents(content: str, file: str) -> Tree[Instruction]:
    file = os.path.normpath(os.path.abspath(file))
    try:
        tree = _parser.parse(content + "\n")
        return (
            StringQuotationTransformer()
            * FileContextTransformer(file)
            * ArgumentToStringTransformer()
            * InstructionTransformer()
        ).transform(tree)
    except UnexpectedToken as e:
        unexpected_token = e.token
        allowed = e.expected
        message = f"Did not expect token: {unexpected_token}. Expected one of {allowed}"
        raise ConfigValidationError.from_info(
            ErrorInfo(
                message=message,
                line=e.line,
                end_line=e.line + 1,
                column=e.column,
                end_column=e.column + 1,
                filename=file,
            )
        ) from e
    except UnexpectedCharacters as e:
        unexpected_char = e.char
        allowed = e.allowed
        message = (
            f"Did not expect character: {unexpected_char}. Expected one of {allowed}"
        )
        raise ConfigValidationError.from_info(
            ErrorInfo(
                message=message,
                line=e.line,
                end_line=e.line + 1,
                column=e.column,
                end_column=e.column + 1,
                filename=file,
            )
        ) from e


def _parse_file(file: str) -> Tree[Instruction]:
    return _parse_contents(read_file(file), file)


def parse(
    file: str,
    schema: SchemaItemDict,
    pre_defines: Defines | None = None,
) -> ConfigDict:
    return _transform_tree(_parse_file(file), file, schema, pre_defines)


def parse_contents(
    contents: str,
    schema: SchemaItemDict,
    file_name: str,
    pre_defines: list[tuple[str, str]] | None = None,
) -> ConfigDict:
    return _transform_tree(
        _parse_contents(contents, file_name), file_name, schema, pre_defines
    )


def _transform_tree(
    tree: Tree[Instruction],
    file: str,
    schema: SchemaItemDict,
    pre_defines: Defines | None = None,
) -> ConfigDict:
    filepath = os.path.normpath(os.path.abspath(file))
    config_dir = os.path.dirname(filepath)
    config_file_name = os.path.basename(file)
    config_file_base = config_file_name.split(".")[0]

    if pre_defines is None:
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
        schema=schema,
    )
