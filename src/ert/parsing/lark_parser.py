# mypy: ignore-errors
import datetime
import logging
import os
import os.path
import warnings
from typing import List, Optional, Tuple, Union

from lark import Discard, Lark, Token, Transformer, Tree, UnexpectedCharacters
from typing_extensions import Self

from .config_errors import ConfigValidationError, ConfigWarning
from .config_schema import SchemaItem, define_keyword
from .error_info import ErrorInfo
from .schema_dict import SchemaItemDict
from .types import ConfigDict, Defines, FileContextToken, Instruction, MaybeWithContext

grammar = r"""
WHITESPACE: (" "|"\t")+
%ignore WHITESPACE

%import common.CNAME
%import common.SIGNED_NUMBER    -> NUMBER
%import common.NEWLINE          -> NEWLINE


_STRING_INNER: /.+?/
_STRING_ESC_INNER: _STRING_INNER /(?<!\\)(\\\\)*?/

STRING: "\"" _STRING_ESC_INNER "\"" | "'" _STRING_ESC_INNER "'" | "\"\""

DIGIT: "0".."9"
LCASE_LETTER: "a".."z"
UCASE_LETTER: "A".."Z"

LETTER: UCASE_LETTER | LCASE_LETTER
WORD: LETTER+

CHAR: /[&$\[\]=,.\*a-zæøåA-ZÆØÅ10-9_%:\<\>\/\\?+-\^]/
UNQUOTED: CHAR+

CHAR_NO_EQ: /[+.\*a-zæøåA-ZÆØÅ10-9_%:\<\>\/-]/
UNQUOTED_NO_EQ: /(?!([ ]))/ CHAR_NO_EQ+
STR_NO_EQ_COM_PARN: /[^=,)( ]+/

CHAR_KW: /[a-zæøåA-ZÆØÅ10-9_:-]/
UNQUOTED_KW: CHAR_KW+
ENV_STRING: "$" UNQUOTED_NO_EQ

arg: STRING | UNQUOTED

kw_list: "(" [ kw_pair ("," kw_pair)*] ")"
kw_val: UNQUOTED_NO_EQ | STRING | ENV_STRING | STR_NO_EQ_COM_PARN
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
        token.value = token.value[1 : len(token.value) - 1]
        return token

    def STR_NO_EQ_COM_PARN(self, token: Token) -> Token:
        token.value = token.value.replace('"', "")
        return token


class ArgumentToStringTransformer(Transformer):
    """Flattens all argument types to just tokens or
    relevant python datastructures"""

    def arg(self, rule: List[FileContextToken]) -> FileContextToken:
        return rule[0]

    def kw_val(self, rule: List[FileContextToken]) -> FileContextToken:
        return rule[0]

    def kw_list(self, kw_list) -> List[Tuple[FileContextToken, FileContextToken]]:
        args = []
        for kw_pair in kw_list:
            if kw_pair is not None:
                key, val = kw_pair.children
                args.append((key, val))
        return args


class FileContextTransformer(Transformer):
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


def _substitute_token(
    defines: Defines,
    token: FileContextToken,
    expand_env: bool = True,
) -> FileContextToken:
    current: FileContextToken = token

    # replace from env
    if expand_env:
        for key, val in os.environ.items():
            current = current.replace_value(f"${key}", val)
    if not defines:
        return current

    # replace from defines
    prev = None
    n = 0
    while prev != current and n < 100:
        n = n + 1
        for key, val in defines:
            prev = current
            current = current.replace_value(key, str(val))

    for key, val in defines:
        if key in current:
            warnings.warn(
                f"Gave up replacing in {token}.\n"
                f"After replacing the value is now: {current}.\n"
                f"This still contains the replacement value: {key}, "
                f"which would be replaced by {val}. "
                f"Probably this causes a loop.",
                category=ConfigWarning,
            )

    return current


def _tree_to_dict(
    config_file: str,
    pre_defines: Defines,
    tree: Tree[Instruction],
    schema: SchemaItemDict,
    site_config: Optional[ConfigDict] = None,
) -> ConfigDict:
    config_dict: ConfigDict = site_config if site_config else {}
    defines = pre_defines.copy()
    config_dict["DEFINE"] = defines  # type: ignore

    errors = []
    cwd = os.path.dirname(os.path.abspath(config_file))

    for node in tree.children:
        args: List[FileContextToken]
        kw: FileContextToken
        kw, *args = node  # type: ignore
        if kw not in schema:
            warnings.warn(f"Unknown keyword {kw!r}", category=ConfigWarning)
            continue

        constraints = schema[kw]

        try:
            args = constraints.join_args(args)
            args = _substitute_args(args, constraints, defines)
            value_list = constraints.apply_constraints(args, kw, cwd)

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


ArgPairList = List[Tuple[FileContextToken]]
ParsedArgList = List[Union[FileContextToken, ArgPairList]]


def _substitute_args(
    args: ParsedArgList,
    constraints: SchemaItem,
    defines: Defines,
) -> ParsedArgList:
    if constraints.substitute_from < 1:
        return args

    def substitute_arglist_tuple(
        tup: Tuple[FileContextToken],
    ) -> Tuple[FileContextToken]:
        key, value = tup
        substituted_value = _substitute_token(defines, value, constraints.expand_envvar)

        return tuple([key, substituted_value])

    def substitute_arg(
        arg: Union[FileContextToken, List[Tuple[FileContextToken]]]
    ) -> Union[FileContextToken, List[Tuple[FileContextToken]]]:
        if isinstance(arg, FileContextToken):
            return _substitute_token(defines, arg, constraints.expand_envvar)

        if isinstance(arg, list):
            # It is a list of keyword tuples
            return [substitute_arglist_tuple(x) for x in arg]

        raise ValueError(
            f"Expected "
            f"Union[FileContextToken, List[Tuple[FileContextToken]]], "
            f"got {arg}"
        )

    return [
        substitute_arg(arg) if (i + 1) >= constraints.substitute_from else arg
        for i, arg in enumerate(args)
    ]


class IncludedFile:
    def __init__(self, included_from: "IncludedFile", filename: str):
        self.included_from = included_from
        self.filename = filename
        self.context = None

    def __contains__(self, filename: str):
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
    current_included_file: Optional[IncludedFile] = None,
):
    if current_included_file is None:
        current_included_file = IncludedFile(included_from=None, filename=config_file)

    config_dir = os.path.dirname(config_file)
    to_include = []

    errors = []
    for i, node in enumerate(tree.children):
        args: List[FileContextToken]
        kw: FileContextToken
        kw, *args = node  # type: ignore
        if kw == "DEFINE":
            constraints = define_keyword()
            args = constraints.join_args(args)
            args = _substitute_args(args, constraints, defines)
            defines.append(args)  # type: ignore
        if kw == "INCLUDE":
            if len(args) > 1:
                superfluous_tokens: List[FileContextToken] = args[1:]
                superfluous = FileContextToken.join_tokens(superfluous_tokens)

                error_context = (
                    superfluous
                    if current_included_file.included_from is None
                    else current_included_file.context
                )

                errors.append(
                    ErrorInfo(
                        message="Keyword:INCLUDE must have exactly one argument "
                        f"at ({current_included_file.filename} "
                        f"line {superfluous.line})",
                        filename=config_file,
                    ).set_context(error_context)
                )
                continue

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
                sub_tree = _parse_file(file_to_include, "INCLUDE")
            except ConfigValidationError as err:
                errors.append(
                    ErrorInfo(
                        message=str(err),
                        filename=config_file,
                    ).set_context(args[0])
                )
                continue

            child_included_file = IncludedFile(
                included_from=current_included_file, filename=args[0]
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


def _parse_file(file: str, error_context_string: str = "") -> Tree[Instruction]:
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
                f"{error_context_string} file: {str(file)} not found"
            )
        raise IOError(f"{error_context_string} file: {str(file)} not found")
    except UnexpectedCharacters as e:
        unexpected_char = e.char
        allowed_chars = e.allowed
        message = (
            f"Did not expect character: {unexpected_char}. "
            f"Expected one of {allowed_chars}"
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
        )
    except UnicodeDecodeError as e:
        error_words = str(e).split(" ")
        hex_str = error_words[error_words.index("byte") + 1]
        try:
            unknown_char = chr(int(hex_str, 16))
        except ValueError:
            unknown_char = f"hex:{hex_str}"

        # Find the first line in the file with decode error
        bad_byte_lines: List[int] = []
        with open(file, "rb") as f:
            all_lines = []
            for line in f:
                all_lines.append(line)

        for i, line in enumerate(all_lines):
            try:
                line.decode("utf-8")
            except UnicodeDecodeError:
                # The error occurs on this line, so make this entire line red
                # (Figuring column if it is not 0 is tricky and prob not necessary)
                # Use 1-indexed lines like lark and for errors in ert
                bad_byte_lines.append(i + 1)

        assert len(bad_byte_lines) != -1

        raise ConfigValidationError(
            [
                ErrorInfo(
                    message=f"Unsupported non UTF-8 character {unknown_char!r} "
                    f"found in file: {file!r}",
                    filename=str(file),
                    column=0,
                    line=bad_line,
                    end_column=-1,
                    end_line=bad_line,
                )
                for bad_line in bad_byte_lines
            ]
        )


def parse(
    file: str,
    schema: SchemaItemDict,
    site_config: Optional[ConfigDict] = None,
) -> ConfigDict:
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
        schema=schema,
    )
