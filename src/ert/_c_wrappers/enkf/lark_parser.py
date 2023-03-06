import datetime
import logging
import os
import os.path
import shutil
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from lark import Lark, Token, Transformer, Tree, UnexpectedCharacters

from ert._c_wrappers.config import ConfigValidationError, ConfigWarning
from ert._c_wrappers.enkf.config_keywords import (
    CONFIG_DEFAULT_ARG_MAX,
    CONFIG_DEFAULT_ARG_MIN,
    SchemaItem,
    SchemaType,
    init_site_config,
    init_user_config,
)


class FileContextToken(Token):
    filename: str

    def __new__(cls, token, filename):
        inst = super(FileContextToken, cls).__new__(
            cls,
            token.type,
            token.value,
            token.start_pos,
            token.line,
            token.column,
            token.end_line,
            token.end_column,
            token.end_pos,
        )
        inst.filename = filename
        return inst

    def __repr__(self):
        return f"{self.value!r}"

    def __str__(self):
        return self.value


class StringQuotationTransformer(Transformer):
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


class FileContextTransformer(Transformer):
    def __init__(self, filename):
        self.filename = filename
        super().__init__(visit_tokens=True)

    def __default_token__(self, token) -> FileContextToken:
        return FileContextToken(token, self.filename)


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

arg: NUMBER | STRING | UNQUOTED
arglist: UNQUOTED kw_list

kw_list: "(" [ kw_pair ("," kw_pair)*] ")"
kw_val: NUMBER | UNQUOTED_NO_EQ | STRING | ENV_STRING
kw_pair: KW_NAME "=" kw_val
KW_NAME_STRICT: "<" UNQUOTED_KW ">"
KW_NAME: UNQUOTED_KW | "<" KEYWORD_NAME ">"

COMMENT: /--.*/ NEWLINE

KEYWORD_NAME: /(?!(INCLUDE|DATA_KW|DEFINE))/ LETTER (LETTER | DIGIT | "_" | "-" | "<" | ">" )*

start: instruction+

inst: "DEFINE" KW_NAME kw_val -> define
    | "DATA_KW" KW_NAME kw_val -> data_kw
    | "INCLUDE" arg* -> include
    | KEYWORD_NAME (arg* | arglist) -> keyword

instruction: inst COMMENT | COMMENT | inst NEWLINE | NEWLINE
"""  # noqa: E501

logger = logging.getLogger(__name__)


def substitute(
    defines: Iterable[Tuple[str, str]],
    token: FileContextToken,
    expand_env: bool = True,
) -> FileContextToken:
    prev = None
    current = token
    if expand_env:
        for key, val in os.environ.items():
            current = current.replace(f"${key}", val)
    n = 0
    while defines and prev != current and n < 100:
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

    if isinstance(current, str):
        return FileContextToken(Token(token.type, current), token.filename)
    else:
        return current


@dataclass
class Instruction:
    keyword: FileContextToken
    args: List[FileContextToken]


@dataclass
class JobInstruction:
    keyword: FileContextToken
    job_name: FileContextToken
    args: List[FileContextToken]


class _TreeToDictTransformer:
    def __init__(
        self,
        config_file: str,
        config_dir: str,
        config_file_base: str,
    ):
        self.defines = []
        self.data_kws = []
        self.keywords: List[Union[Instruction, JobInstruction]] = []
        self.config_dict = None
        self.config_dir = config_dir
        self.config_file_base = config_file_base
        self.config_file = config_file

    def __call__(
        self, tree: Tree[FileContextToken], site_config: Optional[Dict[str, Any]] = None
    ):
        schema: Mapping[str, SchemaItem] = (
            init_user_config() if site_config is not None else init_site_config()
        )

        self.defines.append(["<CONFIG_PATH>", self.config_dir])
        self.defines.append(["<CONFIG_FILE_BASE>", self.config_file_base])
        self.defines.append(["<DATE>", datetime.date.today().isoformat()])
        self.defines.append(["<CWD>", self.config_dir])
        self.defines.append(["<CONFIG_FILE>", os.path.basename(self.config_file)])

        self.schema = schema
        self.config_dict = site_config if site_config else {}
        for node in tree.children:
            if isinstance(node, Tree) and node.data == "instruction":
                if isinstance(node.children[0], Token):
                    pass  # newline or comment
                elif node.children[0].data == "define":
                    self.define(node.children[0])
                elif node.children[0].data == "data_kw":
                    self.data_kw(node.children[0])
                elif node.children[0].data == "keyword":
                    self.keyword(node.children[0])

        def check_valid(val: FileContextToken, item: SchemaItem, index: int):
            if index in item.indexed_selection_set:
                if val not in item.indexed_selection_set[index]:
                    raise ConfigValidationError(
                        f"{item.kw!r} argument {index!r} must be one of"
                        f" {item.indexed_selection_set[index]!r} was {val.value!r}",
                        config_file=val.filename,
                    )

        def convert(
            val: FileContextToken, item: SchemaItem, index: int
        ) -> Optional[Union[str, int, float]]:
            check_valid(val, item, index)
            if not len(item.type_map) > index:
                return str(val)
            val_type = item.type_map[index]
            if val_type is None:
                return val
            if val_type == SchemaType.CONFIG_BOOL:
                if val.lower() == "true":
                    return True
                elif val.lower() == "false":
                    return False
                else:
                    raise ConfigValidationError(
                        f"{item.kw!r} must have a boolean value"
                        f" as argument {index + 1!r}",
                        config_file=val.filename,
                    ) from None
            if val_type == SchemaType.CONFIG_INT:
                try:
                    return int(val)
                except ValueError:
                    raise ConfigValidationError(
                        f"{item.kw!r} must have an integer value"
                        f" as argument {index + 1!r}",
                        config_file=val.filename,
                    ) from None
            if val_type == SchemaType.CONFIG_FLOAT:
                try:
                    return float(val)
                except ValueError:
                    raise ConfigValidationError(
                        f"{item.kw!r} must have a number as argument {index + 1!r}",
                        config_file=val.filename,
                    ) from None
            if val_type in [SchemaType.CONFIG_PATH, SchemaType.CONFIG_EXISTING_PATH]:
                path = str(val)
                if not os.path.isabs(val):
                    path = os.path.normpath(
                        os.path.join(os.path.dirname(val.filename), val)
                    )
                if val_type == SchemaType.CONFIG_EXISTING_PATH and not os.path.exists(
                    path
                ):
                    err = f'Cannot find file or directory "{val.value}" \n'
                    if path != val:
                        err += f"The configured value was {path!r} "
                    raise ConfigValidationError(err, config_file=val.filename)
                return path
            if val_type == SchemaType.CONFIG_EXECUTABLE:
                path = str(val)
                if not os.path.isabs(val) and not os.path.exists(val):
                    path = shutil.which(val)
                    if path is None:
                        raise ConfigValidationError(
                            f"Could not find executable {val.value!r}",
                            config_file=val.filename,
                        )
                return path
            return str(val)

        def with_types(args: List[FileContextToken], item: SchemaItem):
            return [convert(x, item, i) for i, x in enumerate(args)]

        def get_values(item: SchemaItem, line: List[FileContextToken]):
            if item.argc_max != -1 and item.argc_max < len(line) - 1:
                raise ConfigValidationError(
                    f"Keyword: {item.kw!r} takes at most {item.argc_max!r} arguments"
                )
            if item.join_after > 0:
                n = item.join_after
                args = " ".join(str(x) for x in line[n:])
                new_line = line[0:n]
                if len(args) > 0:
                    new_line.append(
                        FileContextToken(Token(line[n].type, args), line[n].filename)
                    )
                return with_types(new_line, item)
            if item.argc_max > 1 or item.argc_max == CONFIG_DEFAULT_ARG_MAX:
                return with_types(line, item)
            else:
                return with_types([line[0]], item)[0] if len(line) > 0 else None

        for line in self.keywords:
            try:
                key = line.keyword
                if key not in schema:
                    warnings.warn(f"unknown key {key!r}", category=ConfigWarning)
                    continue
                item = schema[key]
                if item.multi_occurrence:
                    val = self.config_dict.get(key, [])
                    if isinstance(line, Instruction):
                        value = get_values(item, line.args)
                        val.append(value)
                        self.config_dict[key] = val
                    elif isinstance(line, JobInstruction):
                        val.append((line.job_name, line.args))
                        self.config_dict[key] = val
                else:
                    if isinstance(line, Instruction):
                        self.config_dict[key] = get_values(item, line.args)
                    elif isinstance(line, JobInstruction):
                        self.config_dict[key] = (line.job_name, line.args)
            except ConfigValidationError as e:
                token: Token = line.keyword
                raise ConfigValidationError(
                    f"{e.errors}\nWas used in {token.value} at line {token.line}",
                    config_file=token.filename,
                ) from e

        for _, item in schema.items():
            if item.required_set:
                if item.kw not in self.config_dict:
                    raise ConfigValidationError(
                        f"{item.kw} must be set.", config_file=self.config_file
                    )

        if self.defines:
            self.config_dict["DEFINE"] = []
            for define in self.defines:
                self.config_dict["DEFINE"].append(define)

        if self.data_kws:
            self.config_dict["DATA_KW"] = []
            for data_kw in self.data_kws:
                self.config_dict["DATA_KW"].append(data_kw)

        return self.config_dict

    def define(self, tree: Tree):
        kw = tree.children[0]
        value = tree.children[1].children[0]
        if not isinstance(value, FileContextToken):
            raise ConfigValidationError(
                f"Cannot define {kw!r} to {value!r}", config_file=self.config_file
            )
        self.defines.append([kw, substitute(self.defines, value)])

    def data_kw(self, tree: Tree):
        kw = tree.children[0]
        value = tree.children[1].children[0]
        if not isinstance(value, FileContextToken):
            raise ConfigValidationError(
                f"Cannot define {kw!r} to {value!r}", config_file=self.config_file
            )
        self.data_kws.append([kw, substitute(self.defines, value)])

    def keyword(self, tree: Tree):
        arguments = []
        kw = tree.children[0]
        if not isinstance(kw, FileContextToken):
            raise ConfigValidationError(
                f"Unrecognized keyword {kw!r}", config_file=self.config_file
            )
        do_env = True
        if kw in self.schema:
            do_env = self.schema[kw].expand_envvar
        instruction: Union[Instruction, JobInstruction]
        if len(tree.children) == 2 and tree.children[1].data == "arglist":
            node = tree.children[1]
            name = node.children[0]
            args = []
            kw_list = node.children[1]
            for kw_pair in kw_list.children:
                if kw_pair is None:
                    break
                key = kw_pair.children[0]
                val = kw_pair.children[1].children[0]
                if not isinstance(val, FileContextToken):
                    raise ConfigValidationError(
                        f"Could not read keyword value {kw!r} for {key!r}",
                        config_file=self.config_file,
                    )
                args.append((key, val))
            job_name = name
            arguments = args
            instruction = JobInstruction(kw, job_name, arguments)
        else:
            for node in tree.children[1:]:
                if node.data != "arg":
                    raise ConfigValidationError(
                        "Cannot mix argument list with"
                        f" parenthesis and without in {node}"
                    )

                val = node.children[0]

                if not isinstance(val, FileContextToken):
                    raise ConfigValidationError(
                        f"Could not read argument {val!r}",
                        config_file=self.config_file,
                    )
                arguments.append(substitute(self.defines, val, expand_env=do_env))
            instruction = Instruction(kw, arguments)
        if kw in self.schema:
            item = self.schema[kw]
            if (
                item.argc_min != CONFIG_DEFAULT_ARG_MIN
                and len(instruction.args) < item.argc_min
            ):
                raise ConfigValidationError(
                    f"{instruction.keyword!r} needs at least {item.argc_min} arguments",
                    config_file=kw.filename,
                )
            if (
                item.argc_max != CONFIG_DEFAULT_ARG_MAX
                and len(instruction.args) > item.argc_max
            ):
                raise ConfigValidationError(
                    f"{kw!r} takes maximum {item.argc_max} arguments",
                    config_file=kw.filename,
                )

        self.keywords.append(instruction)


def handle_includes(tree: Tree, config_file: str):
    config_dir = os.path.dirname(config_file)
    to_include = []
    for i, node in enumerate(tree.children):
        if not isinstance(node, Tree):
            raise ConfigValidationError(
                f"Unexpected top level statement {node!r}", config_file=config_file
            )
        if isinstance(node.children[0], Token):
            continue  # This is either a newline or a comment
        if node.children[0].data == "include":
            inc_node = node.children[0]
            if len(inc_node.children) > 1:
                raise ConfigValidationError(
                    "Keyword:INCLUDE must have exactly one argument",
                    config_file=config_file,
                )
            val = inc_node.children[0].children[0]
            if not isinstance(val, FileContextToken):
                raise ConfigValidationError(
                    f"INCLUDE keyword must be given filepath, got {val!r}",
                    config_file=config_file,
                )
            if not os.path.isabs(val):
                val = os.path.normpath(os.path.join(config_dir, val))

            sub_tree = _parse_file(val, "INCLUDE")
            handle_includes(sub_tree, val)

            to_include.append((sub_tree, i))

    to_include.reverse()
    for sub_tree, i in to_include:
        tree.children.pop(i)
        tree.children[i:i] = sub_tree.children


def _parse_file(
    file: Union[str, bytes, os.PathLike], error_context_string: str = ""
) -> Tree[FileContextToken]:
    try:
        with open(file, encoding="utf-8") as f:
            content = f.read()
        parser = Lark(grammar, propagate_positions=True)
        tree = parser.parse(content + "\n")
        return FileContextTransformer(file).transform(
            StringQuotationTransformer().transform(tree)
        )
    except FileNotFoundError:
        if error_context_string == "INCLUDE":
            raise ConfigValidationError(
                f"{error_context_string} file: {file} not found"
            )
        else:
            raise IOError(f"{error_context_string} file: {file} not found")
    except UnexpectedCharacters as e:
        msg = str(e)
        if "DEFINE" in msg or "DATA_KW" in msg:
            msg = (
                f"\nKeyword:{'DEFINE' if 'DEFINE' in msg else 'DATA_KW'} "
                f"must have two or more arguments.\n"
                "It must be of the form: <ABC>  Inside the angle brackets, only"
                " characters, numbers, _ or - is allowed.\n"
                "\n"
                f"The parser said: {msg!r}"
            )
        raise ConfigValidationError(msg, config_file=file)


def parse(
    file: str,
    site_config: Optional[Dict[str, Any]] = None,
):
    filepath = os.path.normpath(os.path.abspath(file))
    tree = _parse_file(filepath)
    config_dir = os.path.dirname(filepath)

    handle_includes(tree, filepath)
    config_file_name = os.path.basename(file)
    config_file_base = config_file_name.split(".")[0]
    return _TreeToDictTransformer(
        config_dir=config_dir,
        config_file_base=config_file_base,
        config_file=file,
    )(tree, site_config)
