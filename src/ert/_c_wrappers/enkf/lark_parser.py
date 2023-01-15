import datetime
import os.path
from typing import List, Mapping

from lark import Lark, Token, Tree, UnexpectedCharacters

from ert._c_wrappers.config import ConfigValidationError
from ert._c_wrappers.enkf.config_keywords import (
    CONFIG_DEFAULT_ARG_MAX,
    CONFIG_DEFAULT_ARG_MIN,
    SchemaItem,
    SchemaType,
    init_site_config,
    init_user_config,
)

grammar = r"""
WHITESPACE: (" ")+
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

CHAR: /[$\[\]=,.\*a-zæøåA-ZÆØÅ10-9_%:\<\>\/-]/
UNQUOTED: CHAR+

CHAR_NO_EQ: /[.\*a-zæøåA-ZÆØÅ10-9_%:\<\>\/-]/
UNQUOTED_NO_EQ: CHAR_NO_EQ+

CHAR_KW: /[a-zæøåA-ZÆØÅ10-9_:-]/
UNQUOTED_KW: CHAR_KW+


arg: NUMBER | STRING | UNQUOTED
arglist: UNQUOTED kw_list

kw_list: "(" [ kw_pair ("," kw_pair)*] ")"
kw_val: NUMBER | UNQUOTED_NO_EQ | STRING
kw_pair: KW_NAME "=" kw_val
KW_NAME_STRICT: "<" UNQUOTED_KW ">"
KW_NAME: UNQUOTED_KW | "<" KEYWORD_NAME ">"

COMMENT: /--.*/ NEWLINE

KEYWORD_NAME: /(?!(INCLUDE|DATA_KW|DEFINE))/ LETTER (LETTER | DIGIT | "_" | "-" | "<" | ">" )*

start: instruction+

inst: "DEFINE" KW_NAME kw_val -> define
    | "DATA_KW" KW_NAME kw_val -> data_kw
    | "INCLUDE" arg -> include
    | KEYWORD_NAME (arg* | arglist) -> keyword

instruction: inst COMMENT | COMMENT | inst NEWLINE | NEWLINE
"""  # noqa: E501


def substitute(defines, string: str):
    prev = None
    current = string
    n = 0
    while defines and prev != current and n < 100:
        n = n + 1
        for key, val in defines:
            prev = current
            current = current.replace(key, str(val))

    if n >= 100:
        print(f"reached max iterations for {string}")

    return current


class MakeDict:
    def do_it(self, tree, site_config=None):
        schema: Mapping[str, SchemaItem] = (
            init_user_config() if site_config is not None else init_site_config()
        )

        self.defines.append(["<CONFIG_PATH>", self.config_dir])
        self.defines.append(["<CONFIG_FILE_BASE>", self.config_file_base])
        self.defines.append(["<DATE>", datetime.date.today().isoformat()])
        self.schema = schema
        self.config_dict = {} if not site_config else site_config
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
                elif node.children[0].data == "include":
                    self.include(node.children[0])

        def check_valid(val, item: SchemaItem, index: int):
            if index in item.indexed_selection_set:
                if val not in item.indexed_selection_set[index]:
                    raise ConfigValidationError(
                        f"{item.kw} argument {index} must be one of"
                        f" {item.indexed_selection_set[index]}"
                    )

            return val

        def convert(val, item: SchemaItem, index: int):
            if not len(item.type_map) > index:
                return check_valid(val, item, index)
            val_type = item.type_map[index]
            if val_type is None:
                return check_valid(val, item, index)
            if val_type == SchemaType.CONFIG_INT:
                return int(check_valid(val, item, index))
            if val_type == SchemaType.CONFIG_FLOAT:
                return float(check_valid(val, item, index))
            if val_type in [SchemaType.CONFIG_PATH, SchemaType.CONFIG_EXISTING_PATH]:
                path = val
                if not os.path.isabs(val):
                    path = os.path.normpath(os.path.join(self.config_dir, val))
                if val_type == SchemaType.CONFIG_EXISTING_PATH and not os.path.exists(
                    path
                ):
                    raise ConfigValidationError(f"File {path} does not exist")
                return path
            return check_valid(val, item, index)

        def with_types(args, item: SchemaItem):
            if isinstance(args, list):
                return [convert(x, item, i) for i, x in enumerate(args)]
            else:
                return convert(args, item, 0)

        def get_value(item: SchemaItem, line: List):
            if item.join_after > 0:
                n = item.join_after + 1
                args = " ".join(str(x) for x in line[n:])
                new_line = line[1:n]
                if len(args) > 0:
                    new_line.append(args)
                return with_types(new_line, item)
            if item.argc_max > 1 or item.argc_max == CONFIG_DEFAULT_ARG_MAX:
                return with_types(line[1:], item)
            else:
                return with_types(line[1], item) if len(line) > 1 else None

        for line in self.keywords:
            key = line[0]
            if key not in schema:
                if self.add_invalid:
                    self.config_dict[key] = line[1:]
                self.errors.append(f"unknown key {key}")
                continue
            item = schema[key]
            if item.multi_occurrence:
                val = self.config_dict.get(key, [])
                value = get_value(item, line)
                val.append(value)
                self.config_dict[key] = val
            else:
                self.config_dict[key] = get_value(item, line)

        for kw, item in schema.items():
            if item.required_set:
                if item.kw not in self.config_dict:
                    raise ConfigValidationError(f"{item.kw} must be set.")

        if self.defines:
            self.config_dict["DEFINE"] = []
            for define in self.defines:
                self.config_dict["DEFINE"].append(define)

        if self.data_kws:
            self.config_dict["DATA_KW"] = []
            for data_kw in self.data_kws:
                self.config_dict["DATA_KW"].append(data_kw)

        return self.config_dict

    def __init__(self, config_dir, config_file_base, add_invalid=False):
        self.defines = []
        self.data_kws = []
        self.keywords = []
        self.config_dict = None
        self.errors = []
        self.add_invalid = add_invalid
        self.config_dir = config_dir
        self.config_file_base = config_file_base

    def include(self, tree):
        pass

    def define(self, tree):
        self.defines.append(
            [tree.children[0], substitute(self.defines, tree.children[1].children[0])]
        )

    def data_kw(self, tree):
        self.data_kws.append(
            [tree.children[0], substitute(self.defines, tree.children[1].children[0])]
        )

    def keyword(self, tree):
        inst = []
        # print(tree)
        for node in tree.children:
            if isinstance(node, Token):
                if node.type == "STRING":
                    # remove quotation marks
                    node.value = node[1 : len(node) - 1]
                inst.append(substitute(self.defines, node))
            elif node.data == "arglist":
                name = node.children[0]
                args = []
                kw_list = node.children[1]
                for kw_pair in kw_list.children:
                    if kw_pair is None:
                        break
                    key = kw_pair.children[0]
                    val = kw_pair.children[1].children[0]
                    if isinstance(val, Token) and val.type == "STRING":
                        # remove quotation marks
                        val = val[1 : len(val) - 1]
                    val = substitute(self.defines, val)
                    # args.append(f"{key}={val}")
                    args.append((key, val))
                # argstring = ",".join(args)
                inst.append(name)
                inst.append(args)
            elif node.data == "arg":
                val = node.children[0]
                inst.append(substitute(self.defines, val))

        kw = inst[0]
        if kw in self.schema:
            item = self.schema[kw]
            if (
                item.argc_min != CONFIG_DEFAULT_ARG_MIN
                and len(inst) - 1 < item.argc_min
            ):
                self.errors.append(f"{inst} does not have required number of arguments")
                if not self.add_invalid:
                    return
            if (
                item.argc_max != CONFIG_DEFAULT_ARG_MAX
                and len(inst) - 1 > item.argc_max
            ):
                self.errors.append(f"{inst} does has too many arguments")
                if not self.add_invalid:
                    return

        self.keywords.append(inst)


def do_includes(tree: Tree, config_dir):
    to_include = []
    for i, node in enumerate(tree.children):
        if not isinstance(node, Tree):
            continue
        if isinstance(node.children[0], Token):
            continue  # This is either a newline or a comment
        if node.children[0].data == "include":
            inc_node = node.children[0]
            val = inc_node.children[0].children[0]
            if not os.path.isabs(val):
                val = os.path.normpath(os.path.join(config_dir, val))

            sub_tree = _parse_file(val, "INCLUDE")
            do_includes(sub_tree, os.path.dirname(val))

            to_include.append((sub_tree, i))

    to_include.reverse()
    for sub_tree, i in to_include:
        tree.children.pop(i)
        tree.children[i:i] = sub_tree.children


def _parse_file(file, error_context_string=""):
    try:
        with open(file, encoding="utf-8") as f:
            content = f.read()
        parser = Lark(grammar, propagate_positions=True)
        tree = parser.parse(content + "\n")
        return tree
    except FileNotFoundError:
        raise ConfigValidationError(f"{error_context_string} file: {file} not found")
    except UnexpectedCharacters as e:
        msg = str(e)
        if "DEFINE" in msg or "DATA_KW" in msg:
            msg = (
                "\nA DEFINE or DATA_KW must be followed by a valid substitution "
                "keyword.\n"
                "It must be of the form: <ABC>  Inside the angle brackets, only"
                " characters, numbers, _ or - is allowed.\n"
                "\n"
                f"The parser said: {msg}"
            )
        raise ConfigValidationError(msg, config_file=file)


def parse(file, site_config=None, add_invalid=False):
    tree = _parse_file(file)
    config_dir = os.path.dirname(os.path.normpath(os.path.abspath(file)))

    do_includes(tree, config_dir)
    config_file_name = os.path.basename(file)
    config_file_base = config_file_name.split(".")[0]
    do_defines = MakeDict(
        config_dir=config_dir,
        config_file_base=config_file_base,
        add_invalid=add_invalid,
    )
    config_dict = do_defines.do_it(tree, site_config)
    # import json

    # print(json.dumps(config_dict, indent=2))
    return config_dict
