from lark import Lark, Tree, Token
from lark.visitors import Interpreter, Visitor, Transformer

from ert._c_wrappers.enkf._config_content_as_dict import SINGLE_OCCURRENCE_SINGLE_ARG_KEYS, \
    SINGLE_OCCURRENCE_MULTI_ARG_KEYS, MULTI_OCCURRENCE_SINGLE_ARG_KEYS, JOIN_KEYS

grammar = r"""
WHITESPACE: (" ")+
%ignore WHITESPACE
%ignore COMMENT


%import common.ESCAPED_STRING   -> STRING
%import common.CNAME
%import common.SIGNED_NUMBER    -> NUMBER
%import common.NEWLINE          -> NEWLINE


CHAR: /[.a-zæøåA-ZÆØÅ10-9_%:\<\>\/-]/
UNQUOTED: CHAR+

arg: NUMBER | STRING | UNQUOTED
arglist: CNAME kw_list

kw_list: "(" [ kw_pair ("," kw_pair)*] ")"
kw_val: NUMBER | UNQUOTED | STRING 
kw_pair: KW_NAME "=" kw_val
KW_NAME: "<" CNAME ">"

COMMENT: /--.*\n/

start: instruction+

inst: "DEFINE" KW_NAME kw_val -> define
    | "DATA_KW" KW_NAME kw_val -> data_kw
    | "INCLUDE" arg -> include
    | CNAME (arg* | arglist) -> keyword


instruction: inst (NEWLINE)


"""
#%import common._STRING_ESC_INNER -> UNQUOTED

def substitute(defines, string: str):
    prev = None
    current = string
    n = 0
    while defines and prev != current and n < 100:
        n = n + 1
        for key, val in defines:
            current = current.replace(key, val)
    if n >= 100:
        print(f"reached max iterations for {string}")
    try:
        current = float(current)
        current = int(current)
    except ValueError:
        pass
    return current

class MakeDict:

    def do_it(self, tree, site_config=None):
        self.config_dict = {} if not site_config else site_config
        for node in tree.children:
            if isinstance(node, Tree) and node.data == "instruction":
                if node.children[0].data == "define":
                    self.define(node.children[0])
                elif node.children[0].data == "data_kw":
                    self.data_kw(node.children[0])
                elif node.children[0].data == "keyword":
                    self.keyword(node.children[0])

        for line in self.keywords:
            key = line[0]
            if key in SINGLE_OCCURRENCE_SINGLE_ARG_KEYS:
                self.config_dict[key] = line[1]
            elif key in SINGLE_OCCURRENCE_MULTI_ARG_KEYS:
                self.config_dict[key] = line[1:]
            elif key in MULTI_OCCURRENCE_SINGLE_ARG_KEYS:
                val = self.config_dict.get(key, [])
                val.append(line[1])
                self.config_dict[key] = val
            elif key == "QUEUE_OPTION":
                args = " ".join(str(x) for x in line[3:])
                new_line = line[:3]
                new_line.append(args)
                val = self.config_dict.get(key, [])
                val.append(new_line[1:])
                self.config_dict[key] = val

            else:
                val = self.config_dict.get(key, [])
                val.append(line[1:])
                self.config_dict[key] = val

        return self.config_dict


    def __init__(self):
        self.defines = []
        self.my_data_kw = []
        self.keywords = []
        self.config_dict = None

    def define(self, tree):
        self.defines.append([tree.children[0], substitute(self.defines, tree.children[1].children[0])])

    def data_kw(self, tree):
        self.my_data_kw.append([tree.children[0], substitute(self.defines, tree.children[1].children[0])])

    def keyword(self, tree):
        inst = []
        print(tree)
        for node in tree.children:
            if isinstance(node, Token):
                inst.append(substitute(self.defines, node))
            elif node.data == "arglist":
                name = node.children[0]
                args = []
                kw_list = node.children[1]
                for kw_pair in kw_list.children:
                    key = kw_pair.children[0]
                    val = kw_pair.children[1].children[0]
                    val = substitute(self.defines, val)
                    args.append(f"{key}={val}")
                argstring = ", ".join(args)
                inst.append(name)
                inst.append(argstring)
            elif node.data == "arg":
                val = node.children[0]
                inst.append(substitute(self.defines, val))
        self.keywords.append(inst)

    def skip(self, tree):
        pass


def parse(file, site_config=None):
    with open(file) as f:
        parser = Lark(grammar, propagate_positions=True)
        tree = parser.parse(f.read() + "\n")
        do_defines = MakeDict()
        config_dict = do_defines.do_it(tree, site_config)
        import json
        print(json.dumps(config_dict, indent=2))
        return config_dict
