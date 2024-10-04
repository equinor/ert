import dataclasses
import inspect
import re
import sys
from typing import Dict, List, Optional

from pydantic.fields import FieldInfo

from everest.config import EverestConfig

CONFIGS = {
    name: obj
    for name, obj in inspect.getmembers(sys.modules["everest.config"])
    if inspect.isclass(obj)
}


@dataclasses.dataclass
class ParsedField:
    name: str
    description: str
    type: str
    is_required: bool
    subfields: Optional[List["ParsedField"]]

    def doc_title(self) -> str:
        return f"{self.name} ({'required' if self.is_required else 'optional'})"

    def doc_description(self) -> str:
        lines = self.description.splitlines()

        return "\n".join(
            [
                (line if line.startswith("|") else " ".join(line.split()))
                for line in lines
            ]
        )

    def clean_type(self) -> str:
        if sys.version_info < (3, 10):
            self.type = self.type.replace("Annotated[int, Ge(ge=0)]", "NonNegativeInt")
            self.type = self.type.replace("Annotated[int, Gt(gt=0)]", "PositiveInt")
            self.type = self.type.replace("Annotated[float, Gt(gt=0)]", "PositiveFloat")
            self.type = re.sub(r"Union\[(.+),\s+NoneType\]", r"Optional[\1]", self.type)

        self.type = self.type.replace("Annotated[int, Ge]", "NonNegativeInt")
        self.type = self.type.replace("Annotated[int, Gt]", "PositiveInt")
        self.type = self.type.replace("Annotated[float, Gt]", "PositiveFloat")
        self.type = self.type.replace("Dict[", "Mapping[")

        self.type = re.sub(r"Union\[(.+),\s+NoneType\]", r"Optional[\1]", self.type)
        return self.type


def parse_field_info(field_infos: Dict[str, FieldInfo]):
    """
    Extracts relevant info from a list of pydantic model fields into a convenient
    format of ParsedField items, to be used for further generation of docs
    """
    parsed_fields = []
    for name, model_field in field_infos.items():
        annotation = str(dict(model_field.__repr_args__()).get("annotation"))
        parsed_subfields = None
        try:
            if model_field.annotation is not None and hasattr(
                model_field.annotation, "model_fields"
            ):
                subfields = model_field.annotation.model_fields
                parsed_subfields = parse_field_info(subfields)
            elif "everest.config" in str(model_field.annotation):
                for cls_name, value in CONFIGS.items():
                    if cls_name in annotation:
                        subfields = value.model_fields
                        annotation = re.sub(
                            r"List\[(.*)\.(\w+)\]", r"List[\2]", annotation
                        )
                        parsed_subfields = parse_field_info(subfields)

        except Exception:
            # If it failed, we simply assume the field was a str, int, float etc.
            pass

        parsed_fields.append(
            ParsedField(
                name=name,
                description=model_field.description or "",
                type=annotation,
                is_required=model_field.is_required(),
                subfields=parsed_subfields,
            )
        )

    return parsed_fields


class DocBuilder:
    def __init__(self, extended: bool):
        self.doc = ""
        self.extended = extended

    @staticmethod
    def _box_doc(msg, level):
        def indent(line):
            return " " * (2 * level) + f"| {line}"

        if not msg:
            return ""
        lines = msg.split("\n")

        return "\n".join(map(indent, map(str.strip, lines)))

    @staticmethod
    def _indent(text: str, level: int):
        stripped_lines = [line.strip() for line in text.split("\n")]
        if level > 0:
            stripped_lines = [
                "    " * (level - 1) + line if line else "" for line in stripped_lines
            ]

        return "\n".join(stripped_lines)

    def add_title(self, title: str, level: int):
        if level == 0:
            self.doc += title + "\n"
            self.doc += "-" * len(title)
        else:
            self.doc += self._indent(f"**{title}**", level)

    def add_description(self, description: str, level: int):
        if self.extended:
            self.doc += self._indent(
                text=self._box_doc(f"\nDocumentation: {description}\n", level),
                level=level,
            )
        else:
            self.doc += self._indent(description, level)

        self.doc += "\n"

    def add_type(self, type: str, level: int):
        self.doc += self._indent(f"Type: *{type}*", level)
        self.doc += "\n"

    def add_newline(self):
        self.doc += "\n"


def _generate_rst(
    parsed_fields: List[ParsedField],
    level: int = 0,
    builder: Optional[DocBuilder] = None,
    extended=False,
):
    if not builder:
        builder = DocBuilder(extended=extended)

    for f in parsed_fields:
        builder.add_title(title=f.doc_title(), level=level)
        builder.add_newline()

        builder.add_type(f.clean_type(), level=level + 1)
        builder.add_newline()

        builder.add_description(f.doc_description(), level=level + 1)
        builder.add_newline()

        if f.subfields:
            _generate_rst(f.subfields, level=level + 1, builder=builder)

        builder.add_newline()

    return builder


def generate_docs_pydantic_to_rst(extended: bool = False):
    order = (
        "name",
        "model",
        "controls",
        "optimization",
        "objective_functions",
        "environment",
        "wells",
        "input_constraints",
        "output_constraints",
        "simulator",
        "install_jobs",
        "install_workflow_jobs",
        "install_data",
        "install_templates",
        "forward_model",
        "workflows",
        "source",
        "target",
        "template",
        "server",
        "export",
        "definitions",
    )

    fields = EverestConfig.model_fields
    list_of_fields = {n: fields[n] for n in order if n in fields}

    parsed_field_info = parse_field_info(list_of_fields)

    return _generate_rst(parsed_field_info, extended=extended).doc.strip() + "\n"
