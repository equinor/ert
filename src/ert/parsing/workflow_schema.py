from typing import Any, Dict

from .config_schema_item import SchemaItem
from .schema_dict import SchemaItemDict
from .workflow_keywords import WorkflowKeys


def define_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowKeys.DEFINE,
        required_set=False,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        substitute_from=2,
        join_after=1,
    )


class WorkflowSchemaDict(SchemaItemDict):
    def check_required(self, config_dict: Dict[str, Any], filename: str) -> None:
        pass

    def __contains__(self, item: Any) -> bool:
        return True

    def __getitem__(self, kw: str) -> SchemaItem:
        if kw == "DEFINE":
            return define_keyword()
        # Since workflow keywords are arbitrary, we create
        # a schema item on the fly when
        # it is requested by the lark parser via
        # [kw]
        return SchemaItem(kw=kw, argc_min=-1, argc_max=-1, multi_occurrence=True)


def init_workflow_schema() -> SchemaItemDict:
    schema = WorkflowSchemaDict()
    return schema
