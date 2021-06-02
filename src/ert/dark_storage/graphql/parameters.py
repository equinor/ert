from typing import Any, List, Optional, TYPE_CHECKING
import graphene as gr
from sqlalchemy.orm import Session

from ert_storage.ext.graphene_sqlalchemy import SQLAlchemyObjectType
from ert_storage import database_schema as ds
from ert_storage.endpoints.experiments import prior_to_dict


if TYPE_CHECKING:
    from graphql.execution.base import ResolveInfo


class Parameter(SQLAlchemyObjectType):
    class Meta:
        model = ds.Record

    name = gr.String()
    prior = gr.JSONString()

    def resolve_name(root: ds.Record, info: "ResolveInfo") -> str:
        return root.name

    def resolve_prior(root: ds.Record, info: "ResolveInfo") -> Optional[dict]:
        prior = root.record_info.prior
        return prior_to_dict(prior) if prior is not None else None
