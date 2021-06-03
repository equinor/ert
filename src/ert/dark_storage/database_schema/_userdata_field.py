from typing import Any, Mapping
import sqlalchemy as sa


class UserdataField:
    userdata = sa.Column(sa.JSON, nullable=False, default=dict)
