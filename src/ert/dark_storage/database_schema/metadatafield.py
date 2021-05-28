from typing import Any, Mapping
import sqlalchemy as sa


class MetadataField:
    _metadata = sa.Column("metadata", sa.JSON, nullable=True)

    @property
    def metadata_dict(self) -> Mapping[str, Any]:
        if self._metadata is None:
            return dict()
        return self._metadata
