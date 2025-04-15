from typing import Any

from pydantic import BaseModel, Field, model_validator

from ert.config import ConfigWarning


class ExportConfig(BaseModel, extra="forbid"):
    keywords: list[str] = Field(
        default_factory=list,
        description="List of eclipse keywords to be exported.",
    )

    @model_validator(mode="before")
    @classmethod
    def deprecate_export_keys(cls, values: Any) -> Any:
        for key in list(values.keys()):
            if key in {
                "csv_output_filepath",
                "discard_gradient",
                "discard_rejected",
                "batches",
                "skip_export",
            }:
                values.pop(key)
                ConfigWarning.deprecation_warn(
                    f"'{key}' key is deprecated."
                    " You can safely remove it from the config file"
                )

        return values
