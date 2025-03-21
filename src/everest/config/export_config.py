from typing import Any

from pydantic import BaseModel, Field, model_validator

from ert.config import ConfigWarning


class ExportConfig(BaseModel, extra="forbid"):
    csv_output_filepath: str | None = Field(
        default=None,
        description="'csv_output_filepath' key is deprecated. You can safely remove it from the config file",
    )
    discard_gradient: bool | None = Field(
        default=None,
        description="'discard_gradient' key is deprecated. You can safely remove it from the config file",
    )
    discard_rejected: bool | None = Field(
        default=None,
        description="'discard_rejected' key is deprecated. You can safely remove it from the config file",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="List of eclipse keywords to be exported.",
    )
    batches: list[int] | None = Field(
        default=None,
        description="'batches' key is deprecated. You can safely remove it from the config file",
    )
    skip_export: bool | None = Field(
        default=None,
        description="'skip_export' key is deprecated. You can safely remove it from the config file",
    )

    @model_validator(mode="before")
    @classmethod
    def deprecate_export_keys(cls, values: Any) -> Any:  # pylint: disable=E0213
        for key in list(values.keys()):
            match key:
                case "csv_output_filepath":
                    values["csv_output_filepath"] = None
                    ConfigWarning.deprecation_warn(
                        "'csv_output_filepath' key is deprecated."
                        " You can safely remove it from the config file"
                    )
                case "discard_gradient":
                    values["discard_gradient"] = None
                    ConfigWarning.deprecation_warn(
                        "'discard_gradient' key is deprecated."
                        " You can safely remove it from the config file"
                    )
                case "discard_rejected":
                    values["discard_rejected"] = None
                    ConfigWarning.deprecation_warn(
                        "'discard_rejected' key is deprecated."
                        " You can safely remove it from the config file"
                    )
                case "batches":
                    values["batches"] = None
                    ConfigWarning.deprecation_warn(
                        "'batches' key is deprecated."
                        " You can safely remove it from the config file"
                    )
                case "skip_export":
                    values["skip_export"] = None
                    ConfigWarning.deprecation_warn(
                        "'skip_export' key is deprecated."
                        " You can safely remove it from the config file"
                    )
        return values
