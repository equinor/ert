from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from everest.config.validation_utils import check_writable_filepath


class ExportConfig(BaseModel, extra="forbid"):  # type: ignore
    csv_output_filepath: Optional[str] = Field(
        default=None,
        description="""Specifies which file to write the export to.
        Defaults to <config_file_name>.csv in output folder.""",
    )
    discard_gradient: Optional[bool] = Field(
        default=None,
        description="If set to True, Everest export will not contain "
        "gradient simulation data.",
    )
    discard_rejected: Optional[bool] = Field(
        default=None,
        description="""If set to True, Everest export will contain only simulations
         that have the increase_merit flag set to true.""",
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="List of eclipse keywords to be exported into csv.",
    )
    batches: Optional[List[int]] = Field(
        default=None,
        description="list of batches to be exported, default is all batches.",
    )
    skip_export: Optional[bool] = Field(
        default=None,
        description="""set to True if export should not
                     be run after the optimization case.
                     Default value is False.""",
    )

    @field_validator("csv_output_filepath", mode="before")
    @classmethod
    def validate_output_file_writable(cls, csv_output_filepath):  # pylint:disable=E0213
        check_writable_filepath(csv_output_filepath)
        return csv_output_filepath
