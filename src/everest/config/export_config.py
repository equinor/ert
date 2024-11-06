from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from everest.config.validation_utils import check_writable_filepath
from everest.export import available_batches, get_internalized_keys


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

    def check_for_errors(
        self,
        optimization_output_path: str,
        storage_path: str,
        data_file_path: Optional[str],
    ):
        """
        Checks for possible errors when attempting to export current optimization
        case.
        """
        export_ecl = True
        export_errors: List[str] = []

        if self.batches:
            _available_batches = available_batches(optimization_output_path)
            for batch in set(self.batches).difference(_available_batches):
                export_errors.append(
                    "Batch {} not found in optimization "
                    "results. Skipping for current export."
                    "".format(batch)
                )
            self.batches = list(set(self.batches).intersection(_available_batches))

        if self.batches == []:
            export_errors.append(
                "No batches selected for export. "
                "Only optimization data will be exported."
            )
            return export_errors, False

        if not data_file_path:
            export_ecl = False
            export_errors.append(
                "No data file found in config."
                "Only optimization data will be exported."
            )

        # If no user defined keywords are present it is no longer possible to check
        # availability in internal storage
        if self.keywords is None:
            return export_errors, export_ecl

        if not self.keywords:
            export_ecl = False
            export_errors.append(
                "No eclipse keywords selected for export. Only"
                " optimization data will be exported."
            )

        internal_keys = get_internalized_keys(
            config=self,
            storage_path=storage_path,
            optimization_output_path=optimization_output_path,
            batch_ids=set(self.batches) if self.batches else None,
        )

        extra_keys = set(self.keywords).difference(set(internal_keys))
        if extra_keys:
            export_ecl = False
            export_errors.append(
                f"Non-internalized ecl keys selected for export '{' '.join(extra_keys)}'."
                " in order to internalize missing keywords "
                f"run 'everest load <config_file>'. "
                "Only optimization data will be exported."
            )

        return export_errors, export_ecl
