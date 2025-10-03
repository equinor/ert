import copy
import logging
from textwrap import dedent

from pydantic import BaseModel, Field

from ert.config import ForwardModelStep
from everest.strings import EVEREST


class InstallTemplateConfig(BaseModel, extra="forbid"):
    template: str = Field(
        description=dedent(
            """
            The jinja2 template file.
            """
        )
    )
    output_file: str = Field(
        description=dedent(
            """
            The name of the output file.
            """
        )
    )
    extra_data: str | None = Field(
        description=dedent(
            """
            Extra input files.

            The content of each extra JSON or YAML file is exposed to the jinja2
            renderer, using the name of the file as a variable name.
            """
        ),
        default=None,
    )

    def to_ert_forward_model_step(
        self,
        control_names: list[str],
        installed_fm_steps: dict[str, ForwardModelStep],
        well_path: str,
    ) -> ForwardModelStep:
        fm_step_instance = copy.deepcopy(installed_fm_steps.get("template_render"))
        assert fm_step_instance is not None
        res_input = (
            control_names  # [control.name for control in everest_config.controls]
        )
        res_input = [fn + ".json" for fn in res_input]
        res_input.append(well_path)
        if self.extra_data is not None:
            res_input.append(self.extra_data)

        fm_step_instance.arglist = [
            "--output",
            self.output_file,
            "--template",
            self.template,
            "--input_files",
            *res_input,
        ]
        logging.getLogger(EVEREST).info(
            f"template_render {' '.join(fm_step_instance.arglist)}"
        )
        # User can define a template w/ extra data to be used with it,
        # append file as arg to input_files if declared.
        return fm_step_instance
