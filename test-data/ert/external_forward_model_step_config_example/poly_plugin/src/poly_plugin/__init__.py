from typing import Optional

from ert import (
    ForwardModelStepDocumentation,
    ForwardModelStepJSON,
    ForwardModelStepPlugin,
)
from ert import (
    ForwardModelStepValidationError as ForwardModelStepValidationError,
)


class CreatePolyConfig(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="create_poly_config",
            command=[
                "poly_create_config",
                "-c",
                "<CONFIG_FILE>",
                "<A>",
                "<B>",
                "<C>",
            ],
            default_mapping={"<A>": 1, "<B>": 2, "<C>": 3},
        )

    def validate_pre_experiment(self, _: ForwardModelStepJSON) -> None:
        pass

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="testing.stuff",
            examples="""
This forward model plugin invokes create_poly_config
""",
        )


class RunPolyConfig(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="poly_run_config",
            command=[
                "run_poly_config",
                "-c",
                "<CONFIG_FILE>",
                "<A>",
                "<B>",
                "<C>",
            ],
            default_mapping={"<A>": 1, "<B>": 2, "<C>": 3},
        )

    def validate_pre_experiment(self, fmjson: ForwardModelStepJSON) -> None:
        pass

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="testing.stuff",
            examples="""
This forward model plugin runs the poly config
""",
        )
