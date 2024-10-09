from ert.shared._doc_utils.ert_jobs import (
    ErtForwardModelDocumentation,
    ErtWorkflowDocumentation,
)


def setup(app):
    app.add_directive("everest_forward_model", ErtForwardModelDocumentation)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
