from ert.shared._doc_utils.everest_jobs import EverestForwardModelDocumentation


def setup(app):
    app.add_directive("everest_forward_model", EverestForwardModelDocumentation)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
