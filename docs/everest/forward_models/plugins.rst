.. _cha_forward_model_plugins:

******************************
Forward model plugin authoring
******************************

EVEREST forward model plugins are Python packages that register hook
implementations in the ``everest`` entry point group. A plugin can add schema
validation, parsing, linting, workflow jobs, and documentation for forward model
jobs.

Registering a plugin
====================

Expose a module or object through the ``everest`` entry point group:

.. code-block:: toml

    [project.entry-points.everest]
    my_forward_models = "my_package.everest_plugin"

In that module, decorate hook implementations with
``everest.plugins.hookimpl``:

.. code-block:: python

    from collections.abc import Sequence

    from pydantic import BaseModel

    from everest.plugins import hookimpl


    class MyJobConfig(BaseModel):
        value: float


    @hookimpl
    def get_forward_models_schemas() -> dict[str, dict[str, type[BaseModel]]]:
        return {"my_job": {"-c/--config": MyJobConfig}}


    @hookimpl
    def lint_forward_model(job: str, args: Sequence[str]) -> list[str] | None:
        if job != "my_job":
            return None
        return [] if args else ["my_job requires arguments"]

Available hooks
===============

``get_forward_models_schemas()``
    Return a dictionary mapping forward model job names to pydantic schemas for
    file-based arguments.

``parse_forward_model_schema(path, schema)``
    Parse a configuration file into the requested pydantic schema type.

``lint_forward_model(job, args)``
    Return linting errors for a forward model job, or ``None`` when the plugin
    does not handle that job.

``check_forward_model_arguments(forward_model_steps)``
    Validate configured forward model step strings and raise an error if they
    are invalid.

``installable_workflow_jobs()``
    Return workflow jobs that EVEREST can install from the plugin.

``get_forward_model_documentations()``
    Return documentation metadata for plugin-provided forward model jobs.
