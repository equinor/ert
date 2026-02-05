Plugin system
=============

Introduction
------------

Assuming the following package structure::

    your_package
    ├── pyproject.toml        # or setup.py
    └── src
        └── your_module
            ├── __init__.py
            ├── your_plugins.py
            └── ...

where the ert plugins are defined in :code:`your_plugins.py`, then discovery is done
by registering your plugin via a setuptools entry point, with the namespace :code:`ert`:

.. code-block:: python

    # setup.py
    setup(
        ...
        entry_points={"ert": ["your_module_jobs = your_module.your_plugins"]},
        ...
    )

.. code-block:: toml

    # pyproject.toml
    [project.entry-points.ert]
    your_module_jobs = "your_module.your_plugins"

This entry point should point to the module where your ert plugin(s) exists.
(Notice that the entry point expects a list, so you can register multiple
modules).


Kinds of plugins
----------------

A plugin is created with the `ert.plugin` decorator and the function name describes what kind of plugin it is.

Forward models
~~~~~~~~~~~~~~
To install forward model steps that you want to have available in ERT you can use
:code:`installable_forward_model_steps`.

.. code-block:: python

    from typing import Optional

    import ert

    class MyForwardModel(ert.ForwardModelStepPlugin):
        def __init__(self):
            super().__init__(
                name="MY_FORWARD_MODEL",
                command=["my_executable", "<parameter1>", "<parameter2>"],
            )

        def validate_pre_realization_run(
            self, fm_step_json: ert.ForwardModelStepJSON
        ) -> ert.ForwardModelStepJSON:
            if fm_step_json["argList"][0] not in ["start", "stop"]:
                raise ert.ForwardModelStepValidationError(
                    "First argument to MY_FORWARD_MODEL must be either start or stop"
                )
            return fm_step_json

        def validate_pre_experiment(self, fm_step_json: ert.ForwardModelStepJSON) -> None:
            pass

        @staticmethod
        def documentation() -> Optional[ert.ForwardModelStepDocumentation]:
            return ert.ForwardModelStepDocumentation(
                category="utility.templating",
                source_package="my_plugin",
                source_function_name="MyForwardModel",
                description="my plugin description",
            )


    @ert.plugin(name="my_plugin")
    def installable_forward_model_steps() -> list[ert.ForwardModelStepPlugin]:
        return [MyForwardModel]


Notice that by using :code:`installable_forward_model_steps`, validation can be added
where the methods ``validate_pre_experiment`` or ``validate_pre_realization_run`` can
throw ``ForwardModelStepValidationError`` to indicate that the configuration of the
forward model step is invalid (which ert then handles gracefully and presents nicely
to the user). If you want to show a warning in cases where the configuration cannot be
validated pre-experiment, you can use the ``ForwardModelStepWarning.warn(...)`` method.

.. code-block:: python

   import ert

   @ert.plugin(name="my_plugin")
   def job_documentation(step_name: str):
       if step_name == "my_step":
            return {
                "description": "step description",
                "examples": "...",
                "category": "test.category.for.step",
            }

When creating documentation in ERT, forward model steps will be grouped by their
main categories (ie. the category listed before the first dot).

Forward model configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forward model steps can be configured through the plugin system by using the ``forward_model_configuration`` hook.
For each forward model step name in the config dict, a set of configuration parameters specific to that forward model can be specified.
These configurations will be injected as environment variables when starting the forward model, and will be isolated from other forward models.

.. code-block:: python

   import ert

   @ert.plugin(name="my_plugin")
   def forward_model_configuration():
        return {
            "forward_model_step_name": {
                "config_key" : <config value>
            },
        }

* ``<config value>`` can be of any python type, but will be exported to env by calling ``str(<config value>)``.
* ``config_key`` will always be exposed as upper case environment variables.


Workflow job
~~~~~~~~~~~~

There are two ways to install workflow jobs in ERT.
Depending on whether you already have a configuration file or need to include additional documentation,
you can choose between the ``installable_workflow_jobs`` hook or the ``ertscript_workflow`` hook.

1. **Using the installable_workflow_jobs hook**

The hook is specified as follows:

.. code-block:: python

   import ert

   @ert.plugin(name="my_plugin")
   def installable_workflow_jobs():
        return {
            "wf_job_name": "/path/to/workflow_job.config",
        }


The configuration file needed to use the ``installable_workflow_jobs`` hook must point to an executable
and specify its arguments.
The built-in internal ``CSV_EXPORT`` workflow job is shown as an example:

.. code-block:: text

   INTERNAL    True
   SCRIPT      /path/to/csv_export.py
   MIN_ARG     1
   MAX_ARG     2
   ARG_TYPE    0  STRING
   ARG_TYPE    1  STRING


Implement the hook specification as follows to register the workflow job ``CSV_EXPORT``:

.. code-block:: python

   import ert

   @ert.plugin(name="ert")
   def installable_workflow_jobs() -> Dict[str, str]:
        return {
            "CSV_EXPORT": "/path/to/csv_export"
        }

.. _legacy_ert_workflow_jobs:

2. **Using the ertscript_workflow hook**

The second approach does not require creating a workflow job configuration file up-front,
and allows adding documentation.

.. literalinclude:: ../../../../src/ert/plugins/hook_specifications/jobs.py
   :pyobject: ertscript_workflow

Minimal example:

.. code-block:: python

   import ert

   class MyJob(ert.ErtScript):
       def run(self):
           print("Hello World")

   @ert.plugin(name="my_plugin")
    def ertscript_workflow(config):
        config.add_workflow(MyJob, "MY_JOB")


Full example:

.. code-block:: python

   import ert

   class MyJob(ert.ErtScript):
       def run(self):
           print("Hello World")

   @ert.plugin(name="my_plugin")
    def ertscript_workflow(config: ert.WorkflowConfigs):
        config.add_workflow(
            MyJob,
            "MY_JOB",
            parser=my_job_parser,
            description="My job description", # optional
            examples="example of use", # optional
        )

The configuration object and properties are as follows.

.. autofunction:: ert.plugins.hook_specifications.jobs.ertscript_workflow
.. automodule:: ert
   :members: ErtScript
.. autoclass:: ert.WorkflowConfigs
    :members: add_workflow
    :undoc-members:

Logging configuration
~~~~~~~~~~~~~~~~~~~~~
The logging can be configured by plugins to add custom log handlers.

.. autofunction:: ert.plugins.hook_specifications.logging.add_log_handle_to_root

Minimal example to log to a new file:

.. code-block:: python

   import ert

    @ert.plugin(name="my_plugin")
    def add_log_handle_to_root():
        import logging
        fh = logging.FileHandler('spam.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        return fh
