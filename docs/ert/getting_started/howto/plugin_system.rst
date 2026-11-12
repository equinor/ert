Plugin system
=============

Introduction
------------

Assuming the following package structure::

    your_package
    ├── pyproject.toml
    └── src
        └── your_module
            ├── __init__.py
            ├── your_plugins.py
            └── ...

where the ert plugins are defined in :code:`your_plugins.py`, then discovery is done
by registering your plugin via a setuptools entry point, with the namespace :code:`ert`:

.. code-block:: toml

    # pyproject.toml
    [project.entry-points.ert]
    your_module_jobs = "your_module.your_plugins"

This entry point should point to the module where your ert plugin(s) exists.
If you have more than one module, add additional lines.


Writing plugins
---------------

A plugin is created with the `ert.plugin` decorator and the function name
describes what kind of plugin it is.

Forward model steps
~~~~~~~~~~~~~~~~~~~

To install forward model steps that you want to have available in ERT you can use
:code:`installable_forward_model_steps`.

.. code-block:: python

    import ert

    class MyForwardModel(ert.ForwardModelStepPlugin):
        def __init__(self):
            super().__init__(
                name="MY_FORWARD_MODEL",
                command=["my_executable", "<parameter1>", "<parameter2>"],
                required_keywords=["<parameter1>"],
                allowed_keywords=["<parameter2>"],
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
        def documentation() -> ert.ForwardModelStepDocumentation | None:
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

Use ``required_keywords`` to declare named arguments that the user must provide. Use
``allowed_keywords`` to reject named arguments not supported by the forward model
step. The allowed-keyword check is opt-in: omitting ``allowed_keywords`` preserves the
default of accepting any named argument. Required keywords and keywords with a
``default_mapping`` are implicitly allowed. Setting ``allowed_keywords`` to an empty
list rejects all other named arguments. These declarations validate argument names
only; use the validation methods for constraints on argument values.

When creating documentation in ERT, forward model steps will be grouped by their
main categories (ie. the category listed before the first dot).

Setting environment variables for forward model steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shell environment in which steps are run can be configured through the
plugin system by using the ``forward_model_configuration`` hook. For each
forward model step name in the config dict, a set of configuration parameters
specific to that forward model can be specified. These configurations will be
injected as environment variables when starting the forward model, and will be
isolated from other forward models.

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


IP address configuration
~~~~~~~~~~~~~~~~~~~~~~~~~
The way Ert chooses IP addresses for communicating with cluster jobs can be configured by plugins.

.. autofunction:: ert.plugins.hook_specifications.net_utils.get_ip_address

Minimal example to override how Ert chooses the IP address:

.. code-block:: python

   import ert

    @ert.plugin(name="my_plugin")
    def get_ip_address():
        return "127.0.0.1"

Status event listener
~~~~~~~~~~~~~~~~~~~~~
A plugin can listen to the status events emitted while an experiment runs,
such as progress updates and snapshot updates.

The listener is called for experiments started from the Ert GUI and for
Everest optimizations. Experiments started from the Ert command line do not
call the listener yet.

Several experiments can run at the same time, so each event is passed
together with an ``experiment_run_id`` (a UUID string) that identifies which
run emitted it. Use this ID to tell runs apart. It is not the ID of the
experiment in storage.

.. autofunction:: ert.plugins.hook_specifications.event_listener.on_status_event

Only one plugin may implement this hook. If a second plugin implements it,
that plugin is rejected with a ``ValueError`` when the plugins are loaded.

Exceptions raised by the listener are logged. They do not stop the
experiment, and the listener is still called for later events.

The listener receives a copy of the event, so modifying it does not change
what is shown to the user in the GUI and in progress monitoring.

End events and reruns
^^^^^^^^^^^^^^^^^^^^^
Every run ends with an ``EndEvent``, also when the run fails or is cancelled.
Its ``failed`` attribute tells whether the run failed, and ``msg`` gives the
reason.

An ``EndEvent`` does not mean that no more events will arrive for that
``experiment_run_id``. When the user reruns failed realizations from the Ert
GUI, the rerun reuses the same ``experiment_run_id``. The listener then
receives a new sequence of events, ending with another ``EndEvent``. This
can happen after an ``EndEvent`` where ``failed`` is ``False``, since a run
can finish with some failed realizations.

The events of a rerun are passed to the listener only after it has returned
from the ``EndEvent`` of the previous run with the same
``experiment_run_id``.

As a consequence, a listener should:

- expect more than one ``EndEvent`` per ``experiment_run_id``,
- not release or delete state for an ``experiment_run_id`` on its first
  ``EndEvent`` if it needs that state for a rerun,
- treat the latest ``EndEvent`` as the outcome of the experiment run.

Example
^^^^^^^
Minimal example of logging the type of each event:

.. code-block:: python

   import logging

   import ert

   logger = logging.getLogger(__name__)

   @ert.plugin(name="event_logger")
   def on_status_event(experiment_run_id, event):
       logger.info(f"Experiment run {experiment_run_id} emitted {event.event_type}")

Threading and ordering
^^^^^^^^^^^^^^^^^^^^^^
The listener is called in a worker thread, so it does not delay the
experiment or the progress shown to the user.

The events of one experiment run are passed one at a time, in the order
they were emitted. This also holds across reruns of failed realizations, so
the listener is never called by two threads at the same time for the same
``experiment_run_id``. Events from different experiment runs can be passed
at the same time from different threads, so state shared between runs must
be protected, for example with a lock.

Ert does not wait for the listener. A slow listener receives its events
later than they were emitted, so a run can be shown as finished to the
user, and a rerun can be started, before the listener has received the
``EndEvent`` of that run.

No events are skipped while the Ert server is running, but events that have
not yet been passed to the listener when the server stops are lost:

- Closing the Ert GUI stops the server without waiting for the listener.
- After an Everest optimization has finished, the server waits up to 30
  seconds for the listener to receive the remaining events before it stops.

The listener must not block indefinitely. A listener that never returns
keeps one of the four shared worker threads busy and receives no more
events for that experiment run. It can also delay events for other
experiment runs and prevent the Ert server from shutting down.
