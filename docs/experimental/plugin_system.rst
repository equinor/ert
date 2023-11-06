Plugin system
=============


.. DANGER::
   The plugin system is experimental, meaning that the interface and/or API
   are subject to breaking changes. In addition, such breaking changes might
   not follow rules for semantic versioning, as the feature is considered experimental
   and not part of the public interface.


Introduction
------------

The plugin system in ERT uses `Pluggy <https://pluggy.readthedocs.io/en/latest/>`_ as its
foundation, and it is recommended that you familiarize yourself with this module before you create an ERT plugin.
Discovery is done by registering your plugin via a setuptools entry point, with the namespace :code:`ert`:

.. code-block:: python

   setup(
     ...
     entry_points={"ert": ["your_module_jobs = your_module.hook_implementations.jobs"]},
     ...
   )

This entry point should point to the module where your hook implementations exists. (Notice that the entry point expects a list, so you can register multiple modules).

Hook implementation
-------------------

All hook implementations expect a ``PluginResponse`` to be returned.
The ``PluginResponse`` is a wrapper around the actual data that you want to return,
with additional metadata about which plugin the hook response is coming from.
To avoid having to deal with the details of this a decorator is provided, which can be used like this.

.. code-block:: python

   from ert.shared.plugins.plugin_response import plugin_response
   @hook_implementation
   @plugin_response(plugin_name="ert")
   def installable_jobs():
      return {"SOME_JOB": "/path/to/some/job/config"}

This way you only need to return the data of interest, which is specified in the hook specification.

Hook specifications
-------------------

The following hook specifications are available to use:

Forward models
~~~~~~~~~~~~~~
To install forward models that you want to have available in ERT you can use the following hook specification:

.. code-block:: python

   @hook_specification
   def installable_jobs():
      """
      :return: dict with job names as keys and path to config as value
      :rtype: PluginResponse with data as dict[str,str]
      """



Forward model documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To provide documentation for a forward model, the following hook_specification can be used.
If you are the plugin that provided the job with the name :code:`job_name`,
then you respond with the documentation as specified, else respond with :code:`None`.

.. code-block:: python

   @hook_specification(firstresult=True)
   def job_documentation(job_name):
      """
      :return: If job_name is from your plugin return
               dict with documentation fields as keys and corresponding
               text as value (See below for details), else None.
      :rtype: PluginResponse with data as dict[str,str]

      Valid fields:
      description: RST markdown as a string. Example: "This is a **dummy** description"
      examples: RST markdown as a string. Example: "This is an example"
      category: Dot seperated list categories (main_category.sub_category) for job.
               Example: "simulator.reservoir". When generating documentation in ERT the
               main category (category before the first dot) will be used to group
               the jobs into sections.
      """

When creating documentation in ERT, forward models will be grouped by their
main categories (ie. the category listed before the first dot).

Workflow jobs
~~~~~~~~~~~~~
There are two ways to install a workflow job in ERT. To install workflow jobs that you
want to have available in ERT you can use one of the following hook specifications:

.. code-block:: python

   @hook_specification
   def installable_workflow_jobs():
      """
      :return: dict with workflow job names as keys and path to config as value
      :rtype: PluginResponse with data as dict[str,str]
      """

This hook specification relies on creating a config file, and has no utility functionality.

The second hook registers individual jobs, with the option of adding documentation in the form
of a description, examples, parser, etc. It passes in a config object where the job is added.

.. literalinclude:: ../../src/ert/shared/plugins/hook_specifications/jobs.py
   :pyobject: legacy_ertscript_workflow

Minimal example:

.. code-block:: python

   from ert import ErtScript
   from ert.shared.plugins.plugin_manager import hook_implementation

   class MyJob(ErtScript)
       def run()
           print("Hello World")

   @hook_implementation
    def legacy_ertscript_workflow(config):
        config.add_workflow(MyJob, "MY_JOB")


Full example:

.. code-block:: python

   from ert import ErtScript
   from ert.shared.plugins.plugin_manager import hook_implementation

   class MyJob(ErtScript)
       def run()
           print("Hello World")

   @hook_implementation
    def legacy_ertscript_workflow(config):
        workflow = config.add_workflow(MyJob, "MY_JOB")
        workflow.parser = my_job_parser  # Optional
        workflow.description = "My job description"  # Optional
        workflow.examples = "example of use"  # Optional

The configuration object and properties are documented in: :class:`ert.shared.plugins.workflow_config.WorkflowConfig`

.. autofunction:: ert.shared.plugins.hook_specifications.jobs.legacy_ertscript_workflow

.. autoclass:: ert.shared.plugins.workflow_config.WorkflowConfigs
    :members: add_workflow
    :undoc-members:


.. autoclass:: ert.shared.plugins.workflow_config.WorkflowConfig
    :members:
    :undoc-members:

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~
The logging can be configured by plugins to add custom log handlers.

.. autofunction:: ert.shared.plugins.hook_specifications.logging.add_log_handle_to_root

Minimal example to log to a new file:

.. code-block:: python


   from ert.shared.plugins.plugin_manager import hook_implementation

    @hook_implementation
    def add_log_handle_to_root():
        import logging
        fh = logging.FileHandler('spam.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        return fh
