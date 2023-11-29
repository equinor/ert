Configuring workflow jobs
=========================

Workflow jobs are configured with a small configuration file much like
the configuration file used to install forward model jobs. The
keywords used in the configuration file are in two *classes* - those
related to how the job should be located/run and the arguments which
should be passed from the workflow to the job.

Configure an internal job
-------------------------

If you wish to implement your job as a Python class, derived from
:code:`ErtScript` you should use the :code:`SCRIPT` keyword to point to an
existing Python script:

::

   INTERNAL  TRUE                     -- The job will call an internal function or script of the currently running ERT instance.
   SCRIPT scripts/my_script.py        -- An existing Python script

Observe that the important thing here is the fact that we are writing
an *internal* Python script; if you are writing an external script to
loop through all your realization folders that will typically be an
*external* script, and in that case the implementation language -
i.e. Python, Perl, C++, F77 ... has no relevance.

NB: note that relative paths are resolved from the location of the job
configuration file, not the configuration file provided to ert.

Configure an external job
-------------------------

An *external* job is a workflow job which is implemented in an
external executable, i.e. typically a script written in for instance
Python. When configuring an external job the most important keyword is
:code:`EXECUTABLE` which is used to give the path to the external
executable:

::

    INTERNAL   FALSE                    -- Optional - Set to FALSE by default.
    EXECUTABLE path/to/program          -- Path to a program/script which will be invoked by the job.


NB: note that relative paths are resolved from the location of the job
configuration file, not the configuration file provided to ert.

Stop Ert execution upon job failure
-----------------------------------
By default, failing jobs (both internal and external) will not stop the entire ert simulation.
In some cases it is best to cancel the entire simulation if a job fails.
This behavior can be achieved by adding the below line to the job file:

::

    STOP_ON_FAIL TRUE

For example, if a job is defined as follows:

::

    INTERNAL   FALSE
    EXECUTABLE script.sh
    STOP_ON_FAIL TRUE                   -- Tell the job to stop ert on failure

STOP_ON_FAIL can also be specified within the internal (python) or external (executable) job script.
For example, this internal job script will stop on failure

::

    from ert import ErtScript
    class AScript(ErtScript):
        stop_on_fail = True

        def run(self):
            assert False, "failure"
    """

As will external .sh executables if they contain the line STOP_ON_FAIL=TRUE:

::

    #!/bin/bash
    STOP_ON_FAIL=True #
    ekho helo wordl


Configuring the arguments
-------------------------

In addition to the :code:`INTERNAL` and :code:`EXECUTABLE` keys
which are used to configure what the job should do, there are some keys
which can be used to configure the number of arguments and their
type. These arguments apply to both internal and external jobs:

::

	MIN_ARG    2                 -- The job should have at least 2 arguments.
	MAX_ARG    3                 -- The job should have maximum 3 arguments.
	ARG_TYPE   0    INT          -- The first argument should be an integer.
	ARG_TYPE   1    FLOAT        -- The second argument should be a float value.
	ARG_TYPE   2    STRING       -- The third argument should be a string - the default.
	ARGLIST    <ARG0> <ARG1>     -- A list of arguments to pass on to the executable.

The :code:`MIN_ARG`, :code:`MAX_ARG` and :code:`ARG_TYPE` arguments are used to validate workflows.

Note
____

When configuring :code:`ARGLIST` for workflow jobs,
named options such as :code:`--some-option` cannot be used
since they are treated as comments by the configuration compiler.
Single letter options, i.e. :code:`-s`, are needed.

**Example : Run external script**

::

	-- FILE: ECL_HIST --
	EXECUTABLE  Script/ecl_hist.py
	MIN_ARG     3

This job will invoke the external script :code:`Script/ecl_host.py`
which is expected to have at least three command line arguments. The path to
the script, :code:`Script/ecl_hist.py` is interpreted relative to the location
of the configuration file.

Loading workflow jobs into ERT
------------------------------

Before the jobs can be used in workflows they must be "loaded" into
ERT. This can be done either by specifying jobs by name,
or by specifying a directory containing jobs.

Use the keyword :code:`LOAD_WORKFLOW_JOB` to specify jobs by name:

::

	LOAD_WORKFLOW_JOB     jobConfigFile     JobName

The :code:`LOAD_WORKFLOW_JOB` keyword will load one workflow job.
The name of the job is optional, and will be fetched from the configuration file if not provided.
Alternatively, you can use the command
:code:`WORKFLOW_JOB_DIRECTORY` which will load all the jobs in a
directory.

Use the keyword :code:`WORKFLOW_JOB_DIRECTORY` to specify a directory containing jobs:

::

	WORKFLOW_JOB_DIRECTORY /path/to/jobs

The :code:`WORKFLOW_JOB_DIRECTORY` loads all workflow jobs found in the `/path/to/jobs` directory.
Observe that all the files in the `/path/to/jobs` directory
should be job configuration files. The jobs loaded in this way will
all get the name of the file as the name of the job. The
:code:`WORKFLOW_JOB_DIRECTORY` keyword will *not* load configuration
files recursively.
