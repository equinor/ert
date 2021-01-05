Configuring Workflow Jobs
=========================

Workflow jobs are configured with a small configuration file much like
the configuration file used to install forward model jobs. The
keywords used in the configuration file are in two *classes* - those
related to how the job should be located/run and the arguments which
should be passed from the workflow to the job.


Configure an internal job
-------------------------

When configuring an internal workflow job the keyword :code:`INTERNAL`
is given the value :code:`TRUE` to indicate that this is an internal
job. In addition you give the name of the C function you wish to
invoke. By default the workflow job will search for the function
symbol in the current process space, but by passing the :code:`MODULE`
keyword you can request the loading of an external shared library:

::

    INTERNAL  TRUE                     -- The job will call an internal function of the current running ERT instance.
    FUNCTION  enkf_main_plot_all       -- Name of the ERT function we are calling; must be marked exportable.
    MODULE    /name/of/shared/library  -- Very optional - to load an extra shared library.


Configure an internal job: Python
-----------------------------------

If you wish to implement your job as a Python class, derived from
:code:`ErtScript` you should use the :code:`SCRIPT` keyword instead of
:code:`FUNCTION`, to point to an existing Python script:

::

   INTERNAL  TRUE                     -- The job will call an internal function of the current running ERT instance.
   SCRIPT scripts/my_script.py         -- An existing Python script

Observe that the important thing here is the fact that we are writing
an *internal* Python script; if you are writing an external script to
loop through all your realization folders that will typically be an
*external* script, and in that case the implementation language -
i.e. Python, Perl, C++, F77 ... has no relevance.

NB: note that relative paths are resolved from the location of the job
configuration file, not the configuration file provided to ert

Configure an external job
-------------------------

An *external* job is a workflow job which is implemented in an
external executable, i.e. typically a script written in for instance
Python. When configuring an external job the most important keyword is
:code:`EXECUTABLE` which is used to give the path to the external
executable:

::

    INTERNAL   FALSE                    -- This is the default - not necessary to include.
    EXECUTABLE path/to/program          -- Path to a program/script which will be invoked by the job.

NB: note that relative paths are resolved from the location of the job
configuration file, not the configuration file provided to ert

Configuring the arguments
-------------------------

In addition to the INTERNAL, FUNCTION, MODULE and EXECUTABLE keys
which are used to configure what the job should do, there are some keys
which can be used to configure the number of arguments and their
type. These arguments apply to both internal and external jobs:

::

	MIN_ARG    2                 -- The job should have at least 2 arguments.
	MAX_ARG    3                 -- The job should have maximum 3 arguments.
	ARG_TYPE   0    INT          -- The first argument should be an integer
	ARG_TYPE   1    FLOAT        -- The second argument should be a float value
	ARG_TYPE   2    STRING       -- The third argument should be a string - the default.
	ARGLIST    <ARG0> <ARG1>     -- A list of arguments to pass on to the executable

The MIN_ARG, MAX_ARG and ARG_TYPE arguments are used to validate workflows.

Note
____
When configuring ARGLIST for WORKFLOW_JOB jobs it is not suitable to use
:code:`--some-option` for named options as it treated as a comment by the
configuration compiler. Single letter options, i.e. :code:`-s`, are needed.

**Example 1 : Plot variables**

::

	-- FILE: PLOT --
	INTERNAL  TRUE
	FUNCTION  ert_tui_plot_JOB
	MIN_ARG   1

This job will use the ERT internal function ert_tui_plot_JOB to plot
an ensemble of an arbitrary ERT variable. The job needs at least one
argument; there is no upper limit on the number of arguments.


**Example 2 : Run external script**

::

	-- FILE: ECL_HIST --
	EXECUTABLE  Script/ecl_hist.py
	MIN_ARG     3

This job will invoke the external script Script/ecl_host.py; the
script should have at least three commandline arguments. The path to
the script, Script/ecl_hist.py is interpreted relative to the location
of the configuration file.


Loading workflow jobs into ERT
------------------------------

Before the jobs can be used in workflows they must be 'loaded' into
ERT. This is done with two different ERT keywords:

::

	LOAD_WORKFLOW_JOB     jobConfigFile     JobName

The LOAD_WORKFLOW_JOB keyword will load one workflow job. The name of
the job is optional, if not provided the job will get name from the
configuration file. Alternatively you can use the command
WORKFLOW_JOB_DIRECTORY which will load all the jobs in a
directory. The command:

::

	WORKFLOW_JOB_DIRECTORY /path/to/jobs

will load all the workflow jobs in the /path/to/jobs
directory. Observe that all the files in the /path/to/jobs directory
should be job configuration files. The jobs loaded in this way will
all get the name of the file as the name of the job. The
:code:`WORKFLOW_JOB_DIRECTORY` keyword will *not* load configuration
files recursively.


