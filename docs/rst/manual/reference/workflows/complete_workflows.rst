.. _complete_workflows_chapter:

Complete Workflows
==================

A workflow is a list of calls to jobs, with additional arguments. The
job name should be the first element on each line. Based on the two
jobs PLOT and ECL_HIST we can create a small workflow example:

::

	PLOT      WWCT:OP_1   WWCT:OP_3  PRESSURE:10,10,10
	PLOT      FGPT        FOPT
	ECL_HIST  <RUNPATH_FILE>   <QC_PATH>/<ERTCASE>/wwct_hist   WWCT:OP_1  WWCT:OP_2

In this workflow we create plots of the nodes
WWCT:OP_1, WWCT:OP_3, PRESSURE:10,10,10, FGPT and FOPT. The plot job we
have created in this example is general, if we limited
ourselves to ECLIPSE summary variables we could get wildcard
support. Then we invoke the ECL_HIST example job to create a
histogram. See below for documentation of <RUNPATH_FILE> and
<ERTCASE>.


Loading workflows
-----------------

Workflows are loaded with the configuration option LOAD_WORKFLOW:

::

	LOAD_WORKFLOW  /path/to/workflow/WFLOW1
	LOAD_WORKFLOW  /path/to/workflow/workflow2  WFLOW2

The LOAD_WORKFLOW takes the path to a workflow file as the first
argument. By default the workflow will be labeled with the filename
internally in ERT, but optionally you can supply a second extra
argument which will be used as name for the workflow. Alternatively
you can load a workflow interactively.


.. _hook_workflow:

Automatically run workflows : HOOK_WORKFLOW
-------------------------------------------

With the keyword :code:`HOOK_WORKFLOW` you can configure workflow
'hooks'; meaning workflows which will be run automatically at certain
points during ERTs execution. Currently there are four points in ERTs
flow of execution where you can hook in a workflow, before the
simulations start, :code:`PRE_SIMULATION`;  after all the simulations
have completed :code:`POST_SIMULATION`; before the update step,
:code:`PRE_UPDATE` and after the update step, :code:`POST_UPDATE`. The
:code:`POST_SIMULATION` hook is typically used to trigger QC
workflows:

::

   HOOK_WORKFLOW initWFLOW        PRE_SIMULATION
   HOOK_WORKFLOW preUpdateWFLOW   PRE_UPDATE
   HOOK_WORKFLOW postUpdateWFLOW  POST_UPDATE
   HOOK_WORKFLOW QC_WFLOW1        POST_SIMULATION
   HOOK_WORKFLOW QC_WFLOW2        POST_SIMULATION


In this example the workflow :code:`initWFLOW` will run after all the
simulation directories have been created, just before the forward
model is submitted to the queue. The workflow :code:`preUpdateWFLOW`
will be run before the update step and :code:`postUpdateWFLOW` will be
run after the update step. When all the simulations have completed the
two workflows :code:`QC_WFLOW1` and :code:`QC_WFLOW2` will be run.

Observe that the workflows being 'hooked in' with the
:code:`HOOK_WORKFLOW` must be loaded with the :code:`LOAD_WORKFLOW`
keyword.


Locating the realisations: <RUNPATH_FILE>
-----------------------------------------

Context must be passed between the main ERT process and the script
through the use of string substitution, in particular the 'magic' key
<RUNPATH_FILE> has been introduced for this purpose.

Many of the external workflow jobs involve looping over all the
realisations in a construction like this:

::

	for each realisation:
	    // Do something for realisation
	summarize()

When running an external job in a workflow there is no direct transfer
of information between the main ERT process and the external
script. We therefore must have a convention for transferring the
information of which realisations we have simulated on, and where they
are located in the filesystem. This is done through a file which looks
like this:

::

	0   /path/to/real0  CASE_0000
	1   /path/to/real1  CASE_0001
	...
	9   /path/to/real9  CASE_0009

The name and location of this file is available as the magical string
<RUNPATH_FILE> and that is typically used as the first argument to
external workflow jobs which should iterate over all realisations. The
realisations referred to in the <RUNPATH_FILE> are meant to be last
simulations you have run; the file is updated every time you run
simulations. This implies that it is (currently) not so convenient to
alter which directories should be used when running a workflow.

