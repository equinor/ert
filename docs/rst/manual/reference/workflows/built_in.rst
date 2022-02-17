.. _built_in_workflow_jobs:

Built in Workflow Jobs
======================

.. todo::
   Make sure list of available workflows is complete

ERT comes with a list of default workflow jobs which invoke internal
ERT functionality. The internal workflow jobs include:

Jobs related to case management
-------------------------------

**SELECT_CASE**

The job :code:`SELECT_CASE` can be used to change the currently selected
case. The :code:`SELECT_CASE` job should be used as:

::

	SELECT_CASE  newCase

if the case `newCase` does not exist it will be created.

**CREATE_CASE**

The job :code:`CREATE_CASE` can be used to create a new case without selecting
it. The :code:`CREATE_CASE` job should be used as:

::

	CREATE_CASE  newCase

**INIT_CASE_FROM_EXISTING**

The job :code:`INIT_CASE_FROM_EXISTING` can be used to initialize a case from
an existing case. The argument to the workflow should be the name of
the workflow you are initializing from; so to initialize the current
case from the existing case `oldCase`:

::

	INIT_CASE_FROM_EXISTING oldCase

By default the job will initialize the 'current case', but optionally
you can give the name of a second case which should be initialized. In
this example we will initialize `newCase` from `oldCase`:

::

	INIT_CASE_FROM_EXISTING oldCase newCase

When giving the name of a second case as target for the initialization
job, the 'current' case will not be affected.

Jobs related to export
----------------------

**EXPORT_FIELD**

The :code:`EXPORT_FIELD` workflow job exports field data to `roff` or `grdecl`
format depending on the extension of the export file argument. The job
takes the following arguments:

#. Field to be exported
#. Filename for export file, must contain %d
#. Report_step
#. Realization range

The filename must contain a `%d`. This will be replaced with the
realization number.

The realization range parameter is optional with the default being all realizations.

Example usage of this job in a workflow:

::

	EXPORT_FIELD PERMZ path_to_export/filename%d.grdecl 0 0,2

**EXPORT_FIELD_RMS_ROFF**

The :code:`EXPORT_FIELD_RMS_ROFF` workflow job exports field data to `roff`
format. The job takes the following arguments:

#. Field to be exported
#. Filename for export file, must contain %d
#. Report_step
#. Realization range

The filename must contain a `%d`, which will be replaced with the realization number.

The realization range parameter is optional with the default being all realizations.

Example usage of this job in a workflow:

::

	EXPORT_FIELD_RMS_ROFF PERMZ path_to_export/filename%d.roff 0 
	EXPORT_FIELD_RMS_ROFF PERMX path_to_export/filename%d 0 0-5 

**EXPORT_FIELD_ECL_GRDECL**

The :code:`EXPORT_FIELD_ECL_GRDECL` workflow job exports field data to `grdecl`
format. The job takes the following arguments:

#. Field to be exported
#. Filename for export file, must contain %d
#. Report_step
#. Realization range

The filename must contain a `%d` which will be replaced with the realization number.

The realization range parameter is optional with the default being all realizations.

Example usage of this job in a workflow:

::

	EXPORT_FIELD_ECL_GRDECL PERMZ path_to_export/filename%d.grdecl 0 
	EXPORT_FIELD_ECL_GRDECL PERMX path_to_export/filename%d 0 0-5 

**EXPORT_RUNPATH**

The :code:`EXPORT_RUNPATH` workflow job writes the runpath file :code:`RUNPATH_FILE`
for the selected case.

The job can have no arguments, or one can set a range of realizations
and a range of iterations as arguments.

Example usage of this job in a workflow:

::

	EXPORT_RUNPATH 

With no arguments, entries for all realizations are written to the
runpath file. If the runpath supports iterations, entries for all
realizations in `iter0` are written to the runpath file.

::

	EXPORT_RUNPATH 0-5 | *

A range of realizations and a range of iterations can be given. "|" is
used as a delimiter to separate realizations and iterations. "*" can
be used to select all realizations or iterations. In the example
above, entries for realizations 0-5 for all iterations are written to
the runpath file.

Jobs related to analysis update
-------------------------------

**ANALYSIS_UPDATE**

This job will perform a update and store the updated parameters as
initial parameters of a different case. The name of the source case
and the target case must be given as arguments:

::

   ANALYSIS_UPDATE prior posterior

Fetches prior parameters and simulated responses from the
case:`prior` and stores updated parameters in the case: `posterior`. If
you have configured local updates they will be respected, otherwise
all available data will be used - and all parameters will be updated.


Jobs related to running simulations - including updates
-------------------------------------------------------

**RUN_SMOOTHER**

The :code:`RUN_SMOOTHER` job will run a simulation and perform an update. The
job has one required argument - the name of a case where the updated
parameters are stored. Optionally the job can take a second boolean
argument, which if set to true will re-run the job based on the updated parameters.

Run a simulation and an update. Store the updated parameters in the
specified case. This case is created if it does not exist:

::

	RUN_SMOOTHER new_case

Run a simulation and an update. Store the updated parameters in the
specified case, then run a simulation on this case:

::

	RUN_SMOOTHER new_case true

**RUN_SMOOTHER_WITH_ITER**

This is exactly like the :code:`RUN_SMOOTHER` job,
but with an additional first argument `iter`, 
which can be used to control the `iter`-number in the :code:`RUNPATH`.
When using the RUN_SMOOTHER job the iter number will be
defaulted to zero, and one in the optional rerun.

**ENSEMBLE_RUN**

The :code:`ENSEMBLE_RUN` job will run a simulation, no update. The job takes as
optional arguments a range and/or list of which realizations to run.

::

	ENSEMBLE_RUN

::

	ENSEMBLE_RUN 1-5, 8

**LOAD_RESULTS**

The :code:`LOAD_RESULTS` loads results from a single, or from multiple simulations. The job takes as
optional arguments a range and/or list of which realizations to load
results from. If no realizations are specified, results for all
realizations are loaded.

::

	LOAD_RESULTS 

::

	LOAD_RESULTS 1-5, 8

In the case of multi-iteration jobs, e.g. the integrated smoother
update, the :code:`LOAD_RESULTS` job will load the results from `iter==0`. To
control which iteration is loaded from, you can use the
:code:`LOAD_RESULTS_ITER` job.

**LOAD_RESULTS_ITER**

The :code:`LOAD_RESULTS_ITER` job is similar to the :code:`LOAD_RESULTS` job,
but it takes an additional first argument which specifies which iteration number to load from. 
This should be used when manually loading results from multi-iteration workflows:

::

	LOAD_RESULTS_ITER 

::

	LOAD_RESULTS_ITER 3 1-3, 8-10

Will load the realisations 1,2,3 and 8,9,10 from the fourth iteration
(counting starts at zero).

**MDA_ES**

This workflow job (plugin) is used to run the *Multiple Data
Assimilation Ensemble Smoother* :code:`MDA ES`.  Only two arguments
are required to start the :code:`MDA_ES` process; target case format and
iteration weights. The weights implicitly indicate the number of
iterations and the normalized global standard deviation scaling
applied to the update step.

::

	MDA_ES target_case_%d observations/obs.txt

This command will use the weights specified in the `obs.txt` file. This
file should have a single floating point number per line.
Alternatively, the weights can be given as arguments as shown here.

::

	MDA_ES target_case_%d 8,4,2,1

This command will use the normalized version of the weights 8,4,2,1
and run for four iterations. The prior will be in *target_case_0* and
the results from the last iteration will be in *target_case_4*.
**Note: the weights must be listed with no spaces and separated with
commas.**

If this is run as a plugin from Ertshell or the GUI a convenient user
interface can be shown.

Jobs for ranking realizations
-----------------------------

**EXPORT_RANKING**

The :code:`EXPORT_RANKING` job exports ranking results to file. The job takes
two parameters; the name of the ranking to export and the file to
export to.

::

	EXPORT_RANKING Dataranking1 /tmp/dataranking1.txt

**INIT_MISFIT_TABLE**

Calculating the misfit for all observations and all timesteps can
potentially be a bit time consuming, the results are therefore cached
internally. If you need to force the recalculation of this cache you
can use the :code:`INIT_MISFIT_TABLE` job to initialize the misfit table that
is used in observation ranking.

::

	INIT_MISFIT_TABLE
