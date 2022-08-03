.. _built_in_workflow_jobs:

Built in workflow jobs
======================

.. todo::
   Make sure list of available workflows is complete

ERT comes with a list of default workflow jobs which invoke internal
ERT functionality. The internal workflow jobs include:

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
