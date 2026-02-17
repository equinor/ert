.. _ert_kw_full_doc:

List of keywords
================

For your convenience, the description of the keywords in the ERT configuration file
are divided into the following groups:

* Commonly used keywords not related to parametrization. I.e. keywords giving
  the data, grid, and observation file, defining how to run simulations
  and how to store results. These keywords are described in :ref:`Commonly used
  keywords.<commonly_used_keywords>`
* Keywords related to parametrization of the ECLIPSE model. These keywords are
  described in :ref:`Parametrization keywords<parameterization_keywords>`.
* Keywords related to the simulation in :ref:`Keywords controlling the simulation<keywords_controlling_the_simulations>`.
* Advanced keywords not related to parametrization. These keywords are described
  in :ref:`Advanced keywords<advanced_keywords>`.


Table of keywords
-----------------

=====================================================================   ======================================  ==============================  ==============================================================================================================================================
Keyword name                                                            Required                                Default value                   Purpose
=====================================================================   ======================================  ==============================  ==============================================================================================================================================
:ref:`ANALYSIS_SET_VAR <analysis_set_var>`                              NO                                                                      Set analysis module internal state variable
:ref:`CASE_TABLE <case_table>`                                          NO                                                                      Deprecated
:ref:`DATA_FILE <data_file>`                                            NO                                                                      Provide an ECLIPSE data file for the problem
:ref:`DATA_KW <data_kw>`                                                NO                                                                      Replace strings in ECLIPSE .DATA files
:ref:`DEFINE <define>`                                                  NO                                                                      Define keywords with config scope
:ref:`DESIGN_MATRIX <design_matrix>`                                    NO                                                                      Add design matrix parameters from a spreadsheet
:ref:`ECLBASE <eclbase>`                                                NO                                                                      Define a name for the ECLIPSE simulations.
:ref:`STD_CUTOFF <std_cutoff>`                                          NO                                      1e-6                            Determines the threshold for ensemble variation in a measurement
:ref:`ENKF_ALPHA <enkf_alpha>`                                          NO                                      3.0                             Parameter controlling outlier behaviour in EnKF algorithm
:ref:`ENKF_TRUNCATION <enkf_truncation>`                                NO                                      0.98                            Cutoff used on singular value spectrum
:ref:`ENSPATH <enspath>`                                                NO                                      storage                         Folder used for storage of simulation results
:ref:`FIELD <field>`                                                    NO                                                                      Adds grid parameters
:ref:`FORWARD_MODEL <forward_model>`                                    NO                                                                      Add the running of a job to the simulation forward model
:ref:`GEN_DATA <gen_data>`                                              NO                                                                      Specify a general type of data created/updated by the forward model
:ref:`GEN_KW <gen_kw>`                                                  NO                                                                      Add a scalar parameter
:ref:`GRID <grid>`                                                      NO                                                                      Provide an ECLIPSE grid for the reservoir model
:ref:`HOOK_WORKFLOW <hook_workflow>`                                    NO                                                                      Install a workflow to be run automatically
:ref:`INCLUDE <include>`                                                NO                                                                      Include contents from another ert config
:ref:`INSTALL_JOB <install_job>`                                        NO                                                                      Install a job for use in a forward model
:ref:`INVERSION <inversion_algorithm>`                                  NO                                                                      Set inversion method for analysis module
:ref:`JOBNAME <jobname>`                                                NO                                      <CONFIG_FILE>-<IENS>            Name used for simulation files.
:ref:`LOAD_WORKFLOW <load_workflow>`                                    NO                                                                      Load a workflow into ERT
:ref:`LOAD_WORKFLOW_JOB <load_workflow_job>`                            NO                                                                      Load a workflow job into ERT
:ref:`LOCALIZATION <localization>`                                      NO                                      False                           Enable experimental adaptive localization correlation
:ref:`LOCALIZATION_CORRELATION_THRESHOLD <local_corr_threshold>`        NO                                      0.30                            Specifying adaptive localization correlation threshold
:ref:`MAX_RUNNING <max_running>`                                        NO                                      0                               Set the maximum number of simultaneously submitted and running realizations a positive integer (> 0) is required
:ref:`MAX_RUNTIME <max_runtime>`                                        NO                                      0                               Set the maximum runtime in seconds for a realization (0 means no runtime limit)
:ref:`MAX_SUBMIT <max_submit>`                                          NO                                      2                               How many times the queue system should retry a simulation
:ref:`MIN_REALIZATIONS <min_realizations>`                              NO                                      0                               Set the number of minimum realizations that has to succeed in order for the run to continue (0 means identical to NUM_REALIZATIONS - all must pass).
:ref:`NUM_CPU <num_cpu>`                                                NO                                      1                               Set the number of CPUs. Intepretation varies depending on context
:ref:`NUM_REALIZATIONS <num_realizations>`                              YES                                                                     Set the number of reservoir realizations to use
:ref:`OBS_CONFIG <obs_config>`                                          NO                                                                      File specifying observations with uncertainties
:ref:`QUEUE_OPTION <queue_option>`                                      NO                                                                      Set options for an ERT queue system
:ref:`QUEUE_SYSTEM <queue_system>`                                      NO                                      LOCAL_DRIVER                    System used for running simulation jobs
:ref:`REALIZATION_MEMORY <realization_memory>`                          NO                                                                      Set the expected memory requirements for a realization
:ref:`RFT <rft>`                                                        NO                                                                      Specify which RFT data to load from simulator output
:ref:`RUNPATH <runpath>`                                                NO                                      realization-<IENS>/iter-<ITER>  Directory to run simulations; simulations/realization-<IENS>/iter-<ITER>
:ref:`RUNPATH_FILE <runpath_file>`                                      NO                                      .ert_runpath_list               Name of file with path for all forward models that ERT has run. To be used by user defined scripts to find the realizations
:ref:`RUN_TEMPLATE <run_template>`                                      NO                                                                      Install arbitrary files in the runpath directory
:ref:`SETENV <setenv>`                                                  NO                                                                      You can modify the UNIX environment with SETENV calls
:ref:`STOP_LONG_RUNNING <stop_long_running>`                            NO                                      FALSE                           Stop long running realizations after minimum number of realizations (MIN_REALIZATIONS) have run
:ref:`SUBMIT_SLEEP  <submit_sleep>`                                     NO                                      0.0                             Determines for how long in seconds the system will sleep between submitting jobs.
:ref:`SUMMARY  <summary>`                                               NO                                                                      Add summary vectors for internalization
:ref:`SURFACE <surface>`                                                NO                                                                      Surface parameter read from RMS IRAP file
:ref:`UPDATE_LOG_PATH  <update_log_path>`                               NO                                      update_log                      Summary of the update steps are stored in this directory
:ref:`WORKFLOW_JOB_DIRECTORY  <workflow_job_directory>`                 NO                                                                      Directory containing workflow jobs
:ref:`ZONEMAP <zonemap>`                                                NO                                                                      Map grid layers to geological zone names for RFT validation
=====================================================================   ======================================  ==============================  ==============================================================================================================================================



.. _commonly_used_keywords:

Commonly used keywords
======================

.. _num_realizations:

NUM_REALIZATIONS
----------------

This is the size of the ensemble, i.e. the number of
realizations/members in the ensemble. All configs must contain this
keyword. Bear in mind that experiments that require update step must contain
at least 2 realizations.

*Example:*

::

        -- Use 200 realizations/members
        NUM_REALIZATIONS 200


.. note::
         If used alongside :ref:`DESIGN_MATRIX <design_matrix>`, Ert will
         compare the value of NUM_REALIZATIONS and the number of realizations
         found in the design matrix, and use the minimum of the two as the
         number of realizations to run. See :ref:`DESIGN_MATRIX notes <design_matrix_notes>`
         for details on how the number of realizations is calculated in the design matrix.

.. _define:

DEFINE
------

With the DEFINE keyword you can define key-value pairs which will be
substituted in the rest of the configuration file. The DEFINE keyword expects
two arguments: a key and a value to replace for that key. Later instances of
the key enclosed in '<' and '>' will be substituted with the value. The value
can consist of several strings, in that case they will be joined by one single
space.

*Example:*

::

        -- Define ECLIPSE_PATH and ECLIPSE_BASE
        DEFINE  <ECLIPSE_PATH>  /path/to/eclipse/run
        DEFINE  <ECLIPSE_BASE>  STATF02
        DEFINE  <KEY>           VALUE1       VALUE2 VALUE3            VALUE4

        -- Set the GRID in terms of the ECLIPSE_PATH
        -- and ECLIPSE_BASE keys.
        GRID    <ECLIPSE_PATH>/<ECLIPSE_BASE>.EGRID

The last key defined above (KEY) will be replaced with VALUE1 VALUE2
VALUE3 VALUE4 - i.e. the extra spaces will be discarded.


Note that redefining a key to a different value takes effect for the rest of the file:

::

        DEFINE <X> 3
        SETENV VAR <X>
        DEFINE <X> 4
        SETENV VAR2 <X>

In the above example, the environment variable VAR is set to 3 while the environment variable
VAR2 is set to 4.

.. _data_file:

DATA_FILE
---------

Specify the filepath to the ``.DATA`` file of Eclipse/flow.
This does two things:

1. Template the ``DATA_FILE`` similarly to :ref:`RUN_TEMPLATE <run_template>`.

   The templated file will be named according to :ref:`ECLBASE <ECLBASE>` and
   copied to the runpath folder. Magic strings and DEFINEs in the template
   contents will be replaced by their associated values.

2. If :ref:`NUM_CPU <num_cpu>` is not set explicitly in the config, Ert will
   search for ``PARALLEL`` in the data file and infer the number of CPUs each
   realization will need, and update :ref:`NUM_CPU <num_cpu>` accordingly .

   If the Eclipse DATA file represents a coupled simulation setup, it will sum
   the needed CPU count for each slave model from the ``SLAVES`` keyword and
   add 1 for the parent simulation.

Example:

.. code-block::

    -- Load the data file called ECLIPSE.DATA
    DATA_FILE ECLIPSE.DATA

.. note::
    See the :ref:`DATA_KW <data_kw>` keyword which can be used to utilize more template
    functionality in the Eclipse/flow datafile.

.. _design_matrix:

DESIGN_MATRIX
-------------

DESIGN_MATRIX is used to read and validate parameters given in XLSX-format.
:code:`DESIGN_MATRIX` supports 1 positional argument, which points to a XLSX file.

*Example:*

::

        DESIGN_MATRIX poly_design.xlsx


Additionally, there are three optional named arguments:

::

        DESIGN_MATRIX <file> DESIGN_SHEET:<name_of_design_sheet> DEFAULT_SHEET:<name_of_default_sheet> PRIORITY:<design_matrix|sampled>

where:

1. DESIGN_SHEET:<name_of_design_sheet> - the name of the :ref:`design_sheet` (defaults to DesignSheet)

2. DEFAULT_SHEET:<name_of_default_sheet> - the name of the :ref:`default_sheet`, which if not present will not be included

3. PRIORITY:<design_matrix|sampled> - specifies which source has priority when merging with :ref:`GEN_KW <gen_kw>` parameters.
   If set to `design_matrix`, parameters found in the design matrix will override values from :ref:`GEN_KW <gen_kw>`.
   If set to `sampled`, parameters will be sampled normally overwriting values from :ref:`DESIGN_MATRIX  <design_matrix>`.
   Default is `design_matrix`.


*Example:*

::

        DESIGN_MATRIX poly_design.xlsx DESIGN_SHEET:DesignSheet DEFAULT_SHEET:DefaultSheet PRIORITY:design_matrix


The XLSX file must contain a design sheet, where in the columns represents different parameters and rows represent realizations.


*Example of a design sheet:*

.. _design_sheet:

.. list-table:: design sheet
   :widths: 30 30 30 30
   :header-rows: 1

   * - REAL
     - a
     - b
     - c
   * - 0
     - 1
     - 1
     - 2
   * - 1
     - 1
     - 1
     - 2
   * - 3
     - 1
     - 1
     - 3


The XLSX file can optionally contain a default sheet, where there are two columns, the first column specifies parameter names and the second
default values distributed across all the realizations defined by :ref:`design_sheet`. In the following example the realization 0,1 and 3 will
contain parameters d and e, where d=0 and e=1 for all of the them.

*Example of a default sheet:*

.. _default_sheet:

.. list-table:: default sheet
   :widths: 100 100
   :width: 60%

   * - d
     - 0
   * - e
     - 1

The final design matrix would be then as follows:

.. _final_design_matrix:

.. list-table:: final design matrix
   :widths: 20 20 20 20 20 20
   :width: 80%
   :header-rows: 1

   * - REAL
     - a
     - b
     - c
     - d
     - e
   * - 0
     - 1
     - 1
     - 2
     - 0
     - 1
   * - 1
     - 1
     - 1
     - 2
     - 0
     - 1
   * - 3
     - 1
     - 1
     - 3
     - 0
     - 1

If the :ref:`design_sheet` contains column `REAL`, ert will automatically set the active realizations to what is
specified in `REAL` column; i.e.; 0,1 and 3 in the example. If the `REAL` column is not present, ert will enumerate individual rows
as realization; i.e. 0,1 and 2 in the example.

Multiple :code:`DESIGN_MATRIX` keywords can be added to the configuration file and ert will validate that
 - the realizations overlap on each instance of the keyword.
 - the parameter names are either unique or the values need to be the same for the overlapping parameters in the different instances of the keyword.

The combination with :ref:`GEN_KW <gen_kw>` parameters is supported. In case of overlapping names, eg. the ert config would contain:

::

        GEN_KW COEFFS coeff_priors
        DESIGN_MATRIX poly_design.xlsx DESIGN_SHEET:DesignSheet DEFAULT_SHEET:DefaultSheet PRIORITY:design_matrix


wherein coeff_priors

::

        b UNIFORM 0 1
        c UNIFORM 0 2
        d UNIFORM 0 5
        f UNIFORM 0 10


The overlapping parameters are b, c and d and the values will read from the design matrix
and the update will be disabled, while f would be sampled and the update flag will remain.
In this case the final set of parameters (for example in parameters.txt in real==0) would be:
::

        a=1 <- design value
        b=1 <- design value
        c=2 <- design value
        d=0 <- design value
        e=1 <- design value
        COEFFS:f=5.7 <- sampled value

So the design matrix parameters would be without the prefix.

In case we specify sampled priority:
::

        GEN_KW COEFFS coeff_priors
        DESIGN_MATRIX poly_design.xlsx DESIGN_SHEET:DesignSheet DEFAULT_SHEET:DefaultSheet PRIORITY:sampled

the final set of parameters (for example in parameters.txt in real==0) would be:
::

        a=1 <- design value
        COEFFS:b=0.2 <- sampled value
        COEFFS:c=1.3 <- sampled value
        COEFFS:d=7.8 <- sampled value
        e=1 <- design value
        COEFFS:f=5.7 <- sampled value

.. _design_matrix_notes:
.. note::
    The number of realizations in the design matrix is calculated by the max realization id found in the
    `REALS` column + 1. If the `REALS` column is missing some realizations, they will be set
    as inactive in the ensemble and not run. For example, if the DESIGN_MATRIX only contains realization
    id 3, then the ensemble_size will be four. Here the realizations 0, 1, and 2 will be marked as inactive and not run.

.. _eclbase:

ECLBASE
-------

The ECLBASE keyword sets the basename for the ECLIPSE simulations which will
be generated by ERT. It can (and should, for your convenience) contain <IENS>
specifier, which will be replaced with the realization numbers when running
ECLIPSE. Note that due to limitations in ECLIPSE, the ECLBASE string must be
in strictly upper or lower case.

*Example:*

::

        -- Use eclipse/model/MY_VERY_OWN_OIL_FIELD-<IENS> etc. as basename.
        -- When ECLIPSE is running, the <IENS> will be, replaced with
        -- realization number, and directories ''eclipse/model''
        -- will be generated by ERT if they do not already exist, giving:
        --
        -- eclipse/model/MY_VERY_OWN_OIL_FIELD-0
        -- eclipse/model/MY_VERY_OWN_OIL_FIELD-1
        -- eclipse/model/MY_VERY_OWN_OIL_FIELD-2
        -- ...
        -- and so on.

        ECLBASE eclipse/model/MY_VERY_OWN_OIL_FIELD-<IENS>

If not supplied, ECLBASE will default to JOBNAME, and if JOBNAME is not set,
it will default to "<CONFIG_FILE>-<IENS>".

.. _jobname:

JOBNAME
-------

Sets the name of the job submitted to the queue system. Will default to
ECLBASE if that is set, otherwise it defaults to "<CONFIG_FILE>-<IENS>".
If JOBNAME is set, and not ECLBASE, it will also be used as the value for
ECLBASE.

.. _grid:

GRID
----

Sets the global grid to use when working with fields.
This keyword specifies the name of an existing GRID/EGRID file for
your ECLIPSE model and enables parametrization via the FIELD keyword.
Note that this keyword is optional,
as you can alternatively specify the grid directly in the FIELD keyword.

This is the name of an existing GRID/EGRID file for your ECLIPSE model.
It is used to enable parametrization via the FIELD keyword. If you had
to create a new grid file when preparing your ECLIPSE reservoir model
for use with ERT, this should point to the new .EGRID file. The main
use of the grid is to map out active and inactive cells when using
FIELD data and define the dimension of the property parameter files in
the FIELD keyword. The grid argument will only be used by the main ERT
application and not passed down to the forward model in any way.

Property values for the FIELD keyword can be handled using a zone-specific
grid. The GRID keyword should in this case specify a grid file (in EGRID format)
with dimensions matching the zone requirements. This grid has the same spatial
location and rotation (x,y location) as the modeling grid, but may have fewer
layers - only enough to store properties for one zone rather than the whole
modeling grid.

The number of layers must at least be as large as the number of layers
in the zone in the modelling grid with most layers. The properties used
in the FIELD keyword have the dimension of the ERTBOX grid and
represents properties of one zone from the modelling grid. Each grid
cell in the modelling grid for a given zone corresponds to one unique
grid cell in the ERTBOX grid. Inactive grid cells in the modelling grid
also corresponds to grid cells in the ERTBOX grid. There may exists
layers of grid cells in the ERTBOX grid that does not corresponds to
grid cells in the modelling grid. It is recommended to let all grid
cells in the ERTBOX grid be active and have realistic values and not a
'missing code'. For cases where the modelling grid is kept fixed for
all realisations, this is not important, but for cases where the number
of layers for the zones in the modelling grid may vary from realisation
to realisation, this approach is more robust. It avoids mixing real
physical values from one realisation with missing code value from
another realization when calculating updated ensemble vectors.


*Example:*

::

        -- Load the .EGRID file called MY_GRID.EGRID
        GRID MY_GRID.EGRID

.. _num_cpu:

NUM_CPU
-------

This keyword tells the compute system (LSF/Torque/Slurm) how many cpus/cores
each realization needs.

*Example:*

.. code-block:: none

  NUM_CPU 4

Note that if you are using Eclipse and the :ref:`DATA_FILE <data_file>` keyword,
this is implicitly set. If you need to override, use ``NUM_CPU`` explicitly.

This number affects scheduling in the queue system, in that a realization will
not start until sufficient CPU resources are assumed available. Setting this
incorrectly can cause instability for yours and others realizations.

For the local queue system, ``NUM_CPU`` is ignored.

Default is 1.

.. _realization_memory:

REALIZATION_MEMORY
------------------

This keyword is set right in your configuration file:

.. code-block:: none

  REALIZATION_MEMORY 12Gb

and this information is propagated to the queue system as the amount of memory to
reserve/book for a realization to complete. It is up to the configuration of
the queuing system how to treat this information, but usually it will stop more
realizations being assigned to a compute node if the compute nodes memory is already
fully booked.

Setting this number lower than the peak memory consumption of each realization puts
the realization at risk of being killed in an out-of-memory situation. Setting this
number higher than needed will give longer wait times in the queue.

For the local queue system, this keyword has no effect. In that scenario, you
can use MAX_RUNNING to choke the memory consumption.

.. _data_kw:


DATA_KW
-------
The keyword DATA_KW can be used for inserting strings into placeholders in the
ECLIPSE data file. For instance, it can be used to insert include paths.

*Example:*

::

        -- Define the alias MY_PATH using DATA_KW. Any instances of <MY_PATH> (yes, with brackets)
        -- in the ECLIPSE data file will now be replaced with /mnt/my_own_disk/my_reservoir_model
        -- when running the ECLIPSE step.
        DATA_KW  MY_PATH  /mnt/my_own_disk/my_reservoir_model

The DATA_KW keyword is optional. Note also that ERT has some built in magic strings.

.. _random_seed:

RANDOM_SEED
-----------
Optional keyword, if provided must be an integer. Use a specific
seed for reproducibility. The default is that fresh unpredictable
entropy is used. Which seed is used is logged, and can then be used
to reproduce the results.

.. _enspath:

ENSPATH
-------

The ENSPATH should give the name of a folder that will be used
for storage by ERT. Note that the contents of
this folder is not intended for human inspection. By default,
ENSPATH is set to "storage".

*Example:*

::

        -- Use internal storage in /mnt/my_big_enkf_disk
        ENSPATH /mnt/my_big_enkf_disk

The ENSPATH keyword is optional.

.. _install_job:

INSTALL_JOB
-----------

The INSTALL_JOB keyword is used to instruct ERT how to run
external applications and scripts, i.e. defining a step. After a step has been
defined with INSTALL_JOB, it can be used with the FORWARD_MODEL keyword. For
example, if you have a script which generates relative permeability curves
from a set of parameters, it can be added as a step, allowing you to do history
matching and sensitivity analysis on the parameters defining the relative
permeability curves.

The INSTALL_JOB keyword takes two arguments, a step name and the name of a
configuration file for that particular step.

*Example:*

::

        -- Define a Lomeland relative permeabilty step.
        -- The file lomeland.txt contains a detailed
        -- specification of the step.
        INSTALL_JOB LOMELAND lomeland.txt

The configuration file used to specify an external step is easy to use and very
flexible. It is documented in Customizing the simulation workflow in ERT.

The INSTALL_JOB keyword is optional.

.. _include:

INCLUDE
-------

The INCLUDE keyword is used to include the contents from another ERT workflow.

*Example:*

::

        INCLUDE other_config.ert

.. _obs_config:

OBS_CONFIG
----------


The OBS_CONFIG key should point to a file defining observations and associated
uncertainties. The file should be in plain text and formatted according to the
guidelines given in :ref:`Creating an observation file for use with ERT<Configuring_observations_for_ERT>`.

*Example:*

::

        -- Use the observations in my_observations.txt
        OBS_CONFIG my_observations.txt

The OBS_CONFIG keyword is optional, but for your own convenience, it is
strongly recommended to provide an observation file.

.. _runpath:

RUNPATH
-------

The RUNPATH keyword should give the name of the folders where the ECLIPSE
simulations are executed. It should contain <IENS> and <ITER>, which
will be replaced by the realization number and iteration number when ERT creates the folders.
By default, RUNPATH is set to "simulations/realization-<IENS>/iter-<ITER>".

Deprecated syntax still allow use of two `%d` specifers. Use of more than two `%d` specifiers,
using multiple `<IENS>` or `<ITER>` keywords or mixing styles is prohibited.

*Example:*

::

        -- Using <IENS> & <ITER> specifiers for RUNPATH.
        RUNPATH /mnt/my_scratch_disk/realization-<IENS>/iter-<ITER>

*Example deprecated syntax:*

::

        -- Using RUNPATH with two %d specifers.
        RUNPATH /mnt/my_scratch_disk/realization-%d/iteration-%d

The RUNPATH keyword is optional.

.. _runpath_file:

RUNPATH_FILE
------------

When running workflows based on external scripts, it is necessary to 'tell' the
external script where all the realisations are located in
the filesystem. Since the number of realisations can be quite high this will
easily overflow the commandline buffer; the solution used is
to let ERT write a regular file. It looks like this::

            003  /cwd/realization-3/iteration-0  case3  000
            004  /cwd/realization-4/iteration-0  case4  000
            003  /cwd/realization-3/iteration-1  case3  001
            004  /cwd/realization-4/iteration-1  case4  001

The first column is the realization number, the second column is the runpath,
the third column is `<ECLBASE>` or `<JOBNAME>` if `<ECLBASE>` is not set, and
the last column is the iteration number.

Note that several tools (such as fmu-ensemble) often expect the third column to
be the path to the reservoir simulator case, but when there is no reservoir
simulator involved, the third column is not a path at all but just the job
name.

The path to this file can then be passed to the scripts using the
magic string <RUNPATH_FILE>. The RUNPATH_FILE will by default be
stored as .ert_runpath_list in the same directory as the configuration
file, but you can set it to something else with the RUNPATH_FILE key.

.. _run_template:

RUN_TEMPLATE
------------


``RUN_TEMPLATE`` can be used to copy files to the run path while doing magic string
replacement in the file content and the file name. All the scalar parameters
(provided as :ref:`GEN_KW <gen_kw>` or :ref:`DESIGN_MATRIX <design_matrix>`) are automatically
available as magic strings.

.. note::
        In case of GEN_KW prior file contains `a UNIFORM 0 1`, the magic strings is <a>.
        In case DESIGN_MATRIX contains column `SENSNAME` the magic string is <SENSNAME>.

*Example:*

::

        RUN_TEMPLATE my_text_file_template.txt my_text_file.txt


this will copy ``my_text_file_template`` into the run path, and perform magic string
replacements in the file. If no magic strings are present, the file will be copied
as it is.

It is also possible to perform replacements in target file names:

*Example:*

::

    DEFINE <MY_FILE_NAME> result.txt
    RUN_TEMPLATE template.tmpl <MY_FILE_NAME>


For templating the Eclipse or Flow DATA file, use the :ref:`DATA_FILE
<data_file>` keyword. In addition to templating it will also ensure that Ert
knows if the DATA file is to be executed in parallel.

.. _keywords_controlling_the_simulations:

Keywords controlling the experiment
-----------------------------------

.. _min_realizations:

MIN_REALIZATIONS
----------------

MIN_REALIZATIONS is the minimum number of realizations that
must have succeeded for the ensemble to be regarded as a
success.

MIN_REALIZATIONS can also be used in combination with
STOP_LONG_RUNNING, see the documentation for STOP_LONG_RUNNING
for a description of this.

*Example:*

::

        MIN_REALIZATIONS  20

The MIN_REALIZATIONS key can also be set as a percentage of
NUM_REALIZATIONS

::

        MIN_REALIZATIONS  10%

The MIN_REALIZATIONS key is optional, but if it has not been
set *all* the realisations must succeed.

Please note that MIN_REALIZATIONS = 0 means all realizations must succeed
(this happens to be the default value). Note MIN_REALIZATIONS is rounded up
e.g. 2% of 20 realizations is rounded to 1.

.. _submit_sleep:

SUBMIT_SLEEP
-----------------

Determines for how long the system will sleep between submitting jobs.
Default: ``0.0``. To change it to 1.0 s

::

    SUBMIT_SLEEP 1

.. _stop_long_running:

STOP_LONG_RUNNING
-----------------

The STOP_LONG_RUNNING key is used in combination with the MIN_REALIZATIONS key
to control the runtime of simulations. When STOP_LONG_RUNNING is set to TRUE,
MIN_REALIZATIONS is the minimum number of realizations run before the
simulation is stopped. After MIN_REALIZATIONS have succeded successfully, the
realizations left are allowed to run for 25% of the average runtime for
successful realizations, and then killed.

*Example:*

::

        -- Stop long running realizations after 20 realizations have succeeded
        MIN_REALIZATIONS  20
        STOP_LONG_RUNNING TRUE

The STOP_LONG_RUNNING key is optional. The MIN_REALIZATIONS key must be set
when STOP_LONG_RUNNING is set to TRUE.

.. _max_running:

MAX_RUNNING
-----------

The MAX_RUNNING keyword controls the maximum number of simultaneously
submitted and running realizations, where ``n`` is a positive integer::

  MAX_RUNNING n

If ``n`` is zero (the default), then there is no limit, and all realizations
will be started as soon as possible.

.. _max_runtime:

MAX_RUNTIME
-----------

The MAX_RUNTIME keyword is used to control the runtime of simulations. When
MAX_RUNTIME is set, a job is only allowed to run for MAX_RUNTIME, given in
seconds. A value of 0 means unlimited runtime.

*Example:*

::

        -- Let each realization run for a maximum of 50 seconds
        MAX_RUNTIME 50

The MAX_RUNTIME key is optional.

.. _parameterization_keywords:

Parameterization keywords
=========================

The keywords in this section are used to define a parametrization of the ECLIPSE
model. I.e. defining which parameters to change in a sensitivity analysis
and/or history matching project.

CASE_TABLE
----------
.. _case_table:

``CASE_TABLE`` is deprecated.

.. _field:

FIELD
-----

The ``FIELD`` keyword is used to parametrize quantities that span the entire grid,
with porosity and permeability being the most common examples.
In order to use the ``FIELD`` keyword, the :ref:`GRID<grid>` keyword must be supplied.

The ``FIELD`` keyword can either use a global grid specified by the :ref:`GRID<grid>` keyword,
or specify a grid directly using the GRID: option as shown below.

Field parameters (e.g. porosity, permeability or Gaussian Random Fields from APS) are defined as follows:

::

        FIELD  ID  PARAMETER  <OUTPUT_FILE>  INIT_FILES:/path/<IENS>  FORWARD_INIT:True  INIT_TRANSFORM:FUNC  OUTPUT_TRANSFORM:FUNC  MIN:X  MAX:Y GRID:CASE.EGRID

- **ID**
  String identifier with maximum 8 characters that must match the name of the parameter specified in ``INIT_FILES``.

- **PARAMETER**
  Legacy from when ERT supported EnKF and needed to handle dynamic fields like pressure and saturations.

- **OUTPUT_FILE**
  Name of file ERT will create, for example ``poro.grdecl``. Note that the Eclipse data file must include this file:

::

   INCLUDE
       'poro.grdecl' /

- **INIT_FILES**
  Filename to load initial field from. Must contain ``<IENS>`` if ``FORWARD_INIT`` is set to ``False``.
  Can be RMS ROFF format, ECLIPSE restart format or ECLIPSE GRDECL format.
  For details, see :ref:`init-files`

- **FORWARD_INIT**

  - ``FORWARD_INIT:True``
    Indicates that the specified files are generated by a forward model and do not require an embedded ``<IENS>``.

  - ``FORWARD_INIT:False``
    Means that the files must be pre-generated before running ERT and require an embedded ``<IENS>`` to differentiate between different realizations.

  For details, see :ref:`field-initialization` and :ref:`forward-init`

- **INIT_TRANSFORM** (Optional)
  Specifies the transformation to apply when the field is loaded into ERT. For details, see :ref:`field-transformations`.

- **OUTPUT_TRANSFORM** (Optional)
  Specifies the transformation to apply before the field is exported. For details, see :ref:`field-transformations`.

- **MIN** (Optional)
  Specifies the minimum value possible after applying ``OUTPUT_TRANSFORM``.

- **MAX** (Optional)
  Specifies the maximum value possible after applying ``OUTPUT_TRANSFORM``.

- **GRID** (Optional)
  Specifies the grid file to use for this specific field parameter, e.g., GRID:CASE.EGRID.
  If not specified, the global grid from the :ref:`GRID<grid>` keyword will be used.

.. _init-files:

Initialization with INIT_FILES
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the situation where you do not have geo modelling as a part of the forward
model you will typically use the geo modelling software to create an ensemble of
geological realisations up front. Assuming you intend to update the porosity
these realisations should typically be in the form of files
``/path/poro_0.grdecl, /path/poro_1.grdecl, ... /path/poro_99.grdecl``. The
``INIT_FILES:`` directive is used to configure ERT to load those files when ERT
is initializing the data. The number ``0, 1, 2, ...`` should be replaced with
the integer format specified ``<IENS>`` - which ERT will replace with the
realization number runtime, i.e.

::

   FIELD ... INIT_FILES:/path/poro_<IENS>.grdecl

in this case. The files can be in eclipse grdecl format or rms roff format; the
type is determined from the extension so you should use the common extensions
``grdecl`` or ``roff``.

.. _field-initialization:

Field initialization
^^^^^^^^^^^^^^^^^^^^

Observe that ERT can *not* sample field variables internally, they must be
supplied through another application - typically geo modelling software like
RMS; so to use the FIELD datatype you must have a workflow external to ERT which
can create/sample the fields. When you have established a workflow for
generating these fields externally there are *two* ways to load them into ERT:
`INIT_FILES` to load pregenerated initial fields or `FORWARD_INIT` to load as
part of the forward model.

.. _forward-init:

Initialization with FORWARD_INIT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When geomodelling is an integrated part of the forward model it is more
attractive to let the forward model generate the parameter fields. To enable
this we must pass the ``FORWARD_INIT:True`` when configuring the field, and also
pass a name in the ``INIT_FILES:poro.grdecl`` for the file which should be
generated by the forward model component.

Observe that there are two important differences to the ``INIT_FILES:``
attribute when it used as *the way* to initialize fields, and when it is used in
combination with ``FORWARD_INIT:True``. When ``INIT_FILES:`` is used alone the
filename given should contain a ``<IENS>`` which will be replaced with realization
number, when used with ``FORWARD_INIT:True`` that is not necessary. Furthermore
in the ``FORWARD_INIT:True`` case *the path is interpreted relative to the
runpath folder*, whereas in the other case the path is interpreted relative to
the location of the main ERT configuration file.

When using ``FORWARD_INIT:True`` together with an update algorithm in ERT the
field generated by the geo modelling software should only be used in the first
iteration (prior), in the subsequent iterations the forward model should use the
field as it comes out from ERT. The typical way to achieve this is:

1. The forward model component outputs to a temporary file ``tmp_poro.grdecl``.
2. In the first iteration ERT will *not* output a file ``poro.grdecl``, but in
   the second and subsequent iterations a ``poro.grdecl`` file will be created
   by ERT - this is at the core of the ``FORWARD_INIT:True`` functionality.
3. In the forward model there should be a step ``CAREFUL_COPY_FILE`` which will copy
   ``tmp_poro.grdecl`` *only if* ``poro.grdecl`` does not already exist. The
   rest of the forward model components should use ``poro.grdecl``.

.. note:
  With regards to behavior relative to the values in storage;
  What is really happening is that if ERT has values, those will be dumped
  to the runpath, and if not, it will read those from the runpath after the
  forward model finishes. However, if you change your runpath and "case" in
  the config file, but not your storage case, you will end up with the same
  parameter values but different RMS seed.

.. _field-transformations:

Field transformations
^^^^^^^^^^^^^^^^^^^^^

The algorithms used for Assisted History Matching (AHM) work best with normally distributed variables.
Therefore, fields that are not normally distributed must be transformed by specifying ``INIT_TRANSFORM:FUNC``.
Here, ``FUNC`` refers to one of the functions listed in the table :ref:`transformation-functions` which is applied when the field is loaded into ERT.
Similarly, ``OUTPUT_TRANSFORM:FUNC`` specifies which function to apply to the field before it is exported.

.. _transformation-functions:

.. list-table:: Transformation Functions
   :widths: 50 150
   :header-rows: 1

   * - Function
     - Description
   * - POW10
     - This function will raise 10 to the power of x: :math:`y = 10^x`
   * - TRUNC_POW10
     - This function will raise 10 to the power of x and truncate lower values at 0.001.
   * - LOG
     - This function will take the NATURAL logarithm of :math:`x: y = \ln{x}`
   * - LN
     - This function will take the NATURAL logarithm of :math:`x: y = \ln{x}`
   * - LOG10
     - This function will take the log10 logarithm of :math:`x: y = \log_{10}{x}`
   * - EXP
     - This function will calculate :math:`y = e^x`.
   * - LN0
     - This function will calculate :math:`y = \ln{(x + 0.000001)}`
   * - EXP0
     - This function will calculate :math:`y = e^x - 0.000001`


In a common scenario, log-normally distributed permeability from geo-modeling software is transformed to become normally distributed in ERT.
To achieve this:

1. ``INIT_TRANSFORM:LOG`` Transforms variables that were initially log-normally distributed into a normal distribution when loaded into ERT.

2. ``OUTPUT_TRANSFORM:EXP`` Re-exponentiates the variables to restore their log-normal distribution before they are exported to Eclipse.


.. note::
    Regarding format of OUTPUT_FILE: The default format for the parameter fields
    is binary format of the same type as used in the ECLIPSE restart files. This
    requires that the ECLIPSE datafile contains an IMPORT statement. The advantage
    with using a binary format is that the files are smaller, and reading/writing
    is faster than for plain text files. If you give the OUTPUT_FILE with the
    extension .grdecl (arbitrary case), ERT will produce ordinary .grdecl files,
    which are loaded with an INCLUDE statement. This is probably what most users
    are used to beforehand - but we recommend the IMPORT form. When using RMS APS
    plugin to create Gaussian Random Fields, the recommended file format is ROFF binary.

*Example A:*

.. code-block:: none

        -- Use Gaussian Random Fields (GRF) from APS for zone Volon.
        -- RMS APSGUI plugin will create the files specified in INIT_FILES.
        -- ERT will read the INIT_FILES in iteration 0 and write the updated GRF's
        -- to the files following the keyword PARAMETER after updating.
        FIELD  aps_Volon_GRF1  PARAMETER  aps_Volon_GRF1.roff  INIT_FILES:rms/output/aps/aps_Volon_GRF1.roff  MIN:-5.5  MAX:5.5  FORWARD_INIT:True
        FIELD  aps_Volon_GRF2  PARAMETER  aps_Volon_GRF2.roff  INIT_FILES:rms/output/aps/aps_Volon_GRF2.roff  MIN:-5.5  MAX:5.5  FORWARD_INIT:True
        FIELD  aps_Volon_GRF3  PARAMETER  aps_Volon_GRF3.roff  INIT_FILES:rms/output/aps/aps_Volon_GRF3.roff  MIN:-5.5  MAX:5.5  FORWARD_INIT:True

*Example B:*

.. code-block:: none

        -- Use perm field for zone A
        -- The GRID keyword should refer to the ERTBOX grid defining the size of the field.
        -- Permeability must be sampled from the geomodel/simulation grid zone into the ERTBOX grid
        -- and exported to /some/path/filename. Note that the name of the property in the input file
        -- in INIT_FILES must be the same as the ID.
        FIELD  perm_zone_A  PARAMETER  perm_zone_A.roff  INIT_FILES:/some/path/perm_zone_A.roff  INIT_TRANSFORM:LOG  OUTPUT_TRANSFORM:EXP  MIN:0  MAX:5000  FORWARD_INIT:True

.. _gen_data:

GEN_DATA
--------

The GEN_DATA key is used to declare a response which corresponds to a
:ref:`GENERAL_OBSERVATION <general_observation>`. It expects to read a
text file produced by the forward model, which will be loaded by ert when
loading general observations. These text files are expected to follow the
same naming scheme for all realizations (ex: ``gd_%d`` which may resolve to
``gd_0``, ``gd_1`` where `%d` is the report step).
The contents of these result are always of this format:
**exactly one floating point number per line**.
Indexing ``GEN_DATA``refers to row number in the forward model's output file,
where the index 0 refers to the first row.
``GEN_DATA`` will only affect the simulation if it is referred to by a
:ref:`GENERAL_OBSERVATION <general_observation>`.

The GEN_DATA keyword has several options, each of them required:

RESULT_FILE
^^^^^^^^^^^

This is the name of the file generated by the forward
model and read by ERT. If ``REPORT_STEPS`` are specified, this filename _must_ have a %d as part of the
name, that %d will be replaced by report step when loading. If ``REPORT_STEPS`` are not specified,
the filename does not need to contain %d.

REPORT_STEPS
^^^^^^^^^^^^

A list of the report step(s) where you expect the
forward model to create a result file. I.e. if the forward model
should create a result file for report steps 50 and 100 this setting
should be: REPORT_STEPS:50,100. If you have observations of this
GEN_DATA data the RESTART setting of the corresponding
GENERAL_OBSERVATION must match one of the values given by
REPORT_STEPS.

*Example:*

::

        GEN_DATA 4DWOC   RESULT_FILE:SimulatedWOC%d.txt   REPORT_STEPS:10,100

Here we introduce a GEN_DATA instance with name 4DWOC. When the forward
model has run it should create two files with name SimulatedWOC10.txt
and SimulatedWOC100.txt. For every realization, ERT will look within its storage
for these result files and load the content. **The files must always contain one number per line.**

ERT does not have any awareness of the type of data
encoded in a ``GEN_DATA`` keyword; it could be the result of gravimetric
calculation or the pressure difference across a barrier in the reservoir. This
means that the ``GEN_DATA`` keyword is extremely flexible, but also slightly
complicated to configure. Assume a ``GEN_DATA`` keyword is used to represent the
result of an estimated position of the oil water contact which should be
compared with a oil water contact from 4D seismic; this could be achieved with
the configuration:

::

        GEN_DATA 4DWOC  RESULT_FILE:SimulatedWOC_%d.txt   REPORT_STEPS:0

The ``4DWOC`` is an arbitrary unique key, ``RESULT_FILE:SimulatedWOC%d.txt``
means that ERT will look for results in the file ``<runpath>/SimulatedWOC_0.txt``.

The ``REPORT_STEPS:0`` is tightly bound to the ``%d`` integer format specifier
in the result file - at load time the ``%d`` is replaced with the integer values
given in the ``REPORT_STEPS:`` option, for the example given above that means
that ``%d`` will be replaced with ``0`` and ERT will look for the file
``SimulatedWOC_0.txt``. In principle it is possible to configure several report
steps like: ``REPORT_STEPS:0,10,20`` - then ERT will look for all three files
``SimulatedWOC_0.txt, SimultedWOC_10.txt`` and ``SimulatedWOC_20.txt``. It is
quite challenging to get this right, and the recommendation is to just stick
with *one* result file at report step 0 [#]_, in the future the possibility to
load one keyword ``GEN_DATA`` for multiple report steps will probably be
removed, but for now the ``GEN_DATA`` configuration is *quite strict* - it will
fail if the ``RESULT_FILE`` attribute does not contain a ``%d``.

.. [#] The option is called *report step* - but the time aspect is not really
        important. You could just as well see it as an arbitrary label, the only
        important thing is that *if* you have a corresponding ``GEN_OBS``
        observation of this ``GEN_DATA`` vector you must match the report step
        used when configuring the ``GEN_DATA`` and the ``GEN_OBS``.

.. note::
    Since the actual result file should be generated by the forward
    model, it is not possible for ERT to fully validate the ``GEN_DATA`` keyword
    at configure time. If for instance your forward model generates a file
    ``SimulatedWOC_0`` (without the ``.txt`` extension you have configured), the
    configuration problem will not be detected before ERT eventually fails to load
    the file ``SimulatedWOC_0.txt``.

.. _gen_kw:

GEN_KW
------

The General Keyword, or :code:`GEN_KW` is meant used for specifying a limited number of parameters.
:code:`GEN_KW` supports either 2 or 4 positional arguments, as well as a few keyword arguments. The first
parameter is always the name of the parameter group. If given two positional arguments, those are:

::

        GEN_KW  <name of parameter group>  <prior distribution file>


where:

| :code:`<name of parameter group>` is an arbitrary unique identifier
| :code:`<prior distribution file>` is a file containing :ref:`parameter definitions <prior_distributions>`


In the case of 4 positional arguments, those are:

::

        GEN_KW  <name of parameter group>  <template file> <output file on runpath> <prior distribution file>

where:

| :code:`<name of parameter group>` is an arbitrary unique identifier
| :code:`<template file>` is an input :ref:`template file <gen_kw_template_file>`,
| :code:`<output file on runpath>` name of the output file from ert containing templated values,
| :code:`<prior distribution file>` is a file containing :ref:`parameter definitions <prior_distributions>`

Keyword arguments:

::

        GEN_KW  ... UPDATE:TRUE/FALSE

Where the :code:`UPDATE` keyword argument specifies whether a parameter group should be included during the
history matching process. It must be set to either TRUE or FALSE. The parameters are still sampled in the prior.

.. note::

        The ``INIT_FILES:`` named attribute that was used to provide externally sampled values has been removed from GEN_KW.
        To provide values sampled outside of ERT, please see :ref:`DESIGN_MATRIX <design_matrix>`.
        Note that only parameters sampled internally in ERT will be updated during assisted history matching, and
        parameters provided through ``DESIGN_MATRIX`` will be constant.

A configuration example is shown below:

::

        GEN_KW  ID  priors.txt

where :code:`ID` is an arbitrary unique identifier,
and :code:`priors.txt` is a file containing a list of parameters and a prior distribution for each.

Given a :code:`priors.txt` file with the following distribution:

::

        A UNIFORM 0 1

where :code:`A` is an arbitrary unique identifier for this parameter,
and :code:`UNIFORM 0 1` is the distribution.

The various prior distributions available for the ``GEN_KW``
keyword are described :ref:`here <prior_distributions>`.

When the forward model is started the parameter values are added to a file located in
runpath called: ``parameters.json``.

.. code-block:: json


        {
        "A" : {"value" : 0.88},
        }


This can then be used in a forward model, an example from python below:

.. code-block:: python

    #!/usr/bin/env python
    import json

    if __name__ == "__main__":
        with open("parameters.json", encoding="utf-8") as f:
            parameters = json.load(f)
        # parameters is a dict with {"A": {"value": <value>}}



.. note::
    A file named ``parameters.txt`` is also create which contains the same information,
    but it is recommended to use ``parameters.json``.

:code:`GEN_KW` also has an optional templating functionality, an example
of the specification is as follows;

::

        GEN_KW  ID  templates/template.txt  include.txt  priors.txt

where :code:`ID` is an arbitrary unique identifier,
:code:`templates/template.txt` is the name of a template file,
:code:`include.txt` is the name of the file created for each realization
based on the template file,
and :code:`priors.txt` is a file containing a list of parameters and a prior distribution for each.

As a more concrete example, let's configure :code:`GEN_KW` to estimate pore volume multipliers,
or :code:`MULTPV`, by for example adding the following line to an ERT config-file:

::

        GEN_KW PAR_MULTPV multpv_template.txt multpv.txt multpv_priors.txt

In the GRID or EDIT section of the ECLIPSE data file, we would insert the
following include statement:

::

        INCLUDE
         'multpv.txt' /

The template file :code:`multpv_template.txt` would contain some parametrized ECLIPSE
statements:

::

        BOX
         1 10 1 30 13 13 /
        MULTPV
         300*<MULTPV_BOX1> /
        ENDBOX

        BOX
         1 10 1 30 14 14 /
        MULTPV
         300*<MULTPV_BOX2> /
        ENDBOX

Here, :code:`<MULTPV_BOX1>` and :code:`<MULTPV_BOX2>`` will act as magic
strings. Note that the ``<`` and ``>`` must be present around the magic
strings. In this case, the parameter configuration file
:code:`multpv_priors.txt` could look like this:

::

        MULTPV_BOX2 UNIFORM 0.98 1.03
        MULTPV_BOX1 UNIFORM 0.85 1.00

In general, the first keyword on each line in the parameter configuration file
defines a key, which when found in the template file enclosed in ``<`` and ``>``,
is replaced with a value. The rest of the line defines a prior distribution
for the key.

**Example: Using GEN_KW to estimate fault transmissibility multipliers**

Previously ERT supported a datatype MULTFLT for estimating fault
transmissibility multipliers. This has now been deprecated, as the
functionality can be easily achieved with the help of GEN_KW. In the ERT
config file:

::

        GEN_KW  MY-FAULTS   MULTFLT.tmpl   MULTFLT.INC   MULTFLT.txt

Here ``MY-FAULTS`` is the (arbitrary) key assigned to the fault multiplers,
``MULTFLT.tmpl`` is the template file, which can look like this:

::

        MULTFLT
         'FAULT1'   <FAULT1>  /
         'FAULT2'   <FAULT2>  /
        /

and finally the initial distribution of the parameters FAULT1 and FAULT2 are
defined in the file ``MULTFLT.txt``:

::

        FAULT1   LOGUNIF   0.00001   0.1
        FAULT2   UNIFORM   0.00      1.0


.. _gen_kw_template_file:

**Regarding templates:** You may supply the arguments TEMPLATE:/template/file
and KEY:MaGiCKEY. The template file is an arbitrary existing text file, and
KEY is a magic string found in this file. When ERT is running the magic string
is replaced with parameter data when the OUTPUT_FILE is written to the
directory where the simulation is run from. Consider for example the following
configuration:

::

        TEMPLATE:/some/file   KEY:Magic123

The template file can look like this (only the Magic123 is special):

::

        Header line1
        Header line2
        ============
        Magic123
        ============
        Footer line1
        Footer line2

When ERT is running the string Magic123 is replaced with parameter values,
and the resulting file will look like this:

::

        Header line1
        Header line2
        ============
        1.6723
        5.9731
        4.8881
        .....
        ============
        Footer line1
        Footer line2


.. _surface:

SURFACE
-------

The SURFACE keyword can be used to work with surface from RMS in the irap
format. The surface keyword is configured like this:

::

        SURFACE TOP   OUTPUT_FILE:surf.irap   INIT_FILES:Surfaces/surf<IENS>.irap   BASE_SURFACE:Surfaces/surf0.irap

The first argument, TOP in the example above, is the identifier you want to
use for this surface in ERT. The OUTPUT_FILE key is the name of surface file
which ERT will generate for you, INIT_FILES points to a list of files which
are used to initialize, and BASE_SURFACE must point to one existing surface
file. When loading the surfaces ERT will check that all the headers are
compatible. An example of a surface IRAP file is:

::

        -996   511     50.000000     50.000000
        444229.9688   457179.9688  6809537.0000  6835037.0000
        260      -30.0000   444229.9688  6809537.0000
        0     0     0     0     0     0     0
        2735.7461    2734.8909    2736.9705    2737.4048    2736.2539    2737.0122
        2740.2644    2738.4014    2735.3770    2735.7327    2733.4944    2731.6448
        2731.5454    2731.4810    2730.4644    2730.5591    2729.8997    2726.2217
        2721.0996    2716.5913    2711.4338    2707.7791    2705.4504    2701.9187
        ....

The surface data will typically be fed into other programs like Cohiba or RMS.
The data can be updated using e.g. the smoother.

**Initializing from the FORWARD MODEL**

Parameter types like FIELD and SURFACE (not GEN_KW) can be
initialized from the forward model. To achieve this you just add the setting
FORWARD_INIT:True to the configuration. When using forward init the
initialization will work like this:

#. The explicit initialization from the case menu, or when you start a
   simulation, will be ignored.
#. When the FORWARD_MODEL is complete ERT will try to initialize the node
   based on files created by the forward model. If the init fails the job as a
   whole will fail.
#. If a node has been initialized, it will not be initialized again if you run
   again.

When using FORWARD_INIT:True ERT will consider the INIT_FILES setting to find
which file to initialize from. If the INIT_FILES setting contains a relative
filename, it will be interpreted relatively to the runpath directory. In the
example below we assume that RMS has created a file petro.grdecl which
contains both the PERMX and the PORO fields in grdecl format; we wish to
initialize PERMX and PORO nodes from these files:

::

        FIELD   PORO  PARAMETER    poro.grdecl     INIT_FILES:petro.grdecl  FORWARD_INIT:True
        FIELD   PERMX PARAMETER    permx.grdecl    INIT_FILES:petro.grdecl  FORWARD_INIT:True

Observe that forward model has created the file petro.grdecl and the nodes
PORO and PERMX create the ECLIPSE input files poro.grdecl and permx.grdecl, to
ensure that ECLIPSE finds the input files poro.grdecl and permx.grdecl the
forward model should contain a step which will copy/convert petro.grdecl ->
(poro.grdecl,permx.grdecl), this step should not overwrite existing versions of
permx.grdecl and poro.grdecl. This extra hoops is not strictly needed in all
cases, but strongly recommended to ensure that you have control over which
data is used, and that everything is consistent in the case where the forward
model is run again.

.. _summary:

SUMMARY
-------

The SUMMARY keyword is used to add summary vectors from the summary files to
be read from the runpath.

Ert uses the following textual format, called :term:`summary key`, to refer to a summary vector.

So for example for field variables, no additional information is required:
`FOPR`, `FWPR`, etc. For well variables, the well name need to be specified:
`WOPR:WELL_NAME`, `WAQR:MY_WELL` etc. For block variables, the index has to be
given: `BOPR:10,9,50`. For a local completion, both the lgr name, well name,
and index has to be given: `LWWITH:LGR1:WELL2:3,5,5`.

The SUMMARY keyword accepts wildcard '*'. This adds all summary vectors where
the corresponding :term:`summary key` matches the pattern.

*Example:*

::

        -- Using the SUMMARY keyword to add diagnostic variables
        SUMMARY WOPR:MY_WELL
        SUMMARY RPR:8
        SUMMARY *           -- Select all keys
        SUMMARY F*          -- Select all keys starting with F

.. note::
    Properties added using the SUMMARY keyword are only
    diagnostic. I.e. they have no effect on the sensitivity analysis or
    history match.

.. _rft:

RFT
---

The RFT keyword is used to specify which :term:`RFT` data should be loaded from
the simulator's RFT output files. This keyword tells ERT which wells, dates,
and properties to read from the RFT files produced by the forward model.

The RFT keyword requires three pieces of information specified using keyword arguments:

- **WELL**: The name of the well to read RFT data from
- **DATE**: The date when the RFT measurement was taken (ISO format: YYYY-MM-DD)
- **PROPERTIES**: Comma-separated list of properties to load (e.g., PRESSURE, SWAT, SGAS, SOIL)

The RFT files are read from the location specified by :ref:`ECLBASE <eclbase>`.
For example, if ``ECLBASE`` is set to ``MY_FIELD``, ERT will look for the file
``MY_FIELD.RFT`` in each realization's runpath.

*Example:*

::

        -- Load pressure and water saturation from well PROD_01 on two dates
        RFT WELL:PROD_01 DATE:2015-03-15 PROPERTIES:PRESSURE,SWAT
        RFT WELL:PROD_01 DATE:2015-06-20 PROPERTIES:PRESSURE,SWAT

        -- Load pressure from multiple wells
        RFT WELL:INJ_01 DATE:2015-01-01 PROPERTIES:PRESSURE
        RFT WELL:INJ_02 DATE:2015-01-01 PROPERTIES:PRESSURE

The RFT keyword can be repeated multiple times to specify different wells, dates,
and properties. This allows ERT to load the simulated RFT data that can be compared
with :ref:`RFT_OBSERVATION <rft_observation>` entries in your observation file.

**Wildcard support:**

The RFT keyword supports wildcards (``*``) in well names, dates, and properties,
allowing you to load data for multiple wells or properties at once:

::

        -- Load pressure from all wells on a specific date
        RFT WELL:* DATE:2015-03-15 PROPERTIES:PRESSURE

        -- Load all properties for a specific well and date
        RFT WELL:PROD_01 DATE:2015-03-15 PROPERTIES:*

.. _analysis_module:

Analysis module
===============

The term analysis module refers to the underlying algorithm used for the analysis,
or update step of data assimilation.
The keywords to load, select and modify the analysis modules are documented here.

.. _analysis_set_var:

ANALYSIS_SET_VAR
----------------

The analysis modules can have internal state, like e.g. truncation cutoff
values. These can be manipulated from the config file using the
ANALYSIS_SET_VAR keyword for the `STD_ENKF` module.

::

    ANALYSIS_SET_VAR  STD_ENKF  ENKF_TRUNCATION  0.98


.. _inversion_algorithm:

INVERSION
^^^^^^^^^

The analysis modules can specify inversion algorithm used.
These can be manipulated from the config file using the
ANALYSIS_SET_VAR keyword for the `STD_ENKF` module.

**STD_ENKF**


.. list-table:: Inversion Algorithms for Ensemble Smoother
   :widths: 50 50 50
   :header-rows: 1

   * - Description
     - INVERSION
     - Note
   * - Exact inversion with diagonal R=I
     - Deprecated: exact, 0
     - Preferred name: EXACT
   * - Subspace inversion with exact R
     - Deprecated: SUBSPACE_EXACT_R, subspace, 1
     - Preferred name: SUBSPACE

.. _localization:

LOCALIZATION
^^^^^^^^^^^^

The analysis module has capability for enabling adaptive localization
correlation threshold.
This can be enabled from the config file using the
ANALYSIS_SET_VAR keyword but is valid for the ``STD_ENKF`` module only.
This is default ``False``.

::

        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True

.. _local_corr_threshold:

LOCALIZATION_CORRELATION_THRESHOLD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The analysis module has capability for specifying the adaptive
localization correlation threshold value.
This can be specified from the config file using the
ANALYSIS_SET_VAR keyword but is valid for the ``STD_ENKF`` module only.
This is default ``0.30``.

::

        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD 0.30

.. _auto_scale_observations_keyword:

AUTO_SCALE_OBSERVATIONS
^^^^^^^^^^^^^^^^^^^^^^^
.. _auto_scale_observations:

The analysis can try to find correlated observations and scale those, to decrease
the impact of correlated observations, this can be specified from the config file:

::

        ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE *

This will go through all the observations and scale them according to how correlated they are. If
you would like to only scale some observations, you can use wildcard matching:

.. code-block:: text

    ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE OBS_1*
    ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE OBS_2*

This will find correlations in all observations starting with: 'OBS_1' and scale those, then
find correlations in all observations starting with: 'OBS_2', and scale those, independent of 'OBS_1*'

.. _enkf_truncation:

ENKF_TRUNCATION
^^^^^^^^^^^^^^^

Truncation factor for the SVD-based EnKF algorithm (see Evensen, 2007). In
this algorithm, the forecasted data will be projected into a low dimensional
subspace before assimilation. This can substantially improve on the results
obtained with the EnKF, especially if the data ensemble matrix is highly
collinear (Saetrom and Omre, 2010). The subspace dimension, p, is selected
such that


:math:`\frac{\sum_{i=1}^{p} s_i^2}{\sum_{i=1}^r s_i^2} \geq \mathrm{ENKF\_TRUNCATION}`

where si is the ith singular value of the centered data ensemble matrix and r
is the rank of this matrix. This criterion is similar to the explained
variance criterion used in Principal Component Analysis (see e.g. Mardia et
al. 1979).

::

    ANALYSIS_SET_VAR  STD_ENKF  ENKF_TRUNCATION  0.98

The default value of ENKF_TRUNCATION is 0.98. If ensemble collapse is a big
problem, a smaller value should be used (e.g 0.90 or smaller). However, this
does not guarantee that the problem of ensemble collapse will disappear. Note
that setting the truncation factor to 1.00, will recover the Standard-EnKF
algorithm if and only if the covariance matrix for the observation errors is
proportional to the identity matrix.


**References**

* Evensen, G. (2007). "Data Assimilation, the Ensemble Kalman Filter", Springer.
* Mardia, K. V., Kent, J. T. and Bibby, J. M. (1979). "Multivariate Analysis", Academic Press.
* Saetrom, J. and Omre, H. (2010). "Ensemble Kalman filtering with shrinkage regression techniques", Computational Geosciences (online first).


.. _keywords_controlling_the_es_algorithm:

Keywords controlling the ES algorithm
=====================================

.. _enkf_alpha:

ENKF_ALPHA
----------

The scaling factor used when detecting outliers. Increasing this
factor means that more observations will potentially be included in the
assimilation. The default value is 3.00.

Including outliers in the Smoother algorithm can dramatically increase the
coupling between the ensemble members. It is therefore important to filter out
these outliers prior to data assimilation. An observation,
:math:`d^o_i`, will be classified as an outlier if

:math:`|d^o_i - \bar{\mathbf{d}}| > \mathrm{ENKF\_ALPHA} \left( s_{d_i} + s^o_{d_i} \right)`

where :math:`\mathbf{d}^o` is the vector of observed data,
:math:`\mathbf{\bar{d}}` is the average of the forecasted data ensemble,
:math:`\mathbf{s}_\mathbf{d}` is the vector of estimated standard deviations
for the forecasted data ensemble, and :math:`\mathbf{s}^o_\mathbf{d}` is the
vector of standard deviations for the observation error (specified a priori).

Observe that for the updates many settings should be applied on the analysis
module in question.

.. _std_cutoff:

STD_CUTOFF
----------

If the ensemble variation for one particular measurement is below
this limit the observation will be deactivated. The default value for
this cutoff is 1e-6.

Observe that for the updates many settings should be applied on the analysis
module in question.

.. _update_log_path:

UPDATE_LOG_PATH
---------------

A summary of the data used for updates are stored in this directory.

.. _max_submit:

MAX_SUBMIT
----------

How many times a realization can be submitted to the queue system in case of
realization failures. Default is 1, meaning there will be no resubmission upon
failures.

.. _advanced_keywords:

Advanced keywords
=================

The keywords in this section, controls advanced features of ERT. Insight in
the internals of ERT and/or ECLIPSE may
be required to fully understand their effect. Moreover, many of these keywords
are defined in the site configuration, and thus optional to set for the user,
but required when installing ERT at a new site.

.. _keywords_related_to_running_the_forward_model:

Keywords related to running the forward model
=============================================

.. _forward_model:

FORWARD_MODEL
-------------

    The FORWARD_MODEL keyword is used to define how the simulations are executed.
    E.g., which version of ECLIPSE to use, which rel.perm script to run, which
    rock physics model to use etc. Steps (i.e. programs and scripts) that are to be
    used in the FORWARD_MODEL keyword must be defined using the INSTALL_JOB
    keyword. A set of default steps is available, and by default FORWARD_MODEL
    takes the value ECLIPSE100.

    The FORWARD_MODEL keyword expects one keyword defined with INSTALL_JOB.

    *Example:*

    ::

            -- Suppose that "MY_RELPERM_SCRIPT" has been defined with
            -- the INSTALL_JOB keyword. This FORWARD_MODEL will execute
            -- "MY_RELPERM_SCRIPT" before ECLIPSE100.
            FORWARD_MODEL MY_RELPERM_SCRIPT
            FORWARD_MODEL ECLIPSE100

    In available steps in ERT you can see a list of the steps which are available.


.. _queue_system:

QUEUE_SYSTEM
------------

The keyword QUEUE_SYSTEM can be used to control where the forward model is
executed. It can take the values LSF, TORQUE, SLURM and LOCAL.

::

        -- Tell ERT to use the LSF cluster.
        QUEUE_SYSTEM LSF

The QUEUE_SYSTEM keyword is optional, and usually defaults to LSF (this is
site dependent).

.. _queue_option:

QUEUE_OPTION
------------

The chosen queue system can be configured further to for instance define the
resources it is using. The different queues have individual options that are
configurable.


Queue configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are configuration options for the various queue systems, described in detail
in :ref:`queue-system-chapter`. In brief, the queue systems have the following options:

* :ref:`LOCAL <local-queue>` — no queue options.
* :ref:`LSF <lsf-systems>` — ``LSF_QUEUE``, ``LSF_RESOURCE``,
  ``BSUB_CMD``, ``BJOBS_CMD``, ``BKILL_CMD``,
  ``BHIST_CMD``, ``SUBMIT_SLEEP``, ``PROJECT_CODE``, ``EXCLUDE_HOST``,
  ``MAX_RUNNING``
* :ref:`TORQUE <pbs-systems>` — ``QSUB_CMD``, ``QSTAT_CMD``, ``QDEL_CMD``,
  ``QUEUE``, ``CLUSTER_LABEL``, ``MAX_RUNNING``, ``KEEP_QSUB_OUTPUT``,
  ``SUBMIT_SLEEP``
* :ref:`SLURM <slurm-systems>` — ``SBATCH``, ``SCANCEL``, ``SCONTROL``, ``SACCT``,
  ``SQUEUE``, ``PARTITION``, ``SQUEUE_TIMEOUT``, ``MAX_RUNTIME``, ``INCLUDE_HOST``,
  ``EXCLUDE_HOST``, ``MAX_RUNNING``

In addition, some options apply to all queue systems:



Workflow hooks
==============

.. _hook_workflow:

HOOK_WORKFLOW
-------------

With the keyword :code:`HOOK_WORKFLOW` you can configure workflow
'hooks'; meaning workflows which will be run automatically at certain
points during ERTs execution. Currently there are five points in ERTs
flow of execution where you can hook in a workflow:

- Before the simulations (all forward models for a realization) start using :code:`PRE_SIMULATION`,
- after all the simulations have completed using :code:`POST_SIMULATION`,
- before the update step using :code:`PRE_UPDATE`
- after the update step using :code:`POST_UPDATE` and
- only before the first update using :code:`PRE_FIRST_UPDATE`.

For non-iterative algorithms, :code:`PRE_FIRST_UPDATE` is equal to :code:`PRE_UPDATE`.
The :code:`POST_SIMULATION` hook is typically used to trigger QC workflows.

::

   HOOK_WORKFLOW initWFLOW        PRE_SIMULATION
   HOOK_WORKFLOW preUpdateWFLOW   PRE_UPDATE
   HOOK_WORKFLOW postUpdateWFLOW  POST_UPDATE
   HOOK_WORKFLOW QC_WFLOW1        POST_SIMULATION
   HOOK_WORKFLOW QC_WFLOW2        POST_SIMULATION

In this example the workflow :code:`initWFLOW` will run after all the
:term:`simulation` directories have been created, just before the forward
model is submitted to the queue. The workflow :code:`preUpdateWFLOW`
will be run before the update step and :code:`postUpdateWFLOW` will be
run after the update step. When all the simulations have completed the
two workflows :code:`QC_WFLOW1` and :code:`QC_WFLOW2` will be run.

Observe that the workflows being 'hooked in' with the
:code:`HOOK_WORKFLOW` must be loaded with the :code:`LOAD_WORKFLOW` keyword.

.. _load_workflow:

LOAD_WORKFLOW
-------------

Workflows are loaded with the configuration option :code:`LOAD_WORKFLOW`:

::

    LOAD_WORKFLOW  /path/to/workflow/WFLOW1
    LOAD_WORKFLOW  /path/to/workflow/workflow2  WFLOW2

The :code:`LOAD_WORKFLOW` takes the path to a workflow file as the first
argument. By default the workflow will be labeled with the filename
internally in ERT, but you can optionally supply a second extra argument
which will be used as the name for the workflow.  Alternatively,
you can load a workflow interactively.

.. _load_workflow_job:

LOAD_WORKFLOW_JOB
-----------------

Before the jobs can be used in workflows they must be "loaded" into
ERT. This can be done either by specifying jobs by name,
or by specifying a directory containing jobs.

Use the keyword :code:`LOAD_WORKFLOW_JOB` to specify jobs by name:

::

    LOAD_WORKFLOW_JOB     jobConfigFile     JobName

The :code:`LOAD_WORKFLOW_JOB` keyword will load one workflow job.
The name of the job is optional, and will be fetched from the configuration file if not provided.

.. _workflow_job_directory:

WORKFLOW_JOB_DIRECTORY
----------------------

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

.. _zonemap:

ZONEMAP
-------

The ZONEMAP keyword specifies a file that maps simulation grid layers (K indices)
to geological zone names. This is primarily used to validate :ref:`RFT_OBSERVATION <rft_observation>`
locations to ensure that observations fall within the expected geological zones.

The zonemap file format is simple: each line contains a K-layer index (1-indexed)
followed by one or more zone names that apply to that layer:

::

    -- Format: K_layer zone1 [zone2 ...]
    1  TopZone
    2  TopZone
    3  MiddleZone
    4  MiddleZone
    5  BottomZone
    6  BottomZone

*Example usage:*

::

        ZONEMAP my_zones.txt

The file my_zones.txt should be located in the runpath in the above example. In
general, when the ZONEMAP is set to a relative path then it is interpreted as relative
to the runpath, otherwise it should be located at the absolute path.

When RFT observations include a ZONE specification (either in the CSV file or
directly in the observation), ERT will validate that the observation's grid
location matches the expected zone from the ZONEMAP. If an observation is
located in a different zone than specified, it will be deactivated with a warning.

.. note::
    Grid layers are 1-indexed in the ZONEMAP file. Multiple zone names can be
    specified for a single layer if it spans multiple geological zones.

Manipulating the environment variables
--------------------------------------

.. _setenv:

SETENV
------

You can use the SETENV keyword to alter the environment variables where ERT runs
forward models.

*Example:*

::

        SETENV  MY_VAR          World
        SETENV  MY_OTHER_VAR    Hello$MY_VAR

This will result in two environment variables being set in the compute side
and available to all step. MY_VAR will be "World", and MY_OTHER_VAR will be
"HelloWorld". The variables are expanded in order on the compute side, so
the environment where ERT is running has no impact, and is not changed.
