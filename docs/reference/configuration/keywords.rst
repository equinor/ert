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
:ref:`ANALYSIS_SELECT <analysis_select>`                                NO                                      STD_ENKF                        Select analysis module to use in update
:ref:`CASE_TABLE <case_table>`                                          NO                                                                      Deprecated
:ref:`DATA_FILE <data_file>`                                            NO                                                                      Provide an ECLIPSE data file for the problem
:ref:`DATA_KW <data_kw>`                                                NO                                                                      Replace strings in ECLIPSE .DATA files
:ref:`DEFINE <define>`                                                  NO                                                                      Define keywords with config scope
:ref:`ECLBASE <eclbase>`                                                NO                                                                      Define a name for the ECLIPSE simulations.
:ref:`STD_CUTOFF <std_cutoff>`                                          NO                                      1e-6                            Determines the threshold for ensemble variation in a measurement
:ref:`ENKF_ALPHA <enkf_alpha>`                                          NO                                      3.0                             Parameter controlling outlier behaviour in EnKF algorithm
:ref:`ENKF_FORCE_NCOMP <enkf_force_ncomp>`                              NO                                      0                               Indicate if ERT should force a specific number of principal components
:ref:`ENKF_NCOMP <enkf_ncomp>`                                          NO                                                                      Number of PC to use when forcing a fixed number; used in combination with kw ENKF_FORCE_NCOMP
:ref:`ENKF_TRUNCATION <enkf_truncation>`                                NO                                      0.99                            Cutoff used on singular value spectrum
:ref:`ENSPATH <enspath>`                                                NO                                      storage                         Folder used for storage of simulation results
:ref:`FIELD <field>`                                                    NO                                                                      Adds grid parameters
:ref:`FORWARD_MODEL <forward_model>`                                    NO                                                                      Add the running of a job to the simulation forward model
:ref:`GEN_DATA <gen_data>`                                              NO                                                                      Specify a general type of data created/updated by the forward model
:ref:`GEN_KW <gen_kw>`                                                  NO                                                                      Add a scalar parameter
:ref:`GRID <grid>`                                                      NO                                                                      Provide an ECLIPSE grid for the reservoir model
:ref:`HISTORY_SOURCE <history_source>`                                  NO                                      REFCASE_HISTORY                 Source used for historical values
:ref:`HOOK_WORKFLOW <hook_workflow>`                                    NO                                                                      Install a workflow to be run automatically
:ref:`INCLUDE <include>`                                                NO                                                                      Include contents from another ert config
:ref:`INSTALL_JOB <install_job>`                                        NO                                                                      Install a job for use in a forward model
:ref:`ITER_CASE <iter_Case>`                                            NO                                      IES%d                           Case name format - iterated ensemble smoother
:ref:`ITER_COUNT <iter_count>`                                          NO                                      4                               Number of iterations - iterated ensemble smoother
:ref:`ITER_RETRY_COUNT <iter_retry_count>`                              NO                                      4                               Number of retries for a iteration - iterated ensemble smoother
:ref:`JOBNAME <jobname>`                                                NO                                      <CONFIG_FILE>-<IENS>            Name used for simulation files.
:ref:`JOB_SCRIPT <job_script>`                                          NO                                                                      Python script managing the forward model
:ref:`LOAD_WORKFLOW <load_workflow>`                                    NO                                                                      Load a workflow into ERT
:ref:`LOAD_WORKFLOW_JOB <load_workflow_job>`                            NO                                                                      Load a workflow job into ERT
:ref:`LICENSE_PATH <license_path>`                                      NO                                                                      A path where ert-licenses to e.g. RMS are stored
:ref:`MAX_RUNTIME <max_runtime>`                                        NO                                      0                               Set the maximum runtime in seconds for a realization (0 means no runtime limit)
:ref:`MAX_SUBMIT <max_submit>`                                          NO                                      2                               How many times the queue system should retry a simulation
:ref:`MIN_REALIZATIONS <min_realizations>`                              NO                                      0                               Set the number of minimum realizations that has to succeed in order for the run to continue (0 means identical to NUM_REALIZATIONS - all must pass).
:ref:`NUM_CPU <num_cpu>`                                                NO                                      1                               Set the number of CPUs. Intepretation varies depending on context
:ref:`NUM_REALIZATIONS <num_realizations>`                              YES                                                                     Set the number of reservoir realizations to use
:ref:`OBS_CONFIG <obs_config>`                                          NO                                                                      File specifying observations with uncertainties
:ref:`QUEUE_OPTION <queue_option>`                                      NO                                                                      Set options for an ERT queue system
:ref:`QUEUE_SYSTEM <queue_system>`                                      NO                                      LOCAL_DRIVER                                System used for running simulation jobs
:ref:`REFCASE <refcase>`                                                NO                                                                      Reference case used for observations and plotting (See HISTORY_SOURCE and SUMMARY)
:ref:`RESULT_PATH  <result_path>`                                       NO                                      results/step_%d                 Define where ERT should store results
:ref:`RUNPATH <runpath>`                                                NO                                      realization-<IENS>/iter-<ITER>  Directory to run simulations; simulations/realization-<IENS>/iter-<ITER>
:ref:`RUNPATH_FILE <runpath_file>`                                      NO                                      .ert_runpath_list               Name of file with path for all forward models that ERT has run. To be used by user defined scripts to find the realizations
:ref:`RUN_TEMPLATE <run_template>`                                      NO                                                                      Install arbitrary files in the runpath directory
:ref:`SETENV <setenv>`                                                  NO                                                                      You can modify the UNIX environment with SETENV calls
:ref:`SIMULATION_JOB <simulation_job>`                                  NO                                                                      Lightweight alternative FORWARD_MODEL
:ref:`STOP_LONG_RUNNING <stop_long_running>`                            NO                                      FALSE                           Stop long running realizations after minimum number of realizations (MIN_REALIZATIONS) have run
:ref:`SUMMARY  <summary>`                                               NO                                                                      Add summary variables for internalization
:ref:`SURFACE <surface>`                                                NO                                                                      Surface parameter read from RMS IRAP file
:ref:`TIME_MAP  <time_map>`                                             NO                                                                      Ability to manually enter a list of dates to establish report step <-> dates mapping
:ref:`UPDATE_LOG_PATH  <update_log_path>`                               NO                                      update_log                      Summary of the update steps are stored in this directory
:ref:`WORKFLOW_JOB_DIRECTORY  <workflow_job_directory>`                 NO                                                                      Directory containing workflow jobs
=====================================================================   ======================================  ==============================  ==============================================================================================================================================



Commonly used keywords
-----------------------
.. _commonly_used_keywords:

.. _num_realizations:
.. topic:: NUM_REALIZATIONS

        This is the size of the ensemble, i.e. the number of
        realizations/members in the ensemble. All configs must contain this
        keyword.

        *Example:*

        ::

                -- Use 200 realizations/members
                NUM_REALIZATIONS 200

.. _define:
.. topic:: DEFINE

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


.. _data_file:
.. topic:: DATA_FILE

        Meant to be set to the filepath of an eclipse simulator input, when such
        a simulator is used. This does two things. First, the DATA_FILE will be
        templated, see :ref:`RUN_TEMPLATE <run_template>`. Second, ert will look
        for the PARALLEL keyword in this file in order to set :ref:`NUM_CPU <num_cpu>`.

        The templated file will be named according to :ref:`ECLBASE <ECLBASE>`
        and copied to the runpath folder. Note that support for parsing the
        ECLIPSE data file is limited, and using explicit templating with
        :ref:`RUN_TEMPLATE <run_template>` is recommended where possible.


        *Example:*

        ::

                -- Load the data file called ECLIPSE.DATA
                DATA_FILE ECLIPSE.DATA

        See the ``DATA_KW`` keyword which can be used to utilize more template
        functionality in the eclipse datafile.

        This is used to replace ERT magic strings into the data file, as well as
        update the number of cpus that are reserved for ERT in the queue system.

        It searches for PARALLEL in the data file, and if that is not found it
        will search for SLAVE and update <NUM_CPU> according to how many nodes are
        found, note that it does *not* parse the data files of the nodes, and will
        assume one cpu per node where entry number 5 is not set, and the number of
        entry number 5 otherwise plus one cpu for the master node.

        It is strongly recommended to use the :ref:`RUN_TEMPLATE <run_template>`
        for magic string replacement and resource allocation instead. Combined
        with :ref:`NUM_CPU <num_cpu>` the resources for the cluster are specified
        directly in the ERT configuration, and can be templated into the ECLIPSE
        data file, see  :ref:`RUN_TEMPLATE <run_template>`.




.. _eclbase:
.. topic:: ECLBASE

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
.. topic::  JOBNAME

        Sets the name of the job submitted to the queue system. Will default to
        ECLBASE If that is set, otherwise it defaults to "<CONFIG_FILE>-<IENS>". If JOBNAME
        is set, and not ECLBASE, it will also be used as the value for ECLBASE.

.. _grid:
.. topic:: GRID

        This is the name of an existing GRID/EGRID file for your ECLIPSE model.
        It is used to enable parametrization via the FIELD keyword. If you had
        to create a new grid file when preparing your ECLIPSE reservoir model
        for use with ERT, this should point to the new .EGRID file. The main
        use of the grid is to map out active and inactive cells when using
        FIELD data and define the dimension of the property parameter files in
        the FIELD keyword. The grid argument will only be used by the main ERT
        application and not passed down to the forward model in any way.

        A new way of handling property values for the FIELD keyword is to use a
        help grid called ERTBOX grid. The GRID keyword should in this case
        specify the ERTBOX filename (which is in EGRID format). The ERTBOX grid
        is a grid with the same spatial location and rotation (x,y location) as
        the modelling grid, but it is a regular grid in a rectangular box. The
        dimensions of the ERTBOX grid laterally is the same as the modelling
        grid, but the number of layers is only large enough to store the
        properties for one zone, not the whole modelling grid.

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
.. topic:: NUM_CPU

    Equates to the ``-n`` argument in the context of LSF. For TORQUE, it is
    simply a upper bound for the product of nodes and CPUs per node.


    *Example:*

    ::

        NUM_CPU 2

.. _data_kw:
.. topic:: DATA_KW

        The keyword DATA_KW can be used for inserting strings into placeholders in the
        ECLIPSE data file. For instance, it can be used to insert include paths.

        *Example:*

        ::

                -- Define the alias MY_PATH using DATA_KW. Any instances of <MY_PATH> (yes, with brackets)
                -- in the ECLIPSE data file will now be replaced with /mnt/my_own_disk/my_reservoir_model
                -- when running the ECLIPSE jobs.
                DATA_KW  MY_PATH  /mnt/my_own_disk/my_reservoir_model

        The DATA_KW keyword is of course optional. Note also that ERT has some
        built in magic strings.

.. _random_seed:
.. topic:: RANDOM_SEED

        Optional keyword, if provided must be an integer. Use a specific
        seed for reproducibility. The default is that fresh unpredictable
        entropy is used. Which seed is used is logged, and can then be used
        to reproduce the results.

.. _enspath:
.. topic:: ENSPATH

        The ENSPATH should give the name of a folder that will be used
        for storage by ERT. Note that the contents of
        this folder is not intended for human inspection. By default,
        ENSPATH is set to "storage".

        *Example:*

        ::

                -- Use internal storage in /mnt/my_big_enkf_disk
                ENSPATH /mnt/my_big_enkf_disk

        The ENSPATH keyword is optional.


.. _history_source:
.. topic:: HISTORY_SOURCE

        In the observation configuration file you can enter
        observations with the keyword HISTORY_OBSERVATION; this means
        that ERT will extract observed values from the model
        historical summary vectors of the reference case. What source
        to use for the  historical values can be controlled with the
        HISTORY_SOURCE keyword. The different possible values for the
        HISTORY_SOURCE keyword are:


        REFCASE_HISTORY
                This is the default value for HISTORY_SOURCE,
                ERT will fetch the historical values from the *xxxH*
                keywords in the refcase summary, e.g. observations of
                WGOR:OP_1 is based the WGORH:OP_1 vector from the
                refcase summary.

        REFCASE_SIMULATED
                In this case the historical values are based on the
                simulated values from the refcase, this is mostly relevant when you want
                compare with another case which serves as 'the truth'.

        When setting HISTORY_SOURCE to either REFCASE_SIMULATED or REFCASE_HISTORY you
        must also set the REFCASE variable to point to the ECLIPSE data file in an
        existing reference case (should be created with the same schedule file as you
        are using now).

        *Example:*

        ::

                -- Use historic data from reference case
                HISTORY_SOURCE  REFCASE_HISTORY
                REFCASE         /somefolder/ECLIPSE.DATA

        The HISTORY_SOURCE keyword is optional.

.. _refcase:
.. topic:: REFCASE

        The REFCASE key is used to provide ERT an existing ECLIPSE simulation
        from which it can read various information at startup. The intention is
        to ease the configuration needs for the user. Functionality provided with the
        refcase:

        * extract observed values from the refcase using the
          :ref:`HISTORY_OBSERVATION <HISTORY_OBSERVATION>` and
          :ref:`HISTORY_SOURCE <HISTORY_SOURCE>` keys.


        The REFCASE keyword should point to an existing ECLIPSE simulation;
        ert will then look up and load the corresponding summary results.

        *Example:*

        ::

                -- The REFCASE keyword points to the datafile of an existing ECLIPSE simulation.
                REFCASE /path/to/somewhere/SIM_01_BASE.DATA


        The refcase is used when loading HISTORY_OBSERVATION and in some scenarios when using SUMMARY_OBSERVATION.
        With HISTORY_OBSERVATION the values are read directly from the REFCASE. When using
        SUMMARY_OBSERVATION the REFCASE is not strictly required. If using DATE in the observation
        configuration the REFCASE can be omitted, and the observation will be compared with the summary
        response configured with ECLBASE. If REFCASE is provided it will validated that the DATE
        exists in the REFCASE, and if there is a mismatch a configuration error will be raised.
        If using HOURS, DAYS, or RESTART in the observation configuration, the REFCASE is required and will
        be used to look up the date of the observation in the REFCASE.


.. _install_job:
.. topic:: INSTALL_JOB

        The INSTALL_JOB keyword is used to instruct ERT how to run
        external applications and scripts, i.e. defining a job. After a job has been
        defined with INSTALL_JOB, it can be used with the FORWARD_MODEL keyword. For
        example, if you have a script which generates relative permeability curves
        from a set of parameters, it can be added as a job, allowing you to do history
        matching and sensitivity analysis on the parameters defining the relative
        permeability curves.

        The INSTALL_JOB keyword takes two arguments, a job name and the name of a
        configuration file for that particular job.

        *Example:*

        ::

                -- Define a Lomeland relative permeabilty job.
                -- The file jobs/lomeland.txt contains a detailed
                -- specification of the job.
                INSTALL_JOB LOMELAND jobs/lomeland.txt

        The configuration file used to specify an external job is easy to use and very
        flexible. It is documented in Customizing the simulation workflow in ERT.

        The INSTALL_JOB keyword is optional.

.. _include:
.. topic:: INCLUDE

        The INCLUDE keyword is used to include the contents from another ERT workflow.

        *Example:*

        ::

                INCLUDE other_config.ert


.. _obs_config:
.. topic:: OBS_CONFIG

        The OBS_CONFIG key should point to a file defining observations and associated
        uncertainties. The file should be in plain text and formatted according to the
        guidelines given in :ref:`Creating an observation file for use with ERT<Configuring_observations_for_ERT>`.

	    If you include HISTORY_OBSERVATION in the observation file, you must
	    provide a reference Eclipse case through the REFCASE keyword.

        *Example:*

        ::

                -- Use the observations in my_observations.txt
                OBS_CONFIG my_observations.txt

        The OBS_CONFIG keyword is optional, but for your own convenience, it is
        strongly recommended to provide an observation file.

.. _result_path:
.. topic:: RESULT_PATH

        ERT will print some simple tabulated results at each report
        step. The RESULT_PATH keyword should point to a folder where the tabulated
        results are to be written. It can contain a %d specifier, which will be
        replaced with the report step. The default value for RESULT_PATH is
        "results/step_%d".

        *Example:*

        ::

                -- Changing RESULT_PATH
                RESULT_PATH my_nice_results/step-%d

        The RESULT_PATH keyword is optional.

.. _runpath:
.. topic:: RUNPATH

        The RUNPATH keyword should give the name of the folders where the ECLIPSE
        simulations are executed. It should contain <IENS> and <ITER>, which
        will be replaced by the realization number and iteration number when ERT creates the folders.
        By default, RUNPATH is set to "simulations/realization-<IENS>/iter-<ITER>".

        Deprecated syntax still allow use of two %d specifers. Use of less than two %d is prohibited.
        The behaviour is identical to the default substitution.

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
.. topic:: RUNPATH_FILE

        When running workflows based on external scripts it is necessary to 'tell' the
        external script in some way or another were all the realisations are located in
        the filesystem. Since the number of realisations can be quite high this will
        easily overflow the commandline buffer; the solution which is used is therefore
        to let ERT write a regular file which looks like this::

                0   /path/to/realization-0   CASE0   iter
                1   /path/to/realization-1   CASE1   iter
                ...
                N   /path/to/realization-N   CASEN   iter

        The path to this file can then be passed to the scripts using the
        magic string <RUNPATH_FILE>. The RUNPATH_FILE will by default be
        stored as .ert_runpath_list in the same directory as the configuration
        file, but you can set it to something else with the RUNPATH_FILE key.


.. _run_template:
.. topic:: RUN_TEMPLATE

        ``RUN_TEMPLATE`` can be used to copy files to the run path while doing magic string
        replacement in the file content and the file name.

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




        If one would like to do substitutions in the ECLIPSE data file, that can be
        done like this:

        *Example:*

        ::

                ECLBASE BASE_ECL_NAME%d
                RUN_TEMPLATE MY_DATA_FILE.DATA <ECLBASE>.DATA

        This will copy ``MY_DATA_FILE.DATA`` into the run path and name it ``BASE_ECL_NAME0.DATA``
        while doing magic string replacement in the contents.

        If you would like to substitute in the realization number as a part of ECLBASE using
        ``<IENS>`` instead of ``%d`` is a better option:

        *Example:*

        ::

                ECLBASE BASE_ECL_NAME-<IENS>
                RUN_TEMPLATE MY_DATA_FILE.DATA <ECLBASE>.DATA



        To control the number of CPUs that are reserved for ECLIPSE use `RUN_TEMPLATE` with
        :ref:`NUM_CPU <num_cpu>` and keep them in sync:

        ::

                NUM_CPU 4
                ECLBASE BASE_ECL_NAME-<IENS>
                RUN_TEMPLATE MY_DATA_FILE.DATA <ECLBASE>.DATA

        In the ECLIPSE data file:

        ::

                PARALLEL <NUM_CPU>


Keywords controlling the simulations
------------------------------------
.. _keywords_controlling_the_simulations:

.. _min_realizations:
.. topic:: MIN_REALIZATIONS

        MIN_REALIZATIONS is the minimum number of realizations that
        must have succeeded for the simulation to be regarded as a
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

        Please note that MIN_REALIZATIONS = 0 means all simulations must succeed
        (this happens to be the default value). Note MIN_REALIZATIONS is rounded up
        e.g. 2% of 20 realizations is rounded to 1.


.. _stop_long_running:
.. topic:: STOP_LONG_RUNNING

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


.. _max_runtime:
.. topic:: MAX_RUNTIME

        The MAX_RUNTIME keyword is used to control the runtime of simulations. When
        MAX_RUNTIME is set, a job is only allowed to run for MAX_RUNTIME, given in
        seconds. A value of 0 means unlimited runtime.

        *Example:*

        ::

                -- Let each realization run for a maximum of 50 seconds
                MAX_RUNTIME 50

        The MAX_RUNTIME key is optional.


Parameterization keywords
-------------------------
.. _parameterization_keywords:

The keywords in this section are used to define a parametrization of the ECLIPSE
model. I.e. defining which parameters to change in a sensitivity analysis
and/or history matching project.


.. _case_table:
.. topic:: CASE_TABLE

        `CASE_TABLE` is deprecated.


.. _field:
.. topic:: FIELD

        The FIELD keyword is used to parametrize quantities which have extent over the
        full grid. In order to use the FIELD keyword, the GRID keyword must be supplied.

        A parameter field (e.g. porosity or permeability or Gaussian Random Fields from APS) is defined as follows:

        ::

                FIELD  ID PARAMETER   <ECLIPSE_FILE>  INIT_FILES:/path/%d  MIN:X MAX:Y OUTPUT_TRANSFORM:FUNC INIT_TRANSFORM:FUNC  FORWARD_INIT:True

	Here ID must be the same as the name of the parameter in the INIT_FILES.
        ECLIPSE_FILE is the name of the file ERT will export this field to when
        running simulations. Note that there should be an IMPORT statement in
        the ECLIPSE data file corresponding to the name given with ECLIPSE_FILE in case
        the field parameter is a field used in ECLIPSE data file like perm or poro.
        INIT_FILES is a filename (with an embedded %d if FORWARD_INIT is set to False)
        to load the initial field from. Can be RMS ROFF format, ECLIPSE restart format
        or ECLIPSE GRDECL format.

        FORWARD_INIT:True means that the files specified in the INIT_FILES are expected
        to be created by a forward model, and does not need any embedded %d.
	FORWARD_INIT:False means that the files must have been created before running
        ERT and need an embedded %d.

        The input arguments MIN, MAX, INIT_TRANSFORM and OUTPUT_TRANSFORM are all
        optional.

        MIN and MAX allows you to add a minimum and/or a maximum value with MIN:X and MAX:Y.

        For Assisted history matching, the variables in ERT should be normally
        distributed internally - the purpose of the transformations is to enable
        working with normally distributed variables internally in ERT. Thus, the
        optional arguments ``INIT_TRANSFORM:FUNC`` and ``OUTPUT_TRANSFORM:FUNC`` are used to
        transform the user input of parameter distribution. ``INIT_TRANSFORM:FUNC`` is a
        function which will be applied when the field is loaded into ERT.
        ``OUTPUT_TRANSFORM:FUNC`` is a function which will be applied to the field when it
        is exported from ERT, and ``FUNC`` is the name of a transformation function to be
        applied. The available functions are listed below:

        | "POW10"                       : This function will raise x to the power of 10: :math:`y = 10^x`
        | "TRUNC_POW10" : This function will raise x to the power of 10 - and truncate lower values at 0.001.
        | "LOG"                 : This function will take the NATURAL logarithm of :math:`x: y = \ln{x}`
        | "LN"                  : This function will take the NATURAL logarithm of :math:`x: y = \ln{x}`
        | "LOG10"                       : This function will take the log10 logarithm of :math:`x: y = \log_{10}{x}`
        | "EXP"                 : This function will calculate :math:`y = e^x`.
        | "LN0"                 : This function will calculate :math:`y = \ln{x} + 0.000001`
        | "EXP0"                        : This function will calculate :math:`y = e^x - 0.000001`

        The most common scenario is that underlying log-normal distributed permeability in the
        geo modelling software is transformed to become normally distributed in ERT, to achieve this you do:

        1. ``INIT_TRANSFORM:LOG`` To ensure that the variables which were initially
        log-normal distributed are transformed to normal distribution when they are
        loaded into ERT.

        2. ``OUTPUT_TRANSFORM:EXP`` To ensure that the variables are reexponentiated to be
        log-normal distributed before going out to Eclipse.

        Regarding format of ECLIPSE_FILE: The default format for the parameter fields
        is binary format of the same type as used in the ECLIPSE restart files. This
        requires that the ECLIPSE datafile contains an IMPORT statement. The advantage
        with using a binary format is that the files are smaller, and reading/writing
        is faster than for plain text files. If you give the ECLIPSE_FILE with the
        extension .grdecl (arbitrary case), ERT will produce ordinary .grdecl files,
        which are loaded with an INCLUDE statement. This is probably what most users
        are used to beforehand - but we recommend the IMPORT form. When using RMS APS
        plugin to create Gaussian Random Fields, the recommended file format is ROFF binary.

        *Example A:*

        ::

                -- Use Gaussian Random Fields from APS for zone Volon.
		-- RMS APSGUI plugin will create the files specified in INIT_FILES.
		-- ERT will read the INIT_FILES in iteration 0 and write the updated GRF
		-- fields to the files following the keyword PARAMETER after updating.
		-- NOTE: The ERTBOX grid is a container for GRF values (or perm or poro values) and
		-- is used to define the dimension of the fields. It is NOT the modelling grid
		-- used in RMS or the simulation grid used by ECLIPSE.
                FIELD  aps_Volon_GRF1  PARAMETER  aps_Volon_GRF1.roff  INIT_FILES:rms/output/aps/aps_Volon_GRF1.roff   MIN:-5.5  MAX:5.5  FORWARD_INIT:True
                FIELD  aps_Volon_GRF2  PARAMETER  aps_Volon_GRF2.roff  INIT_FILES:rms/output/aps/aps_Volon_GRF2.roff   MIN:-5.5  MAX:5.5  FORWARD_INIT:True
                FIELD  aps_Volon_GRF3  PARAMETER  aps_Volon_GRF3.roff  INIT_FILES:rms/output/aps/aps_Volon_GRF3.roff   MIN:-5.5  MAX:5.5  FORWARD_INIT:True

        *Example B:*

        ::

                -- Use perm field for zone A
		-- The GRID keyword should refer to the ERTBOX grid defining the size of the field.
		-- Permeability must be sampled from the geomodel/simulation grid zone into the ERTBOX grid
		-- and exported to /some/path/filename. Note that the name of the property in the input file
		-- in INIT_FILES must be the same as the ID.
                FIELD  perm_zone_A   PARAMETER  perm_zone_A.roff  INIT_FILES:/some/path/perm_zone_A.roff     INIT_TRANSFORM:LOG  OUTPUT_TRANSFORM:EXP   MIN:-5.5  MAX:5.5  FORWARD_INIT:True


.. _gen_data:
.. topic:: GEN_DATA

        The ``GEN_DATA`` keyword is used to load text files which have been generated
        by the forward model.

        The GEN_DATA keyword has several options, each of them required:

        * RESULT_FILE - This is the name of the file generated by the forward
          model and read by ERT. This filename _must_ have a %d as part of the
          name, that %d will be replaced by report step when loading.
        * INPUT_FORMAT - The format of the file written by the forward model
          (i.e. RESULT_FILE) and read by ERT, the only valid value is ASCII.
        * REPORT_STEPS - A list of the report step(s) where you expect the
          forward model to create a result file. I.e. if the forward model
          should create a result file for report steps 50 and 100 this setting
          should be: REPORT_STEPS:50,100. If you have observations of this
          GEN_DATA data the RESTART setting of the corresponding
          GENERAL_OBSERVATION must match one of the values given by
          REPORT_STEPS.

        *Example:*

        ::

                GEN_DATA 4DWOC  INPUT_FORMAT:ASCII   RESULT_FILE:SimulatedWOC%d.txt   REPORT_STEPS:10,100

        Here we introduce a GEN_DATA instance with name 4DWOC. When the forward
        model has run it should create two files with name SimulatedWOC10.txt
        and SimulatedWOC100.txt. The result files are in ASCII format, ERT will
        look for these files and load the content. The files should be pure
        numbers - without any header.

        ERT does not have any awareness of the type of data
        encoded in a ``GEN_DATA`` keyword; it could be the result of gravimetric
        calculation or the pressure difference across a barrier in the reservoir. This
        means that the ``GEN_DATA`` keyword is extremely flexible, but also slightly
        complicated to configure. Assume a ``GEN_DATA`` keyword is used to represent the
        result of an estimated position of the oil water contact which should be
        compared with a oil water contact from 4D seismic; this could be achieved with
        the configuration:

        ::

                GEN_DATA 4DWOC  RESULT_FILE:SimulatedWOC_%d.txt  INPUT_FORMAT:ASCII   REPORT_STEPS:0

        The ``4DWOC`` is an arbitrary unique key, ``RESULT_FILE:SimulatedWOC%d.txt``
        means that ERT will look for results in the file ``SimulatedWOC_0.txt``. The
        ``INPUT_FORMAT:ASCII`` means that ERT will expect the result file to be
        formatted as an ASCII file.

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

        Observe that since the actual result file should be generated by the forward
        model, it is not possible for ERT to fully validate the ``GEN_DATA`` keyword
        at configure time. If for instance your forward model generates a file
        ``SimulatedWOC_0`` (without the ``.txt`` extension you have configured), the
        configuration problem will not be detected before ERT eventuallly fails to load
        the file ``SimulatedWOC_0.txt``.


.. _gen_kw:
.. topic:: GEN_KW

        The General Keyword, or :code:`GEN_KW` is meant used for specifying a limited number of parameters.
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
                "ID" : {
                "A" : 0.88,
                },
                "ID:A" : 0.88,
                }


        This can then be used in a forward model, an example from python below:

        .. code-block:: python

            #!/usr/bin/env python
            import json

            if __name__ == "__main__":
                with open("parameters.json", encoding="utf-8") as f:
                    parameters = json.load(f)
                # parameters is a dict with {"ID": {"A": <value>}}



        Note: A file named ``parameters.txt`` is also create which contains the same information,
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

        **Note that ERT only stores values sampled from a standard normal distribution,**
        **and a transformation is performed based on the configuration that is loaded**
        **from file. This means that if the distribution file is changed, the transformed**
        **values written to the run path will be different the next time ERT is started,**
        **even though the underlying value stored by ERT has not changed**

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


        **Loading GEN_KW values from an external file**

        The default use of the GEN_KW keyword is to let the ERT application sample
        random values for the elements in the GEN_KW instance, but it is also possible
        to tell ERT to load a precreated set of data files, this can for instance be
        used as a component in an experimental design based workflow. When using
        external files to initialize the GEN_KW instances you supply an extra keyword
        ``INIT_FILE:/path/to/priors/files%d`` which tells where the prior files are:

        ::

                GEN_KW  MY-FAULTS   MULTFLT.tmpl   MULTFLT.INC   MULTFLT.txt    INIT_FILES:priors/multflt/faults%d

        In the example above you must prepare files ``priors/multflt/faults0``,
        ``priors/multflt/faults1``, ... ``priors/multflt/faultsn`` which ERT
        will load when you initialize the case. The format of the GEN_KW input
        files can be of two varieties:

        1. The files can be plain ASCII text files with a list of numbers:

        ::

                1.25
                2.67

        The numbers will be assigned to parameters in the order found in the
        ``MULTFLT.txt`` file.

        2. Alternatively values and keywords can be interleaved as in:

        ::

                FAULT1 1.25
                FAULT2 2.56

        in this case the ordering can differ in the init files and the parameter file.

        The heritage of the ERT program is based on the EnKF algorithm, and the EnKF
        algorithm evolves around Gaussian variables - internally the GEN_KW variables
        are assumed to be samples from the N(0,1) distribution, and the distributions
        specified in the parameters file are based on transformations starting with a
        N(0,1) distributed variable. The slightly awkward consequence of this is that
        to let your sampled values pass through ERT unmodified you must configure the
        distribution NORMAL 0 1 in the parameter file; alternatively if you do not
        intend to update the GEN_KW variable you can use the distribution RAW.

        **Regarding templates:** You may supply the arguments TEMPLATE:/template/file
        and KEY:MaGiCKEY. The template file is an arbitrary existing text file, and
        KEY is a magic string found in this file. When ERT is running the magic string
        is replaced with parameter data when the ECLIPSE_FILE is written to the
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
.. topic:: SURFACE

        The SURFACE keyword can be used to work with surface from RMS in the irap
        format. The surface keyword is configured like this:

        ::

                SURFACE TOP   OUTPUT_FILE:surf.irap   INIT_FILES:Surfaces/surf%d.irap   BASE_SURFACE:Surfaces/surf0.irap

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
        forward model should contain a job which will copy/convert petro.grdecl ->
        (poro.grdecl,permx.grdecl), this job should not overwrite existing versions of
        permx.grdecl and poro.grdecl. This extra hoops is not strictly needed in all
        cases, but strongly recommended to ensure that you have control over which
        data is used, and that everything is consistent in the case where the forward
        model is run again.


.. _summary:
.. topic:: SUMMARY

        The SUMMARY keyword is used to add variables from the ECLIPSE summary file to
        the parametrization. The keyword expects a string, which should have the
        format VAR:WGRNAME. Here, VAR should be a quantity, such as WOPR, WGOR, RPR or
        GWCT. Moreover, WGRNAME should refer to a well, group or region. If it is a
        field property, such as FOPT, WGRNAME need not be set to FIELD.

        *Example:*

        ::

                -- Using the SUMMARY keyword to add diagnostic variables
                SUMMARY WOPR:MY_WELL
                SUMMARY RPR:8
                SUMMARY F*          -- Use of wildcards requires that you have entered a REFCASE.


        The SUMMARY keyword has limited support for '*' wildcards, if your key
        contains one or more '*' characters all matching variables from the refcase
        are selected. Observe that if your summary key contains wildcards you must
        supply a refcase with the REFCASE key - otherwise only fully expanded keywords will be used.

        **Note:** Properties added using the SUMMARY keyword are only
        diagnostic. I.e. they have no effect on the sensitivity analysis or
        history match.


.. _keywords_controlling_the_es_algorithm:

Keywords controlling the ES algorithm
-------------------------------------


.. _enkf_alpha:
.. topic:: ENKF_ALPHA

        This controls the scaling factor used when detecting outliers. Increasing this
        factor means that more observations will potentially be included in the
        assimilation. The default value is 3.00.

        Including outliers in the Smoother algorithm can dramatically increase the
        coupling between the ensemble members. It is therefore important to filter out
        these outlier data prior to data assimilation. An observation, :math:`\textstyle
        d^o_i`, will be classified as an outlier if

        :math:`|d^o_i - \bar{d}_i| > \mathrm{ENKF\_ALPHA} \left(s_{d_i} + \sigma_{d^o_i}\right)`

        where :math:`\textstyle\boldsymbol{d}^o` is the vector of observed data,
        :math:`\textstyle\boldsymbol{\bar{d}}` is the average of the forecasted data ensemble,
        :math:`\textstyle\boldsymbol{s_{d}}` is the vector of estimated standard deviations
        for the forecasted data ensemble, and :math:`\textstyle\boldsymbol{s_{d}^o}` is the
        vector standard deviations for the observation error (specified a priori).

        Observe that for the updates many settings should be applied on the analysis
        module in question.

.. _enkf_bootstrap:
.. topic:: ENKF_BOOTSTRAP

        Boolean specifying if we want to resample the Kalman gain matrix in the update
        step. The purpose is to avoid that the ensemble covariance collapses. When
        this keyword is true each ensemble member will be updated based on a Kalman
        gain matrix estimated from a resampling with replacement of the full ensemble.

        In theory and in practice this has worked well when one uses a small number of
        ensemble members.


.. _enkf_force_ncomp:
.. topic:: ENKF_FORCE_NCOMP

        Bool specifying if we want to force the subspace dimension we want to use in
        the EnKF updating scheme (SVD-based) to a specific integer. This is an
        alternative to selecting the dimension using ENKF_TRUNCATION.

        *Example:*

        ::

                -- Setting the the subspace dimension to 2
                ENKF_FORCE_NCOMP     TRUE
                ENKF_NCOMP              2


.. _enkf_mode:
.. topic:: ENKF_MODE

        The ENKF_MODE keyword is used to select which EnKF algorithm to use. Use the
        value STANDARD for the original EnKF algorithm, or SQRT for the so-called
        square root scheme. The default value for ENKF_MODE is STANDARD.

        *Example A:*

        ::

                -- Using the square root update
                ENKF_MODE SQRT

        *Example B:*

        ::

                -- Using the standard update
                ENKF_MODE STANDARD

        The ENKF_MODE keyword is optional.


.. _enkf_ncomp:
.. topic:: ENKF_NCOMP

        Integer specifying the subspace dimension. Requires that ENKF_FORCE_NCOMP is
        TRUE.

.. _enkf_truncation:
.. topic:: ENKF_TRUNCATION

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

        The default value of ENKF_TRUNCATION is 0.98. If ensemble collapse is a big
        problem, a smaller value should be used (e.g 0.90 or smaller). However, this
        does not guarantee that the problem of ensemble collapse will disappear. Note
        that setting the truncation factor to 1.00, will recover the Standard-EnKF
        algorithm if and only if the covariance matrix for the observation errors is
        proportional to the identity matrix.

.. _std_cutoff:
.. topic:: STD_CUTOFF

        If the ensemble variation for one particular measurement is below
        this limit the observation will be deactivated. The default value for
        this cutoff is 1e-6.

        Observe that for the updates many settings should be applied on the analysis
        module in question.

.. _update_log_path:
.. topic:: UPDATE_LOG_PATH

        A summary of the data used for updates are stored in this directory.

**References**

* Evensen, G. (2007). "Data Assimilation, the Ensemble Kalman Filter", Springer.
* Mardia, K. V., Kent, J. T. and Bibby, J. M. (1979). "Multivariate Analysis", Academic Press.
* Saetrom, J. and Omre, H. (2010). "Ensemble Kalman filtering with shrinkage regression techniques", Computational Geosciences (online first).


Analysis module
---------------
.. _analysis_module:

The term analysis module refers to the underlying algorithm used for the analysis,
or update step of data assimilation.
The keywords to load, select and modify the analysis modules are documented here.

.. _analysis_select:
.. topic:: ANALYSIS_SELECT

        This command is used to select which analysis module to use in the
        updates:

        ::

                ANALYSIS_SELECT ANAME


.. _analysis_set_var:
.. topic:: ANALYSIS_SET_VAR

        The analysis modules can have internal state, like e.g. truncation cutoff
        values. These can be manipulated from the config file using the
        ANALYSIS_SET_VAR keyword:

        ::

                ANALYSIS_SET_VAR  ANAME  ENKF_TRUNCATION  0.97

        Here `ANAME` must be one of `IES_ENKF` and `STD_ENKF` which are the two
        analysis modules currently available. To use this you must know which
        variables the module supports setting this way. If you try to set an
        unknown variable you will get an error message on stderr.

.. _iter_case:
.. topic:: ITER_CASE


        Case name format - iterated ensemble smoother.
                By default, this value is set to `default_%d`.


.. _iter_count:
.. topic:: ITER_COUNT

        Number of iterations - iterated ensemble smoother.
                Default is 4.


.. _iter_retry_count:
.. topic:: ITER_RETRY_COUNT

        Number of retries for a iteration - iterated ensemble smoother.
                Defaults to 4.


.. _max_submit:
.. topic:: MAX_SUBMIT

        How many times the queue system should retry a simulation.
                Default is 2.


Advanced keywords
--------------------------
.. _advanced_keywords:

The keywords in this section, controls advanced features of ERT. Insight in
the internals of ERT and/or ECLIPSE may
be required to fully understand their effect. Moreover, many of these keywords
are defined in the site configuration, and thus optional to set for the user,
but required when installing ERT at a new site.


.. _time_map:
.. topic:: TIME_MAP

        Normally the mapping between report steps and true dates is inferred by
        ERT indirectly by loading the ECLIPSE summary files. In cases where you
        do not have any ECLIPSE summary files you can use the TIME_MAP keyword
        to specify a file with dates which are used to establish this mapping.
        This is only needed in cases where GEN_OBSERVATION is used with the
        DATE keyword, or cases with SUMMARY observations without REFCASE.

        *Example:*

        ::

                -- Load a list of dates from external file: "time_map.txt"
                TIME_MAP time_map.txt

        The format of the TIME_MAP file should just be a list of dates formatted as
        YYYY-MM-DD. The example file below has four dates:

        ::

                2000-01-01
                2000-07-01
                2001-01-01
                2001-07-01


.. _license_path:
.. topic:: LICENSE_PATH

    A path where ert-licenses to e.g. RMS are stored.




Keywords related to running the forward model
---------------------------------------------
.. _keywords_related_to_running_the_forward_model:



.. _forward_model:
.. topic:: FORWARD_MODEL

        The FORWARD_MODEL keyword is used to define how the simulations are executed.
        E.g., which version of ECLIPSE to use, which rel.perm script to run, which
        rock physics model to use etc. Jobs (i.e. programs and scripts) that are to be
        used in the FORWARD_MODEL keyword must be defined using the INSTALL_JOB
        keyword. A set of default jobs is available, and by default FORWARD_MODEL
        takes the value ECLIPSE100.

        The FORWARD_MODEL keyword expects one keyword defined with INSTALL_JOB.

        *Example:*

        ::

                -- Suppose that "MY_RELPERM_SCRIPT" has been defined with
                -- the INSTALL_JOB keyword. This FORWARD_MODEL will execute
                -- "MY_RELPERM_SCRIPT" before ECLIPSE100.
                FORWARD_MODEL MY_RELPERM_SCRIPT
                FORWARD_MODEL ECLIPSE100

      In available jobs in ERT you can see a list of the jobs which are available.


.. _simulation_job:
.. topic:: SIMULATION_JOB

        ``SIMULATION_JOB`` is a lightweight version of ``FORWARD_MODEL`` that allows passing
        raw command line arguments to executable.
        It is heavily used in Everest as the Everest configuration transpiles all jobs
        into ``SIMULATION_JOB``.

.. _job_script:
.. topic:: JOB_SCRIPT

        Running the forward model from ERT is a multi-level process which can be
        summarized as follows:

        #. A Python module called jobs.py is written and stored in the directory where
           the forward simulation is run. The jobs.py module contains a list of
           job-elements, where each element is a Python representation of the code
           entered when installing the job.
        #. ERT submits a Python script to the enkf queue system, this
           script then loads the jobs.py module to find out which programs to run, and
           how to run them.
        #. The job_script starts and monitors the individual jobs in the jobs.py
           module.

        The JOB_SCRIPT variable should point at the Python script which is managing
        the forward model. This should normally be set in the site wide configuration
        file.

.. _queue_system:
.. topic:: QUEUE_SYSTEM

        The keyword QUEUE_SYSTEM can be used to control where the simulation jobs are
        executed. It can take the values LSF, TORQUE, SLURM and LOCAL.

        ::

                -- Tell ERT to use the LSF cluster.
                QUEUE_SYSTEM LSF

        The QUEUE_SYSTEM keyword is optional, and usually defaults to LSF (this is
        site dependent).

.. _queue_option:
.. topic:: QUEUE_OPTION

        The chosen queue system can be configured further to for instance define the
        resources it is using. The different queues have individual options that are
        configurable.

.. _lsf_list_of_kwds:

Available LSF configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _lsf_server:
.. topic:: LSF_SERVER

        By using the LSF_SERVER option you essentially tell ERT two things about how
        jobs should be submitted to LSF:

        #. You tell ERT that jobs should be submitted using shell commands.
        #. You tell ERT which server should be used when submitting.

        So when your configuration file has the setting:

        ::

                QUEUE_OPTION LSF LSF_SERVER   be-grid01

        ERT will use ssh to submit your jobs using shell commands on the server
        be-grid01. For this to work you must have passwordless ssh to the server
        be-grid01. If you give the special server name LOCAL ERT will submit using
        shell commands on the current workstation.

        **bsub/bjobs/bkill options**

        By default ERT will use the shell commands bsub, bjobs and bkill to interact
        with the queue system, i.e. whatever binaries are first in your PATH will be
        used. For fine grained control of the shell based submission you can tell ERT
        which programs to use:

        ::

                QUEUE_OPTION   LSF  BJOBS_CMD  /path/to/my/bjobs
                QUEUE_OPTION   LSF  BSUB_CMD   /path/to/my/bsub

        *Example 1*

        ::

                LSF_SERVER    be-grid01
                QUEUE_OPTION  LSF     BJOBS_CMD   /path/to/my/bjobs
                QUEUE_OPTION  LSF     BSUB_CMD    /path/to/my/bsub

        In this example we tell ERT to submit jobs from the workstation be-grid01
        using custom binaries for bsub and bjobs.

.. _lsf_queue:
.. topic:: LSF_QUEUE

        ::

                QUEUE_OPTION LSF LSF_QUEUE name_of_queue

        The name of the LSF queue you are running simulations in.
        For example, ``bsub``, this option will be passed to the ``-q`` parameter:
        https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_command_ref/bsub.q.1.html

.. _lsf_resource:
.. topic:: LSF_RESOURCE

        ::

                QUEUE_OPTION LSF LSF_RESOURCE resource_string

        From https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_admin/res_req_strings_about.html:

        Most LSF commands accept a -R res_req argument to specify resource
        requirements. The exact behavior depends on the command. For
        example, specifying a resource requirement for the lsload command
        displays the load levels for all hosts that have the requested resources.

        Specifying resource requirements for the lsrun command causes LSF to
        select the best host out of the set of hosts that have the requested
        resources.

        A resource requirement string describes the resources that a job needs.
        LSF uses resource requirements to select hosts for remote execution and
        job execution.

        Resource requirement strings can be simple (applying to the entire job)
        or compound (applying to the specified number of slots).

.. _lsf_rsh_cmd:
.. topic:: LSF_RSH_CMD

        ::

                QUEUE_OPTION LSF LSF_RSH_CMD name_of_queue

        This option sets the *remote shell* command, which defaults to ``/usr/bin/ssh``.

.. _lsf_login_shell:
.. topic:: LSF_LOGIN_SHELL

        ::

                QUEUE_OPTION LSF LSF_LOGIN_SHELL name_of_queue

        Equates to the ``-L`` parameter of e.g. ``bsub``:
        https://www.ibm.com/support/knowledgecenter/en/SSWRJV_10.1.0/lsf_command_ref/bsub.__l.1.html
        Useful if you need to force the ``bsub`` command to use e.g. ``/bin/csh``.

.. _bsub_cmd:
.. topic:: BSUB_CMD

        The ``bsub`` command. Default: ``bsub``.

        ::

                QUEUE_OPTION LSF BSUB_CMD command

.. _bjobs_cmd:
.. topic:: BJOBS_CMD

        The ``bjobs`` command. Default: ``bjobs``.

        ::

                QUEUE_OPTION LSF BJOBS_CMD command


.. _bkill_cmd:
.. topic:: BKILL_CMD

        The ``bkill`` command. Default: ``bkill``.

        ::

                QUEUE_OPTION LSF BKILL_CMD command


.. _bhist_cmd:
.. topic:: BHIST_CMD

        The ``bhist`` command. Default: ``bhist``.

        ::

                QUEUE_OPTION LSF BHIST_CMD command


.. _bjobs_timeout:
.. topic:: BJOBS_TIMEOUT

        Determines how long-lived the job cache is. Default: ``0`` (i.e. no cache).

        ::

                QUEUE_OPTION LSF BJOBS_TIMEOUT 0


.. _submit_sleep:
.. topic:: SUBMIT_SLEEP

        Determines for how long the system will sleep between submitting jobs.
        Defaults to 0.

        ::

                QUEUE_OPTION LSF SUBMIT_SLEEP 5


.. _project_code:
.. topic:: PROJECT_CODE

        Equates to the ``-P`` parameter for e.g. ``bsub``. See https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_command_ref/bsub.__p.1.html

        ::

                QUEUE_OPTION LSF PROJECT_CODE command


.. _exclude_host:
.. topic:: EXCLUDE_HOST

        Comma separated list of hosts to be excluded. The LSF system will pass this
        list of hosts to the ``-R`` argument of e.g. ``bsub`` with the criteria
        ``hname!=<exluded_host_1>``.

        ::

                QUEUE_OPTION LSF EXCLUDE_HOST host1,host2


.. _lsf_max_running:
.. topic:: MAX_RUNNING

        The queue option MAX_RUNNING controls the maximum number of simultaneous jobs
        submitted to the queue when using (in this case) the LSF option in
        QUEUE_SYSTEM.

        ::

                QUEUE_SYSTEM LSF
                -- Submit no more than 30 simultaneous jobs
                -- to the TORQUE cluster.
                QUEUE_OPTION LSF MAX_RUNNING 30


.. _torque_list_of_kwds:

Available TORQUE configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _torque_sub_stat_del_cmd:
.. topic:: QSUB_CMD, QSTAT_CMD, QDEL_CMD

        By default ERT will use the shell commands qsub, qstat and qdel to interact with
        the queue system, i.e. whatever binaries are first in your PATH will be used.
        For fine grained control of the shell based submission you can tell ERT which
        programs to use:

        ::

                QUEUE_SYSTEM TORQUE
                QUEUE_OPTION TORQUE QSUB_CMD /path/to/my/qsub
                QUEUE_OPTION TORQUE QSTAT_CMD /path/to/my/qstat
                QUEUE_OPTION TORQUE QDEL_CMD /path/to/my/qdel

In this example we tell ERT to submit jobs using custom binaries for bsub and
bjobs.


.. _torque_qstat_options:
.. topic:: QSTAT_OPTIONS

        Options to be supplied to the qstat command. This is defaulted to :code:`-x`,
        which would tell the qstat command to include exited processes.

.. _torque_queue:
.. topic:: QUEUE

        The name of the TORQUE queue you are running simulations in.

        ::

                QUEUE_OPTION TORQUE QUEUE name_of_queue

.. _torque_cluster_label:
.. topic:: CLUSTER_LABEL

        The name of the TORQUE cluster you are running simulations in. This
        might be a label (serveral clusters), or a single one, as in this example baloo.

        ::

                QUEUE_OPTION TORQUE CLUSTER_LABEL baloo

.. _torque_max_running:
.. topic:: MAX_RUNNING

        The queue option MAX_RUNNING controls the maximum number of simultaneous jobs
        submitted to the queue when using (in this case) the TORQUE option in
        QUEUE_SYSTEM.

        ::

                QUEUE_SYSTEM TORQUE
                -- Submit no more than 30 simultaneous jobs
                -- to the TORQUE cluster.
                QUEUE_OPTION TORQUE MAX_RUNNING 30


.. _torque_nodes_cpus:
.. topic:: NUM_NODES, NUM_CPUS_PER_NODE

        When using TORQUE, you can specify how many nodes a single job should
        use, and how many CPUs per node. The default setup in ERT will use one node and
        one CPU. These options are called NUM_NODES and NUM_CPUS_PER_NODE.

        If the numbers specified is higher than supported by the cluster (i.e. use 32
        CPUs, but no node has more than 16), the job will not start.

        If you wish to increase this number, the program running (typically ECLIPSE)
        will usually also have to be told to correspondingly use more processing units
        (keyword PARALLEL)

        ::

                QUEUE_SYSTEM TORQUE
                -- Use more nodes and CPUs
                -- in the TORQUE cluster per job submitted
                -- This should (in theory) allow for 24 processing
                -- units to be used by eg. ECLIPSE
                QUEUE_OPTION TORQUE NUM_NODES 3
                QUEUE_OPTION TORQUE NUM_CPUS_PER_NODE 8

.. _torque_memory_per_job:
.. topic:: MEMORY_PER_JOB


        You can specify the amount of memory you will need for running your
        job. This will ensure that not too many jobs will run on a single
        shared memory node at once, possibly crashing the compute node if it
        goes out of memory.

        You can get an indication of the memory requirement by watching the
        course of a local run using the `htop` utility. Whether you should set
        the peak memory usage as your requirement or a lower figure depends on
        how simultaneously each job will run.

        The option to be supplied will be used as a string in the `qsub`
        argument, meaning you must specify the unit, either `gb` or `mb` as in
        the example:

        By default, this value is not set.

        ::

                QUEUE_OPTION TORQUE MEMORY_PER_JOB 16gb

.. _torque_keep_qsub_output:
.. topic:: KEEP_QSUB_OUTPUT

        Sometimes the error messages from qsub can be useful, if something is seriously
        wrong with the environment or setup. To keep this output (stored in your home
        folder), use this:

        ::

                QUEUE_OPTION TORQUE KEEP_QSUB_OUTPUT 1


.. _torque_submit_sleep:
.. topic:: SUBMIT_SLEEP

        To be more gentle with the TORQUE system you can instruct the driver to sleep
        for every submit request. The argument to the SUBMIT_SLEEP is the number of
        seconds to sleep for every submit, which can be a fraction like 0.5.

        ::

                QUEUE_OPTION TORQUE SUBMIT_SLEEP 0.5

.. _torque_queue_query_timeout:
.. topic:: QUEUE_QUERY_TIMEOUT

        The driver allows the backend Torque system to be flaky, i.e. it may
        intermittently not respond and give error messages when submitting jobs
        or asking for job statuses. The timeout (in seconds) determines how long
        ERT will wait before it will give up. Applies to job submission (qsub)
        and job status queries (qstat). Default is 126 seconds.

        ERT will do exponential sleeps, starting at 2 seconds, and the provided
        timeout is a maximum. Let the timeout be sums of series like 2+4+8+16+32+64
        in order to be explicit about the number of retries. Set to zero to disallow
        flakyness, setting it to 2 will allow for one re-attempt, and 6 will give two
        re-attempts.

        ::

                QUEUE_OPTION TORQUE QUEUE_QUERY_TIMEOUT 254

.. _configuring_the_slurm_queue:

Configuring the SLURM queue
---------------------------

        The slurm queue managing tool has a very fine grained control. In ERT only the options that
        are the most necessary have been added.

.. _slurm_sbatch:
.. topic:: SBATCH

        Command used to submit the jobs.

        ::

                QUEUE_OPTION SLURM SBATCH


.. _slurm_scancel:
.. topic:: SCANCEL

        Command used to cancel the jobs.

        ::

                QUEUE_OPTION SLURM SCANCEL


.. _slurm_scontrol:
.. topic:: SCONTROL

        Command to modify configuration and state

        ::

                QUEUE_OPTION SLURM SCONTROL


.. _slurm_squeue:
.. topic:: SQUEUE

        Command to view information about the queue

        ::

                QUEUE_OPTION SLURM SQUEUE


.. _slurm_partition:
.. topic:: PARTITION

        Partition/queue in which to run the jobs

        ::

                QUEUE_OPTION SLURM PARTITION


.. _slurm_squeue_timeout:
.. topic:: SQUEUE_TIMEOUT

        Specify timeout used when querying for status of the jobs while running.

        ::

                QUEUE_OPTION SLURM SQUEUE_TIMEOUT 10

.. _slurm_smax_runtime:
.. topic:: MAX_RUNTIME

        Specify the maximum runtime (in seconds) for how long a job can run.

        ::

                QUEUE_OPTION SLURM MAX_RUNTIME 100

.. _slurm_memory:
.. topic:: MEMORY

        Memory required per node (MB).
        ::

                QUEUE_OPTION SLURM MEMORY 16000

.. _slurm_memory_per_cpu:
.. topic:: MEMORY_PER_CPU (MB).


        Memory required per allocated CPU
        ::

                QUEUE_OPTION SLURM MEMORY_PER_CPU 4000

.. _slurm_include_host:
.. topic:: INCLUDE_HOST

        Specific host names to use when running the jobs. It is possible to add multiple
        hosts separated by space or comma in one option call

        ::

                QUEUE_OPTION SLURM INCLUDE_HOST host1,host2

.. _slurm_exclude_host:
.. topic:: EXCLUDE_HOST

        Specific host names to exclude when running the jobs. It is possible to add multiple
        hosts separated by space or comma in one option call

        ::

                QUEUE_OPTION SLURM EXCLUDE_HOST host3,host4


.. _max_running_slurm:
.. topic:: MAX_RUNNING

        The queue option keyword MAX_RUNNING controls the maximum number of simultaneous
        jobs running when (in this case) using the SLURM option in QUEUE_SYSTEM.

        *Example:*

        ::

                QUEUE_SYSTEM SLURM
                -- No more than 10 simultaneous jobs
                -- running via SLURM.
                QUEUE_OPTION SLURM MAX_RUNNING 10

Workflow hooks
----------------------------

.. _hook_workflow:
.. topic:: HOOK_WORKFLOW

        With the keyword :code:`HOOK_WORKFLOW` you can configure workflow
        'hooks'; meaning workflows which will be run automatically at certain
        points during ERTs execution. Currently there are five points in ERTs
        flow of execution where you can hook in a workflow:

        - Before the simulations (all forward models for a realization) start using :code:`PRE_SIMULATION`,
        - after all the simulations have completed using :code:`POST_SIMULATION`,
        - before the update step using :code:`PRE_UPDATE`
        - after the update step using :code:`POST_UPDATE` and
        - only before the first update using :code:`PRE_FIRST_UPDATE`.

        For non interactive algorithms, :code:`PRE_FIRST_UPDATE` is equal to :code:`PRE_UPDATE`.
        The :code:`POST_SIMULATION` hook is typically used to trigger QC workflows.

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
        :code:`HOOK_WORKFLOW` must be loaded with the :code:`LOAD_WORKFLOW` keyword.

.. _load_workflow:
.. topic:: LOAD_WORKFLOW

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
.. topic:: LOAD_WORKFLOW_JOB

        Before the jobs can be used in workflows they must be "loaded" into
        ERT. This can be done either by specifying jobs by name,
        or by specifying a directory containing jobs.

        Use the keyword :code:`LOAD_WORKFLOW_JOB` to specify jobs by name:

        ::

            LOAD_WORKFLOW_JOB     jobConfigFile     JobName

        The :code:`LOAD_WORKFLOW_JOB` keyword will load one workflow job.
        The name of the job is optional, and will be fetched from the configuration file if not provided.

.. _workflow_job_directory:
.. topic:: WORKFLOW_JOB_DIRECTORY

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

Manipulating the Unix environment
---------------------------------

.. _setenv:
.. topic:: SETENV

        You can use the SETENV keyword to alter the unix environment where ERT runs
        forward models.

        *Example:*

        ::

                -- Setting up LSF
                SETENV  MY_VAR          World
                SETENV  MY_OTHER_VAR    Hello$MY_VAR

        This will result in two environment variables being set in the compute side
        and available to all jobs. MY_VAR will be "World", and MY_OTHER_VAR will be
        "HelloWorld". The variables are expanded in order on the compute side, so
        the environment where ERT is running has no impact, and is not changed.
