Release Notes
=============


.. Release notes template
 Version <MAJOR.MINOR>
 ------------

 Breaking changes:
   -

 Bugfixes:
   -

 New features:
   -

 Improvements:
   -

 Deprecations:
   -

 Dependencies:
   -

 Miscellaneous:
   -

Version 2.25
------------

Bugfixes:
  - Fix initial ensemble state
  - Fix flaky legacy ensemble test by making job queue always launch jobs (`#1794 <https://github.com/equinor/ert/issues/1794>`_)
  - Fix GUI crash (`#457 <https://github.com/equinor/ert/issues/457>`_)
  - Fix bug where the last summary obs was not loaded (`#1813 <https://github.com/equinor/ert/issues/1813>`_)
  - Fix bug in progress calculation (`#1830 <https://github.com/equinor/ert/issues/1830>`_)

New features:
  - Make it possible to visualise ERT3 results
  - Check the status of ert3 services
  - Support space and comma in forward model job arguments (`#1472 <https://github.com/equinor/ert/issues/1472>`_)

Improvements:
  - Rename: userdata and ensemble_ids endpoints
  - Add section on how to restart ES-MDA in GUI (`#1290 <https://github.com/equinor/ert/issues/1290>`_)
  - Generate narratives on the fly when building the docs
  - Support building the documenation on ReadTheDocs (`#1610 <https://github.com/equinor/ert/issues/1610>`_)
  - ert2 use servermonitor for fetching ert-storage auth
  - Add ert-storage, clean experiment and webviz-ert to spe1-README (`#1736 <https://github.com/equinor/ert/issues/1736>`_)
  - Fix typo in RMS documantion and CLI (`#1438 <https://github.com/equinor/ert/issues/1438>`_)
  - Add output records to ert3 ensemble config
  - Implement time out for legacy evaluator
  - Fix resolve socket family (INET/INET6) (`#1676 <https://github.com/equinor/ert/issues/1676>`_)

Deprecations:
  - Deprecate PORTABLE_EXE (`#1718 <https://github.com/equinor/ert/issues/1718>`_)

Dependencies:
  - Add type stub packages to types-requirements
  - Require semeio>=1.1.3rc0
  - Remove ert-storage from dev-requirements.txt
  - Add scikit-build as a dev-requirement
  - Add setuptools_scm as a dev-requirement

Miscellaneous:
  - Move all the libres code into the ert repository
  - ert3: Introduce common RecordIndex
  - Add integration tests for post_update_data (`#1671 <https://github.com/equinor/ert/issues/1671>`_)
  - Add some temporary debuging of events
  - Make cancel test more consistent (`#1755 <https://github.com/equinor/ert/issues/1755>`_)
  - Fix flaky prefect retry test by ignoring order of events in the test (`#1730 <https://github.com/equinor/ert/issues/1730>`_)
  - Use example servers in comments
  - Ignore some numpy type annotions that are difficult to handle in python
  - Added Docs Section In README.md
  - Split ERT 3 parameters into separate records
  - Have mypy ignore missing numpy imports
  - Make deploy to PyPi also depend on ctests
  - Change workflows to not trigger on all push events (`#1739 <https://github.com/equinor/ert/issues/1739>`_)
  - Reduce the number of macos builds on GA (`#1756 <https://github.com/equinor/ert/issues/1756>`_)
  - Run ert2 test data setup on python 3.8 instead of 3.7
  - Clone with tags in style and typing workflows
  - Merge ert and libres test-data
  - Delete libres test-data
  - Remove init from tests (`#1734 <https://github.com/equinor/ert/issues/1734>`_)
  - Remove use of temp test folder in GA
  - Unite the resolving of port and sockets to use (`#1676 <https://github.com/equinor/ert/issues/1676>`_)
  - Make verification tests wait for the ensemble evaluator to finish (`#1819 <https://github.com/equinor/ert/issues/1819>`_)

Version 2.24
------------

Bugfixes:
  - ert3: Fix wrongly typed distribution index
  - Fix bug in prefect ensemble error handling
  - Fix retry connection in evaluator tracker
  - Fix rounding error in realization progress visualisation (`#1672 <https://github.com/equinor/ert/issues/1672>`_)
  - Re-add `stderr` and `stdout` info
  - Re-enable retries for Tasks
  - Add timeout when waiting for cancelled dispatchers
  - Use isoformat for timestamps when converting to str (`#1637 <https://github.com/equinor/ert/issues/1637>`_)
  - Pass `ee_id` to `execute_queue_async`
  - Fix running event loop in gui sim thread
  - Fix cancelling of ensemble hanging for ever
  - Handle dns operation timeout (`#1625 <https://github.com/equinor/ert/issues/1625>`_)
  - Fix JOB_QUEUE_DO_KILL_NODE_FAILURE spelling error
  - Update detailed progress after failure (`#1658 <https://github.com/equinor/ert/issues/1658>`_)

New features:
  - Add support for Python 3.9
  - Add callback function for catching MAX_RUNTIME failure (`#1525 <https://github.com/equinor/ert/issues/1525>`_)
  - Add method keys() to LocalDataset

Improvements:
  - Pass ert3 records to storage as numerical data
  - Define, test and document communication protocols in EE (`#1235 <https://github.com/equinor/ert/issues/1235>`_)
  - Connection error handling in EvaluatorTracker (`#1679 <https://github.com/equinor/ert/issues/1679>`_)
  - Batch all event types
  - Order real status according to state transitions
  - ert3: Ensure immutable stages config
  - ert3: Ensure immutable ensemble config
  - ert3: Ensure immutable experiment config
  - ert3: Introduce parameters config
  - ert3: Make distributions expose their arguments
  - Fix usage of .closed in socket code (`#1600 <https://github.com/equinor/ert/issues/1600>`_)
  - ERT 3: Feature/step type class
  - Use the prefect context to pass url, token and certification info
  - Force x_axis values to be strings before json serialization
  - Extract priors to new storage
  - Add certificates and tokens to websocket communication (`#1326 <https://github.com/equinor/ert/issues/1326>`_)
  - Add batching of events in ensemble evaluator (`#1683 <https://github.com/equinor/ert/issues/1683>`_)
  - Log evaluator cross talk (`#1647 <https://github.com/equinor/ert/pull/1647>`_)
  - Remove ensemble evaluator feature flag warning
  - Use `phaseCount` in progress calculation, drop phase (`#1635 <https://github.com/equinor/ert/issues/1635>`_)
  - Timeout CI pipeline after 30 mins
  - Refactor evaluator utils
  - Refactor `update_step` to refer to step entities
  - Refactor and extend testing of PartialSnapshot
  - Remove size cap on ensemble evaluator msg queue
  - Add 'ensemble_size' as param to 'post_ensemble_data'
  - Add record class 'response' to extracted responses
  - Add test to make sure total progress is updated (`#1608 <https://github.com/equinor/ert/issues/1608>`_)
  - Keep tracker iteration snapshot up to date
  - Use ERT Storage's TestClient
  - Improve cancellation of ensembles
  - Add token and certificates to websocket communication
  - Re-add stderr and stdout info
  - Use public MonkeyPatch
  - Re-use Client for dispatch lifecycle
  - Add --summary-conversion flag with default no to eclipse run
  - Require `ee_id` in `execute_queue_async`
  - Allow certificate to be `None`

Dependencies:
  - Upgrade to websockets 9 (`#1615 <https://github.com/equinor/ert/issues/1615>`_)
  - Depend on `dnspython>=2` and `pydantic>=1.8.1`

Miscellaneous:
  - Cleanup exceptions in ert3.storage
  - Introduce SyncWebsocketDuplexer (`#1538 <https://github.com/equinor/ert/issues/1538>`_)
  - Refactor: Remove handlers from evaluator
  - Pin pytest-qt<4
  - Add integration tests for post_ensemble_data (`#1669 <https://github.com/equinor/ert/issues/1669>`_)
  - ert3: Use public interface when testing ensemble config
  - Add classifiers to setup.py
  - Cleanup prefect ensemble test for function defined outside the python environment
  - Add logging to development strategy
  - Add type hinting to make mypy pass on ert3 in strict mode

Version 2.23
------------

Bugfixes:
  - Fix 1307 by removing the signaller from the pool (`#1307 <https://github.com/equinor/ert/issues/1307>`_)
  - Fix extraction bug when no observations
  - Fix function ensemble run on PBS cluster by cloudpickling function (`#1505 <https://github.com/equinor/ert/issues/1505>`_)

New features:
  - Use experiment in ert-storage for ert3 (`#1554 <https://github.com/equinor/ert/issues/1554>`_)

Improvements:
  - Stop using ArgParse FileType (`#1500 <https://github.com/equinor/ert/issues/1500>`_)
  - Remove and stop using the nfs_adaptor for status messages (`#1344 <https://github.com/equinor/ert/issues/1344>`_)
  - Make legacy ensemble members connect through websocket
  - Updates related to row scaling
  - Adapt websockets events to new event model
  - Introduce RecordTransmitter (`#1334 <https://github.com/equinor/ert/issues/1334>`_, `#1447 <https://github.com/equinor/ert/issues/1447>`_, `#1328 <https://github.com/equinor/ert/issues/1328>`_, `#1502 <https://github.com/equinor/ert/issues/1502>`_)
  - Make ert3 ensemble config forward model point to a single stage (`#1553 <https://github.com/equinor/ert/issues/1553>`_)
  - Move data extraction to new ert-storage rest api (`#1544 <https://github.com/equinor/ert/issues/1544>`_)
  - Extract priors to new storage
  - Force x_axis values to be strings before json serialization
  - Fix x axis to str, when posting update data
  - Update detailed progress after failure (`#1658 <https://github.com/equinor/ert/issues/1658>`_)
  - Use snapshot instead of run_context (`#1658 <https://github.com/equinor/ert/issues/1658>`_)

Miscellaneous:
  - Refactor Realization/Stages/Steps (`#1220 <https://github.com/equinor/ert/issues/1220>`_)
  - Drop broken ertplot script (`#547 <https://github.com/equinor/ert/issues/547>`_)
  - Add development strategy
  - Remove new legacy storage db (`#1544 <https://github.com/equinor/ert/issues/1544>`_)
  - Advertise ert_shared entry point as ert (`#418 <https://github.com/equinor/ert/issues/418>`_)
  - Add function step tests for function defined outside python environment (`#1556 <https://github.com/equinor/ert/issues/1556>`_)

Version 2.22
------------

Bugfixes:
  - Fix wrong use of STDERR identifier

New features:
  - Add ert3 status command (`#1457 <https://github.com/equinor/ert/issues/1457>`_)
  - Add ert3 clean sub-command
  - Use the new storage to fetch/store results running ert3
  - Add possibility to initialise examples from cli for ert3
  - Add forward model function step
  - Make TEMPLATE_RENDER support parameters.json not being present

Improvements:
  - Revert "Add retry to ert3"
  - Look up correct stage in stages_config
  - Remove redundant engine code
  - Have ert3.engine stop closing passed streams (`#1498 <https://github.com/equinor/ert/issues/1498>`_)
  - Use forkserver as strategy with multiprocessing
  - Ensure fresh loop in prefect ensemble to fix the Bad file descriptor
  - Make observation active mask not a list
  - Reintroduce rendering of job statuses
  - Set prefect log level to WARNING when running

Dependencies:
  - Add ert-storage as extras to setup.py

Miscellaneous:
  - Reorder ert3 submodule imports
  - Use conditionals to import Literal
  - Add __all__ to ert3.data
  - Introduce type checking for ert3 in CI
  - Add type hints to ert3.storage
  - Run strict type checking for ert3.storage
  - Add type hints to ert3.engine
  - Add pylintrc
  - Run pylint for ert3 as part of style CI workflow
  - Replace usage of ert3 examples folder with generated test data in ert3
  - Remove used of example folder in ert3 evaluator tests
  - Remove used of example folder in ert3 cli tests
  - Fix flake8 errors
  - Remove used of example folder in ert3 stages config tests
  - More removal of examples folder reference from ert3 cli tests
  - Remove unused imports
  - Enable pylint error: unused-import
  - Reposition imports
  - Enable pylint error: wrong-import-position
  - Reorder imports
  - Enable pylint error: wrong-import-order
  - Enable pylint error: ungrouped-imports
  - Improve reporting of validation errors
  - Refactor UnixStep
  - Replacing magic strings
  - Remove faulty col resize and add tooltip
  - Improve storage development workflow
  - Add tests for the prefect ensemble
  - Use pydantic to instantiate all dicts
  - Move conftest out to tests/gui
  - Improve ensemble client and add new tests
  - Set timeout of all jobs to 15 minutes
  - Increase build timeout to 30 minutes
  - Drop SPE1 example templating hack
  - Fix examples/polynomial Github Actions workflow

Version 2.21
------------

Bugfixes:
  - Set correct phase count in ESMDA model
  - Prevent double startup of storage server
  - Seperate Update creation from ensemble creation and link observation transformation
  - Don't assume singular snapshot in CLI. Fixes a problem where ERT would crash on iiteration 1 if a realization failed in iteration 0.

New features:
  - Add obs_location property to misfits and corresponding test (`#1373 <https://github.com/equinor/ert/issues/1373>`_)
  - Implement oat sensitivity algorithm
  - Add ert3 support for sensitivity studies
  - Apply row scaling in the smoother update

Improvements:
  - Add failure event and use it in the legacy and prefect ensembles (`#1301 <https://github.com/equinor/ert/issues/1301>`_)
  - Push ensembles separarate from responses to new storage
  - Use numpy_vector instead of deprecated method
  - Fix snake_oil_field ert config
  - Remove the prefect option from ert
  - Remove coefficient generation in Prefect Ensemble
  - Use LocalDaskCluster to run local ensembles
  - Add index and ppf to distributions
  - Introduce experiment folder in workspace
  - Add uniform polynomial experiments
  - Add ert3 reservoir example based on SPE1 and flow
  - Refactor of Qt Graphical User Interface (`#566 <https://github.com/equinor/ert/issues/566>`_)
  - Add error if not response could be loaded for `MeasuredData`
  - Introduce record data types
  - Add retry to the ert3 evaluator
  - Check queue hash when updating in LegacyTracker
  - Use forkserver as strategy with multiprocessing
  - Always use queue from map in LegacyTracker (`#1476 <https://github.com/equinor/ert/issues/1476>`_)
  - Check if partial_snapshot is None before merging
  - Reintroduce rendering of job statuses
  - `JobQueue.snapshot` provide the user with a snapshot of the queue state that can be queried independently of `run_context` and `run_arg`
  - `JobQueue.execute_queue_async` and `JobQueue.stop_jobs_async` provides asynchronous execution and stopping of queue
  - Remove fs dependency for summary observation collector
  - Force sequential execution of callbacks
  - Export shared rng to Python

Deprecations:
  - Depecate loading functions

Miscellaneous:
  - Turn monitor into a context manager (`#1332 <https://github.com/equinor/ert/issues/1332>`_)
  - Load config close to ensemble evaluator evaluation
  - Refactor data loading
  - Refactor plot api
  - Black plot api
  - Run test-data as a part of CI
  - Change patch import in ert3 cli test
  - Add base distribution
  - Fix Literal imports
  - Run polynomial demo during CI
  - Remove trailing whitespace
  - Break before binary operators
  - Make lambda's into def's
  - Run pylint during CI
  - Create CODE_OF_CONDUCT.md (`#1414 <https://github.com/equinor/ert/issues/1414>`_)
  - Add black badge to README
  - Run black on everything in CI
  - Format all files
  - Update badges
  - Move flake8 settings into .flake8 config
  - Fix test that was testing a (now fixed) bug in `libres`
  - Run flake on tests/ert3 during style testing
  - Stop using single character variable names in tests
  - Stop storing unused return values in tests
  - Fix deprecated escape characters
  - Drop support for variables, input and ouput data in storage
  - Pass data as records in ert3
  - Move conftest out to tests/gui
  - Keep ensemble config nodes in an ordered data structure to avoid sampling differences over different build machines
  - Write all elements in grdecl test data

Version 2.20
------------

Bugfixes:
  - Fix for default tabs selection (`#1282 <https://github.com/equinor/ert/issues/1282>`_)

New features:
  - Run eclipse using eclrun when available
  - Add row scaling API
  - Fist working iteration of prefect evaluator (`#1125 <https://github.com/equinor/ert/issues/1125>`_)
  - Introduce ert3 (`#1311 <https://github.com/equinor/ert/issues/1311>`_)

Improvements:
  - Disable logging in forward models and workflows
  - Unify code paths with and without monitoring
  - Graceful exit if storage_server.json exists
  - Clarify how rel paths in workflow are resolved
  - Return empty list in create_observations when no obs are available
  - Add --host setting to ert api
  - Storage: Allow NaNs to be returned (`#1284 <https://github.com/equinor/ert/issues/1284>`_)
  - Storage: Move database to /tmp while using it (`#1309 <https://github.com/equinor/ert/issues/1309>`_)
  - Make evaluator input files configurable
  - Avoid unnecessary stack trace during `ert vis` (`#1306 <https://github.com/equinor/ert/issues/1306>`_)
  - Storage: Combine ParameterDefinition & Parameter

Dependencies:
  - Add `pyrsistent` (`#1376 <https://github.com/equinor/ert/issues/1376>`_)

Miscellaneous:
  - Remove builtin and __future__ imports
  - Fix wrong tests folder in Github Actions
  - Introduce exceptions module for ert3 workspace errors
  - Create CONTRIBUTING.md
  - Refactor Storage API Server (`#1102 <https://github.com/equinor/ert/issues/1102>`_, `#1116 <https://github.com/equinor/ert/issues/1116>`_)
  - Fix extraction.py's create_update
  - Correct spelling of modelling to British variant
  - Copy examples when running Jenkins CI
  - Run flake8 on the ert3 module as part of CI


Version 2.19
------------

Highlighted changes
~~~~~~~~~~~~~~~~~~~

ERT is now pip-installable
##########################
ERT can now be installed via PyPI by running

.. code-block:: none

   >>>> pip install ert

2.19.0 ERT
~~~~~~~~~~ 
Improvements:
  - Improve observation format documentation
  - Fix plotting warnings
  - Introduce sub categories to job documentation section
  - Remove legacy logo
  - Improve documentation of installed workflows
  - Various improvements to the new (but for now optional) data storage 
  - Various improvements to the new (but for now optional) internal workflow manager

Miscellaneous:
  - Remove upper limit on matplotlib version
  - Use the Qt5 backend

8.0.0 libres
~~~~~~~~~~~~
Improvements:
  - pip installable libres
  - Catch version import error
  - Rename all shell workflows to uppercase
  - Improve RMS forward model documentation

Miscellaneous:
  - Remove unused EnKF update checks
  - Move tests to ease running them
  - Remove legacy logo
  - Update requirement list

Version 2.16
------------

Highlighted changes
~~~~~~~~~~~~~~~~~~~

Isolated RMS environment
########################

We recommend all users to remove ``RMS_PYTHONPATH`` from their
ERT configuration files when using ERT 2.16 or newer. Users can experience
problems with not having access to Python modules they earlier had access to
within RMS. If so, contact your ERT admins to evaluate the problem.

For Equinor users we have installed a `run_external` command in the RMS
environment that allows our users to reconstruct the environment prior to
launching RMS to allow for executing commands within the standard user
environment from RMS.

2.16.0 ERT
~~~~~~~~~~ 
New features:
  - Make it possible to run the IES via the command line interface
  - New workflow hook ``PRE_FIRST_UPDATE_HOOK``

Improvements:
  - Improvements to the documentation
  - Use gunicorn instead of werkzeug for data server
  - Authenticate towards data server
  - Have job_dispatch propagate events to prepare for a new ensemble evaluator
  - Have the RMS-job keep track of default Python environment

7.0.0 libres
~~~~~~~~~~~~
See ERT release notes

0.6.4 semeio
~~~~~~~~~~~~
New features:
  - Make data export from overburden_timeshift optional
  - Add all forward model jobs as command line tools
  - Extract saturations from RFT files

Bug fixes:
  - Make CSV_EXPORT2 robust towards empty parameters.txt
  - Disallow whitespaces in parameter names
  - Update summary data when running ``MISFIT_PREPROCESSOR``
  - Install the STEA job correctly
  - design2param forbids invalid parameter names ``ENSEMBLE``, ``DATE`` and ``REAL``

Version 2.15
------------

Highlighted changes
~~~~~~~~~~~~~~~~~~~

Python3.6-only
##############

This version of ERT is now incompatible with Python version less than 3.6.

2.15.0 ERT
~~~~~~~~~~~~
New features:
  - Replace Data export button functionality with a CSV-export
  - Add file operation jobs as workflow jobs

Improvements:
  - Document magic strings
  - Clean up documentation with respect to outdated keywords

Miscellaneous:
  - Deprecate workflow ``STD_SCALE_CORRELATED_OBS``. Recommended to use ``MISFIT_PREPROCESSOR`` instead.
  - Drop support for Python < 3.6
  - Drop ``CUSTOM_KW`` support
  - Drop deprecated analysis keywords
  - Drop deprecated ecl config keywords
  - Drop deprecated ``PLOT_SETTINGS`` keyword
  - Drop deprecated model config keywords
  - Drop support for deprecated keywords ``{STORE, LOAD}_SEED``
  - Drop support for jobs with relative paths to the config
  - Drop support for creating ``EnkfMain`` with filename
  - Drop support for ``QC_{PATH, WORKFLOW}`` keywords
  - Drop support for non enum log levels
  - Remove warning for deprecated ``ERT_LIBRARY_PATH`` env variable
  - Remove unused code
  - New libres version ``6.0.0``

Bug fixes:
  - Handle empty observation set in Data API
  - Alpha and std_cutoff passed wrongly to the now deprecated observation correlation scaling in libres


6.0.0 libres
~~~~~~~~~~~~
See ERT release notes.

0.6.0 semeio
~~~~~~~~~~~~
New features:
  - Add --outputdirectory option to gendata_rft
  - Missing namespace support added to ``design_kw``
  - New option, ``auto_scale`` added to MisfitPreprocessor
  - Add new forward model job, for ``overburden_timeshift`` ( ``OTS``)

Other changes:
  - Refactor scaling factor calculation
  - Reports moved from being in the ``storage`` folder to next to the config file
  - Fixed a bug where ``csv_export2`` was not executable
  - Changed default linkage option from ``single`` to ``average`` for MisfitPreprocessor
  - Decreased likelihood of ``storage`` folders generated in source tree when running tests
  - Fixed a bug where user input observations to MisfitPreprocessor were not being used
  - Add documentation to ``csv_export2``.
  - Add warning for existing keys in parameters.txt for ``design2params``


Version 2.14
------------

Highlighted changes
~~~~~~~~~~~~~~~~~~~

Restarting ES-MDA
#################

It is now possible to restart ES-MDA runs from an intermediate iteration. Note
that this requires a bit of care due to the lack of metadata in current storage.
We are aiming at resolving this in the future in the new storage that soon will
be the standard.

After selecting the ES-MDA algorithm, you first need to set `Current case` to
the case you intend to be your first case to reevaluate. After which the
iteration number will be wrongly injected into the `Target case format`, which
you have to remove manually (reset it to the original `Target case format`).
After which you have to set `Start iteration` to the iteration for which you
are to start from; this number must correspond to the iteration number of the
case you have selected as your `Current case`. We recognize that this induces
some manual checking, but this is due to the lack of metadata mentioned above.
We still hope that this can aid our users and prevent the need of a complete
restart due to cluster issues etc.

2.14.0 ERT
~~~~~~~~~~~~
New features:
  - It's now possible to restart ES-MDA

Improvements:
  - Clean up ENKF_ALPHA keyword usage and documentation
  - Improved queue documentation
  - Warn user if invalid target is used to run analysis

Miscellaneous:
  - Find right extension based on system when loading rml_enkf analysis module

Bug fixes:
  - Ensure py2 QString conversion through py3 str
  - Correctly initialize active mask for IES
  - Fix early int conversion lead to rounding error and graphical glitches in the detailed view

5.0.1 libres
~~~~~~~~~~~~
Improvements:
  - Pretty print status.json and jobs.json
  - Add job index to elements in jobs.json
  - Print update log even if points are missing or inactive

Miscellaneous:
  - Remove deprecated BUILT_PYTHON option
  - Deprecate CUSTOM_KW
  - Stop generating EXIT files (only ERROR file is created)

0.5.4 semeio
~~~~~~~~~~~~
Improvements:
  - All data reported by workflows are persisted

Bug fixes:
  - Fix crash on emmpty defaults sheet for design matrices
  - Fix GENDATA_RFT job config

Version 2.13
------------

2.13.0 ERT
~~~~~~~~~~~~

New features:
  - Jobs can provide documentation via the plugin system

Improvements:
  - Resolve Python 3.8 deprecation warnings
  - Document job plugin system
  - Update COPY_DIRECTORY job docs

4.2.2 libres
~~~~~~~~~~~~

Improvements:
  - Label configuring slurm jobs as running
  - Remove deprecated queue configuration keys

0.5.3 semeio
~~~~~~~~~~~~

New features:
  - Pass job documentation of jobs to ERT via plugin system

Version 2.12
------------

2.12.0 ERT
~~~~~~~~~~~~
New features:
  - Support Slurm as a queue system

Improvements:
  - Support for --version in CLI

4.1.0 libres
~~~~~~~~~~~~
New features:
  - Support Slurm as a queue system

Improvements:
  - Backup PYTHONPATH when running RMS to facilitate external scripts

Miscellaneous:
  - Improve tmp-file usage in tests
  - Remove unused configsuite dependency

0.5.1 semeio
~~~~~~~~~~~~
New features:
  - Add INSERT_NOSIM and REMOVE_NOSIM

Improvements:
  - Add name to ensemble set in CSV_EXPORT2
  - Support configsuite 0.6
  - Have design2params support spaces in values
  - SpearmanJob exposes data via the reporter

Version 2.11
------------

Highlighted changes
~~~~~~~~~~~~~~~~~~~

New database
############

A new storage implementation has been made. The aim is that this will end up
as a more robust storage solution and that it will serve as a large step
towards the future of ERT. The solution is still experimental and **should not be
used for classified data** as of now. To enable the new storate solution use the
command line option `--enable-new-storage` when launching ERT. Note that it
will have to be used while running a case for the data to reside in the new
storage, but that the same data will also be available in the old storage if
you afterwards open ERT without the new storage enabled.

.. code-block:: none

   >>>> ert .... --enable-new-storage

4.0.2 libres
~~~~~~~~~~~~
Bugfixes:
  - Always load GEN_PARAM to ensure correct state before update

0.4.0 semeio
~~~~~~~~~~~~
New features:
  - Implemented Misfit preprocessor which will allow the user to run a pipeline of jobs to account for correlated observations
  - Implemented new CSV_EXPORT2 job which utilizes fmu-ensemble to do the export.

Improvements:
  - Added reporter functionality to output data to common storage
  - Correlated observations scaling uses SemeioScript with reporter in place of ErtScript
  - Improve error messages in design_kw
  - Correlated observation scaling will report singular values
  - Correlated observation scaling will report scale factor

Version 2.10
------------

Highlighted changes
~~~~~~~~~~~~~~~~~~~

Change in shell script behaviour
################################

The shell script jobs will no longer interpret the first path relative to the
configuration file. This implies that if you want to copy a file from the area
around your configuration file to the runpath, the following copying will not
work anymore:

.. code-block:: none

   FORWARD_MODEL COPY_FILE(<FROM>=my_files/data.txt, <TO>=data.txt)

And the reason is that it is not possible to deduce whether the intent was to
copy a file relative to your configuration file or whether you wanted to copy
(or delete) something that is already in your runpath. This led to mysterious
and strange errors. From now on, you will have to explicitly ask for the path
to be relative to your configuration file using the ``<CONFIG_PATH>`` magic
string:

.. code-block:: none

   FORWARD_MODEL COPY_FILE(<FROM>=<CONFIG_PATH>/my_files/data.txt, <TO>=data.txt)

The above change takes effect for the following shell scripts:
  - ``CAREFUL_COPY_FILE``
  - ``COPY_DIRECTORY``
  - ``COPY_FILE``
  - ``DELETE_DIRECTORY``
  - ``DELETE_FILE``
  - ``MAKE_DIRECTORY``
  - ``MAKE_SYMLINK``
  - ``MOVE_FILE``
  - ``SCRIPT``
  - ``SYMLINK``

Python 3 compatible CSV-export
##############################

``CSV_EXPORT2`` has been added as Python 3 compatible alternative to ``CSV_EXPORT1``.

2.10 ERT application
~~~~~~~~~~~~~~~~~~~~

Improvements:
  - Improve documentation on ARGSLIST
  - Enable RML_ENKF in default ERT installation

Bug fixes:
  - Fixed jumping cursor when filling in ES MDA weights
  - Fixed Python exception on exit
  - Logarithmic distributions are now plotted with correct axis type. The
    values themselves on a logarithmic scale, and the exponents on a linear
    scale.
  - Do not terminate on exception in RunContext due to race condition
  - [Python 2] Fix saving unicode configuration files

Other changes:
  - Lower bound matplotlib version for Python 3
  - Remove cwrap from install requirements
  - Separate plotting from data by an API
  - Add Jinja2 as an install dependency

4.0 libres
~~~~~~~~~~~~

Breaking changes:
  - The shell script jobs will no longer interpret the first path relative to
    the configuration file.

Bug fixes:
  - Use workflow future to determine running status
  - Give queue index in queue error message
  - Ensure integer division when making CPU list

2.9 libecl
~~~~~~~~~~

Improvements:
  - Pip-installable libecl
  - Improved identification of rate and total (cumulative) keywords

0.2.2 semeio
~~~~~~~~~~~~

Highlighted changes
  - Add CSV_EXPORT2 as Python 3 compatible alternative to CSV_EXPORT1

Bug fixes:
  - Add string representation to TrajectoryPoint for backwards compatibility


Version 2.9
-----------

2.9 ERT application
~~~~~~~~~~~~~~~~~~~

Improvements:
  - Fix bug where changing dataset for plotting would crash ERT
  - Fix bug in ERT data API where inactive summary observations would exist

3.2 libres
~~~~~~~~~~~~

Improvements:
  - Fix bug in normal distribution which could generate -∞ when sampled

Other changes:
  - Removed ecl version from jobs.json
  - Remove possibility to specify iens when creating runpath

Version 2.8
-----------

Highlighted changes
~~~~~~~~~~~~~~~~~~~

Improvements to ERT
###################
When running ERT in komodo, the forward models will now run in the same komodo version as the ERT application. 
This happens even if the stable komodo version changes while ERT is running. 

Improvements to ERT CLI
#######################
Defining current case is now available in the cli. The see the usage and complete list of available commands, go to :doc:`../reference/running_ert`.


Improvements to ERT GUI
#######################
The viewer for job-output in the detailed monitor widget is now improved to handle larger outputs. 

This will improve the experience for jobs like eclipse.

2.8 ERT application
~~~~~~~~~~~~~~~~~~~

New features:
  - CLI support current case
  - Output viewer supports large output from jobs like eclipse

Improvements:
  - Improvements to documentation

2.8 libres
~~~~~~~~~~

General bug fixes and improvements
  - Added support in IES_ENKF for using newly activated observations
  - Fixed bug in IES_ENKF when extracting active observations
  - Add filtering on module to prevent warnings from other modules
  - Fix error in triangular distribution (Also backported to 2.6)

2.7 libecl
~~~~~~~~~~
General bug fixes and improvements
  - Add deprecation warnings when import ecl.ecl or import ert.ecl.

0.1 Semeio
~~~~~~~~~~

New workflow jobs:
  - CORRELATED_OBSERVATIONS_SCALING - Experimental
  - SPEARMAN_CORRELATION - Experimental

New jobs (Ported from ert-statoil for python 3): 
  - STEA
  - GENDATA_RFT
  - DESIGN2PARAM
  - DESIGN_KW

Add komodo_job_dispatch from equlibrium

Version 2.6
-----------

Highlighted changes
~~~~~~~~~~~~~~~~~~~

Improvements to ERT CLI
#######################
The text and shell interface has been removed from ERT, but the CLI interface has gotten a upgrade and now
included basic monitoring to show the progress of the running experiment. The CLI now also supports MDA and
running single workflows.

The see the usage and complete list of available commands, go to :doc:`../reference/running_ert`.


Improvements to ERT GUI
#######################
The ERT GUI now includes help links to relevant resources and the job monitoring window now also includes
memory usage information for each job. In addition, the output from the Eclipse job is treated like any
other job and you can now read it from the the job monitoring window.

Experimental features
#####################
The new iterative ensemble smoother algorithm has been added as an update algorithm.


2.6 ERT application
~~~~~~~~~~~~~~~~~~~

New features:
  - Add basic monitoring to ERT cli
  - Memory usage monitoring in GUI
  - CLI supports MDA
  - CLI supports running single workflow

Improvements:
  - Less resource intensive monitoring
  - Display config file name in all GUI window titles
  - Run dialog no longer closes when pressing esc
  - Improved exit monitoring when simulations fail
  - Active realizations field is automatically filled with the runable realizations available
  - Tailored plotting for single data points
  - Algorithm recommendations in menu
  - Check for observation data
  - Better support for lsf-nodes with unknown status
  - Communicate analysis completetion
  - Various improvements to the documentation
  - Document RMS job
  - Help links in the GUI

Code structure and tooling:
  - Removed all C-code and CMake
  - Removed ERT_SHARE_PATH
  - Make CLI independent of Qt

Breaking changes:
  - Removed the text and tui interfaces.

Experimental features:
  - New iterative ensemble smother algorithm
  - Python 3 support
  - PyQt5 support
  - Add plugin system for forward models

2.6 libres
~~~~~~~~~~
New features:
  - Suffix support for External Parameters
  - Back up existing parameters-file
  - Support for lower case shell scripts

Improvements:
  - Make runWorkflows static
  - Exposed enkf_main_have_obs in python
  - Added support for unknown status in the queue driver
  - Ensure that the number of required successful realization are not higher then the ensemble size
  - Fix yaml load warnings in python 3
  - Fix ecl load warnings in python 3
  - Improved support for when lsf executables are temporarily unavailable
  - Use subprocess instead of fork
  - General code and performance improvements

Bug fixes:
  - Log random seed
  - Make sure reporting does not fail
  - Remove double dash arguments from job arglists

Breaking changes:
  - Deprecated various unused keywords
  - Deprecated updating workflows
  - Remove SCHEDULE as HISTORY_SOURCE

Experimental features:
  - Programmatic initialization (Validation will come in a future release)

ERT forward models
~~~~~~~~~~~~~~~~~~~
Improvements:
  - Output from Eclipse job is treated like any other job


2.5 libecl
~~~~~~~~~~
General bug fixes and improvements

Changes:
  - More aggressively close files when loading summary vectors.
  - Dump mapaxes even though they are not applied
  - Ignore wells with blank names
  - Infer format from extension
  - Use 0-based indices for nnc's.
  - Allow for mixed case basenames
  - Reset before active cells are set

Version 2.5
-----------

This is a small release which only contains some improvements to the GUI due to
user feedback. See the *Highlighted* section for the most prominent changes. For a more
in-depth overview, as well as links to the relevant pull requests, we refer the
reader to the repository specific sections.

Highlighted changes
~~~~~~~~~~~~~~~~~~~

Open job logs from the GUI
#############################
Open the montoring by pressing `details`. If you select a realization and then
click either its `stdout` or `stderr` you will get the corresponding output
displayed in the GUI for easier debugging.

Notify user of failing workflows
###################################
If workflows fail a list of the failing workflows will be presented to the
user.

Polishing monitoring window
###################################
Several minor improvements to the monitoring section, including embedding it
into the existinging monitoring window, making the layout vertical to prevent
scrolling, not setting the start and end times before they actually exist,
colors etc is added to the monitoring window.

Debug after ensemble failure
####################################
The above capablilities are available also after the entire ensemble has
finished.


2.5 ert application
~~~~~~~~~~~~~~~~~~~

New functionality:
  - MacOS compatibility
  - Notify user of failed workflows
  - Allow the user to open logs from the forward models in the GUI

Improvement:
  - Revert to old plot style if more than one data point
  - Validate that target is specified when running an update via the `cli`
  - Merge detailed view into the old progress window
  - Vertical layout of monitoring for better usability
  - Fetch queue status for each forward model in the detailed monitoring. Border color indicates:
    Yellow, still running on queue. Black, finished. Red, failed

Others:
  - Documentation for setting up custom jobs.
  - Fix status on finished runs.


2.5 libres
~~~~~~~~~~
Minor code improvement and exposure of status files.

ert forward models
~~~~~~~~~~~~~~~~~~~
No changes.

libecl
~~~~~~~~~~
No changes.


Version 2.4
-----------

See the *Highlighted* section for the most prominent changes. For a more
in-depth overview, as well as links to the relevant pull requests, we refer the
reader to the repository specific sections.

Highlighted changes
~~~~~~~~~~~~~~~~~~~

Unified ERT binary
###########################
All launches of *ERT* is now to happen through the shell command ``ert``. To get
an overview of the various *ERT* tools, you can run ``ert --help``. You will then be
presented with the following overview.

::

    [scout@desktop ert]$ ert --help
    usage: ert [-h] {gui,text,shell,cli} ...

    ERT - Ensemble Reservoir Tool

    optional arguments:
      -h, --help            show this help message and exit

    Available user entries:
      ERT can be accessed through a GUI or CLI interface. Include one of the
      following arguments to change between the interfaces. Note that different
      entry points may require different additional arguments. See the help
      section for each interface for more details.  

      {gui,text,shell,cli}  Available entry points
        gui                 Graphical User Interface - opens up an independent
                            window for the user to interact with ERT.        
        cli                 Command Line Interface - provides a user interface in
                            the terminal.

Hence, ``ert gui my_config_file`` will launch the *ERT*-gui with the specified
configuration file. For detailed support for each of the options, use ``ert gui
--help`` etc.

ERT command line interface
###########################
The **cli** option listed above is new and will run *ERT* as a command line
interface with no further interaction after initialization. This will be the
supported command line interface of *ERT* in the future.

Forward model monitoring
######################################################
An essential new feature of *ERT 2.4* is a monitoring screen in the GUI
displaying the progress of each forward model in your ensemble. After
initiating the run, press the **Details** button to get an overview of the
progress of each of the forward models. In the view appearing you can click on
a specific realization to get even more details regarding that specific
realization.

Restarting failed realizations
####################################################
If some of your forward models failed there will appear a **Restart** button
when the run has finished, which will rerun only the failed realizations.

Run prior and posterior separately
####################################################
Many users have requested the possibility of running the prior and posterior
independently. This feature already exists in the advanced mode of the GUI, but
to make it more accessible to the users we have now made the advanced mode the
only mode.

To run your prior, you run an **Ensemble Experiment**. Then, to run an update you
click **Run Analysis** from the top menu of the main window; you can then specify
the target and source case and the update will be calculated. To evaluate your
posterior, you then run a new **Ensemble Experiment** with your target case.
After this, you can plot and compare just as if you had run an **Ensemble
Smoother** to begin with.

Generic tooling in the forward model
####################################################
As a first step towards more generic tooling in *ERT* forward models *ERT* will now dump all
parameters with their corresponding values to the runpath as *JSON*. The format
of this file is still experimental and will most likely change in a future
release of *ERT*, but one is still welcome to play around with the extra
possibilities this gives.

Generic templating
######################
Jinja based templating has been a great success in *Everest* and will most
likely be standardized in future version of *ERT* also; both with respect to
configuration templating and templating in the forward model itself. As a first
step towards this, a forward model named *TEMPLATE_RENDER* has been added. It
will load the parameter values that is dumped by *ERT* (described above),
optionally together with user specified *json*- and *yaml*-files and render a
specified template. For more on how to write *Jinja* template, see the official
`documentation <http://jinja.pocoo.org/docs/2.10/>`_.

Eclipse version in forward model
#################################
The recommended way of specifying the eclipse version is to pass ``<VERSION>``
as argument to the forward model ``ECLIPSE100`` instead of using
``ECLIPSE100_<MY_ECL_VERSION>``. The old format of putting the version in the
job name will be deprecated in the future.


2.4 ert application
~~~~~~~~~~~~~~~~~~~
PR: 162 - 257

New functionality:
  - Unified ERT binary `[165] <https://github.com/equinor/ert/pull/165>`__
  - Restart failed realizations `[206, 209] <https://github.com/equinor/ert/pull/206>`__
  - Forward model monitoring in GUI `[252] <https://github.com/equinor/ert/pull/252>`__

Improvement:
  - Print warning if decimal point is not `.` `[212] <https://github.com/equinor/ert/pull/212>`__
  - Fixed bug such that initial realization mask contains all `[213] <https://github.com/equinor/ert/pull/213>`__
  - Fixed bug in iterated smoother gui `[215] <https://github.com/equinor/ert/pull/215>`__
  - Always display advanced settings `[216] <https://github.com/equinor/ert/pull/216>`__
  - Change default plot size to emphasize discrete data `[243] <https://github.com/equinor/ert/pull/243>`__

Others:
  - Continued to move documentation into the manual.
  - TUI and shell is deprecated.
  - Improved automatic testing on FMU tutorial.

2.4 ert forward models
~~~~~~~~~~~~~~~~~~~~~~
PR: 114 - 126

New functionality:
  - Forward model for dynamic porevolume geertsma `[114] <https://github.com/equinor/ert-statoil/pull/114>`__

Improvements:
  - Eclipse version should be passed to job ECLIPSE_100 / ECLIPSE_300 as an argument

Others:
  - Major move of forward models from ert-statoil to libres


2.4 libres
~~~~~~~~~~
PR: 411 - 517

New functionality:
 - Job description can set environment vars `[431] <https://github.com/equinor/libres/pull/431/files>`__
 - Experimental dump of parameters to runpath as json `[436] <https://github.com/equinor/libres/pull/436>`__
 - Jinja based rendering forward model `[443] <https://github.com/equinor/libres/pull/443/files>`__
 - New config keyword NUM_CPU to override eclipse PARALLEL keyword `[455] <https://github.com/equinor/libres/pull/455/files>`__
 - Expose the algorithm iteration number as magic string <ITER> `[515] <https://github.com/equinor/libres/pull/515>`__

Improvements:
 - Fix bug in default standard deviation calculations `[513] <https://github.com/equinor/libres/pull/513>`__
 - Start scan for active observations at report step 0, instead of 1 `[439] <https://github.com/equinor/libres/pull/439>`__
 - Bug fixes in linear algebra code `[435] <https://github.com/equinor/libres/pull/435>`__
 - Improved job killing capabilities of local queue `[488] <https://github.com/equinor/libres/pull/488>`__

Others:
 - Various improvements to code due to now being a C++ project
 - Removed traces of WPRO and the RPCServer `[428] <https://github.com/equinor/libres/pull/428>`__
 - CAREFUL_COPY moved to libres `[424] <https://github.com/equinor/libres/pull/424>`__
 - Split simulator configuration into multiple files `[477] <https://github.com/equinor/libres/pull/477>`__


2.4 libecl
~~~~~~~~~~
PR: 506 - 579

New functionality:
 - Ability to compute geertsma based on dynamic porevolume `[530] <https://github.com/equinor/libecl/pull/530>`__
 - Support for Intersect NNC format `[533] <https://github.com/equinor/libecl/pull/533>`__
 - Support for extrapolation when resampling `[534] <https://github.com/equinor/libecl/pull/534>`__
 - Ability to load summary data from .csv-files `[536] <https://github.com/equinor/libecl/pull/536>`__
 - Identify region-to-region variables `[551] <https://github.com/equinor/libecl/pull/551>`__

Improvements:
 - Load LGR info when loading well info `[529] <https://github.com/equinor/libecl/pull/529>`__
 - Do not fail if restart file is missing icon `[549] <https://github.com/equinor/libecl/pull/549>`__

Others:
 - Various improvements to code due to now being a C++ project.
 - Improved documentation for Windows users
 - Improved Python 3 testing
 - Revert fortio changes to increase reading speed `[567] <https://github.com/equinor/libecl/pull/567>`__


Version 2.3
-----------

2.3 ert application
~~~~~~~~~~~~~~~~~~~
PR: 67 - 162


2.3 libres
~~~~~~~~~~
PR: 105 - 411


2.3 libecl
~~~~~~~~~~
PR: 170 - 506




Version 2.2
-----------

2.2: ert application
~~~~~~~~~~~~~~~~~~~~

Version 2.2.1 September 2017 PR: 1 - 66
Cherry-picked: `70 <https://github.com/Equinor/ert/pull/70/>`__

Misc:

 - Using res_config changes from libres `[16] <https://github.com/Equinor/ert/pull/16/>`__
 - files moved from libecl to libres: `[51] <https://github.com/Equinor/ert/pull/51>`__
 - replaced ert.enkf with res.enkf `[56] <https://github.com/Equinor/ert/pull/56/>`__
 - Created ErtVersion: [`61 <https://github.com/Equinor/ert/pull/61/>`__, `66 <https://github.com/Equinor/ert/pull/66/>`__].
 - Using res_config: [`62 <https://github.com/Equinor/ert/pull/62/>`__]
 - Removed dead workflow files: `[64] <https://github.com/Equinor/ert/pull/64/>`__

Build and testing:

 - Cleanup after repo split [`1 <https://github.com/Equinor/ert/pull/1/>`__, `2 <https://github.com/Equinor/ert/pull/2/>`__, `3 <https://github.com/Equinor/ert/pull/3/>`__ , `4 <https://github.com/Equinor/ert/pull/4/>`__, `5 <https://github.com/Equinor/ert/pull/5/>`__ , `6 <https://github.com/Equinor/ert/pull/6/>`__]
 - Added test_install functionality [`7 <https://github.com/Equinor/ert/pull/7/>`__]
 - Added travis build script for libecl+libres+ert:
   [`15 <https://github.com/Equinor/ert/pull/15/>`__, `17 <https://github.com/Equinor/ert/pull/17/>`__, `18 <https://github.com/Equinor/ert/pull/18/>`__, `19 <https://github.com/Equinor/ert/pull/19/>`__, `21 <https://github.com/Equinor/ert/pull/21/>`__, `26 <https://github.com/Equinor/ert/pull/26/>`__, `27 <https://github.com/Equinor/ert/pull/27/>`__, `39, <https://github.com/Equinor/ert/pull/39/>`__ `52 <https://github.com/Equinor/ert/pull/52/>`__-`55 <https://github.com/Equinor/ert/pull/55/>`__, `63 <https://github.com/Equinor/ert/pull/63/>`__]

 - MacOS build error: [`28 <https://github.com/Equinor/ert/pull/28/>`__].
 - Created simple gui_test bin/gui_test [`32 <https://github.com/Equinor/ert/pull/32/>`__]
 - cmake - create symlink: [`41 <https://github.com/Equinor/ert/pull/41/>`__, `42 <https://github.com/Equinor/ert/pull/42/>`__, `43 <https://github.com/Equinor/ert/pull/43/>`__]
 - Initial Python3 testing [`58 <https://github.com/Equinor/ert/pull/58/>`__, `60 <https://github.com/Equinor/ert/pull/60/>`__].


Queue and running:

 - Added base run model - gui model updates: [`20 <https://github.com/Equinor/ert/pull/20/>`__].
 - Added single simulation pretest running [`33 <https://github.com/Equinor/ert/pull/33/>`__, `36 <https://github.com/Equinor/ert/pull/36/>`__, `50 <https://github.com/Equinor/ert/pull/50/>`__, `67 <https://github.com/Equinor/ert/pull/67/>`__].
 - Add run_id to simulation batches.


2.2: libres
~~~~~~~~~~~

Version 2.2.9 September 2017 PR: 1 - 104
Cherry-picks: [`106 <https://github.com/Equinor/res/pull/106/>`__, `108 <https://github.com/Equinor/res/pull/108/>`__, `110 <https://github.com/Equinor/res/pull/110/>`__, `118 <https://github.com/Equinor/res/pull/118/>`__, `121 <https://github.com/Equinor/res/pull/121/>`__, `122 <https://github.com/Equinor/res/pull/122/>`__, `123 <https://github.com/Equinor/res/pull/123/>`__, `127 <https://github.com/Equinor/res/pull/127/>`__]

Misc:

 - implement legacy from ert.xxx [`1, <https://github.com/Equinor/res/pull/1/>`__ `20, <https://github.com/Equinor/res/pull/20/>`__ `21, <https://github.com/Equinor/res/pull/21/>`__ `22 <https://github.com/Equinor/res/pull/22/>`__]
 - Setting up libres_util and moving ert_log there [`13 <https://github.com/Equinor/res/pull/13/>`__, `44 <https://github.com/Equinor/res/pull/44/>`__, `48 <https://github.com/Equinor/res/pull/48/>`__].
 - Added subst_list + block_fs functionality to res_util - moved from
   libecl [`27 <https://github.com/Equinor/res/pull/27/>`__, `68 <https://github.com/Equinor/res/pull/68/>`__, `74 <https://github.com/Equinor/res/pull/74/>`__].
 - Do not generate parameters.txt if no GEN_KW is specified.[`89 <https://github.com/Equinor/res/pull/89/>`__]
 - Started using RES_VERSION [`91 <https://github.com/Equinor/res/pull/91/>`__].
 - CONFIG_PATH subtitution settings - bug fixed[`43 <https://github.com/Equinor/res/pull/43/>`__, `96 <https://github.com/Equinor/res/pull/96/>`__].
 - Will load summary if GEN_DATA is present [`123 <https://github.com/Equinor/res/pull/123/>`__, `127 <https://github.com/Equinor/res/pull/127/>`__]


Build and test fixes:

 - Simple functionality to do post-install testing[`3 <https://github.com/Equinor/res/pull/3/>`__]
 - Use libecl as cmake target[`6 <https://github.com/Equinor/res/pull/6/>`__,`15 <https://github.com/Equinor/res/pull/15/>`__]
 - removed stale binaries [`7 <https://github.com/Equinor/res/pull/7/>`__, `9 <https://github.com/Equinor/res/pull/9/>`__]
 - travis will build all repositories [`23 <https://github.com/Equinor/res/pull/23/>`__].
 - Travis + OSX [`69 <https://github.com/Equinor/res/pull/69/>`__, `72 <https://github.com/Equinor/res/pull/72/>`__]
 - Remove equinor specific settings from build sytem [`38 <https://github.com/Equinor/res/pull/38/>`__].
 - Travis split for parallell builds [`79 <https://github.com/Equinor/res/pull/79/>`__].


Config refactor:

  In this release cycle there have been large amount of changes to the
  code configuring the ERT state; the purpose of these changes has
  been to prepare for further development with Everest. The main net
  change is that a new configuration object - res_config has been
  created ,which holds all the configuration subobjects:

    [`10 <https://github.com/Equinor/res/pull/10/>`__, `14 <https://github.com/Equinor/res/pull/14/>`__, `35 <https://github.com/Equinor/res/pull/35/>`__, `39 <https://github.com/Equinor/res/pull/39/>`__, `45 <https://github.com/Equinor/res/pull/45/>`__, `52 <https://github.com/Equinor/res/pull/52/>`__, `54 <https://github.com/Equinor/res/pull/54/>`__, `58 <https://github.com/Equinor/res/pull/58/>`__-`62 <https://github.com/Equinor/res/pull/62/>`__, `66 <https://github.com/Equinor/res/pull/66/>`__, `75 <https://github.com/Equinor/res/pull/75/>`__]


Queue layer:

 - Improved logging [`17 <https://github.com/Equinor/res/pull/17/>`__, `37 <https://github.com/Equinor/res/pull/37/>`__].
 - Funcionality to create a queue_config object copy [`36 <https://github.com/Equinor/res/pull/36/>`__].

 As part of this development cycle the job_dispatch script has been
 included in the libres distribution. There are many PR's related to
 this script:

    [`28 <https://github.com/Equinor/res/pull/28/>`__, `40 <https://github.com/Equinor/res/pull/40/>`__, `41 <https://github.com/Equinor/res/pull/1/>`__, `51 <https://github.com/Equinor/res/pull/51/>`__, `53 <https://github.com/Equinor/res/pull/53/>`__, `63 <https://github.com/Equinor/res/pull/63/>`__, `64 <https://github.com/Equinor/res/pull/64/>`__, `83 <https://github.com/Equinor/res/pull/83/>`__, `84 <https://github.com/Equinor/res/pull/84/>`__, `85 <https://github.com/Equinor/res/pull/85/>`__, `93 <https://github.com/Equinor/res/pull/93/>`__, `94 <https://github.com/Equinor/res/pull/94/>`__, `95 <https://github.com/Equinor/res/pull/95/>`__, `97 <https://github.com/Equinor/res/pull/97/>`__-`99 <https://github.com/Equinor/res/pull/99/>`__,
     `101 <https://github.com/Equinor/res/pull/101/>`__, `103 <https://github.com/Equinor/res/pull/103/>`__, `108 <https://github.com/Equinor/res/pull/108/>`__, `110 <https://github.com/Equinor/res/pull/110/>`__]

 - Create a common run_id for one batch of simulations, and generally
   treat one batch of simulations as one unit, in a better way than
   previously: [`42 <https://github.com/Equinor/res/pull/42/>`__, `67 <https://github.com/Equinor/res/pull/67/>`__]

 - Added PPU (Paay Per Use) code to LSF driver [`71 <https://github.com/Equinor/res/pull/71/>`__].
 - Workflow job PRE_SIMULATION_COPY [`73 <https://github.com/Equinor/res/pull/73/>`__, `88 <https://github.com/Equinor/res/pull/88/>`__].
 - Allow to unset QUEUE_OPTION [`87 <https://github.com/Equinor/res/pull/87/>`__].
 - Jobs failing due to dead nodes are restarted [`100 <https://github.com/Equinor/res/pull/100/>`__].


Documentation:

  - Formatting bugs: [`49 <https://github.com/Equinor/res/pull/49/>`__, `50 <https://github.com/Equinor/res/pull/50/>`__]
  - Removed doxygen + build rst [`29 <https://github.com/Equinor/res/pull/29/>`__]

2.2: libecl
~~~~~~~~~~~

Version 2.2.0 September 2017 PR: 1 - 169
Open PR: 108, 145

Grid:

 - Extracted implementation ecl_nnc_geometry [`1 <https://github.com/Equinor/libecl/pull/1/>`__, `66 <https://github.com/Equinor/libecl/pull/66/>`__, `75 <https://github.com/Equinor/libecl/pull/75/>`__, `78 <https://github.com/Equinor/libecl/pull/78/>`__, `80 <https://github.com/Equinor/libecl/pull/80/>`__, `109 <https://github.com/Equinor/libecl/pull/109/>`__].
 - Fix bug in cell_contains for mirrored grid [`51 <https://github.com/Equinor/libecl/pull/51/>`__, `53 <https://github.com/Equinor/libecl/pull/53/>`__].
 - Extract subgrid from grid [`56 <https://github.com/Equinor/libecl/pull/56/>`__].
 - Expose mapaxes [`63 <https://github.com/Equinor/libecl/pull/63/>`__, `64 <https://github.com/Equinor/libecl/pull/64/>`__].
 - grid.get_lgr - numbered lookup [`83 <https://github.com/Equinor/libecl/pull/83/>`__]
 - Added NUMRES values to EGRID header [`125 <https://github.com/Equinor/libecl/pull/125/>`__].

Build & testing:

 - Removed warnings - added pylint [`4 <https://github.com/Equinor/libecl/pull/4/>`__, `5 <https://github.com/Equinor/libecl/pull/5/>`__, `6 <https://github.com/Equinor/libecl/pull/6/>`__, `10 <https://github.com/Equinor/libecl/pull/10/>`__, `11 <https://github.com/Equinor/libecl/pull/11/>`__, `12 <https://github.com/Equinor/libecl/pull/12/>`__]
 - Accept any Python 2.7.x version [`17 <https://github.com/Equinor/libecl/pull/17/>`__, `18 <https://github.com/Equinor/libecl/pull/18/>`__]
 - Remove ERT testing & building [`3 <https://github.com/Equinor/libecl/pull/3/>`__, `19 <https://github.com/Equinor/libecl/pull/19/>`__]
 - Changes to Python/cmake machinery [`25 <https://github.com/Equinor/libecl/pull/25/>`__, `30 <https://github.com/Equinor/libecl/pull/3/>`__, `31 <https://github.com/Equinor/libecl/pull/31/>`__, `32 <https://github.com/Equinor/libecl/pull/32/>`__, `49 <https://github.com/Equinor/libecl/pull/49/>`__, `52 <https://github.com/Equinor/libecl/pull/52/>`__, `62 <https://github.com/Equinor/libecl/pull/62/>`__].
 - Added cmake config file [`33 <https://github.com/Equinor/libecl/pull/33/>`__, `44 <https://github.com/Equinor/libecl/pull/44/>`__, `45 <https://github.com/Equinor/libecl/pull/45/>`__, `47 <https://github.com/Equinor/libecl/pull/47/>`__].
 - Only *one* library [`54 <https://github.com/Equinor/libecl/pull/54/>`__, `55 <https://github.com/Equinor/libecl/pull/55/>`__, `58 <https://github.com/Equinor/libecl/pull/58/>`__, `69 <https://github.com/Equinor/libecl/pull/69/>`__, `73 <https://github.com/Equinor/libecl/pull/73/>`__, `77 <https://github.com/Equinor/libecl/pull/77/>`__, `91 <https://github.com/Equinor/libecl/pull/91/>`__, `133 <https://github.com/Equinor/libecl/pull/133/>`__]
 - Removed stale binaries [`59 <https://github.com/Equinor/libecl/pull/59/>`__].
 - Require cmake >= 2.8.12 [`67 <https://github.com/Equinor/libecl/pull/67/>`__].
 - Fix build on OSX [`87 <https://github.com/Equinor/libecl/pull/87/>`__, `88 <https://github.com/Equinor/libecl/pull/88/>`__, `95 <https://github.com/Equinor/libecl/pull/95/>`__, `103 <https://github.com/Equinor/libecl/pull/103/>`__].
 - Fix broken behavior with internal test data [`97 <https://github.com/Equinor/libecl/pull/97/>`__].
 - Travis - compile with -Werror [`122 <https://github.com/Equinor/libecl/pull/122/>`__, `123 <https://github.com/Equinor/libecl/pull/123/>`__, `127 <https://github.com/Equinor/libecl/pull/127/>`__, `130 <https://github.com/Equinor/libecl/pull/130/>`__]
 - Started to support Python3 syntax [`150 <https://github.com/Equinor/libecl/pull/150/>`__, `161 <https://github.com/Equinor/libecl/pull/161/>`__]
 - Add support for paralell builds on Travis [`149 <https://github.com/Equinor/libecl/pull/149/>`__]

libecl now fully supports OSX. On Travis it is compiled with
-Werror=all which should protect against future warnings.

C++:

 - Removed use of deignated initializers [`7 <https://github.com/Equinor/libecl/pull/7/>`__].
 - Memory leak in EclFilename.cpp [`14 <https://github.com/Equinor/libecl/pull/14/>`__].
 - Guarantee C linkage for ecl_data_type [`65 <https://github.com/Equinor/libecl/pull/65/>`__].
 - New smspec overload [`89 <https://github.com/Equinor/libecl/pull/89/>`__].
 - Use -std=c++0x if -std=c++11 is unavailable [`118 <https://github.com/Equinor/libecl/pull/118/>`__]
 - Make all of (previous( libutil compile with C++ [`162 <https://github.com/Equinor/libecl/pull/162/>`__]

Well:

 - Get well rates from restart files [`8 <https://github.com/Equinor/libecl/pull/8/>`__, `20 <https://github.com/Equinor/res/pull/20/>`__].
 - Test if file exists before load [`111 <https://github.com/Equinor/libecl/pull/111/>`__].
 - Fix some warnings [`169 <https://github.com/Equinor/libecl/pull/169/>`__]

Core:

 - Support for variable length strings in binary eclipse files [`13 <https://github.com/Equinor/libecl/pull/13/>`__, `146 <https://github.com/Equinor/libecl/pull/146/>`__].
 - Renamed root package ert -> ecl [`21 <https://github.com/Equinor/libecl/pull/21/>`__]
 - Load INTERSECT summary files with NAMES instead WGNAMES [`34 <https://github.com/Equinor/libecl/pull/34/>`__ - `39 <https://github.com/Equinor/libecl/pull/39/>`__].
 - Possible memory leak: [`61 <https://github.com/Equinor/libecl/pull/61/>`__]
 - Refactored binary time search in __get_index_from_sim_time() [`113 <https://github.com/Equinor/libecl/pull/113/>`__]
 - Possible to mark fortio writer as "failed" - will unlink on close [`119 <https://github.com/Equinor/libecl/pull/119/>`__].
 - Allow keywords of more than 8 characters [`120 <https://github.com/Equinor/libecl/pull/120/>`__, `124 <https://github.com/Equinor/libecl/pull/124/>`__].
 - ecl_sum writer: Should write RESTART keyword [`129 <https://github.com/Equinor/libecl/pull/129/>`__, `131 <https://github.com/Equinor/libecl/pull/131/>`__]
 - Made EclVersion class [`160 <https://github.com/Equinor/libecl/pull/160/>`__]
 - Functionality to dump an index file for binary files: [`155 <https://github.com/Equinor/libecl/pull/155/>`__, `159 <https://github.com/Equinor/libecl/pull/159/>`__, `163 <https://github.com/Equinor/libecl/pull/163/>`__, `166 <https://github.com/Equinor/libecl/pull/166/>`__, `167 <https://github.com/Equinor/libecl/pull/167/>`__]

Misc:

 - Added legacy pacakge ert/ [`48 <https://github.com/Equinor/libecl/pull/48/>`__, `99 <https://github.com/Equinor/libecl/pull/99/>`__]
 - Improved logging - adding enums for og levels [`90 <https://github.com/Equinor/libecl/pull/90/>`__, `140 <https://github.com/Equinor/libecl/pull/140/>`__, `141 <https://github.com/Equinor/libecl/pull/141/>`__]
 - Refactored to use snake_case instead of CamelCase [`144 <https://github.com/Equinor/libecl/pull/144/>`__, `145 <https://github.com/Equinor/libecl/pull/145/>`__]


-----------------------------------------------------------------

Version 2.1.0 February 2017  PR: 1150 - 1415
Open PR: 1352, 1358, 1362

Queue system/workflows:

 - Functionality to blacklist nodes from LSF [1240, 1256, 1258, 1274, 1412, 1415].
 - Use bhist command to check lsf job status if job has expired from bjobs [1301].
 - Debug output from torque goes to stdout [1151].
 - Torque driver will not abort if qstat returns invalid status [1411].
 - Simulation status USER_EXIT - count as failed [1166].
 - Added Enum identifier 'JOB_QUEUE_DO_KILL_NODE_FAILURE [1268].
 - Have deprecated the ability to set queue options directly on the drivers [1288].
 - Added system for version specific loading for workflow job model
   description files [1177].
 - Job loader should not try to load from directory [1187].
 - Refactoring of max runtime - initiated from WPRO [1237].
 - Determine which nodes are running a job [1251].

Build updates:

 - Check if python modules are present in the required version [1150].
 - Do not build ERT_GUI if PyQt4 is not found [1150, 1230].
 - Do not fail build numpy is not found [1153].
 - Allow for user provided CMAKE_C_FLAGS on linux [1300].
 - Require exactly version 2.7 of Python [1307].
 - Travis testing improvements [1363].
 - Removed devel/ directory from source [1196].
 - Setting correct working directory, and build target dependency
   for sphinx-apidoc / html generation [1385].

Eclipse library:

 - C++ move constructor and operator= for smspec_node [1155, 1200].
 - fortio_ftruncate( ) [1161].
 - INIT writer will write keywords DEPTH, DX, DY, DZ [1164, 1172, 1311, 1388].
 - Grid writer will take unit system enum argument [1164].
 - New function ecl_kw_first_different( ) [1165].
 - Completion variables can be treated as totals [1167].
 - Fixed bug in ecl_kw_compare_numeric( ) [1168].
 - Refactored / inlined volume calculations in ecl_grid [1173, 1184].
 - Made function ecl_kw_size_and_type_equal() public [1192].
 - Fixed bug in ecl_grid_cell_contains() [1402, 1404, 1195, 1419].
 - OOM bug in ecl_kw_grdecl loader for large files [1207].
 - Cache cell volumes in ecl_grid [1228].
 - Geertsma / gravity functionality [1227, 1284, 1289, 1292, 1364, 1408].
 - Summary + restart - will allow some keyword differences [1296].
 - Implemented ecl_rst_file_open_write_seek( ) [1236].
 - Optionally apply mapaxes [1242, 1281].
 - Expose and use ecl_file_view datastructere - stop using 'blocks' in ecl_file objects [1250].
 - ecl_sum will internalize Aquifer variables [1275].
 - Make sure region properties RxxT are marked as total + depreecated some properties [1285].
 - ecl_kw_resize() + C++ operator[] [1316]
 - Added small C++ utility to create eclipse filenames[1396].
 - Make sure restart and INIT files are written with correct unit ID [1399, 1407].
 - Skip keyword data type: 'C010' without failing [1406, 1410].
 - Adding parsing of the last (optional) config token for the SLAVES kwd [1409].
 - Add nnc index to the information exported by ecl_nnc_export() [1204].
 - Marked solvent related total keywords ?NIT and ?NPT.* as totals [1241].
 - Top active cell in grid [1322].
 - Added absolute epsilon to ecl_kw comparsion [1345,1351].

Smoother, updates and 'running':

 - Fixed bug with local updates of GEN_DATA [1291].
 - Changed default MDA weights and removed file input [1156, 1190, 1191].
 - Bug in handling of failed realisations [1163].
 - Fix bug missing assignment of analysis module in ES-MDA [1179].
 - OpenMP implementation of fwd_step [1185, 1324,1342].
 - Removes the ability to update dynamic variables [1189].
 - Allow max CV FOLD to be the number of ensembles [1205, 1208].
 - Fix for min_realizations logic [1206].
 - Can assign a specific analyis module for one local update [1224].
 - Handle updates when some summary relaisations are "too short" [1400, 1405].
 - Extending hook manager to support PRE_UPDATE and POST_UPDATE hooks [1340,1360].
 - RML logging is on by default [1318].
 - Changed default ENKF_ALPHA value to 3.0 [??]
 - Implemented subsspacce inversion algorithm [1334, 1344].

libgeometry:

 - Added function to create new geo_surface (i.e. IRAP) object [1308].
 - Get (x, y) pos from surface [1395].

Python code:

 - cwrap has been moved out to stand-alone module, out of ert
   package [1159, 1320, 1325, 1326, 1328, 1332, 1338, 1341, 1343, 1347, 1350, 1353]
 - Simplified loading of shared libraries [1234].
 - Python3 preparations [1231, 1347].
 - Added __repr__ methods: [1266, 1327, 1329, 1331, 1348, 1373, 1375, 1377, 1384, 1387].
 - Implement __getitem__( ) for gen_data [1331].
 - Removed cstring_obj Python class [1387].
 - EclKW.numpy_array returns shared buffer ndarray [1180].
 - Minor bug in ecl_kw.py [1171].
 - Added EclKW.numpyView( ) and EclKW.numpyCopy( ) [1188].
 - Bug in EclKW slice access [1203].
 - Expose active_list better in Python [1392].
 - @TYPE@_vector suppports negative indices in __getitem__ and
   __setitem__; added repr method [1378].
 - added root() methdo ert/__init__.py [1293].

GUI / Configuration / Documentation

 - Bug when viewing plots while simulating [1157.]
 - Bug when plotting short vectors [1303].
 - Completely refactored the ERT Gui event system [1158, 1162].
 - Marked keywords INIT_SECTION and SCHEDULE_FILE as deprecated [1181].
 - Removed outdated keywords from documentation [1390].
 - Documented UMASK keyword [1186].
 - ConfigParser: Can turn off validation + warnings [1233, 1249, 1287].
 - Make ies advanced option only [1401].
 - Removing MAX_RUNNING_LOCAL and MAX_RUNNING_LSF from user doc [1398].
 - Apply plot style to other plots [1397].
 - Fig bug in initialization when out of range [1394].
 - Added new object for generic config settings [1391].
 - Changes to plot settings [11359,376,1380,1382,1386].
 - Fix bug in load case manually [1368].
 - Documentation of plugins [1194].
 - Changed all time handling to UTC. This will affect loading old cases [1229, 1257].
 - Removed keyword QC_PATH + updated GRID [1263].
 - Making sure the ertshell is creating the run path [1280].
 - Create Doxygen [1277,1278,1294,1309,1317].
 - Ability to run analysis from GUI [1314].
 - Improved documentation of priors [1319].
 - Bug in config parsing with relative paths [1333].
 - Field documentation updates [1337].


libwecl_well:

  - Internalize rates for wells and connections in the well library
    [1403].
  - New function well_ts_get_name() [1393].

libutil:

  - Functions for parsing and outputting dates in ISO format[1248].
  - stringlist_join - like Python str.join [1243].
  - bug in matrix_dgemm [1286].
  - Resurrected block_fs utilities from the past [1297].
  - Slicing for runpath_list [1356].
