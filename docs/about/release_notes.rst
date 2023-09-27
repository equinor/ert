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

Version 6.0
------------

Breaking changes:
  - Use parameters from storage instead of ensemble_config (`#5674 <https://github.com/equinor/ert/pull/5674>`_)
  - Remove ANALYSIS_COPY (`#5826 <https://github.com/equinor/ert/pull/5826>`_)
  - Pass in storage to BatchSimulator.start (`#5656 <https://github.com/equinor/ert/pull/5656>`_)
  - Change init for genKw (`#5637 <https://github.com/equinor/ert/pull/5637>`_)
  - Remove ActiveList (`#5695 <https://github.com/equinor/ert/pull/5695>`_)
  - Don't allow creating new case with existing name. (`#5861 <https://github.com/equinor/ert/pull/5861>`_)

Bugfixes:
  - Resolve timeout problems around running a high number of realizations (`#5597 <https://github.com/equinor/ert/pull/5597>`_)
  - Fix FLOW forward model (`#5588 <https://github.com/equinor/ert/pull/5588>`_)
  - Fix bug where log values were not included in facade function for gen kw (`#5643 <https://github.com/equinor/ert/pull/5643>`_)
  - Cover previously unhandled job queue state `JOB_QUEUE_DO_KILL_NODE_FAILURE` (`#5667 <https://github.com/equinor/ert/pull/5667>`_)
  - Fix plotting of parameters from ensembles made with manual update (`#5700 <https://github.com/equinor/ert/pull/5700>`_)
  - Fix bug with gen_kw templating (`#5822 <https://github.com/equinor/ert/pull/5822>`_)
  - Fix validation of general observations ERROR/VALUE (`#5823 <https://github.com/equinor/ert/pull/5823>`_)
  - Do no create experiment and case on start-up (`#5799 <https://github.com/equinor/ert/pull/5799>`_)
  - Fix an issue with double comments (`#5824 <https://github.com/equinor/ert/pull/5824>`_)
  - Fix incorrect suggestion for RUNPATH deprecation (`#5856 <https://github.com/equinor/ert/pull/5856>`_)
  - Fix error message for OBS_FILE not showing correct location (`#5878 <https://github.com/equinor/ert/pull/5878>`_)
  - Show a helpful error message when reading time map fails (`#5882 <https://github.com/equinor/ert/pull/5882>`_)
  - Several mitigations against crashes when running with many realizations (`#5828 <https://github.com/equinor/ert/pull/5828>`_)
  - Backport Fix INDEX_FILE reading (`#5935 <https://github.com/equinor/ert/pull/5935>`_)
  - Improved validation of the FIELD keyword (`#5985 <https://github.com/equinor/ert/pull/5985>`_)
  - Remove duplicates from summary files (`#6117 <https://github.com/equinor/ert/pull/6117>`_)
  - Fix bug where parameter names were not sorted (`#5625 <https://github.com/equinor/ert/pull/5625>`_)
  - Fix observations returned from dark storage (`#5611 <https://github.com/equinor/ert/pull/5611>`_)
  - Fix dimensions of surfaces and fields (`#5660 <https://github.com/equinor/ert/pull/5660>`_)
  - Fix bug in es-mda where returned variable could potentially be unbound (`#5683 <https://github.com/equinor/ert/pull/5683>`_)
  - Fix bug where all deactivated GEN_DATA would crash (`#5784 <https://github.com/equinor/ert/pull/5784>`_)
  - Fix ES-MDA iteration being 0 (`#5846 <https://github.com/equinor/ert/pull/5846>`_)
  - Fix bug where truncated normal was not truncated (`#6110 <https://github.com/equinor/ert/pull/6110>`_)

New features:
  - Make refcase only required if using HISTORY_OBSERVATION (`#5830 <https://github.com/equinor/ert/pull/5830>`_)
  - Remove the old config parser (`#5657 <https://github.com/equinor/ert/pull/5657>`_)
  - Only make time map required if DATE is used in GEN_OBS (`#5805 <https://github.com/equinor/ert/pull/5805>`_)

Improvements:
  - Save template for GenKW in experiment (`#5719 <https://github.com/equinor/ert/pull/5719>`_)
  - Rephrase start simulation button (`#5746 <https://github.com/equinor/ert/pull/5746>`_)
  - Add support for Python 3.11 (`#5789 <https://github.com/equinor/ert/pull/5789>`_)
  - Improve location info in error message (`#5832 <https://github.com/equinor/ert/pull/5832>`_)
  - Display meaningful message upon job queue failure (`#5841 <https://github.com/equinor/ert/pull/5841>`_)
  - Add parameter counts in the GUI (`#5671 <https://github.com/equinor/ert/pull/5671>`_)
  - Exit un-runnable job and fail job (`#5865 <https://github.com/equinor/ert/pull/5865>`_)
  - Validate that observation error is above 0 at startup (`#5879 <https://github.com/equinor/ert/pull/5879>`_)
  - Allow missing observations (`#5658 <https://github.com/equinor/ert/pull/5658>`_)
  - Only write mask to experiment, not full grid (`#5665 <https://github.com/equinor/ert/pull/5665>`_)
  - Allow multiple arguments in workflow job ARGLIST (`#5704 <https://github.com/equinor/ert/pull/5704>`_)
  - Remove websocket connection open/closed from logging (`#5734 <https://github.com/equinor/ert/pull/5734>`_)
  - Move time_map from cpp to python (`#5793 <https://github.com/equinor/ert/pull/5793>`_)
  - Validate field parameter at startup (`#5869 <https://github.com/equinor/ert/pull/5869>`_)
  - Store simulation arguments in experiment folder (`#5710 <https://github.com/equinor/ert/pull/5710>`_)
  - Make sure job_queue will not timeout when sending event (`#5973 <https://github.com/equinor/ert/pull/5973>`_)
  - Drop invalid values, dates when migrating SUMMARY (`#6003 <https://github.com/equinor/ert/pull/6003>`_)
  - Speed up summary loading (`#6133 <https://github.com/equinor/ert/pull/6133>`_)

Miscellaneous:
  - Convert TransferFunction to dataclass (`#5596 <https://github.com/equinor/ert/pull/5596>`_)
  - Remove remnants of prefect (`#5689 <https://github.com/equinor/ert/pull/5689>`_)
  - Remove UPDATE_SETTINGS keyword (`#5783 <https://github.com/equinor/ert/pull/5783>`_)
  - Refactor JobQueue & JobQueueStatus (`#5803 <https://github.com/equinor/ert/pull/5803>`_)
  - Resolve RUNPATH deprecated warnings in generated tests (`#5820 <https://github.com/equinor/ert/pull/5820>`_)
  - Rename Forward models to Jobs in config summary (`#5848 <https://github.com/equinor/ert/pull/5848>`_)
  - Refactor JobQueue & JobQueueStatus (`#5845 <https://github.com/equinor/ert/pull/5845>`_)
  - Ensure that migrating EXT_PARAM throws (`#5618 <https://github.com/equinor/ert/pull/5618>`_)
  - Add logging messages to migration tool (`#5624 <https://github.com/equinor/ert/pull/5624>`_)
  - Remove addHelpToWidget (`#5838 <https://github.com/equinor/ert/pull/5838>`_)


Version 5.0
------------

Breaking changes:
  - ECLBASE now behaves separately from JOBNAME (`#5396 <https://github.com/equinor/ert/pull/5396>`_)
  - New storage solution replacing enkf_fs (`#5030 <https://github.com/equinor/ert/pull/5030>`_)
  - Remove unused field export function (`#5140 <https://github.com/equinor/ert/pull/5140>`_)
  - Changed workflow substitutions to work the same as in the main config file (`#5543 <https://github.com/equinor/ert/pull/5543>`_)
  - Observations parser no longer supports `include` (`#5575 <https://github.com/equinor/ert/pull/5575>`_)
  - DEFINE with whitespace is treated a single key, not multiple (`#5543 <https://github.com/equinor/ert/pull/5543>`_)

Bugfixes:
  - Make steplength settable again in IES (`#5075 <https://github.com/equinor/ert/pull/5075>`_)
  - Verify if active realizations is less than minimum set `#5066 <https://github.com/equinor/ert/pull/5066>`_)
  - Fix check for empty target case `#5125 <https://github.com/equinor/ert/pull/5125>`_)
  - Scale obs errors before outlier detection (`#5126 <https://github.com/equinor/ert/pull/5126>`_)
  - Fix bug where timed out realizations would be marked as success (`#5113 <https://github.com/equinor/ert/pull/5113>`_)
  - Raise expection if no file for refcase `#5163 <https://github.com/equinor/ert/pull/5163>`_)
  - Declare qsub jobs as not rerunnable (`#5173 <https://github.com/equinor/ert/pull/5173>`_)
  - Validate distribution parameters (`#5160 <https://github.com/equinor/ert/pull/5160>`_)
  - Solve race condition in qstat_proxy.sh (`#5182 <https://github.com/equinor/ert/pull/5182>`_)
  - Enable new parser DEFINE key to allow multiple arguments (`#5193 <https://github.com/equinor/ert/pull/5193>`_)
  - Fix strange GUI case name appearing for each run of EnsembleSmoother (`#5223 <https://github.com/equinor/ert/pull/5223>`_)
  - Fix test for missing response (`#5274 <https://github.com/equinor/ert/pull/5274>`_)
  - Change current working directory to config path `#5499 <https://github.com/equinor/ert/pull/5499>`_)
  - Interpret zero length output from qstat as failure (`#5134 <https://github.com/equinor/ert/pull/5134>`_)
  - Fix closing the RunDialog with a file open takes down the entire application (`#5512 <https://github.com/equinor/ert/pull/5512>`_)
  - Trust a nonzero exit value from qstat (`#5534 <https://github.com/equinor/ert/pull/5534>`_)
  - Guard against div-by-zero when min_required_realizations is zero `#5547 <https://github.com/equinor/ert/pull/5547>`_)  -

New features:
  - Added command line option to lint the configuration setup `#5249 <https://github.com/equinor/ert/pull/5249>`_)

Improvements:
  - Move storage meta data from ensemble -> experiment (`#5102 <https://github.com/equinor/ert/pull/5102>`_)
  - Replace text for run button in ERT to 'Open ERT' `#5184 <https://github.com/equinor/ert/pull/5184>`_)
  - Collect config errors before displaying (`#5235 <https://github.com/equinor/ert/pull/5235>`_)
  - Raise ConfigValidationError if max running value or min_realization is not an integer (`#5246 <https://github.com/equinor/ert/pull/5246>`_)
  - Lower memory usage significantly when handling fields/grids (`#5251 <https://github.com/equinor/ert/pull/5251>`_)
  - Add progress updates to Run analysis (`#4986 <https://github.com/equinor/ert/pull/4986>`_)
  - Observation validation errors are now shown in GUI (`#5385 <https://github.com/equinor/ert/pull/5385>`_)
  - New parser for observations used by default (`#5575 <https://github.com/equinor/ert/pull/5575>`_)
  - Reduce memory usage by not unnecessarily creating a copy of the parameters (`#5050 <https://github.com/equinor/ert/pull/5050>`_)
  - Show Error in suggestor when unsupported non-UTF-8 characters are present in the DATA file (`#5072 <https://github.com/equinor/ert/pull/5072>`_)

Miscellaneous:
  - Add timestamps to torque debug log statements (`#5166 <https://github.com/equinor/ert/pull/5166>`_)
  - Add some more logging `#5321 <https://github.com/equinor/ert/pull/5321>`_)
  - Remove duplicate installation from documentation (`#5076 <https://github.com/equinor/ert/pull/5076>`_)
  - Ensemble Config & Enkf Config Node refactor `#5087 <https://github.com/equinor/ert/pull/5087>`_)
  - Change weights via method instead of GUI during test (`#5198 <https://github.com/equinor/ert/pull/5198>`_)
  - Delete unnecessary test of gen_data_config (`#5221 <https://github.com/equinor/ert/pull/5221>`_)
  - Change Torque option TIMEOUT to QUEUE_QUERY_TIMEOUT (`#5218 <https://github.com/equinor/ert/pull/5218>`_)
  - Replace field_config.cpp and surface_config.cpp with dataclass (`#5180 <https://github.com/equinor/ert/pull/5180>`_)
  - Remove .DATA suffix from REFCASE path `#5245 <https://github.com/equinor/ert/pull/5245>`_)
  - Merge tests that were both updating field (`#5175 <https://github.com/equinor/ert/pull/5175>`_)
  - Avoid function call in arguments default (`#5270 <https://github.com/equinor/ert/pull/5270>`_)
  - Remove unused from summary_config (`#5298 <https://github.com/equinor/ert/pull/5298>`_)
  - Refactor gen_kw_config & trans_func `#5265 <https://github.com/equinor/ert/pull/5265>`_)
  - Dont show gui in tests by default (`#5306 <https://github.com/equinor/ert/pull/5306>`_)
  - Remove SCHEDULE_PREDICTION_FILE & GEN_KW PRED `#5317 <https://github.com/equinor/ert/pull/5317>`_)
  - Fix some function names `#5313 <https://github.com/equinor/ert/pull/5313>`_)
  - Refactor _generate_gen_kw_parameter_file `#5335 <https://github.com/equinor/ert/pull/5335>`_)
  - Explicitly use mixed format when converting to dates, avoiding warning (`#5417 <https://github.com/equinor/ert/pull/5417>`_)
  - Refactor GEN_DATA and SUMMARY configuration `#5344 <https://github.com/equinor/ert/pull/5344>`_)
  - Refactor gen_obs extraction of observation values `#5439 <https://github.com/equinor/ert/pull/5439>`_)
  - Refactor EnsembleConfig, EnkfConfigNode & ExtParamConfig `#5436 <https://github.com/equinor/ert/pull/5436>`_)
  - Remove config-node-meta structure, ErtImplType & EnkfVarType enums `#5451 <https://github.com/equinor/ert/pull/5451>`_)
  - Remove unused functions from C codebase `#5490 <https://github.com/equinor/ert/pull/5490>`_)
  - Refactor responses using dataclasses `#5486 <https://github.com/equinor/ert/pull/5486>`_)
  - ExtParamConfig and GenKwConfig refactor (`#5472 <https://github.com/equinor/ert/pull/5472>`_)
  - Remove unused: target format from gen_kw_config (`#5496 <https://github.com/equinor/ert/pull/5496>`_)
  - Clean up following gen_kw_config refactor (`#5497 <https://github.com/equinor/ert/pull/5497>`_)
  - Remove the 'Initialize from existing' tab `#5522 <https://github.com/equinor/ert/pull/5522>`_)
  - Remove unused facade functions (`#5554 <https://github.com/equinor/ert/pull/5554>`_)
  - Remove outdated docs (`#5540 <https://github.com/equinor/ert/pull/5540>`_)
  - Extend summary config to avoid observations adding response configuration (`#5560 <https://github.com/equinor/ert/pull/5560>`_)
  - Unpin SciPy in Ert `#5570 <https://github.com/equinor/ert/pull/5570>`_)
  - Fix bug where y and x increments were set to orientation (`#5573 <https://github.com/equinor/ert/pull/5573>`_)

Other Changes
  - Fix bug where all deactivated GEN_DATA would crash (`#5790 <https://github.com/equinor/ert/pull/5790>`_)
  - Fix saving of fields to use less disk space (`#5666 <https://github.com/equinor/ert/pull/5666>`_)
  - Fix running workflows from CLI (`#5068 <https://github.com/equinor/ert/pull/5068>`_)
  - Show Error in suggestor when unsupported non-UTF-8 characters are present in the config file. (`#5036 <https://github.com/equinor/ert/pull/5036>`_)
  - Open storage inside BatchSimulator (`#5071 <https://github.com/equinor/ert/pull/5071>`_)
  - Rename instances of test_res_config.py to test_ert_config.py and test_ert_config_parsing.py (`#5080 <https://github.com/equinor/ert/pull/5080>`_)
  - Improve documentation for disable_monitoring (`#5116 <https://github.com/equinor/ert/pull/5116>`_)
  - Use case name as ensemble name (`#5136 <https://github.com/equinor/ert/pull/5136>`_)
  - Create hypothesis strategy for observations parsing (`#5117 <https://github.com/equinor/ert/pull/5117>`_)
  - Fix warning and errors not showing up together in the suggestor window (`#5165 <https://github.com/equinor/ert/pull/5165>`_)
  - Expose run information as environmental variables (`#5127 <https://github.com/equinor/ert/pull/5127>`_)
  - Clarify that GEN_KW does not support FORWARD_INIT (`#5232 <https://github.com/equinor/ert/pull/5232>`_)
  - Fix saving surface to file `#5227 <https://github.com/equinor/ert/pull/5227>`_)
  - Sort observation keys before analysis (`#5259 <https://github.com/equinor/ert/pull/5259>`_)
  - Fix flake8-bugbear's B007 on unused loop control variables (`#5286 <https://github.com/equinor/ert/pull/5286>`_)
  - Bump the queue_query timeout in Torque driver (`#5297 <https://github.com/equinor/ert/pull/5297>`_)
  - Add documentation for ERTCASE as a magic string (`#5250 <https://github.com/equinor/ert/pull/5250>`_)
  - Use numpy vectorised funcs in TRANSFORM_FUNCTIONS (`#5268 <https://github.com/equinor/ert/pull/5268>`_)
  - Cleanup old summary key loading logic (`#5329 <https://github.com/equinor/ert/pull/5329>`_)
  - Evaluate min_realizations for ensemble_experiment `#5339 <https://github.com/equinor/ert/pull/5339>`_)
  - Allow quoted substrings as part of the FORWARD_MODEL arglist in new parser (`#5327 <https://github.com/equinor/ert/pull/5327>`_)
  - Add config path and file name to jobs.json (`#5374 <https://github.com/equinor/ert/pull/5374>`_)
  - Add environment variables on client (`#5333 <https://github.com/equinor/ert/pull/5333>`_)
  - Assign all unit tests using snake_oil_case_storage to same xdist thread (`#5390 <https://github.com/equinor/ert/pull/5390>`_)
  - Add to_dict to ParameterConfig `#5372 <https://github.com/equinor/ert/pull/5372>`_)
  - Test out type hints (`#5392 <https://github.com/equinor/ert/pull/5392>`_)
  - Flipping default parser means logging against old (`#5402 <https://github.com/equinor/ert/pull/5402>`_)
  - Refactor keyword handling with regard to meta-data creation (`#5428 <https://github.com/equinor/ert/pull/5428>`_)
  - Make default jobname <CONFIG_FILE>  - (`#5444 <https://github.com/equinor/ert/pull/5444>`_)
  - Completely Simplify gen observation (`#5493 <https://github.com/equinor/ert/pull/5493>`_)
  - Simplify obs_vector (`#5504 <https://github.com/equinor/ert/pull/5504>`_)
  - Fix `--show-gui` (`#5503 <https://github.com/equinor/ert/pull/5503>`_)
  - Make creating jobs.json faster (`#5513 <https://github.com/equinor/ert/pull/5513>`_)
  - Fix an issue where del raises (`#5514 <https://github.com/equinor/ert/pull/5514>`_)
  - Add migration from BlockFs storage (`#4937 <https://github.com/equinor/ert/pull/4937>`_)
  - Fix typos in docs (`#5492 <https://github.com/equinor/ert/pull/5492>`_)
  - Fix order of arguments to parse_arg_types_list (`#5536 <https://github.com/equinor/ert/pull/5536>`_)
  - Add active information loading (`#5326 <https://github.com/equinor/ert/pull/5326>`_)
  - Migration fails gracefully (`#5532 <https://github.com/equinor/ert/pull/5532>`_)
  - Use xarray/netcdf for surfaces (`#5508 <https://github.com/equinor/ert/pull/5508>`_)
  - Generalise parameters in storage (`#5401 <https://github.com/equinor/ert/pull/5401>`_)
  - Do not add _prior suffix to ES runs (`#5594 <https://github.com/equinor/ert/pull/5594>`_)


Version 4.1
------------

Breaking changes:
  - Disable automatic resize of state-map when setting outside map size (`#3951 <https://github.com/equinor/ert/pull/3951>`_)
  - Remove the GEN_PARAM keyword (`#3867 <https://github.com/equinor/ert/pull/3867>`_)
  - Move init and storing of GEN_KW form C to python (`#3943 <https://github.com/equinor/ert/pull/3943>`_)
  - Simplify EclConfig (`#3919 <https://github.com/equinor/ert/pull/3919>`_)
  - Change default runpath (`#4019 <https://github.com/equinor/ert/pull/4019>`_)
  - Remove no longer used min std (`#4057 <https://github.com/equinor/ert/pull/4057>`_)
  - Remove RSH queue driver (`#3962 <https://github.com/equinor/ert/pull/3962>`_)
  - Replace libecl RNG with numpy.random.Generator (`#4033 <https://github.com/equinor/ert/pull/4033>`_)

Bugfixes:
  - Make default ensemble path consistent (`#3982 <https://github.com/equinor/ert/pull/3982>`_)
  - Update torque driver to handle finished jobs (`#3880 <https://github.com/equinor/ert/pull/3880>`_)
  - Fix bug where extra case is created when running ies (`#4049 <https://github.com/equinor/ert/pull/4049>`_)
  - Make sure loading gui produces just one storage folder (`#4053 <https://github.com/equinor/ert/pull/4053>`_)
  - Add grid and grid_file properties back to libres_facade (`#4063 <https://github.com/equinor/ert/pull/4063>`_)
  - Disable "Start simulation" button while running simulations (`#4080 <https://github.com/equinor/ert/pull/4080>`_)
  - Show restart button when all realisations fail (`#4113 <https://github.com/equinor/ert/pull/4113>`_)
  - Propagate error messages from job_queue_node (`#4127 <https://github.com/equinor/ert/pull/4127>`_)
  - Propagate qstat options to qstat executable (`#4138 <https://github.com/equinor/ert/pull/4138>`_)

New features:
  - Consume Protobuf events from dispatcher and pass them to the statemachine (`#3733 <https://github.com/equinor/ert/pull/3733>`_)

Improvements:
  - Improve test coverage of ok callback (`#3860 <https://github.com/equinor/ert/pull/3860>`_)
  - Raise exception if size of gen_kw in storage differs with gen_kw_config (`#3984 <https://github.com/equinor/ert/pull/3984>`_)
  - Save parameters to in-memory storage between update-steps (`#4026 <https://github.com/equinor/ert/pull/4026>`_)
  - Show loading cursor when loading from runpath (`#4094 <https://github.com/equinor/ert/pull/4094>`_)
  - Support Torque job id without a dot character (`#3911 <https://github.com/equinor/ert/pull/3911>`_)
  - Improve error message if a parameter is missing from storage (`#4119 <https://github.com/equinor/ert/pull/4119>`_)
  - Move lock file to ENSPATH (`#4122 <https://github.com/equinor/ert/pull/4122>`_)
  - Mute external DEBUG messages (`#3981 <https://github.com/equinor/ert/pull/3981>`_)
  - Speed up realizations by moving ert.job_runner to _ert_job_runner (`#4076 <https://github.com/equinor/ert/pull/4076>`_)

Dependencies:
  - Relax protobuf pin to "<4" (`#3968 <https://github.com/equinor/ert/pull/3968>`_)
  - Define usage of setuptools_scm in pyproject.toml (`#4081 <https://github.com/equinor/ert/pull/4081>`_)

Miscellaneous:
  - Log experiment type and size when a run model is instantiated (`#3967 <https://github.com/equinor/ert/pull/3967>`_)
  - Remove unused function weakref from EnkfFs (`#3989 <https://github.com/equinor/ert/pull/3989>`_)
  - Remove copyright notices in .py, .cpp and .hpp files (`#3582 <https://github.com/equinor/ert/pull/3582>`_)
  - Change num cpu interface / usage and default value (`#3988 <https://github.com/equinor/ert/pull/3988>`_)
  - Remove outdated GEN_DATA docs (`#3997 <https://github.com/equinor/ert/pull/3997>`_)
  - Move enkf logic from enkf_main to fs_manager (`#3807 <https://github.com/equinor/ert/pull/3807>`_)
  - Remove unused code related to enkf_node (`#4066 <https://github.com/equinor/ert/pull/4066>`_)
  - Remove HistorySourceEnum.SCHEDULE (`#4097 <https://github.com/equinor/ert/pull/4097>`_)
  - Move Eclipse Grid and Refcase to EnsembleConfig (`#4100 <https://github.com/equinor/ert/pull/4100>`_)
  - Remove bunch of stuff from ensemble config (`#4075 <https://github.com/equinor/ert/pull/4075>`_)
  - Initialize res config form minimal dictionary  (`#3952 <https://github.com/equinor/ert/pull/3952>`_)
  - Make default __repr__ on BaseCClasses (`#3963 <https://github.com/equinor/ert/pull/3963>`_)
  - Add automatic typecasting from cwrap to C++` (`#3971 <https://github.com/equinor/ert/pull/3971>`_)
  - Ignore type errors in key_manager (`#3973 <https://github.com/equinor/ert/pull/3973>`_)
  - Convert equinor test to local test (`#3961 <https://github.com/equinor/ert/pull/3961>`_)
  - Update snake oil field test (`#3983 <https://github.com/equinor/ert/pull/3983>`_)
  - Remove local folder from test-data (`#3996 <https://github.com/equinor/ert/pull/3996>`_)
  - Simplify queue_config (`#3957 <https://github.com/equinor/ert/pull/3957>`_)
  - Simplify site config (`#4003 <https://github.com/equinor/ert/pull/4003>`_)
  - Simplify ert config builder (`#4022 <https://github.com/equinor/ert/pull/4022>`_)
  - Simplify run workflows (`#4009 <https://github.com/equinor/ert/pull/4009>`_)
  - Simplify analysis config (`#4034 <https://github.com/equinor/ert/pull/4034>`_)
  - Remove unused hook_manager_run_workflow (`#4008 <https://github.com/equinor/ert/pull/4008>`_)
  - Document the ERT Storage Server database model (`#3683 <https://github.com/equinor/ert/pull/3683>`_)
  - Remove hook manager (`#4012 <https://github.com/equinor/ert/pull/4012>`_)
  - Pass site config to workflow list (`#4016 <https://github.com/equinor/ert/pull/4016>`_)
  - Fix test_config_parsing generation and site config (`#4023 <https://github.com/equinor/ert/pull/4023>`_)
  - Fix ext joblist (`#4025 <https://github.com/equinor/ert/pull/4025>`_)
  - Fix ModelConfig default runpathformat (`#4029 <https://github.com/equinor/ert/pull/4029>`_)
  - Remove collectors and keymanager (`#4027 <https://github.com/equinor/ert/pull/4027>`_)
  - Remove unused functionality from plot_data (`#4031 <https://github.com/equinor/ert/pull/4031>`_)
  - Fix missing config_node free function (`#4058 <https://github.com/equinor/ert/pull/4058>`_)
  - Move forward_model_ok into Python (`#4038 <https://github.com/equinor/ert/pull/4038>`_)
  - Update ConfigContent to_dict functionality (`#4052 <https://github.com/equinor/ert/pull/4052>`_)
  - Remove libecl-style "type-safety" (`#4051 <https://github.com/equinor/ert/pull/4051>`_)
  - Remove references to Equinor test data (`#4040 <https://github.com/equinor/ert/pull/4040>`_)
  - Undeprecate the DATA_FILE keyword and add documentation (`#4017 <https://github.com/equinor/ert/pull/4017>`_)
  - Job runner yields Running event only when memory consumption has significant change (`#4067 <https://github.com/equinor/ert/pull/4067>`_)
  - Remove WORKFLOW and FORWARD_MODEL before logging user config (`#4085 <https://github.com/equinor/ert/pull/4085>`_)
  - Rename tests directories (`#4030 <https://github.com/equinor/ert/pull/4030>`_)
  - Setup log file for job_dispatch logger (`#3999 <https://github.com/equinor/ert/pull/3999>`_)
  - Update webviz-ert documentation (`#4090 <https://github.com/equinor/ert/pull/4090>`_)
  - log_process_usage in finally rather than atexit (`#4087 <https://github.com/equinor/ert/pull/4087>`_)
  - Delete site config c code (`#4020 <https://github.com/equinor/ert/pull/4020>`_)
  - Update documentation of HISTORY_OBSERVATION > ERROR (`#4032 <https://github.com/equinor/ert/pull/4032>`_)
  - Emit deprecation warning for non-ISO dates in observation config files (`#3958 <https://github.com/equinor/ert/pull/3958>`_)
  - Update docs for DATA_FILE (`#4104 <https://github.com/equinor/ert/pull/4104>`_)
  - Analysis iter config pure dataclass (`#4082 <https://github.com/equinor/ert/pull/4082>`_)
  - Make forward_model only called from python (`#4137 <https://github.com/equinor/ert/pull/4137>`_)
  - Fix komodo tests (`#4142 <https://github.com/equinor/ert/pull/4142>`_)
  - Move C implementation for ies_config analysis_config and analysis_module in python (`#4133 <https://github.com/equinor/ert/pull/4133>`_)
  - Account for instance where ERT config file has `MAX_RUNTIME` defined multiple times (`#4147 <https://github.com/equinor/ert/pull/4147>`_)


Version 4.0
------------

Breaking changes:
  - Stop special casing PRED as a GEN_KW (`#3820 <https://github.com/equinor/ert/pull/3820>`_)
  - Remove BLOCK_OBSERVATION keyword (`#3732 <https://github.com/equinor/ert/pull/3732>`_)
  - Remove UMASK config option (`#3892 <https://github.com/equinor/ert/pull/3892>`_)
  - Remove CONTAINER keyword (`#3834 <https://github.com/equinor/ert/pull/3834>`_)
  - Remove CONTAINER keyword (`#3834 <https://github.com/equinor/ert/pull/3834>`_)
  - Remove BINARY_FLOAT and BINARY_DOUBLE file formats (`#3947 <https://github.com/equinor/ert/pull/3947>`_)

Bugfixes:
  - Fix bug where random seed would overflow (`#3863 <https://github.com/equinor/ert/pull/3863>`_)
  - Fix has_data check in initRun (`#3964 <https://github.com/equinor/ert/pull/3964>`_)
  - Free obs_data in load_observations_and_responses (`#3916 <https://github.com/equinor/ert/pull/3916>`_)

New features:
  - Add a cli argument for specifying a log dir (`#3819 <https://github.com/equinor/ert/pull/3819>`_)
  - Add DisableParametersUpdate workflow (`#3861 <https://github.com/equinor/ert/pull/3861>`_)
  - Add Experiment server for CLI (`#3768 <https://github.com/equinor/ert/pull/3768>`_)

Improvements:
  - Overwrite Storage config file if it exits on disk (`#3913 <https://github.com/equinor/ert/pull/3913>`_)
  - Use variable defining matrix start size (`#3856 <https://github.com/equinor/ert/pull/3856>`_)

Dependencies:
  - Pin beartype to less than 0.11 (`#3904 <https://github.com/equinor/ert/pull/3904>`_)
  - Pin protobuf and grpcio-tools (`#3909 <https://github.com/equinor/ert/pull/3909>`_)
  - Run isort and add isort checking to CI (`#3812 <https://github.com/equinor/ert/pull/3812>`_)
  - Fix deprecation warning in py310 from setDaemon (`#3848 <https://github.com/equinor/ert/pull/3848>`_)
  - Move the iterative ensemble smoother to its own repository (`#3844 <https://github.com/equinor/ert/pull/3844>`_)

Miscellaneous:
  - Move rng creation to python (`#3843 <https://github.com/equinor/ert/pull/3843>`_)
  - Remove unused template (`#3827 <https://github.com/equinor/ert/pull/3827>`_)
  - Remove unused function state_map::count_matching (`#3549 <https://github.com/equinor/ert/pull/3459>`_)
  - Mute matplotlib debug messages (`#3826 <https://github.com/equinor/ert/pull/3826>`_)
  - Remove Title Case from documentation (`#3821 <https://github.com/equinor/ert/pull/3821>`_)
  - Fix typo and formatting in keyword documentation (`#3818 <https://github.com/equinor/ert/pull/3818>`_)
  - Test update with externally sampled params (`#3722 <https://github.com/equinor/ert/pull/3722>`_)
  - Remove unused strict-keyword from EnkfMain (`#3835 <https://github.com/equinor/ert/pull/3835>`_)
  - Remove unused functions in queue.py and enkf_config_node.py (`#3852 <https://github.com/equinor/ert/pull/3852>`_)
  - Test update with INIT_FILES and FORWARD_INIT (`#3846 <https://github.com/equinor/ert/pull/3846>`_)
  - Mute connection closed error from websocket (`#3814 <https://github.com/equinor/ert/pull/3814>`_)
  - Close stdin/stdout/stderr files when done (`#3849 <https://github.com/equinor/ert/pull/3849>`_)
  - Remove getters only used in tests from enkf_main.cpp (`#3895 <https://github.com/equinor/ert/pull/3895>`_)
  - Remove non-functional private mode for EnkfNode (`#3874 <https://github.com/equinor/ert/pull/3874>`_)
  - Move enkf_fs method from enkf_main to enkf_fs (`#3900 <https://github.com/equinor/ert/pull/3900>`_)
  - Move parameter keys to ensemble_config (`#3901 <https://github.com/equinor/ert/pull/3901>`_)
  - Solve pylint consider-using-with (`#3850 <https://github.com/equinor/ert/pull/3850>`_)
  - enkf_obs clean-up (`#3917 <https://github.com/equinor/ert/pull/3917>`_)
  - Clean-up of enkf_node (`#3926 <https://github.com/equinor/ert/pull/3926>`_)
  - Remove enkf_main from C (`#3924 <https://github.com/equinor/ert/pull/3924>`_)
  - Solve and enforce 9 pylint messages (`#3730 <https://github.com/equinor/ert/pull/3730>`_)
  - Solve pylint issue no-member (`#3851 <https://github.com/equinor/ert/pull/3851>`_)
  - Remove ert3 (`#3877 <https://github.com/equinor/ert/pull/3877>`_)
  - Fix unnecessary cast from const double to double (`#3832 <https://github.com/equinor/ert/pull/3832>`_)
  - Propagate ensemble id for source when building (`#3793 <https://github.com/equinor/ert/pull/3793>`_)
  - Update documentation of SUMMARY keyword (`#3824 <https://github.com/equinor/ert/pull/3824>`_)
  - Remove unused function enkf_main_load_obs (`#3853 <https://github.com/equinor/ert/pull/3853>`_)
  - Allow c++ as compiler in build script (`#3794 <https://github.com/equinor/ert/pull/3794>`_)
  - Disable flaky test (`#3869 <https://github.com/equinor/ert/pull/3869>`_)
  - Fix config reprs (`#3876 <https://github.com/equinor/ert/pull/3876>`_)
  - Generate experiment_id and propagate to communication channels (`#3811 <https://github.com/equinor/ert/pull/3811>`_)
  - Remove some remaining references to libres (`#3878 <https://github.com/equinor/ert/pull/3878>`_)
  - Remove deprecation limit on ert.data.loader.load_general_data and load.summary.data (`#3883 <https://github.com/equinor/ert/pull/3883>`_)
  - Move responsibility of creating the log folder into the writing of the update report logic (`#3866 <https://github.com/equinor/ert/pull/3866>`_)
  - Minor config fixes (`#3858 <https://github.com/equinor/ert/pull/3858>`_)
  - Remove ert_test_context (`#3879 <https://github.com/equinor/ert/pull/3879>`_)
  - Replace unittest with pytest (`#3888 <https://github.com/equinor/ert/pull/3888>`_)
  - Remove unused function get_observations from EnKFMain (`#3891 <https://github.com/equinor/ert/pull/3891>`_)
  - Remove unused but declared exceptions (`#3896 <https://github.com/equinor/ert/pull/3896>`_)
  - Remove unused history functions (`#3894 <https://github.com/equinor/ert/pull/3894>`_)
  - Resolve some mypy typing issues (`#3898 <https://github.com/equinor/ert/pull/3898>`_)
  - Consolidate unit tests (`#3899 <https://github.com/equinor/ert/pull/3899>`_)
  - Move storage_service.json to enspath  and propagate the ensepath when starting the webviz_ert service (`#3890 <https://github.com/equinor/ert/pull/3890>`_)
  - Simplify res config (`#3908 <https://github.com/equinor/ert/pull/3908>`_)
  - Reconcile location in tests and location in src (`#3914 <https://github.com/equinor/ert/pull/3914>`_)
  - Remove rng_config (`#3920 <https://github.com/equinor/ert/pull/3920>`_)
  - Remove c usage of res_config  (`#3922 <https://github.com/equinor/ert/pull/3922>`_)
  - Remove reading of site_config directly from file (`#3931 <https://github.com/equinor/ert/pull/3931>`_)
  - Remove direct init of job_queue from file (`#3933 <https://github.com/equinor/ert/pull/3933>`_)
  - Remove unused analysis_config directly from file (`#3932 <https://github.com/equinor/ert/pull/3932>`_)
  - Add string representation of ResConfig (`#3928 <https://github.com/equinor/ert/pull/3928>`_)
  - Consolidate config file and dict paths for substitution config (`#3887 <https://github.com/equinor/ert/pull/3887>`_)
  - Remove unused model config functions (`#3934 <https://github.com/equinor/ert/pull/3934>`_)
  - Convert equinor test to local test (`#3944 <https://github.com/equinor/ert/pull/3944>`_)
  - Clean up unused c code and superfluous `extern "C"` (`#3941 <https://github.com/equinor/ert/pull/3941>`_)
  - Fix experiment server iterated ensemble smoother (`#3950 <https://github.com/equinor/ert/pull/3950>`_)
  - Initialize AnalysisIterConfig from dict (`#3946 <https://github.com/equinor/ert/pull/3946>`_)
  - Simplify which keywords are added for parsing (`#3942 <https://github.com/equinor/ert/pull/3942>`_)
  - Make enkf_main.log_seed more C++ (`#3945 <https://github.com/equinor/ert/pull/3945>`_)
  - Remove RSH_DRIVER from test config dict generation (`#3955 <https://github.com/equinor/ert/pull/3955>`_)


Version 3.0
------------

Breaking changes:
  - Remove END_DATE keyword (`#3737 <https://github.com/equinor/ert/pull/3737>`_)
  - Remove RFTPATH keyword (`#3746 <https://github.com/equinor/ert/pull/3746>`_)
  - Remove REFCASE_LIST keyword (`#3745 <https://github.com/equinor/ert/pull/3745>`_)
  - Remove PRED as a reserved name for GEN_KW (`#3820 <https://github.com/equinor/ert/pull/3820>`_)
  - Change dates written by ERT to ISO-8601 (`#3755 <https://github.com/equinor/ert/pull/3755>`_)
  - Fix the logging path when running the GUI (`#3772 <https://github.com/equinor/ert/pull/3772>`_)
  - Simplify ErtRunContext and rename to RunContext (`#3660 <https://github.com/equinor/ert/pull/3660>`_)
  - Remove workflows related to case management (`#3687 <https://github.com/equinor/ert/pull/3687>`_)
  - Remove unused workflows EXPORT_FIELD (`#3715 <https://github.com/equinor/ert/pull/3715>`_)
  - Remove last internal C workflows EXIT_ERT and PRE_SIMULATION_COPY (`#3716 <https://github.com/equinor/ert/pull/3716>`_)
  - Use src/ directory for packages (`#3633 <https://github.com/equinor/ert/pull/3633>`_)
  - Move ert3 to ert/ert3 (`#3648 <https://github.com/equinor/ert/pull/3648>`_)
  - Move ert3 examples (`#3652 <https://github.com/equinor/ert/pull/3652>`_)
  - Move ert_logging to ert.logging (`#3654 <https://github.com/equinor/ert/pull/3654>`_)
  - Move ert_data to ert.data (`#3655 <https://github.com/equinor/ert/pull/3655>`_)
  - Move ert_shared to ert.shared (`#3752 <https://github.com/equinor/ert/pull/3752>`_)
  - Move job_runner to ert.job_runner (`#3684 <https://github.com/equinor/ert/pull/3684>`_)
  - Move ert_gui -> ert/gui (`#3625 <https://github.com/equinor/ert/pull/3625>`_)
  - Move res and make private (`#3761 <https://github.com/equinor/ert/pull/3761>`_)
  - Move out res.test.synthesizer to separate package oil_reservoir_synthesizer by (`#3696 <https://github.com/equinor/ert/pull/3696>`_)
  - Refactor ert/analysis import structure (`#3665 <https://github.com/equinor/ert/pull/3665>`_)
  - Remove case-log file (`#3779 <https://github.com/equinor/ert/pull/3779>`_)
  - Join EnKFMain, _RealEnKFMain, EnkfFsManager and EnkfJobRunner in python (`#3705 <https://github.com/equinor/ert/pull/3705>`_)
  - Remove enkf_main alloc enkf_fs from symlink (`#3773 <https://github.com/equinor/ert/pull/3773>`_)
  - Change how EnkfFs classmethod is instantiated (`#3777 <https://github.com/equinor/ert/pull/3777>`_)
  - Move runSimpleStep from enkf_main to simulation_context and make private (`#3785 <https://github.com/equinor/ert/pull/3785>`_)
  - Remove hidden case concept (`#3786 <https://github.com/equinor/ert/pull/3786>`_)

Bugfixes:
  - Retry qsub and qstat in case of failures (`#3537 <https://github.com/equinor/ert/pull/3537>`_)

New features:
  - Add a cli argument for specifying a log dir (`#3819 <https://github.com/equinor/ert/pull/3819>`_)
  - Introduce basic experiment server (`#3438 <https://github.com/equinor/ert/pull/3438>`_)

Improvements:
  - Add proxy script for qstat (`#3553 <https://github.com/equinor/ert/pull/3553>`_)
  - Send fewer but larger dataChanged signals (`#3605 <https://github.com/equinor/ert/pull/3605>`_)
  - Stop logging to stderr in the config_parser (`#3753 <https://github.com/equinor/ert/pull/3753>`_)
  - Use standard Qt APIs in GUI proxy models (`#3615 <https://github.com/equinor/ert/pull/3615>`_)
  - Sort messages in simulation panel so they more likely ordered by realization (`#3641 <https://github.com/equinor/ert/pull/3641>`_)
  - Fix the logging path when running the GUI (`#3772 <https://github.com/equinor/ert/pull/3772>`_)

Deprecations:
  - Deprecated DATA_FILE keyword (`<https://github.com/equinor/ert/pull/3751>`_)

Miscellaneous:
  - Fix flaky test for qstat concurrency (`#3738 <https://github.com/equinor/ert/pull/3738>`_)
  - Remove forward load context (`#3728 <https://github.com/equinor/ert/pull/3728>`_)
  - Replace util_binary_split_string with standard C++ (`#3702 <https://github.com/equinor/ert/pull/3702>`_)
  - Move docs/rst/manual/ to docs/ (`#3692 <https://github.com/equinor/ert/pull/3692>`_)
  - Solve pylint issue consider-using-generator (`#3585 <https://github.com/equinor/ert/pull/3585>`_)
  - Move handling of runpath and substitutions to python (`#3583 <https://github.com/equinor/ert/pull/3583>`_)
  - Solve pylint issue arguments-renamed (`#3586 <https://github.com/equinor/ert/pull/3586>`_)
  - Add annotation of errors (`#3626 <https://github.com/equinor/ert/pull/3626>`_)
  - Use --strict_markers for pytest (`#3664 <https://github.com/equinor/ert/pull/3664>`_)
  - Clean up ResTest and convert some tests to pytest (`#3635 <https://github.com/equinor/ert/pull/3635>`_)
  - Remove warning about use of restart (`#3632 <https://github.com/equinor/ert/pull/3632>`_)
  - Remove py36 specifics in tests (`#3672 <https://github.com/equinor/ert/pull/3672>`_)
  - Ensure jenkins tests see pyproject.toml (`#3668 <https://github.com/equinor/ert/pull/3668>`_)
  - Remove wrappers of run_arg_alloc (`#3666 <https://github.com/equinor/ert/pull/3666>`_)
  - Make IES-implementation similar to paper (`#3122 <https://github.com/equinor/ert/pull/3122>`_)
  - Remove unused callback_arg (`#3675 <https://github.com/equinor/ert/pull/3675>`_)
  - Add missing logger-method to _Proc (`#3686 <https://github.com/equinor/ert/pull/3686>`_)
  - Remove res_config getters from EnKFMain (`#3679 <https://github.com/equinor/ert/pull/3679>`_)
  - Remove outdated GUI resources (`#3689 <https://github.com/equinor/ert/pull/3689>`_)
  - Remove python implementation of ForwardLoadContext (`#3694 <https://github.com/equinor/ert/pull/3694>`_)
  - Remove outdated install script (`#3695 <https://github.com/equinor/ert/pull/3695>`_)
  - Move test_stop_running into test_job_queue (`#3709 <https://github.com/equinor/ert/pull/3709>`_)
  - Change enkf main init (`#3690 <https://github.com/equinor/ert/pull/3690>`_)
  - Fix flaky bug in test_ert_qstat_proxy (`#3731 <https://github.com/equinor/ert/pull/3731>`_)
  - Adjust the still flaky qstat_concurrent_invocations (`#3744 <https://github.com/equinor/ert/pull/3744>`_)
  - Replace internal C tests with pytest (`#3741 <https://github.com/equinor/ert/pull/3741>`_)
  - Update default branch name in README (`#3742 <https://github.com/equinor/ert/pull/3742>`_)
  - Remove unnecessary fixtures (`#3735 <https://github.com/equinor/ert/pull/3735>`_)
  - Remove unused function enkf_config_node_get_FIELD_fill_file (`#3721 <https://github.com/equinor/ert/pull/3721>`_)
  - Solve warnings emitted during pytesting (`#3764 <https://github.com/equinor/ert/pull/3764>`_)
  - Pin setuptools to <64 (`#3771 <https://github.com/equinor/ert/pull/3771>`_)
  - Remove init_internalization from enkf_main (`#3719 <https://github.com/equinor/ert/pull/3719>`_)
  - Use id for ensemble instead of ensemble evaluator (`#3724 <https://github.com/equinor/ert/pull/3724>`_)
  - Upgrade to cloudevents 1.6.0 (`#3784 <https://github.com/equinor/ert/pull/3784>`_)
  - Refactor FileSystemRotator (`#3788 <https://github.com/equinor/ert/pull/3788>`_)
  - Update undefined (`#3797 <https://github.com/equinor/ert/pull/3797>`_)
  - Remove ecl_write from EnkfNode (`#3750 <https://github.com/equinor/ert/pull/3750>`_)
  - Update cmake build instructions in readme (`#3799 <https://github.com/equinor/ert/pull/3799>`_)
  - Use pytest in test_exporter, test_libres_facade (`#3614 <https://github.com/equinor/ert/pull/3614>`_)
  - Add a read_only flag to enkf_main (`#3550 <https://github.com/equinor/ert/pull/3550>`_)
  - config_parser: Make paths absolute (`#3624 <https://github.com/equinor/ert/pull/3624>`_)
  - Restrict annotated files to existing (`#3634 <https://github.com/equinor/ert/pull/3634>`_)
  - Skip flaky test of experiment server (`#3645 <https://github.com/equinor/ert/pull/3645>`_)
  - Use Ubuntu 20.04 for running spe1 workflow (`#3651 <https://github.com/equinor/ert/pull/3651>`_)
  - Fix linting errors (`#3650 <https://github.com/equinor/ert/pull/3650>`_)
  - Expose MeasuredData through ert (`#3656 <https://github.com/equinor/ert/pull/3656>`_)
  - Remove logging of conn_info (`#3659 <https://github.com/equinor/ert/pull/3659>`_)
  - Run ert3 tests in separation (`#3657 <https://github.com/equinor/ert/pull/3657>`_)
  - Rewrite run context to python only (`#3649 <https://github.com/equinor/ert/pull/3649>`_)
  - Reduce output from pytest in ci (`#3653 <https://github.com/equinor/ert/pull/3653>`_)
  - Migrate to flake8 5.0.0 compatible config (`#3661 <https://github.com/equinor/ert/pull/3661>`_)
  - Fix duplicate missing package data in setup.py (`#3662 <https://github.com/equinor/ert/pull/3662>`_)
  - Small fix for running experiment_server (`#3642 <https://github.com/equinor/ert/pull/3642>`_)
  - Skip failing test (`#3671 <https://github.com/equinor/ert/pull/3671>`_)
  - Increase timeout on experiment_server integration test (`#3685 <https://github.com/equinor/ert/pull/3685>`_)
  - Log maximum memory usage in main ERT thread (`#3357 <https://github.com/equinor/ert/pull/3357>`_)
  - Do not generate certificates for cli tests (`#3691 <https://github.com/equinor/ert/pull/3691>`_)
  - Fix format scripts (`#3701 <https://github.com/equinor/ert/pull/3701>`_)
  - Remove creation of certs and tokens in tests (`#3700 <https://github.com/equinor/ert/pull/3700>`_)
  - Remove unused functions in enkf_main (`#3714 <https://github.com/equinor/ert/pull/3714>`_)
  - Remove ERT splash + contextmanage GUILogHandler (`#3717 <https://github.com/equinor/ert/pull/3717>`_)
  - Resolve circular imports (`#3736 <https://github.com/equinor/ert/pull/3736>`_)
  - Add functionality for semeio in LibresFacade (`#3743 <https://github.com/equinor/ert/pull/3743>`_)
  - Refactor StateMap into C++ & pybind11 (`#3693 <https://github.com/equinor/ert/pull/3693>`_)
  - Avoid circular install of webviz_ert (`#3757 <https://github.com/equinor/ert/pull/3757>`_)
  - Fix broken imports (`#3763 <https://github.com/equinor/ert/pull/3763>`_)
  - Add <except.hpp> for convenient C++ exceptions (`#3762 <https://github.com/equinor/ert/pull/3762>`_)
  - Remove unused modules from coverage test (`#3769 <https://github.com/equinor/ert/pull/3769>`_)
  - Fix broken shell scripts (`#3776 <https://github.com/equinor/ert/pull/3776>`_)
  - Fetch storage server name, i.e. the config name, and use as title in webviz-ert (`#3767 <https://github.com/equinor/ert/pull/3767>`_)
  - Protobuf job_runner.reporter (`#3620 <https://github.com/equinor/ert/pull/3620>`_)
  - Move all enkf_main interaction with enkf_fs into python (`#3775 <https://github.com/equinor/ert/pull/3775>`_)
  - Remove refcounting from enkf_fs (`#3789 <https://github.com/equinor/ert/pull/3789>`_)
  - Remove logging from umount (`#3803 <https://github.com/equinor/ert/pull/3803>`_)
  - Test initializing from config_dict via hypothesis (`#3796 <https://github.com/equinor/ert/pull/3796>`_)

Version 2.37
------------

Breaking changes:
  - Refactor ies_data.cpp (`#3439 <https://github.com/equinor/ert/pull/3439>`_)
  - Make Qt plotter utilizing the ert-api (`#3458 <https://github.com/equinor/ert/pull/3458>`_)
  - Refactor creating data for running analysis (`#3473 <https://github.com/equinor/ert/pull/3473>`_)
  - Refactor code paths for IES and ES (`#3476 <https://github.com/equinor/ert/pull/3476>`_)
  - Remove UPDATE_RUNPATH_LIST workflow (`#3554 <https://github.com/equinor/ert/pull/3554>`_)
  - Drop support for Python 3.6 and Python 3.7 (`#3490 <https://github.com/equinor/ert/pull/3490>`_)
  - Change EXPORT_MISFIT_DATA workflow to only export to single file (`#3573 <https://github.com/equinor/ert/pull/3573>`_)

Bugfixes:
  - Use higher resolution when checking modification-time for target-file (`#3428 <https://github.com/equinor/ert/pull/3428>`_)
  - Fix iteration nr bug in IES and add ies cli integration test (`#3457 <https://github.com/equinor/ert/pull/3457>`_)
  - Release GIL before waiting for (async) futures in C++ (`#3450 <https://github.com/equinor/ert/pull/3450>`_)
  - Add a filter to the log messages in base_run_model (`#3598 <https://github.com/equinor/ert/pull/3598>`_)
  - Make ensemble client handle TimeoutError (`#3612 <https://github.com/equinor/ert/pull/3612>`_)

New features:
  - Expose priors in dark-storage endpoint (`#3522 <https://github.com/equinor/ert/pull/3522>`_)

Improvements:
  - Dont retry forward model if inconsistent time map (`#3427 <https://github.com/equinor/ert/pull/3427>`_)
  - Remember plot type when switching between types (`#3447 <https://github.com/equinor/ert/pull/3447>`_)
  - Fix storing initial A matrix in updatA (`#3453 <https://github.com/equinor/ert/pull/3453>`_)
  - Avoid crashing if not connected to network, fallback to localhost (`#3481 <https://github.com/equinor/ert/pull/3481>`_)
  - Remove module name from GUI (`#3529 <https://github.com/equinor/ert/pull/3529>`_)
  - Improve feedback to users when callbacks fail (`#3534 <https://github.com/equinor/ert/pull/3534>`_)
  - Allow more parameters in the webviz config (`#3609 <https://github.com/equinor/ert/pull/3609>`_)

Dependencies:
  - Add webviz-ert as dependency in setup.py (`#3587 <https://github.com/equinor/ert/pull/3587>`_)

Miscellaneous:
  - Update poly config guide (`#3444 <https://github.com/equinor/ert/pull/3444>`_)
  - Use pd.concat instead of append (`#3449 <https://github.com/equinor/ert/pull/3449>`_)
  - Explicitly load no-self-use pylint extension (`#3468 <https://github.com/equinor/ert/pull/3468>`_)
  - Solve some infrequent pylint issues (`#3479 <https://github.com/equinor/ert/pull/3479>`_)
  - Move create runpath and sample parameter logic from C to Python (`#3467 <https://github.com/equinor/ert/pull/3467>`_)
  - Fix global-statement pylint error (`#3497 <https://github.com/equinor/ert/pull/3497>`_)
  - Type and clean enkf_fs_manager (`#3491 <https://github.com/equinor/ert/pull/3491>`_)
  - Update github issue template for bugs (`#3503 <https://github.com/equinor/ert/pull/3503>`_)
  - Remove unused param arg (`#3507 <https://github.com/equinor/ert/pull/3507>`_)
  - Remove unecessary fs version checks (`#3510 <https://github.com/equinor/ert/pull/3510>`_)
  - Use async context manager for ens_evaluator client (`#3484 <https://github.com/equinor/ert/pull/3484>`_)
  - Remove unused function get_observation_count (`#3513 <https://github.com/equinor/ert/pull/3513>`_)
  - Replace util_split_string and util_alloc_joined_string with C++ (`#3500 <https://github.com/equinor/ert/pull/3500>`_)
  - Remove unneeded extern C (`#3525 <https://github.com/equinor/ert/pull/3525>`_)
  - Inline `time_map_summary_update__` (`#3530 <https://github.com/equinor/ert/pull/3530>`_)
  - Push runpath_list into hook_manager (`#3526 <https://github.com/equinor/ert/pull/3526>`_)
  - Use standard library for string split and file handling (`#3538 <https://github.com/equinor/ert/pull/3538>`_)
  - Type and simplify measured.py (`#3539 <https://github.com/equinor/ert/pull/3539>`_)
  - Select pytest asyncio_mode=auto (`#3540 <https://github.com/equinor/ert/pull/3540>`_)
  - Cleaning and refactoring block fs for readability (`#3552 <https://github.com/equinor/ert/pull/3552>`_)
  - Rewrite test framework for and upgrade test_job_queue_manager (`#3518 <https://github.com/equinor/ert/pull/3518>`_)
  - Replace enkf_main_alloc_caselist with iterdir (`#3563 <https://github.com/equinor/ert/pull/3563>`_)
  - Update MIN_SUPPORTED_FS_VERSION (`#3545 <https://github.com/equinor/ert/pull/3545>`_)
  - Remove unused cases config (`#3565 <https://github.com/equinor/ert/pull/3565>`_)
  - Use copy_if in enkf_main.cpp::get_parameter_keys (`#3577 <https://github.com/equinor/ert/pull/3577>`_)
  - Improve documentation of GEN_KW (`#3576 <https://github.com/equinor/ert/pull/3576>`_)
  - Solve pylint warnings on dangerous-default-value (`#3584 <https://github.com/equinor/ert/pull/3584>`_)
  - Move save/load parameters to EnkfFs (`#3574 <https://github.com/equinor/ert/pull/3574>`_)
  - Delete unused block fs drivers (`#3566 <https://github.com/equinor/ert/pull/3566>`_)
  - Fix test that failed due to new pandas (`#3441 <https://github.com/equinor/ert/pull/3441>`_)
  - Update about-section of readme (`#3442 <https://github.com/equinor/ert/pull/3442>`_)
  - Set docs language to english (`#3446 <https://github.com/equinor/ert/pull/3446>`_)
  - Simplify return type to reflect function behaviour (`#3339 <https://github.com/equinor/ert/pull/3339>`_)
  - Update readme after first setup (`#3166 <https://github.com/equinor/ert/pull/3166>`_)
  - Update jupyter notebook hm examples to new API (`#3460 <https://github.com/equinor/ert/pull/3460>`_)
  - Log plot views (`#3470 <https://github.com/equinor/ert/pull/3470>`_)
  - Handle dying batcher (`#3466 <https://github.com/equinor/ert/pull/3466>`_)
  - Update spe1 readme according to new config layout (`#3472 <https://github.com/equinor/ert/pull/3472>`_)
  - Hoverinfo formatting (`#3475 <https://github.com/equinor/ert/pull/3475>`_)
  - Remove unused queue code (`#3454 <https://github.com/equinor/ert/pull/3454>`_)
  - Reverting an earlier attempt to optimize creation (`#3483 <https://github.com/equinor/ert/pull/3483>`_)
  - Separate benchmarks into different runs (`#3419 <https://github.com/equinor/ert/pull/3419>`_)
  - Remove doc referring to tagged keywords (`#3492 <https://github.com/equinor/ert/pull/3492>`_)
  - Remove unused model_config internalization (`#3480 <https://github.com/equinor/ert/pull/3480>`_)
  - Add experimental feature flag to webviz ert (`#3482 <https://github.com/equinor/ert/pull/3482>`_)
  - Remove unused function gen_kw_ecl_write_template (`#3504 <https://github.com/equinor/ert/pull/3504>`_)
  - Remove unnecessary enkf_main_init_fs (`#3512 <https://github.com/equinor/ert/pull/3512>`_)
  - Refactor enkf_main_write_run_path (`#3494 <https://github.com/equinor/ert/pull/3494>`_)
  - Removal of INIT_MISFIT_TABLE workflow. (`#3477 <https://github.com/equinor/ert/pull/3477>`_)
  - Add missing await in a rare branch of the code, extend logging (`#3519 <https://github.com/equinor/ert/pull/3519>`_)
  - Remove unused function run_path_list_load (`#3520 <https://github.com/equinor/ert/pull/3520>`_)
  - Apply the fire-and-forget strategy when sending updates to clients (`#3531 <https://github.com/equinor/ert/pull/3531>`_)
  - Add safety-check after #3483 because self._dispatchers_connected can be None (`#3533 <https://github.com/equinor/ert/pull/3533>`_)
  - Install pybind11 from PyPI in CMake CI (`#3547 <https://github.com/equinor/ert/pull/3547>`_)
  - Set file dialog to reasonable width and height - simplified approach (`#3461 <https://github.com/equinor/ert/pull/3461>`_)
  - Remove graphql related code (`#3532 <https://github.com/equinor/ert/pull/3532>`_)
  - Refactor `block_fs` `file_node` (`#3555 <https://github.com/equinor/ert/pull/3555>`_)
  - Remove fixing of nodes (`#3562 <https://github.com/equinor/ert/pull/3562>`_)
  - Make dependency on file location in Block explicit (`#3570 <https://github.com/equinor/ert/pull/3570>`_)
  - Hide log statements from console and put storage statements in log file (`#3489 <https://github.com/equinor/ert/pull/3489>`_)
  - Remove unneeded source fs from save_parameters (`#3580 <https://github.com/equinor/ert/pull/3580>`_)
  - Remove unused enum (`#3592 <https://github.com/equinor/ert/pull/3592>`_)
  - Clean up for moving runpath list writing (`#3604 <https://github.com/equinor/ert/pull/3604>`_)
  - Add C tests with EXCLUDE_FROM_ALL (`#3607 <https://github.com/equinor/ert/pull/3607>`_)
  - Use ert_shared Client in job_runner (`#3606 <https://github.com/equinor/ert/pull/3606>`_)
  - Remove logging of conn_info (`#3670 <https://github.com/equinor/ert/pull/3670>`_)

Version 2.36
------------

Breaking changes:
  - Refactor analysis config min_required_realizations (`#3426 <https://github.com/equinor/ert/pull/3426>`_)
  - Raise exception if analysis can not be performed (`#3302 <https://github.com/equinor/ert/pull/3302>`_)
  - Change verbose flag behaviour to output info-level an greater logs (`#3332 <https://github.com/equinor/ert/pull/3332>`_)
  - Change update configuration (`#3322 <https://github.com/equinor/ert/pull/3322>`_)
  - Remove unused functions on EnKFMain (`#3400 <https://github.com/equinor/ert/pull/3400>`_)

Bugfixes:
  - Fix edit analysis varables for run_analysis_panel (`#3330 <https://github.com/equinor/ert/pull/3330>`_)
  - Fix iteration nr bug in IES and add ies cli integration test (`#3457 <https://github.com/equinor/ert/pull/3457>`_)

New features:
  - ert3: Add GUI monitoring (`#3167 <https://github.com/equinor/ert/pull/3167>`_)

Improvements:
  - Remove module name from GUI (`#3529 <https://github.com/equinor/ert/pull/3529>`_)
  - Shorten list of forward models in main GUI (`#3382 <https://github.com/equinor/ert/pull/3382>`_)
  - Remove File menu from main window (`#3395 <https://github.com/equinor/ert/pull/3395>`_)
  - Add timestamp to log file name (`#3334 <https://github.com/equinor/ert/pull/3334>`_)
  - Catch exception and exit with meaningful error in shellscripts (`#3362 <https://github.com/equinor/ert/pull/3362>`_)
  - Allow resize of simulations failed message box (`#3409 <https://github.com/equinor/ert/pull/3409>`_)
  - Dont retry forward model if inconsistent time map (`#3427 <https://github.com/equinor/ert/pull/3427>`_)
  - Make sure newlines are preserved in message box (`#3431 <https://github.com/equinor/ert/pull/3431>`_)
  - Various improvements to analysis (
    `#3439 <https://github.com/equinor/ert/pull/3439>`_,
    `#3473 <https://github.com/equinor/ert/pull/3473>`_,
    `#3476 <https://github.com/equinor/ert/pull/3476>`_
    )
  - Various improvements to runpath initialization (
    `#3475 <https://github.com/equinor/ert/pull/3475>`_,
    `#3492 <https://github.com/equinor/ert/pull/3492>`_,
    `#3480 <https://github.com/equinor/ert/pull/3480>`_,
    `#3504 <https://github.com/equinor/ert/pull/3504>`_,
    `#3512 <https://github.com/equinor/ert/pull/3512>`_,
    `#3494 <https://github.com/equinor/ert/pull/3494>`_,
    `#3477 <https://github.com/equinor/ert/pull/3477>`_,
    `#3520 <https://github.com/equinor/ert/pull/3520>`_,
    `#3526 <https://github.com/equinor/ert/pull/3526>`_,
    `#3467 <https://github.com/equinor/ert/pull/3467>`_
    )
  - Various improvments to stability of status tracking (
    `#3481 <https://github.com/equinor/ert/pull/3481>`_,
    `#3466 <https://github.com/equinor/ert/pull/3466>`_,
    `#3483 <https://github.com/equinor/ert/pull/3483>`_,
    `#3519 <https://github.com/equinor/ert/pull/3519>`_,
    `#3531 <https://github.com/equinor/ert/pull/3531>`_,
    `#3315 <https://github.com/equinor/ert/pull/3315>`_,
    `#3324 <https://github.com/equinor/ert/pull/3324>`_,
    `#3408 <https://github.com/equinor/ert/pull/3408>`_,
    `#3360 <https://github.com/equinor/ert/pull/3360>`_
    )
  - Various improvments reading/writing to storage (
    `#3429 <https://github.com/equinor/ert/pull/3429>`_,
    `#3513 <https://github.com/equinor/ert/pull/3513>`_,
    `#3530 <https://github.com/equinor/ert/pull/3530>`_,
    `#3539 <https://github.com/equinor/ert/pull/3539>`_,
    `#3434 <https://github.com/equinor/ert/pull/3434>`_,
    `#3384 <https://github.com/equinor/ert/pull/3384>`_,
    `#3390 <https://github.com/equinor/ert/pull/3390>`_,
    `#3194 <https://github.com/equinor/ert/pull/3194>`_,
    `#3510 <https://github.com/equinor/ert/pull/3510>`_,
    `#3491 <https://github.com/equinor/ert/pull/3491>`_,
    `#3552 <https://github.com/equinor/ert/pull/3552>`_
    )
  - ert3: Merge the ensemble and experiment config (`#3385 <https://github.com/equinor/ert/pull/3385>`_)
  - ert3: Change "record" to "name" in the ensemble config (`#3364 <https://github.com/equinor/ert/pull/3364>`_)

Version 2.35
------------

Breaking changes:
  - Change default inversion to IES_INVERSION_EXACT (`#3193 <https://github.com/equinor/ert/pull/3193>`_)
  - Remove flag for using aa_projection in IES and ES (`#3230 <https://github.com/equinor/ert/pull/3230>`_)
  - Fix scaling of ESMDA weights (`#3211 <https://github.com/equinor/ert/pull/3211>`_)
  - Replaced fm message with logging statements and remove unused workflows LOAD_RESULTS(_ITER) (`#3252 <https://github.com/equinor/ert/pull/3252>`_)
  - Remove option of loading from non-unified summary files (`#3247 <https://github.com/equinor/ert/pull/3247>`_)
  - Remove setting `MODULE` from workflows (`#3288 <https://github.com/equinor/ert/pull/3288>`_)
  - Remove analysis enums (`#3283 <https://github.com/equinor/ert/pull/3283>`_)

Bugfixes:
  - Fix AnalysisIterConfig._repr_() and add test (`#3171 <https://github.com/equinor/ert/pull/3171>`_)
  - Fix index out of bounds for active realizations < ensemble size in ESMDA (`#3200 <https://github.com/equinor/ert/pull/3200>`_)
  - Fix bug where valid run_path was not recognised (`#3254 <https://github.com/equinor/ert/pull/3254>`_)

New features:
  - Add event viewer tool to gui (`#3136 <https://github.com/equinor/ert/pull/3136>`_)
  - ert3: Add realization selection support in ert3 (`#3095 <https://github.com/equinor/ert/pull/3095>`_)
  - ert3: Add visualise parameter to ert3 for starting webviz-ert (`#3209 <https://github.com/equinor/ert/pull/3209>`_)
  - ert3: Add support for a scalar NumericalRecord (`#2934 <https://github.com/equinor/ert/pull/2934>`_)

Improvements:
  - Update poly_example to use more accurate observations (`#3149 <https://github.com/equinor/ert/pull/3149>`_)
  - Upgrade ERT icon set to Equinor design system (`#3178 <https://github.com/equinor/ert/pull/3178>`_)
  - Drop Title Case In Ert Gui (`#3190 <https://github.com/equinor/ert/pull/3190>`_)
  - Remove .index files from block_fs (`#3185 <https://github.com/equinor/ert/pull/3185>`_)
  - Log the contents of the ERT2 configuration file (`#3218 <https://github.com/equinor/ert/pull/3218>`_)
  - Update algorithm GUI config (`#3213 <https://github.com/equinor/ert/pull/3213>`_)
  - Make image cache singleton (`#3237 <https://github.com/equinor/ert/pull/3237>`_)
  - Upgrade and add more info to log message when failing to read SUMMARY (`#3232 <https://github.com/equinor/ert/pull/3232>`_)
  - Refactor parts of `block_fs.cpp` (`#3233 <https://github.com/equinor/ert/pull/3233>`_)
  - Test realization masks in base_run_model (`#3275 <https://github.com/equinor/ert/pull/3275>`_)
  - dlopen libres with RTLD_LOCAL (`#3210 <https://github.com/equinor/ert/pull/3210>`_)
  - Replace util_abort on time map with logging error and failing realisation (`#3256 <https://github.com/equinor/ert/pull/3256>`_)
  - Add details view to simulations failed including error logs (`#3290 <https://github.com/equinor/ert/pull/3290>`_)
  - Refactor RunDialog to depend less on RunModel (`#3108 <https://github.com/equinor/ert/pull/3108>`_)
  - Remove unused function local_ministep_get_obs_data (`#3158 <https://github.com/equinor/ert/pull/3158>`_)
  - Remove unused function enkf_analysis_deactivate_std_zero (`#3176 <https://github.com/equinor/ert/pull/3176>`_)
  - Remove `thread_pool.cpp`, `arg_pack.cpp` et al (`#3117 <https://github.com/equinor/ert/pull/3117>`_)
  - Undo pinning of click in dev-requirements (`#3208 <https://github.com/equinor/ert/pull/3208>`_)
  - Fix typo recieved (`#3220 <https://github.com/equinor/ert/pull/3220>`_)
  - Use explicit int-value from enum (`#3221 <https://github.com/equinor/ert/pull/3221>`_)
  - Delete unused test_analysis_test_external_module.c (`#3206 <https://github.com/equinor/ert/pull/3206>`_)
  - Extend flake8 linting to ert-directory and ert_tests (`#3203 <https://github.com/equinor/ert/pull/3203>`_)
  - Remove using matrix_type from serializer (`#3236 <https://github.com/equinor/ert/pull/3236>`_)
  - Remove unused covar from obs_data and replace replace matrix_type with eigen in meas_data (`#3234 <https://github.com/equinor/ert/pull/3234>`_)
  - Type and style for ensemble evaluator builder code (`#3219 <https://github.com/equinor/ert/pull/3219>`_)
  - Print warning every time non-ISO time format is used in simulation setup  (`#3238 <https://github.com/equinor/ert/pull/3238>`_)
  - Remove unused functions in enkf_state.cpp (`#3243 <https://github.com/equinor/ert/pull/3243>`_)
  - Ensure duplexer stop entails a websocket close (`#3246 <https://github.com/equinor/ert/pull/3246>`_)
  - Avoid DeprecationWarning from qtbot (`#3245 <https://github.com/equinor/ert/pull/3245>`_)
  - Address comments left over from pull 3219 (`#3253 <https://github.com/equinor/ert/pull/3253>`_)
  - Bring libres_tests up to flake8 standard (`#3250 <https://github.com/equinor/ert/pull/3250>`_)
  - Filter out comments from logged configuration (`#3249 <https://github.com/equinor/ert/pull/3249>`_)
  - Replace deprecated `..index.is_all_dates` in plottery (`#3231 <https://github.com/equinor/ert/pull/3231>`_)
  - Remove unused function -time_map_summary_upgrade107 (`#3257 <https://github.com/equinor/ert/pull/3257>`_)
  - Remove matrixtype from  rowscaling (`#3242 <https://github.com/equinor/ert/pull/3242>`_)
  - Replace deprecated `..index.is_all_dates` in plottery (`#3265 <https://github.com/equinor/ert/pull/3265>`_)
  - Remove matrix type from ert (`#3268 <https://github.com/equinor/ert/pull/3268>`_)
  - Remove documentation about internal C workflows (`#3276 <https://github.com/equinor/ert/pull/3276>`_)
  - Remove pytest-runner (`#3285 <https://github.com/equinor/ert/pull/3285>`_)
  - Avoid BoolVector in Python code (`#3251 <https://github.com/equinor/ert/pull/3251>`_)
  - Remove `setup_requires` from `setup.py` (`#3286 <https://github.com/equinor/ert/pull/3286>`_)
  - Make fixtures cleanup after themselves (`#3287 <https://github.com/equinor/ert/pull/3287>`_)
  - Enable `SortInclude` in clang-format configuration (`#3284 <https://github.com/equinor/ert/pull/3284>`_)
  - Fix all flake8 issues in ert  (`#3281 <https://github.com/equinor/ert/pull/3281>`_)
  - Extend weak pylinting to more directories (`#3289 <https://github.com/equinor/ert/pull/3289>`_)
  - Use caplog to make sure root log level is INFO in test (`#3300 <https://github.com/equinor/ert/pull/3300>`_)
  - Use model factory in gui (`#3294 <https://github.com/equinor/ert/pull/3294>`_)
  - Decrease example size for polynomial doe (`#3304 <https://github.com/equinor/ert/pull/3304>`_)
  - Propagate logs from ensemble_evaluator, storage and status (`#3293 <https://github.com/equinor/ert/pull/3293>`_)
  - Dark storage performance (`#3051 <https://github.com/equinor/ert/pull/3051>`_)

Dependencies:
  - Pin PyQt5-sip to 12.9.1 or lower (`#3261 <https://github.com/equinor/ert/pull/3261>`_)
  - Pin click in dev-requirements to 8.0.2 (`#3172 <https://github.com/equinor/ert/pull/3172>`_)

Miscellaneous:
  - Fix to circumvent pylint bug (`#3163 <https://github.com/equinor/ert/pull/3163>`_)
  - Add additional information on failure in test_http_response (`#3132 <https://github.com/equinor/ert/pull/3132>`_)
  - Add release notes for ert 2.34 (`#3165 <https://github.com/equinor/ert/pull/3165>`_)
  - Update labels for automatic release notes generation (`#3192 <https://github.com/equinor/ert/pull/3192>`_)
  - Add maintenance as release notes category (`#3199 <https://github.com/equinor/ert/pull/3199>`_)
  - Modify ModelConfig.enspath to hold an absolute file-path (`#3186 <https://github.com/equinor/ert/pull/3186>`_)
  - Create example for running history matching in python using analysis module (`#3131 <https://github.com/equinor/ert/pull/3131>`_)
  - Have repeated flaky tests run new python instance (`#3189 <https://github.com/equinor/ert/pull/3189>`_)
  - Add extension to error message (`#3291 <https://github.com/equinor/ert/pull/3291>`_)

Version 2.34
------------

Breaking changes:
  - Remove the IDE (Built-in config editor) (`#3148 <https://github.com/equinor/ert/pull/3148>`_)
  - Remove legacy tracker (`#2965 <https://github.com/equinor/ert/pull/2965>`_)
  - Remove enkf_obs instance from local_obsdata (`#3046 <https://github.com/equinor/ert/pull/3046>`_)

Bugfixes:
  - Remove inactive analysis module options (`#3126 <https://github.com/equinor/ert/pull/3126>`_)
  - Fix row scaling local configuration job (`#2954 <https://github.com/equinor/ert/pull/2954>`_)
  - Improve failure behaviour from ert client to storage (`#2956 <https://github.com/equinor/ert/pull/2956>`_)
  - Add an out of bounds check (`#2969 <https://github.com/equinor/ert/pull/2969>`_)
  - Set strict=True when creating EnkfMain, make testname unique (`#3042 <https://github.com/equinor/ert/pull/3042>`_)
  - Skip lazy_load flag test which is failing on jenkins (`#3038 <https://github.com/equinor/ert/pull/3038>`_)
  - Undo removal of getAll  - collectors (`#3034 <https://github.com/equinor/ert/pull/3034>`_)
  - [ert3] Ignore command location during step execution (`#3147 <https://github.com/equinor/ert/pull/3147>`_)

New features:
  - Support Python 3.10 (`#2834 <https://github.com/equinor/ert/pull/2834>`_)
  - Fix removing duplicates, added test to verify, improved performance (`#2951 <https://github.com/equinor/ert/pull/2951>`_)
  - Disable lazy_loading of summary-data by default (`#2976 <https://github.com/equinor/ert/pull/2976>`_)
  - Ies from python (`#3145 <https://github.com/equinor/ert/pull/3145>`_)
  - [ert3] Auto-generate documentation for plugins (`#3138 <https://github.com/equinor/ert/pull/3138>`_)
  - [ert3] Add support for log-uniform distribution (`#3156 <https://github.com/equinor/ert/pull/3156>`_)
  - [ert3] ERT3 local test run (`#2755 <https://github.com/equinor/ert/pull/2755>`_)
  - [ert3] Plugin-in transformation configuration and wider transformation usage (`#3025 <https://github.com/equinor/ert/pull/3025>`_)
  - [ert3] Cli monitor for ert3 (`#2960 <https://github.com/equinor/ert/pull/2960>`_)

Improvements:
  - Refactor interactions between JobQueue and LegacyEnsemble. (`#3144 <https://github.com/equinor/ert/pull/3144>`_)
  - Remove sorting of variables (`#3128 <https://github.com/equinor/ert/pull/3128>`_)
  - Add logging of EnKFMain usage in workflows (`#3140 <https://github.com/equinor/ert/pull/3140>`_)
  - Make tests use localhost (`#3160 <https://github.com/equinor/ert/pull/3160>`_)
  - Set proper timeout for Storage.start_server() (`#3076 <https://github.com/equinor/ert/pull/3076>`_)
  - Print a message to user about starting Webviz-ert (`#3075 <https://github.com/equinor/ert/pull/3075>`_)
  - Add webviz-ert documentation (`#3065 <https://github.com/equinor/ert/pull/3065>`_)
  - Fix timing in test batch sim (`#3091 <https://github.com/equinor/ert/pull/3091>`_)
  - Force GC to avoid conflict with running C++ dtors later (`#3100 <https://github.com/equinor/ert/pull/3100>`_)
  - Reduce log level for MAX_RUNTIME reached and only log once (`#2770 <https://github.com/equinor/ert/pull/2770>`_)
  - Rename test to avoid conflict (`#3098 <https://github.com/equinor/ert/pull/3098>`_)
  - Test tracker progress (`#3110 <https://github.com/equinor/ert/pull/3110>`_)
  - Fix typo, successfull -> successful (`#3107 <https://github.com/equinor/ert/pull/3107>`_)
  - Automatically increase softlimit for max open files while running tests (`#3112 <https://github.com/equinor/ert/pull/3112>`_)
  - Start statically checking code in ert_shared/models (`#3094 <https://github.com/equinor/ert/pull/3094>`_)
  - Remove global ert (`#3118 <https://github.com/equinor/ert/pull/3118>`_)
  - Increasing default timeout from 20s to 120s in BaseService. (`#3129 <https://github.com/equinor/ert/pull/3129>`_)
  - Handle error publish changes (`#3130 <https://github.com/equinor/ert/pull/3130>`_)
  - Remove usage of global ERT in run models (`#3039 <https://github.com/equinor/ert/pull/3039>`_)
  - Remove usage of threadpool in block_fs_driver (`#3021 <https://github.com/equinor/ert/pull/3021>`_)
  - Various improvements to flaky tests (
    `#3119 <https://github.com/equinor/ert/pull/3119>`_,
    `#3125 <https://github.com/equinor/ert/pull/3125>`_,
    `#2983 <https://github.com/equinor/ert/pull/2983>`_,
    `#2987 <https://github.com/equinor/ert/pull/2987>`_
    )
  - Various improvements to the analysis module (
    `#3060 <https://github.com/equinor/ert/pull/3060>`_,
    `#2913 <https://github.com/equinor/ert/pull/2913>`_,
    `#3082 <https://github.com/equinor/ert/pull/3082>`_,
    `#3087 <https://github.com/equinor/ert/pull/3087>`_,
    `#3083 <https://github.com/equinor/ert/pull/3083>`_,
    `#3097 <https://github.com/equinor/ert/pull/3097>`_,
    `#2958 <https://github.com/equinor/ert/pull/2958>`_,
    `#2948 <https://github.com/equinor/ert/pull/2948>`_,
    `#2999 <https://github.com/equinor/ert/pull/2999>`_,
    `#2964 <https://github.com/equinor/ert/pull/2964>`_,
    `#3018 <https://github.com/equinor/ert/pull/3018>`_,
    `#3028 <https://github.com/equinor/ert/pull/3028>`_,
    `#2962 <https://github.com/equinor/ert/pull/2962>`_,
    `#3035 <https://github.com/equinor/ert/pull/3035>`_,
    `#3005 <https://github.com/equinor/ert/pull/3005>`_,
    `#3058 <https://github.com/equinor/ert/pull/3058>`_,
    `#2966 <https://github.com/equinor/ert/pull/2966>`_
    )

Dependencies:
  - Pin pylint to version <2.13.0 (`#3161 <https://github.com/equinor/ert/pull/3161>`_)
  - Remove requirement mypy < 0.920 (`#3090 <https://github.com/equinor/ert/pull/3090>`_)
  - Remove dependency on semeio (`#2980 <https://github.com/equinor/ert/pull/2980>`_)
  - Remove dependency on BLAS and LAPACK (`#3000 <https://github.com/equinor/ert/pull/3000>`_)

Miscellaneous:
  - Ignore errors in rmtree (`#3155 <https://github.com/equinor/ert/pull/3155>`_)
  - Fix filename typo in tests (`#3072 <https://github.com/equinor/ert/pull/3072>`_)
  - Use self._timeout in fetch_conn_info() (`#3078 <https://github.com/equinor/ert/pull/3078>`_)
  - Add host ensemble server config (`#3096 <https://github.com/equinor/ert/pull/3096>`_)
  - job_dispatch: Makedir in, out, err locations (`#2998 <https://github.com/equinor/ert/pull/2998>`_)
  - Add an optional has_observations flag to the record (`#2979 <https://github.com/equinor/ert/pull/2979>`_)
  - Remove unused job_queue from BaseRunModel (`#3019 <https://github.com/equinor/ert/pull/3019>`_)
  - clang-format: Remove version check (`#3027 <https://github.com/equinor/ert/pull/3027>`_)
  - Make `extern "C"` explicit for each function (`#2963 <https://github.com/equinor/ert/pull/2963>`_)
  - Use the key_manager from enkf_main (`#3026 <https://github.com/equinor/ert/pull/3026>`_)
  - Remove utility functions of the legacy tracker (`#3006 <https://github.com/equinor/ert/pull/3006>`_)
  - Introduce read-only info class derived from EvaluatorServerConfig (`#3045 <https://github.com/equinor/ert/pull/3045>`_)
  - Correct exception name typo (`#3047 <https://github.com/equinor/ert/pull/3047>`_)
  - Make port range larger in test (`#3059 <https://github.com/equinor/ert/pull/3059>`_)
  - [ert3] Drop experiment_run_config from load_resource (`#3102 <https://github.com/equinor/ert/pull/3102>`_)
  - [ert3] Add better error handling when trying to rerun an experiment (`#2891 <https://github.com/equinor/ert/pull/2891>`_)


Version 2.33
------------

Breaking changes:
  - Introduce nested namespace ies::data (`#2828 <https://github.com/equinor/ert/pull/2828>`_)
  - Remove unused python class ConfigSettings (`#2746 <https://github.com/equinor/ert/pull/2746>`_)
  - Remove changing mode of files by ext_job (`#2784 <https://github.com/equinor/ert/pull/2784>`_)
  - Deprecate keyword UMASK and disallow UMASK 0 (`#2777 <https://github.com/equinor/ert/pull/2777>`_)
  - Delete workflows related to obs/data ranking (`#2927 <https://github.com/equinor/ert/pull/2927>`_)
  - Replace bool_vector with stl::vector and return vector instead of mutating in state_map_select_matching (`#2922 <https://github.com/equinor/ert/pull/2922>`_)
  - Remove `Matrix`, `\{Obs,Meas\}\{Data,Block\}` classes from Python (`#2923 <https://github.com/equinor/ert/pull/2923>`_)

Bugfixes:
  - Make AutorunTestEnsemble cancellable (`#2786 <https://github.com/equinor/ert/pull/2786>`_)
  - Do not build vanilla step on unexpected step type (`#2807 <https://github.com/equinor/ert/pull/2807>`_)
  - Free internal resources in ies::data_free() (`#2830 <https://github.com/equinor/ert/pull/2830>`_)
  - Update config path to basename when changing working directory (`#2851 <https://github.com/equinor/ert/pull/2851>`_)
  - Remove redundant parameter in enkf_main_alloc (`#2890 <https://github.com/equinor/ert/pull/2890>`_)
  - [ert3] Fix websocket connection timeout in Unix step (`#2783 <https://github.com/equinor/ert/pull/2783>`_)

New features:
  - Keyword data ndarray copy (`#2806 <https://github.com/equinor/ert/pull/2806>`_)
  - Cleanup redundant parameters (`#2853 <https://github.com/equinor/ert/pull/2853>`_)
  - Remove usage of thread_pool in job_queue (`#2883 <https://github.com/equinor/ert/pull/2883>`_)
  - [ert3] Add snakeoil example for ert3 (`#2703 <https://github.com/equinor/ert/pull/2703>`_)
  - Replace ResLog with Python logging (`#2710 <https://github.com/equinor/ert/pull/2710>`_)
  - [ert3] Implement discrete uniform and constant distribution (`#2224 <https://github.com/equinor/ert/pull/2224>`_)

Improvements:
  - [ert3] Improve error handling around ert3 initialization (`#2779 <https://github.com/equinor/ert/pull/2779>`_)
  - [ert3] Increase worker memory requirements on PBS (`#2832 <https://github.com/equinor/ert/pull/2832>`_)
  - [ert3] Add better error handling when trying to rerun an experiment (`#2891 <https://github.com/equinor/ert/pull/2891>`_)
  - Improve the summary collector (`#2871 <https://github.com/equinor/ert/pull/2871>`_)
  - `matrix_type` -> Eigen (`#2872 <https://github.com/equinor/ert/pull/2872>`_)

Dependencies:
  - Upgrade black to 22.1.0 (`#2946 <https://github.com/equinor/ert/pull/2946>`_)

Miscellaneous:
  - Fix typo ensamble (`#2845 <https://github.com/equinor/ert/pull/2845>`_)
  - Remove cmake feature tests (`#2791 <https://github.com/equinor/ert/pull/2791>`_)
  - Various improvements to analysis module (
    `#2747 <https://github.com/equinor/ert/pull/2747>`_,
    `#2748 <https://github.com/equinor/ert/pull/2748>`_,
    `#2772 <https://github.com/equinor/ert/pull/2772>`_,
    `#2829 <https://github.com/equinor/ert/pull/2829>`_,
    `#2857 <https://github.com/equinor/ert/pull/2857>`_,
    `#2855 <https://github.com/equinor/ert/pull/2855>`_,
    `#2892 <https://github.com/equinor/ert/pull/2892>`_,
    `#2863 <https://github.com/equinor/ert/pull/2863>`_,
    `#2893 <https://github.com/equinor/ert/pull/2893>`_,
    `#2901 <https://github.com/equinor/ert/pull/2901>`_,
    `#2877 <https://github.com/equinor/ert/pull/2877>`_,
    `#2744 <https://github.com/equinor/ert/pull/2744>`_,
    `#2897 <https://github.com/equinor/ert/pull/2897>`_,
    `#2933 <https://github.com/equinor/ert/pull/2933>`_,
    `#2929 <https://github.com/equinor/ert/pull/2929>`_,
    `#2910 <https://github.com/equinor/ert/pull/2910>`_,
    `#2919 <https://github.com/equinor/ert/pull/2919>`_,
    `#2947 <https://github.com/equinor/ert/pull/2947>`_,
    `#2957 <https://github.com/equinor/ert/pull/2957>`_,
    `#2942 <https://github.com/equinor/ert/pull/2942>`_,
    `#2950 <https://github.com/equinor/ert/pull/2950>`_
    )
  - Drop threads when collecting summary-data (`#2808 <https://github.com/equinor/ert/pull/2808>`_)
  - Refactor summary collector (`#2802 <https://github.com/equinor/ert/pull/2802>`_)
  - Revert flaky performance tests (`#2822 <https://github.com/equinor/ert/pull/2822>`_)
  - Retrieve responses for _export via the dedicated endpoint (`#2820 <https://github.com/equinor/ert/pull/2820>`_)
  - Replaced reuse_addr with something more insisting (`#2757 <https://github.com/equinor/ert/pull/2757>`_)
  - Improve snake oil example (
    `#2848 <https://github.com/equinor/ert/pull/2848>`_,
    `#2888 <https://github.com/equinor/ert/pull/2888>`_
    )
  - Add types and perform clean-up of BaseRunModel (`#2854 <https://github.com/equinor/ert/pull/2854>`_)
  - Fix mypy error (`#2876 <https://github.com/equinor/ert/pull/2876>`_)
  - Remove deprecated parameter 'loop' in Queues  (`#2889 <https://github.com/equinor/ert/pull/2889>`_)
  - Update ensemble parameter response from dark storage (`#2856 <https://github.com/equinor/ert/pull/2856>`_)
  - Remove thread_pool in loading/saving parameters (`#2884 <https://github.com/equinor/ert/pull/2884>`_)
  - Remove compiler warning (`#2911 <https://github.com/equinor/ert/pull/2911>`_)
  - Adjust test in extraction due to changes in API (`#2917 <https://github.com/equinor/ert/pull/2917>`_)
  - Parameterize some test instead of having duplicate (`#2810 <https://github.com/equinor/ert/pull/2810>`_)
  - [ert3] Change one_at_the_time() to one_at_a_time() (`#2943 <https://github.com/equinor/ert/pull/2943>`_)
  - Add observations to responses query (`#2912 <https://github.com/equinor/ert/pull/2912>`_)
  - Replace first iteration of convert from cwrap (`#2938 <https://github.com/equinor/ert/pull/2938>`_)
  - Remove unused module util_fprint and res_version (`#2949 <https://github.com/equinor/ert/pull/2949>`_)
  - Use fmt to write error message (`#2974 <https://github.com/equinor/ert/pull/2974>`_)


Version 2.32
------------

Breaking changes:
  - Remove local dataset notion (`#2645 <https://github.com/equinor/ert/pull/2645>`_)
  - Remove unused functions in enkf_fs_manager (`#2664 <https://github.com/equinor/ert/pull/2664>`_)
  - Remove unused class History (`#2718 <https://github.com/equinor/ert/pull/2718>`_)
  - Remove unused python obs and measure (`#2725 <https://github.com/equinor/ert/pull/2725>`_)
  - Remove unused code in python interface with analysis module (`#2716 <https://github.com/equinor/ert/pull/2716>`_)

Bugfixes:
  - Forward database url from argparse to Storage (`#2680 <https://github.com/equinor/ert/pull/2680>`_)
  - Expected boolean return from void-function (`#2768 <https://github.com/equinor/ert/pull/2768>`_)
  - Always handle connection errors when monitoring the ensemble (`#2771 <https://github.com/equinor/ert/pull/2771>`_)
  - [ert3] Use lowercase letters when specifying psb resources (`#2692 <https://github.com/equinor/ert/pull/2692>`_)
  - [ert3] Fix failure when using pbs driver (`#2689 <https://github.com/equinor/ert/pull/2689>`_)
  - [ert3] Fix SPE1 yaml config for smry_keys (`#2685 <https://github.com/equinor/ert/pull/2685>`_)
  - [ert3] Use ert port range for pbs driver (`#2723 <https://github.com/equinor/ert/pull/2723>`_)

New features:
  - Add method getActiveIndexList  to class ActiveList  (`#2323 <https://github.com/equinor/ert/pull/2323>`_)
  - Support YYYY-MM-DD (ISO-8601) in observation files, timemap files and END_DATE keyword (`#2641 <https://github.com/equinor/ert/pull/2641>`_)
  - [ert3] Support numerical characters in parameter names (`#2668 <https://github.com/equinor/ert/pull/2668>`_)
  - [ert3] Make ert3 log to console when realizations complete (`#2732 <https://github.com/equinor/ert/pull/2732>`_)
  - [ert3] Allow the user to setup backend DB url in ert3 (`#2701 <https://github.com/equinor/ert/pull/2701>`_)
  - [ert3] Allow inline shell script in stages (`#2740 <https://github.com/equinor/ert/pull/2740>`_)

Improvements:
  - [ert3] Improve error messages in ert3 config validation (`#2702 <https://github.com/equinor/ert/pull/2702>`_)
  - [ert3] Pretty print json for human readable files (`#2706 <https://github.com/equinor/ert/pull/2706>`_)
  - [ert3] Increase timeout of storage (`#2729 <https://github.com/equinor/ert/pull/2729>`_)
  - Use ert_storage.client (`#2644 <https://github.com/equinor/ert/pull/2644>`_)
  - Introducing class and methods to log memory usage in scope (`#2640 <https://github.com/equinor/ert/pull/2640>`_)
  - Namespace ies (`#2621 <https://github.com/equinor/ert/pull/2621>`_)
  - Remove leftover dd/mm/yyyy mention in docs (`#2696 <https://github.com/equinor/ert/pull/2696>`_)
  - Writeup of posterior properties (`#2699 <https://github.com/equinor/ert/pull/2699>`_)
  - Make ConfigKeys an enum w/o ResPrototype (`#2657 <https://github.com/equinor/ert/pull/2657>`_)
  - Remove broken test configuration file (`#2665 <https://github.com/equinor/ert/pull/2665>`_)
  - Remove outdated documentation about developing analysis modules (`#2662 <https://github.com/equinor/ert/pull/2662>`_)
  - Use `ies::config::config` class for configuration also of std enkf module (`#2681 <https://github.com/equinor/ert/pull/2681>`_)
  - Use std variant (`#2709 <https://github.com/equinor/ert/pull/2709>`_)
  - Replace parsing of excluded hostnames in lsf with standard library functions (`#2638 <https://github.com/equinor/ert/pull/2638>`_)
  - Add function time logger (`#2624 <https://github.com/equinor/ert/pull/2624>`_)
  - Use ies (`#2602 <https://github.com/equinor/ert/pull/2602>`_)
  - Raise meaningful exception when accessing non existing analysis module (`#2727 <https://github.com/equinor/ert/pull/2727>`_)
  - Add logging of analysis configuration (`#2752 <https://github.com/equinor/ert/pull/2752>`_)
  - Describe SIMULATION_JOB (`#2754 <https://github.com/equinor/ert/pull/2754>`_)
  - Gendata ndarray copy (`#2682 <https://github.com/equinor/ert/pull/2682>`_)

Dependencies:
  - Pin fastapi==0.70.1 (`#2677 <https://github.com/equinor/ert/pull/2677>`_)
  - Bump ert-storage >= 0.3.7 (`#2679 <https://github.com/equinor/ert/pull/2679>`_)
  - Don't ask for storage as an extras in CI (`#2695 <https://github.com/equinor/ert/pull/2695>`_)
  - Pin Pandas version (`#2765 <https://github.com/equinor/ert/pull/2765>`_)

Miscellaneous:
  - Various improvements to analysis module (
    `#2412 <https://github.com/equinor/ert/pull/2412>`_,
    `#2527 <https://github.com/equinor/ert/pull/2527>`_,
    `#2497 <https://github.com/equinor/ert/pull/2497>`_,
    `#2628 <https://github.com/equinor/ert/pull/2628>`_,
    `#2690 <https://github.com/equinor/ert/pull/2690>`_,
    `#2705 <https://github.com/equinor/ert/pull/2705>`_,
    `#2697 <https://github.com/equinor/ert/pull/2697>`_,
    `#2711 <https://github.com/equinor/ert/pull/2711>`_,
    `#2717 <https://github.com/equinor/ert/pull/2717>`_,
    `#2721 <https://github.com/equinor/ert/pull/2721>`_,
    )
  - [ert3] Move EclSum support from serializers to transformations (`#2613 <https://github.com/equinor/ert/pull/2613>`_)
  - Fix mypy `unused "type: ignore" comment` (`#2646 <https://github.com/equinor/ert/pull/2646>`_)
  - Add script/ecl-check which counts libecl functions (`#2637 <https://github.com/equinor/ert/pull/2637>`_)
  - Simplify enkf_main_case_is_initialized (`#2656 <https://github.com/equinor/ert/pull/2656>`_)
  - Add information to pull request template (`#2663 <https://github.com/equinor/ert/pull/2663>`_)
  - Simplify enkf_main_copy_ensemble  (`#2654 <https://github.com/equinor/ert/pull/2654>`_)
  - Remove leftover debug-output (`#2693 <https://github.com/equinor/ert/pull/2693>`_)
  - Group all ert3 tagged PRs together in release notes (`#2713 <https://github.com/equinor/ert/pull/2713>`_)
  - Increase timeout from `run_examples_polynomial` CI (`#2726 <https://github.com/equinor/ert/pull/2726>`_)
  - Replace default `gen_kw_export_name` with pybind (`#2719 <https://github.com/equinor/ert/pull/2719>`_)
  - Add `RES_LIB_SUBMODULE` to pybind11 code (`#2737 <https://github.com/equinor/ert/pull/2737>`_)
  - Add tests for performance of data collectors (`#2674 <https://github.com/equinor/ert/pull/2674>`_)
  - Revert flaky performance tests (`#2825 <https://github.com/equinor/ert/pull/2825>`_)


Version 2.31
------------


Breaking changes:
  - Remove support for config keyword ANALYSIS_LOAD (`#2407 <https://github.com/equinor/ert/pull/2407>`_)

Bugfixes:
  - [ert3] Fix forgotten transformation raise statement (`#2608 <https://github.com/equinor/ert/pull/2608>`_)

New features:
  - [ert3] Introduction of RecordTree (`#2255 <https://github.com/equinor/ert/pull/2255>`_)

Improvements:
  - Timeout by default when fetching storage connection information (`#2541 <https://github.com/equinor/ert/pull/2541>`_)
  - Handle connection closed errors in EvaluatorTracker (`#2597 <https://github.com/equinor/ert/pull/2597>`_)
  - Add logging to Python from C (`#2550 <https://github.com/equinor/ert/pull/2550>`_)
  - Add docstrings for ``ert3.data`` module (`#2521 <https://github.com/equinor/ert/pull/2521>`_)
  - [ert3] Return native floats when decoding EclSum files (`#2540 <https://github.com/equinor/ert/pull/2540>`_)
  - [ert3] Refactor transmissions and transformations in order to decouple them (`#2566 <https://github.com/equinor/ert/pull/2566>`_)
  - [ert3] Make Workspace use transformations (`#2604 <https://github.com/equinor/ert/pull/2604>`_)
  - Add dark storage record labels endpoints (`#2491 <https://github.com/equinor/ert/pull/2491>`_)

Dependencies:
  - Ensure compatibility with `beartype==0.9.1` (`#2633 <https://github.com/equinor/ert/pull/2633>`_)

Miscellaneous:
  - Various improvements to analysis module (
    `#2504 <https://github.com/equinor/ert/pull/2504>`_,
    `#2568 <https://github.com/equinor/ert/pull/2568>`_,
    `#2530 <https://github.com/equinor/ert/pull/2530>`_,
    `#2463 <https://github.com/equinor/ert/pull/2463>`_,
    `#2469 <https://github.com/equinor/ert/pull/2469>`_,
    `#2591 <https://github.com/equinor/ert/pull/2591>`_,
    `#2598 <https://github.com/equinor/ert/pull/2598>`_,
    `#2599 <https://github.com/equinor/ert/pull/2599>`_,
    `#2611 <https://github.com/equinor/ert/pull/2611>`_,
    `#2617 <https://github.com/equinor/ert/pull/2617>`_
    )
  - Remove unused code (
    `#2499 <https://github.com/equinor/ert/pull/2499>`_,
    `#2509 <https://github.com/equinor/ert/pull/2509>`_,
    `#2518 <https://github.com/equinor/ert/pull/2518>`_,
    `#2532 <https://github.com/equinor/ert/pull/2532>`_,
    `#2533 <https://github.com/equinor/ert/pull/2533>`_,
    `#2519 <https://github.com/equinor/ert/pull/2519>`_,
    `#2564 <https://github.com/equinor/ert/pull/2564>`_,
    `#2595 <https://github.com/equinor/ert/pull/2595>`_,
    `#2593 <https://github.com/equinor/ert/pull/2593>`_,
    `#2618 <https://github.com/equinor/ert/pull/2618>`_,
    `#2620 <https://github.com/equinor/ert/pull/2620>`_
    )
  - Consistently use realizations, iter and ``-`` as separator (`#2603 <https://github.com/equinor/ert/pull/2603>`_)
  - Improve res imports to satisfy pylint checking (`#2502 <https://github.com/equinor/ert/pull/2502>`_)
  - Reduce calls to fs->refcount at decref (`#2501 <https://github.com/equinor/ert/pull/2501>`_)
  - Cleanup unnecessary use of run_mode (`#2563 <https://github.com/equinor/ert/pull/2563>`_)
  - Pass python executable to cmake (`#2569 <https://github.com/equinor/ert/pull/2569>`_)
  - Add release notes configuration (`#2570 <https://github.com/equinor/ert/pull/2570>`_)
  - Replace util_mkdir_fopen with standard C++ (`#2590 <https://github.com/equinor/ert/pull/2590>`_)
  - Fix spelling of therefore (`#2600 <https://github.com/equinor/ert/pull/2600>`_)
  - Only comment coverage after all reports are sent (`#2623 <https://github.com/equinor/ert/pull/2623>`_)
  - Mute PEP-585 warnings from BearType in Py39 (`#2610 <https://github.com/equinor/ert/pull/2610>`_)


Version 2.30
------------

Breaking changes:
  - Remove support for RML_ENKF (`#2037 <https://github.com/equinor/ert/issues/2037>`_)
  - Remove external analysis module loading (`#2202 <https://github.com/equinor/ert/issues/2202>`_)
  - Remove scale_correlated_obs (`#2358 <https://github.com/equinor/ert/issues/2358>`_)
  - Stop persisting principal component to disk (PC directory) (`#2367 <https://github.com/equinor/ert/issues/2367>`_)
  - Remove some experimental update schemas (`#2399 <https://github.com/equinor/ert/issues/2399>`_)
  - Improve bundling of shared resources, and move them under the ert_shared module (`#2176 <https://github.com/equinor/ert/issues/2176>`_, `#2379 <https://github.com/equinor/ert/issues/2379>`_)

Bugfixes:
  - Fix IES analysis to allow custom initial ensemble mask (`#2074 <https://github.com/equinor/ert/issues/2074>`_)
  - Properly remove RMS environment when using run_external (`#2104 <https://github.com/equinor/ert/issues/2104>`_)
  - Fix crash in CSV Export plugin (`#2157 <https://github.com/equinor/ert/issues/2157>`_)
  - Fix occasional GUI crash in detailed view when opening files (`#2300 <https://github.com/equinor/ert/issues/2300>`_)
  - Fix crash in MISFIT_PREPROCESSOR due to inf values (`#2356 <https://github.com/equinor/ert/issues/2356>`_)

New features:
  - Make IES algorithm available by default (`#2037 <https://github.com/equinor/ert/issues/2037>`_)
  - Introduce API for fetching data from file storage; making it possible to run webviz-ert with ert2 (`#2065 <https://github.com/equinor/ert/issues/2065>`_, `#2154 <https://github.com/equinor/ert/issues/2154>`_, `#2100 <https://github.com/equinor/ert/issues/2100>`_, `#2219 <https://github.com/equinor/ert/issues/2219>`_)
  - Show duration in run_dialog for progress (`#2398 <https://github.com/equinor/ert/issues/2398>`_)
  - [ert3] Introduce uniform/invariant records concept (`#2070 <https://github.com/equinor/ert/issues/2070>`_)
  - [ert3] Add concept of record transformation (`#2040 <https://github.com/equinor/ert/issues/2040>`_)

Improvements:
  - Remove outdated help resources (`#2086 <https://github.com/equinor/ert/issues/2086>`_)
  - Introduce BaseService to unify api and vis interface (`#2018 <https://github.com/equinor/ert/issues/2018>`_, `#2147 <https://github.com/equinor/ert/issues/2147>`_, `#2149 <https://github.com/equinor/ert/issues/2149>`_, `#2258 <https://github.com/equinor/ert/issues/2258>`_, `#2308 <https://github.com/equinor/ert/issues/2308>`_)
  - Log workflow usage (`#2113 <https://github.com/equinor/ert/issues/2113>`_)
  - Log forward model jobs (`#2098 <https://github.com/equinor/ert/issues/2098>`_)
  - Update workflows docs (`#2039 <https://github.com/equinor/ert/issues/2039>`_)
  - Fix spelling mistake in template render docs (`#2152 <https://github.com/equinor/ert/issues/2152>`_)
  - Log util_abort (`#2230 <https://github.com/equinor/ert/issues/2230>`_)
  - Add IES to CLI-docs (`#2234 <https://github.com/equinor/ert/issues/2234>`_)
  - Improve retry logic when communicating with Evaluator (`#2248 <https://github.com/equinor/ert/issues/2248>`_, `#2249 <https://github.com/equinor/ert/issues/2249>`_)
  - Retry check if \*_server.json is deleted (`#2250 <https://github.com/equinor/ert/issues/2250>`_)
  - Ensure error message is logged when CLI fails (`#2281 <https://github.com/equinor/ert/issues/2281>`_)
  - Refer to log files on unexpected crash (`#2400 <https://github.com/equinor/ert/issues/2400>`_)
  - [ert3] Validate ensemble size from the ensemble config against the experiment config (`#2370 <https://github.com/equinor/ert/issues/2370>`_)
  - [ert3] Validate that a stage in an ensemble exists in the stages config (`#2371 <https://github.com/equinor/ert/issues/2371>`_)
  - [ert3] Validate file-based workspace resources (`#2377 <https://github.com/equinor/ert/issues/2377>`_)

Dependencies:
  - Add flake8 to dev-requirements (`#2188 <https://github.com/equinor/ert/issues/2188>`_)
  - Specify version-range for beartype (`#2243 <https://github.com/equinor/ert/issues/2243>`_)
  - Set lower-bound on ert-storage >= 0.3.4 (`#2324 <https://github.com/equinor/ert/issues/2324>`_)
  - Add Conan and Catch2 (`#2350 <https://github.com/equinor/ert/issues/2350>`_)

Miscellaneous:
  - Delete unused const LOG_URL (`#2090 <https://github.com/equinor/ert/issues/2090>`_)
  - Use get to read dict (`#2092 <https://github.com/equinor/ert/issues/2092>`_)
  - Fix environment test (`#2093 <https://github.com/equinor/ert/issues/2093>`_)
  - Remove unused komodo Jenkins file (`#2124 <https://github.com/equinor/ert/issues/2124>`_)
  - Remove outdated tips file from docs (`#2126 <https://github.com/equinor/ert/issues/2126>`_)
  - Remove outdated files in libres source directory (`#2127 <https://github.com/equinor/ert/issues/2127>`_)
  - Refactor ErtSummary and add tests (`#2112 <https://github.com/equinor/ert/issues/2112>`_)
  - Simplify Record design (`#2071 <https://github.com/equinor/ert/issues/2071>`_)
  - Fix flaky test_singleton_start (`#2134 <https://github.com/equinor/ert/issues/2134>`_)
  - Make project_id Optional in connection get_info (`#2131 <https://github.com/equinor/ert/issues/2131>`_)
  - Remove ResLog (`#2138 <https://github.com/equinor/ert/issues/2138>`_)
  - Increase max runs on flaky tests (`#2139 <https://github.com/equinor/ert/issues/2139>`_)
  - Libres cmake cleanup (`#2135 <https://github.com/equinor/ert/issues/2135>`_)
  - Refactor ResConfig input validation (`#2114 <https://github.com/equinor/ert/issues/2114>`_)
  - Improve test for active observations (`#2141 <https://github.com/equinor/ert/issues/2141>`_, `#2148 <https://github.com/equinor/ert/issues/2148>`_)
  - Let EvaluatorServerConfig be responsible for keeping port allocated (`#2097 <https://github.com/equinor/ert/issues/2097>`_, `#2242 <https://github.com/equinor/ert/issues/2242>`_, `#2254 <https://github.com/equinor/ert/issues/2254>`_, `#2260 <https://github.com/equinor/ert/issues/2260>`_)
  - Ensure the service is running when test executes (`#2151 <https://github.com/equinor/ert/issues/2151>`_)
  - Remove unlink_node and unlink_vector (`#2155 <https://github.com/equinor/ert/issues/2155>`_)
  - Provide output when producing error from unix_step (`#2144 <https://github.com/equinor/ert/issues/2144>`_)
  - Replace util_file_exists with exists (`#2142 <https://github.com/equinor/ert/issues/2142>`_)
  - [ert3] Refactor statistical tests (`#2156 <https://github.com/equinor/ert/issues/2156>`_, `#2209 <https://github.com/equinor/ert/issues/2209>`_)
  - [ert3] Test indexed ordered dict (`#2172 <https://github.com/equinor/ert/issues/2172>`_)
  - [ert3] Remove an extra summary.df from summary2json job (`#2182 <https://github.com/equinor/ert/issues/2182>`_)
  - [ert3] Make function step use one transmitter per output (`#2183 <https://github.com/equinor/ert/issues/2183>`_)
  - [ert3] Remove unused _NumericalMetaData class (`#2187 <https://github.com/equinor/ert/issues/2187>`_)
  - [ert3] Set input and output type hints in polynomial function (`#2201 <https://github.com/equinor/ert/issues/2201>`_)
  - [ert3] Make input source configuration more independent (`#2203 <https://github.com/equinor/ert/issues/2203>`_)
  - Use Ubuntu 20.04 for Read The Docs builds (`#2205 <https://github.com/equinor/ert/issues/2205>`_)
  - [ert3] Test changing default mime (`#2185 <https://github.com/equinor/ert/issues/2185>`_)
  - Avoid using mutable instance as default argument in IO builder (`#2212 <https://github.com/equinor/ert/issues/2212>`_)
  - Add section about commits in CONTRIBUTING.md (`#2214 <https://github.com/equinor/ert/issues/2214>`_)
  - Delete empty readme (`#2231 <https://github.com/equinor/ert/issues/2231>`_)
  - Add module docstring to ert_data (`#2232 <https://github.com/equinor/ert/issues/2232>`_)
  - [ert3] Use transformations for outputs in Unix step (`#2208 <https://github.com/equinor/ert/issues/2208>`_)
  - Use caplog context (`#2240 <https://github.com/equinor/ert/issues/2240>`_)
  - Remove unused parameters in RunModel (`#2236 <https://github.com/equinor/ert/issues/2236>`_)
  - Test block_fs_driver_create_fs (`#2302 <https://github.com/equinor/ert/issues/2302>`_)
  - Don't use hardcoded ranges in port-tests (`#2246 <https://github.com/equinor/ert/issues/2246>`_)
  - Move capturing inside context in integration test (`#2252 <https://github.com/equinor/ert/issues/2252>`_)
  - Read file to vec using iterator in es_testdata (`#2253 <https://github.com/equinor/ert/issues/2253>`_)
  - Remove unused run_analysis function (`#2256 <https://github.com/equinor/ert/issues/2256>`_)
  - [ert3] Improve input/output handling (`#2174 <https://github.com/equinor/ert/issues/2174>`_, `#2284 <https://github.com/equinor/ert/issues/2284>`_)
  - Remove unnecessary alloc-funcs in analysis module (`#2257 <https://github.com/equinor/ert/issues/2257>`_)
  - Remove unused enkf_update files (`#2264 <https://github.com/equinor/ert/issues/2264>`_)
  - Remove state_map_select_matching\_\_ (`#2280 <https://github.com/equinor/ert/issues/2280>`_)
  - Avoid using same objects in multiple tests (`#2301 <https://github.com/equinor/ert/issues/2301>`_)
  - Remove call to static private function enkf_main_smoother_update\_\_ (`#2287 <https://github.com/equinor/ert/issues/2287>`_)
  - Avoid passing enkf_main_type to enkf_main_inflate (`#2296 <https://github.com/equinor/ert/issues/2296>`_)
  - Remove unnecessary step list alloc function (`#2295 <https://github.com/equinor/ert/issues/2295>`_)
  - Remove unused rng parameter from IES (`#2286 <https://github.com/equinor/ert/issues/2286>`_)
  - Avoid passing enkf_main_type to enkf_main_log_step_list ()(`#2294 <https://github.com/equinor/ert/issues/2294>`_)
  - Remove more unused files (`#2292 <https://github.com/equinor/ert/issues/2292>`_)
  - Rename _set_dict_from_list to _create_record_mapping (`#2181 <https://github.com/equinor/ert/issues/2181>`_)
  - Prefect tests simplification (`#2192 <https://github.com/equinor/ert/issues/2192>`_, `#2317 <https://github.com/equinor/ert/issues/2317>`_)
  - Remove unused enkf_main_submit_jobs and its callstack in enkf_main (`#2307 <https://github.com/equinor/ert/issues/2307>`_)
  - Convert IES to C ++(`#2312 <https://github.com/equinor/ert/issues/2312>`_)
  - Remove unused ${ies_source }(`#2309 <https://github.com/equinor/ert/issues/2309>`_)
  - Test ies_enkf_linalg_extract_active (`#2306 <https://github.com/equinor/ert/issues/2306>`_)
  - Remove 'ies' as dependency for a test (`#2321 <https://github.com/equinor/ert/issues/2321>`_)
  - Refactor ert3 workspace module (`#2299 <https://github.com/equinor/ert/issues/2299>`_, `#2311 <https://github.com/equinor/ert/issues/2311>`_, `#2303 <https://github.com/equinor/ert/issues/2303>`_, `#2342 <https://github.com/equinor/ert/issues/2342>`_, `#2344 <https://github.com/equinor/ert/issues/2344>`_, `#2365 <https://github.com/equinor/ert/issues/2365>`_, `#2426 <https://github.com/equinor/ert/issues/2426>`_)
  - Fix production of ert narratives for communication protocols (`#2319 <https://github.com/equinor/ert/issues/2319>`_)
  - Reduce enkf main usage in analysis module (`#2333 <https://github.com/equinor/ert/issues/2333>`_)
  - Remove unused enkf_main_run_workflow (`#2337 <https://github.com/equinor/ert/issues/2337>`_)
  - Add tests for workspace pollution (`#2293 <https://github.com/equinor/ert/issues/2293>`_)
  - Replace util_int_min with std::min (`#2341 <https://github.com/equinor/ert/issues/2341>`_)
  - Use pytest instead of ErtTestContext (`#2343 <https://github.com/equinor/ert/issues/2343>`_)
  - Remove unused ResPrototypes (`#2210 <https://github.com/equinor/ert/issues/2210>`_)
  - Update development strategy (`#2244 <https://github.com/equinor/ert/issues/2244>`_)
  - Remove dependency on fs_driver_type (`#2251 <https://github.com/equinor/ert/issues/2251>`_)
  - Improve developer documentation (`#2338 <https://github.com/equinor/ert/issues/2338>`_)
  - Expose ert_share_path (`#2373 <https://github.com/equinor/ert/issues/2373>`_)
  - Test enkf_linalg_genX2 with catch (`#2349 <https://github.com/equinor/ert/issues/2349>`_)
  - Test matrix_subtract_row_mean (`#2378 <https://github.com/equinor/ert/issues/2378>`_)
  - Add workspace documentation (`#2385 <https://github.com/equinor/ert/issues/2385>`_)
  - Provide storage URL to ert-storage via env (`#2316 <https://github.com/equinor/ert/issues/2316>`_)
  - Remove hardcoded path to 'true' executable in test (`#2391 <https://github.com/equinor/ert/issues/2391>`_)
  - [ert3] Let serializers write to disk (`#2390 <https://github.com/equinor/ert/issues/2390>`_)
  - Only register signal handlers in main thread (`#2413 <https://github.com/equinor/ert/issues/2413>`_)
  - Add numerical analysis test for Mac (`#2415 <https://github.com/equinor/ert/issues/2415>`_)
  - Improve cleanup on kill signals (`#2352 <https://github.com/equinor/ert/issues/2352>`_, `#2410 <https://github.com/equinor/ert/issues/2410>`_, `#2414 <https://github.com/equinor/ert/issues/2414>`_, `#2428 <https://github.com/equinor/ert/issues/2428>`_)

Version 2.27
------------

Breaking changes:
  - Notice that the bugfix related to HISTORY_OBSERVATION is expected to cause
    changes in the numerical results for users of that feature.

Bugfixes:
  - Include last report step in HISTORY_OBSERVATION (`#1820 <https://github.com/equinor/ert/issues/1820>`_)

New features:
  - ert3: Add fast sensitivity algorithm (`#1941 <https://github.com/equinor/ert/issues/1941>`_)
  - ert3: Support blob output records (`#1920 <https://github.com/equinor/ert/pull/1920>`_)

Improvements:
  - Fix misspelling of keyword LSF_RESOURCE in documentation (`#1242 <https://github.com/equinor/ert/issues/1242>`_)
  - Improved on-premise integration testing (`#1936 <https://github.com/equinor/ert/pull/1936>`_, `#1938 <https://github.com/equinor/ert/pull/1938>`_)

Dependencies:
  - Loosen semeio requirement (`#1935 <https://github.com/equinor/ert/pull/1935>`_)

Miscellaneous:
  - Add encoding to open statements (`#1960 <https://github.com/equinor/ert/pull/1960>`_)
  - Avoid handling bare exceptions (`#1891 <https://github.com/equinor/ert/issues/1891>`_)
  - Remove commented code in logger (`#1956 <https://github.com/equinor/ert/pull/1956>`_)
  - Refactor record implementation (
    `#1875 <https://github.com/equinor/ert/issues/1875>`_,
    `#1925 <https://github.com/equinor/ert/pull/1925>`_,
    `#1929 <https://github.com/equinor/ert/pull/1929>`_,
    `#1931 <https://github.com/equinor/ert/pull/1931>`_,
    `#1933 <https://github.com/equinor/ert/pull/1933>`_
    )
  - Refactor storage abstraction (`#1942 <https://github.com/equinor/ert/pull/1942>`_, `#1945 <https://github.com/equinor/ert/pull/1945>`_)

Version 2.26
------------

 Breaking changes:
   - Revert "Enable ensemble evaluator by default"

 Bugfixes:
   - Fix #1830 bug in progress calculation

 New features:
   - Log when MAX_RUNTIME is reached
   - Add support for blob records (`#1855 <https://github.com/equinor/ert/issues/1855>`_)
   - Implement an ert-storage transmitter

 Improvements:
   - GUI optimizations
   - Make tail configurable in oat experiments
   - Allow non-stochastic input records to sensitivity
   - Implement a --num-iterations option for IES in the cli
   - Updated FIELD keyword doc and in particular the requirements for the FIELD ID and GRID keyword doc with info about ERTBOX grid usage
   - Updated documentation of the FIELD keyword
   - Create ert package

 Dependencies:
   - Add httpx

 Miscellaneous:
   - Add validated record_field property to EnsembleRecord
   - Unite the resolving of port and sockets to use
   - Change the blob data type from List[StrictBytes] to StrictBytes
   - Use QueueDiffer in legacy tracking
   - Use pyrsistent and pre-rendered data in GUI
   - Introduce separate numerical and blob classes
   - Using mocked lsf bsub and bjobs commands to verify behaviour in ERT
   - Reduce source root implementation to one
   - Add Python version and activity badges to README
   - Stop parsing and exposing Eclipse START_DATE in ERT
   - Add default QUEUE_SYSTEM LOCAL to default site-config lines
   - Merge ert and libres python tests (`#1782 <https://github.com/equinor/ert/issues/1782>`_)
   - Fix verification tests by waiting for EE (`#1819 <https://github.com/equinor/ert/issues/1819>`_)
   - Change __token__ to Token for fetching connection info
   - Move ert3.data to ert.data
   - Run GitHub Actions on tags
   - Add ids to test_legacy_tracker test cases
   - Add timed_out field to job_queue node
   - Reorganize ensemble modules into ensemble package

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
 - Travis split for parallel builds [`79 <https://github.com/Equinor/res/pull/79/>`__].


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
