Release notes for ERT
=====================

Version 2.4
-----------

2.4 ert application
~~~~~~~~~~~~~~~~~~~
PR: 162 - ?


2.4 libres
~~~~~~~~~~
PR: 411 - ?


2.4 libecl
~~~~~~~~~~
PR: 506 - 579

New functionality:
 - Ability to compute geertsma based on dynamic porevolume `[530] https://github.com/equinor/libecl/pull/530?`
 - Support for Intersect NNC format `[533] https://github.com/equinor/libecl/pull/533`
 - Support for extrapolation when resampling `[534] https://github.com/equinor/libecl/pull/534`
 - Ability to load summary data from .csv-files `[536] https://github.com/equinor/libecl/pull/536`
 - Identify region-to-region variables `[551] https://github.com/equinor/libecl/pull/551`

Improvements:
 - Load LGR info when loading well info `[529] https://github.com/equinor/libecl/pull/529`
 - Do not fail if restart file is missing icon `[549] https://github.com/equinor/libecl/pull/549`

Others:
 - Various improvements to code due to now being a C++ project.
 - Improved documentation for Windows users
 - Improved Python 3 testing
 - Revert fortio changes to increase reading speed `[567] https://github.com/equinor/libecl/pull/567`


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
Cherry-picked: `70 <https://github.com/Statoil/ert/pull/70/>`_

Misc:

 - Using res_config changes from libres `[16] <https://github.com/Statoil/ert/pull/16/>`_
 - files moved from libecl to libres: `[51] <https://github.com/Statoil/ert/pull/51>`_
 - replaced ert.enkf with res.enkf `[56] <https://github.com/Statoil/ert/pull/56/>`_
 - Created ErtVersion: [`61 <https://github.com/Statoil/ert/pull/61/>`_, `66 <https://github.com/Statoil/ert/pull/66/>`_].
 - Using res_config: [`62 <https://github.com/Statoil/ert/pull/62/>`_]
 - Removed dead workflow files: `[64] <https://github.com/Statoil/ert/pull/64/>`_

Build and testing:

 - Cleanup after repo split [`1 <https://github.com/Statoil/ert/pull/1/>`_, `2 <https://github.com/Statoil/ert/pull/2/>`_, `3 <https://github.com/Statoil/ert/pull/3/>`_ , `4 <https://github.com/Statoil/ert/pull/4/>`_, `5 <https://github.com/Statoil/ert/pull/5/>`_ , `6 <https://github.com/Statoil/ert/pull/6/>`_]
 - Added test_install functionality [`7 <https://github.com/Statoil/ert/pull/7/>`_]
 - Added travis build script for libecl+libres+ert:
   [`15 <https://github.com/Statoil/ert/pull/15/>`_, `17 <https://github.com/Statoil/ert/pull/17/>`_, `18 <https://github.com/Statoil/ert/pull/18/>`_, `19 <https://github.com/Statoil/ert/pull/19/>`_, `21 <https://github.com/Statoil/ert/pull/21/>`_, `26 <https://github.com/Statoil/ert/pull/26/>`_, `27 <https://github.com/Statoil/ert/pull/27/>`_, `39, <https://github.com/Statoil/ert/pull/39/>`_ `52 <https://github.com/Statoil/ert/pull/52/>`_-`55 <https://github.com/Statoil/ert/pull/55/>`_, `63 <https://github.com/Statoil/ert/pull/63/>`_]

 - MacOS build error: [`28 <https://github.com/Statoil/ert/pull/28/>`_].
 - Created simple gui_test bin/gui_test [`32 <https://github.com/Statoil/ert/pull/32/>`_]
 - cmake - create symlink: [`41 <https://github.com/Statoil/ert/pull/41/>`_, `42 <https://github.com/Statoil/ert/pull/42/>`_, `43 <https://github.com/Statoil/ert/pull/43/>`_]
 - Initial Python3 testing [`58 <https://github.com/Statoil/ert/pull/58/>`_, `60 <https://github.com/Statoil/ert/pull/60/>`_].


Queue and running:

 - Added base run model - gui model updates: [`20 <https://github.com/Statoil/ert/pull/20/>`_].
 - Added single simulation pretest running [`33 <https://github.com/Statoil/ert/pull/33/>`_, `36 <https://github.com/Statoil/ert/pull/36/>`_, `50 <https://github.com/Statoil/ert/pull/50/>`_, `67 <https://github.com/Statoil/ert/pull/67/>`_].
 - Add run_id to simulation batches.


2.2: libres
~~~~~~~~~~~

Version 2.2.9 September 2017 PR: 1 - 104
Cherry-picks: [`106 <https://github.com/Statoil/res/pull/106/>`_, `108 <https://github.com/Statoil/res/pull/108/>`_, `110 <https://github.com/Statoil/res/pull/110/>`_, `118 <https://github.com/Statoil/res/pull/118/>`_, `121 <https://github.com/Statoil/res/pull/121/>`_, `122 <https://github.com/Statoil/res/pull/122/>`_, `123 <https://github.com/Statoil/res/pull/123/>`_, `127 <https://github.com/Statoil/res/pull/127/>`_]

Misc:

 - implement legacy from ert.xxx [`1, <https://github.com/Statoil/res/pull/1/>`_ `20, <https://github.com/Statoil/res/pull/20/>`_ `21, <https://github.com/Statoil/res/pull/21/>`_ `22 <https://github.com/Statoil/res/pull/22/>`_]
 - Setting up libres_util and moving ert_log there [`13 <https://github.com/Statoil/res/pull/13/>`_, `44 <https://github.com/Statoil/res/pull/44/>`_, `48 <https://github.com/Statoil/res/pull/48/>`_].
 - Added subst_list + block_fs functionality to res_util - moved from
   libecl [`27 <https://github.com/Statoil/res/pull/27/>`_, `68 <https://github.com/Statoil/res/pull/68/>`_, `74 <https://github.com/Statoil/res/pull/74/>`_].
 - Do not generate parameters.txt if no GEN_KW is specified.[`89 <https://github.com/Statoil/res/pull/89/>`_]
 - Started using RES_VERSION [`91 <https://github.com/Statoil/res/pull/91/>`_].
 - CONFIG_PATH subtitution settings - bug fixed[`43 <https://github.com/Statoil/res/pull/43/>`_, `96 <https://github.com/Statoil/res/pull/96/>`_].
 - Will load summary if GEN_DATA is present [`123 <https://github.com/Statoil/res/pull/123/>`_, `127 <https://github.com/Statoil/res/pull/127/>`_]


Build and test fixes:

 - Simple functionality to do post-install testing[`3 <https://github.com/Statoil/res/pull/3/>`_]
 - Use libecl as cmake target[`6 <https://github.com/Statoil/res/pull/6/>`_,`15 <https://github.com/Statoil/res/pull/15/>`_]
 - removed stale binaries [`7 <https://github.com/Statoil/res/pull/7/>`_, `9 <https://github.com/Statoil/res/pull/9/>`_]
 - travis will build all repositories [`23 <https://github.com/Statoil/res/pull/23/>`_].
 - Travis + OSX [`69 <https://github.com/Statoil/res/pull/69/>`_, `72 <https://github.com/Statoil/res/pull/72/>`_]
 - Remove statoil specific settings from build sytem [`38 <https://github.com/Statoil/res/pull/38/>`_].
 - Travis split for parallell builds [`79 <https://github.com/Statoil/res/pull/79/>`_].


Config refactor:

  In this release cycle there have been large amount of changes to the
  code configuring the ERT state; the purpose of these changes has
  been to prepare for further development with Everest. The main net
  change is that a new configuration object - res_config has been
  created ,which holds all the configuration subobjects:

    [`10 <https://github.com/Statoil/res/pull/10/>`_, `14 <https://github.com/Statoil/res/pull/14/>`_, `35 <https://github.com/Statoil/res/pull/35/>`_, `39 <https://github.com/Statoil/res/pull/39/>`_, `45 <https://github.com/Statoil/res/pull/45/>`_, `52 <https://github.com/Statoil/res/pull/52/>`_, `54 <https://github.com/Statoil/res/pull/54/>`_, `58 <https://github.com/Statoil/res/pull/58/>`_-`62 <https://github.com/Statoil/res/pull/62/>`_, `66 <https://github.com/Statoil/res/pull/66/>`_, `75 <https://github.com/Statoil/res/pull/75/>`_]


Queue layer:
`
 - Improved logging [`17 <https://github.com/Statoil/res/pull/17/>`_, `37 <https://github.com/Statoil/res/pull/37/>`_].
 - Funcionality to create a queue_config object copy [`36 <https://github.com/Statoil/res/pull/36/>`_].

 As part of this development cycle the job_dispatch script has been
 included in the libres distribution. There are many PR's related to
 this script:

    [`28 <https://github.com/Statoil/res/pull/28/>`_, `40 <https://github.com/Statoil/res/pull/40/>`_, `41 <https://github.com/Statoil/res/pull/1/>`_, `51 <https://github.com/Statoil/res/pull/51/>`_, `53 <https://github.com/Statoil/res/pull/53/>`_, `63 <https://github.com/Statoil/res/pull/63/>`_, `64 <https://github.com/Statoil/res/pull/64/>`_, `83 <https://github.com/Statoil/res/pull/83/>`_, `84 <https://github.com/Statoil/res/pull/84/>`_, `85 <https://github.com/Statoil/res/pull/85/>`_, `93 <https://github.com/Statoil/res/pull/93/>`_, `94 <https://github.com/Statoil/res/pull/94/>`_, `95 <https://github.com/Statoil/res/pull/95/>`_, `97 <https://github.com/Statoil/res/pull/97/>`_-`99 <https://github.com/Statoil/res/pull/99/>`_,
     `101 <https://github.com/Statoil/res/pull/101/>`_, `103 <https://github.com/Statoil/res/pull/103/>`_, `108 <https://github.com/Statoil/res/pull/108/>`_, `110 <https://github.com/Statoil/res/pull/110/>`_]

 - Create a common run_id for one batch of simulations, and generally
   treat one batch of simulations as one unit, in a better way than
   previously: [`42 <https://github.com/Statoil/res/pull/42/>`_, `67 <https://github.com/Statoil/res/pull/67/>`_]

 - Added PPU (Paay Per Use) code to LSF driver [`71 <https://github.com/Statoil/res/pull/71/>`_].
 - Workflow job PRE_SIMULATION_COPY [`73 <https://github.com/Statoil/res/pull/73/>`_, `88 <https://github.com/Statoil/res/pull/88/>`_].
 - Allow to unset QUEUE_OPTION [`87 <https://github.com/Statoil/res/pull/87/>`_].
 - Jobs failing due to dead nodes are restarted [`100 <https://github.com/Statoil/res/pull/100/>`_].


Documentation:

  - Formatting bugs: [`49 <https://github.com/Statoil/res/pull/49/>`_, `50 <https://github.com/Statoil/res/pull/50/>`_]
  - Removed doxygen + build rst [`29 <https://github.com/Statoil/res/pull/29/>`_]

2.2: libecl
~~~~~~~~~~~

Version 2.2.0 September 2017 PR: 1 - 169
Open PR: 108, 145

Grid:

 - Extracted implementation ecl_nnc_geometry [`1 <https://github.com/Statoil/libecl/pull/1/>`_, `66 <https://github.com/Statoil/libecl/pull/66/>`_, `75 <https://github.com/Statoil/libecl/pull/75/>`_, `78 <https://github.com/Statoil/libecl/pull/78/>`_, `80 <https://github.com/Statoil/libecl/pull/80/>`_, `109 <https://github.com/Statoil/libecl/pull/109/>`_].
 - Fix bug in cell_contains for mirrored grid [`51 <https://github.com/Statoil/libecl/pull/51/>`_, `53 <https://github.com/Statoil/libecl/pull/53/>`_].
 - Extract subgrid from grid [`56 <https://github.com/Statoil/libecl/pull/56/>`_].
 - Expose mapaxes [`63 <https://github.com/Statoil/libecl/pull/63/>`_, `64 <https://github.com/Statoil/libecl/pull/64/>`_].
 - grid.get_lgr - numbered lookup [`83 <https://github.com/Statoil/libecl/pull/83/>`_]
 - Added NUMRES values to EGRID header [`125 <https://github.com/Statoil/libecl/pull/125/>`_].

Build & testing:

 - Removed warnings - added pylint [`4 <https://github.com/Statoil/libecl/pull/4/>`_, `5 <https://github.com/Statoil/libecl/pull/5/>`_, `6 <https://github.com/Statoil/libecl/pull/6/>`_, `10 <https://github.com/Statoil/libecl/pull/10/>`_, `11 <https://github.com/Statoil/libecl/pull/11/>`_, `12 <https://github.com/Statoil/libecl/pull/12/>`_]
 - Accept any Python 2.7.x version [`17 <https://github.com/Statoil/libecl/pull/17/>`_, `18 <https://github.com/Statoil/libecl/pull/18/>`_]
 - Remove ERT testing & building [`3 <https://github.com/Statoil/libecl/pull/3/>`_, `19 <https://github.com/Statoil/libecl/pull/19/>`_]
 - Changes to Python/cmake machinery [`25 <https://github.com/Statoil/libecl/pull/25/>`_, `30 <https://github.com/Statoil/libecl/pull/3/>`_, `31 <https://github.com/Statoil/libecl/pull/31/>`_, `32 <https://github.com/Statoil/libecl/pull/32/>`_, `49 <https://github.com/Statoil/libecl/pull/49/>`_, `52 <https://github.com/Statoil/libecl/pull/52/>`_, `62 <https://github.com/Statoil/libecl/pull/62/>`_].
 - Added cmake config file [`33 <https://github.com/Statoil/libecl/pull/33/>`_, `44 <https://github.com/Statoil/libecl/pull/44/>`_, `45 <https://github.com/Statoil/libecl/pull/45/>`_, `47 <https://github.com/Statoil/libecl/pull/47/>`_].
 - Only *one* library [`54 <https://github.com/Statoil/libecl/pull/54/>`_, `55 <https://github.com/Statoil/libecl/pull/55/>`_, `58 <https://github.com/Statoil/libecl/pull/58/>`_,
 `69 <https://github.com/Statoil/libecl/pull/69/>`_, `73 <https://github.com/Statoil/libecl/pull/73/>`_, `77 <https://github.com/Statoil/libecl/pull/77/>`_, `91 <https://github.com/Statoil/libecl/pull/91/>`_, `133 <https://github.com/Statoil/libecl/pull/133/>`_]
 - Removed stale binaries [`59 <https://github.com/Statoil/libecl/pull/59/>`_].
 - Require cmake >= 2.8.12 [`67 <https://github.com/Statoil/libecl/pull/67/>`_].
 - Fix build on OSX [`87 <https://github.com/Statoil/libecl/pull/87/>`_, `88 <https://github.com/Statoil/libecl/pull/88/>`_, `95 <https://github.com/Statoil/libecl/pull/95/>`_, `103 <https://github.com/Statoil/libecl/pull/103/>`_].
 - Fix broken behavior with internal test data [`97 <https://github.com/Statoil/libecl/pull/97/>`_].
 - Travis - compile with -Werror [`122 <https://github.com/Statoil/libecl/pull/122/>`_, `123 <https://github.com/Statoil/libecl/pull/123/>`_, `127 <https://github.com/Statoil/libecl/pull/127/>`_, `130 <https://github.com/Statoil/libecl/pull/130/>`_]
 - Started to support Python3 syntax [`150 <https://github.com/Statoil/libecl/pull/150/>`_, `161 <https://github.com/Statoil/libecl/pull/161/>`_]
 - Add support for paralell builds on Travis [`149 <https://github.com/Statoil/libecl/pull/149/>`_]

libecl now fully supports OSX. On Travis it is compiled with
-Werror=all which should protect against future warnings.

C++:

 - Removed use of deignated initializers [`7 <https://github.com/Statoil/libecl/pull/7/>`_].
 - Memory leak in EclFilename.cpp [`14 <https://github.com/Statoil/libecl/pull/14/>`_].
 - Guarantee C linkage for ecl_data_type [`65 <https://github.com/Statoil/libecl/pull/65/>`_].
 - New smspec overload [`89 <https://github.com/Statoil/libecl/pull/89/>`_].
 - Use -std=c++0x if -std=c++11 is unavailable [`118 <https://github.com/Statoil/libecl/pull/118/>`_]
 - Make all of (previous( libutil compile with C++ [`162 <https://github.com/Statoil/libecl/pull/162/>`_]

Well:

 - Get well rates from restart files [`8 <https://github.com/Statoil/libecl/pull/8/>`_,`20 <https://github.com/Statoil/res/pull/20/>`_].
 - Test if file exists before load [`111 <https://github.com/Statoil/libecl/pull/111/>`_].
 - Fix some warnings [`169 <https://github.com/Statoil/libecl/pull/169/>`_]

Core:

 - Support for variable length strings in binary eclipse files [`13 <https://github.com/Statoil/libecl/pull/13/>`_, `146 <https://github.com/Statoil/libecl/pull/146/>`_].
 - Renamed root package ert -> ecl [`21 <https://github.com/Statoil/libecl/pull/21/>`_]
 - Load INTERSECT summary files with NAMES instead WGNAMES [`34 <https://github.com/Statoil/libecl/pull/34/>`_ - `39 <https://github.com/Statoil/libecl/pull/39/>`_].
 - Possible memory leak: [`61 <https://github.com/Statoil/libecl/pull/61/>`_]
 - Refactored binary time search in __get_index_from_sim_time() [`113 <https://github.com/Statoil/libecl/pull/113/>`_]
 - Possible to mark fortio writer as "failed" - will unlink on close [`119 <https://github.com/Statoil/libecl/pull/119/>`_].
 - Allow keywords of more than 8 characters [`120 <https://github.com/Statoil/libecl/pull/120/>`_, `124 <https://github.com/Statoil/libecl/pull/124/>`_].
 - ecl_sum writer: Should write RESTART keyword [`129 <https://github.com/Statoil/libecl/pull/129/>`_, `131 <https://github.com/Statoil/libecl/pull/131/>`_]
 - Made EclVersion class [`160 <https://github.com/Statoil/libecl/pull/160/>`_]
 - Functionality to dump an index file for binary files: [`155 <https://github.com/Statoil/libecl/pull/155/>`_, `159 <https://github.com/Statoil/libecl/pull/159/>`_, `163 <https://github.com/Statoil/libecl/pull/163/>`_, `166 <https://github.com/Statoil/libecl/pull/166/>`_, `167 <https://github.com/Statoil/libecl/pull/167/>`_]

Misc:

 - Added legacy pacakge ert/ [`48 <https://github.com/Statoil/libecl/pull/48/>`_, `99 <https://github.com/Statoil/libecl/pull/99/>`_]
 - Improved logging - adding enums for og levels [`90 <https://github.com/Statoil/libecl/pull/90/>`_, `140 <https://github.com/Statoil/libecl/pull/140/>`_, `141 <https://github.com/Statoil/libecl/pull/141/>`_]
 - Refactored to use snake_case instead of CamelCase [`144 <https://github.com/Statoil/libecl/pull/144/>`_, `145 <https://github.com/Statoil/libecl/pull/145/>`_]


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

