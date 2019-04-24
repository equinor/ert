#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_programmatic_res_config.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import os
from ecl.util.test import TestAreaContext
from tests import ResTest

from res.test import ErtTestContext
from res.enkf import ResConfig, ConfigKeys

class ProgrammaticResConfigTest(ResTest):

    def setUp(self):
        self.minimum_config = {
                                "INTERNALS" :
                                {
                                  "CONFIG_DIRECTORY" : "simple_config",
                                },

                                "SIMULATION" :
                                {
                                  "QUEUE_SYSTEM" :
                                  {
                                    "JOBNAME" : "Job%d",
                                  },

                                  "RUNPATH"            : "/tmp/simulations/run%d",
                                  "NUM_REALIZATIONS"   : 10,
                                  "MIN_REALIZATIONS"   : 10,
                                  "JOB_SCRIPT"         : "script.sh",
                                  "ENSPATH"            : "Ensemble"
                                }
                              }

        self.minimum_config_extra_key = {
                                "INTERNALS" :
                                {
                                  "CONFIG_DIRECTORY" : "simple_config",
                                },
                                "UNKNOWN_KEY" : "Have/not/got/a/clue",
                                "SIMULATION" :
                                {
                                  "QUEUE_SYSTEM" :
                                  {
                                    "JOBNAME" : "Job%d",
                                  },

                                  "RUNPATH"            : "/tmp/simulations/run%d",
                                  "NUM_REALIZATIONS"   : 10,
                                  "JOB_SCRIPT"         : "script.sh",
                                  "ENSPATH"            : "Ensemble"
                                }
                              }

        self.minimum_config_error = {
                                "INTERNALS" :
                                {
                                  "CONFIG_DIRECTORY" : "simple_config",
                                },
                                "SIMULATION" :
                                {
                                  "QUEUE_SYSTEM" :
                                  {
                                    "JOBNAME" : "Job%d",
                                  },

                                  "RUNPATH"            : "/tmp/simulations/run%d",
                                  "NUM_REALIZATIONS"   : "/should/be/an/integer",
                                  "JOB_SCRIPT"         : "script.sh",
                                  "ENSPATH"            : "Ensemble"
                                }
                              }


        self.minimum_config_cwd = {
                                "INTERNALS" :
                                {
                                },

                                "SIMULATION" :
                                {
                                  "QUEUE_SYSTEM" :
                                  {
                                    "JOBNAME" : "Job%d",
                                  },

                                  "RUNPATH"            : "/tmp/simulations/run%d",
                                  "NUM_REALIZATIONS"   : 10,
                                  "JOB_SCRIPT"         : "script.sh",
                                  "ENSPATH"            : "Ensemble"
                                }
                              }

        self.new_config    = {
                                "INTERNALS" :
                                {
                                },

                                "SIMULATION" :
                                {
                                  "QUEUE_SYSTEM" :
                                  {
                                    "QUEUE_SYSTEM"     : "LOCAL",
                                    "JOBNAME"          : "SIM_KW",
                                  },
                                  "ECLBASE" : "SIM_KW",

                                  "NUM_REALIZATIONS"   : 10,

                                  "INSTALL_JOB" :
                                  [
                                    {
                                      "NAME" : "NEW_JOB_A",
                                      "PATH" : "simulation_model/jobs/NEW_TYPE_A"
                                    },
                                    {
                                      "NAME" : "NEW_JOB_B",
                                      "PATH" : "simulation_model/jobs/NEW_TYPE_B"
                                    },
                                    {
                                        "NAME" : "NEW_JOB_C",
                                        "PATH" : "simulation_model/jobs/NEW_TYPE_C"
                                      }
                                   ],
                                  "SIMULATION_JOB" :
                                  [
                                    {
                                      "NAME"    : "NEW_JOB_A",
                                      "ARGLIST" : ["Hello", True, 3.14, 4]
                                    },
                                    {
                                      "NAME"    : "NEW_JOB_B",
                                      "ARGLIST" : ["word"]
                                    },
                                  ]


                                }

                             }


        self.large_config  = {
                                "DEFINES" :
                                {
                                  "<USER>"         : "TEST_USER",
                                  "<SCRATCH>"      : "scratch/ert",
                                  "<CASE_DIR>"     : "the_extensive_case",
                                  "<ECLIPSE_NAME>" : "XYZ"
                                },

                                "INTERNALS" :
                                {
                                  "CONFIG_DIRECTORY" : "snake_oil_structure/ert/model",
                                },

                                "SIMULATION" :
                                {
                                  "QUEUE_SYSTEM" :
                                  {
                                    "JOBNAME"          : "SNAKE_OIL_STRUCTURE_%d",
                                    "QUEUE_SYSTEM"     : "LSF",
                                    "MAX_RUNTIME"      : 23400,
                                    "MIN_REALIZATIONS" : "50%",
                                    "MAX_SUBMIT"       : 13,
                                    "UMASK"            : "007",
                                    "QUEUE_OPTION"     :
                                    [
                                      {
                                        "DRIVER_NAME" : "LSF",
                                        "OPTION"      : "MAX_RUNNING",
                                        "VALUE"       : 100
                                      },
                                      {
                                        "DRIVER_NAME" : "LSF",
                                        "OPTION"      : "LSF_RESOURCE",
                                        "VALUE"       : "select[x86_64Linux] same[type:model]",
                                      },
                                      {
                                        "DRIVER_NAME" : "LSF",
                                        "OPTION"      : "LSF_SERVER",
                                        "VALUE"       : "simulacrum",
                                      },
                                      {
                                        "DRIVER_NAME" : "LSF",
                                        "OPTION"      : "LSF_QUEUE",
                                        "VALUE"       : "mr",
                                      },
                                    ]
                                  },

                                  "DATA_FILE"        : "../../eclipse/model/SNAKE_OIL.DATA",
                                  "RUNPATH"          : "<SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d",
                                  "RUNPATH_FILE"     : "../output/run_path_file/.ert-runpath-list_<CASE_DIR>",
                                  "ECLBASE"          : "eclipse/model/<ECLIPSE_NAME>-%d",
                                  "NUM_REALIZATIONS" : "10",
                                  "ENSPATH"          : "../output/storage/<CASE_DIR>",
                                  "GRID"             : "../../eclipse/include/grid/CASE.EGRID",
                                  "REFCASE"          : "../input/refcase/SNAKE_OIL_FIELD",
                                  "HISTORY_SOURCE"   : "REFCASE_HISTORY",
                                  "OBS_CONFIG"       : "../input/observations/obsfiles/observations.txt",
                                  "TIME_MAP"         : "../input/refcase/time_map.txt",

                                  "PLOT_SETTINGS" :
                                  {
                                    "PATH" : "../output/results/plot/<CASE_DIR>"
                                  },

                                  "SUMMARY" :
                                  [
                                    "WOPR:PROD",
                                    "WOPT:PROD",
                                    "WWPR:PROD",
                                    "WWCT:PROD",
                                    "WWPT:PROD",
                                    "WBHP:PROD",
                                    "WWIR:INJ",
                                    "WWIT:INJ",
                                    "WBHP:INJ",
                                    "ROE:1"
                                  ],

                                  "INSTALL_JOB" :
                                  [
                                    {
                                      "NAME" : "SNAKE_OIL_SIMULATOR",
                                      "PATH" : "../../snake_oil/jobs/SNAKE_OIL_SIMULATOR"
                                    },
                                    {
                                      "NAME" : "SNAKE_OIL_NPV",
                                      "PATH" : "../../snake_oil/jobs/SNAKE_OIL_NPV"
                                    },
                                    {
                                      "NAME" : "SNAKE_OIL_DIFF",
                                      "PATH" : "../../snake_oil/jobs/SNAKE_OIL_DIFF"
                                    }
                                  ],

                                  "LOAD_WORKFLOW_JOB" :
                                  [
                                    "../bin/workflows/workflowjobs/UBER_PRINT"
                                  ],

                                  "LOAD_WORKFLOW" :
                                  [
                                    "../bin/workflows/MAGIC_PRINT"
                                  ],

                                  "FORWARD_MODEL" :
                                  [
                                      "SNAKE_OIL_SIMULATOR",
                                      "SNAKE_OIL_NPV",
                                      "SNAKE_OIL_DIFF"
                                  ],

                                  "RUN_TEMPLATE" :
                                  [
                                    {
                                      "TEMPLATE" : "../input/templates/seed_template.txt",
                                      "EXPORT"   : "seed.txt"
                                    }
                                  ],

                                  "GEN_KW" :
                                  [
                                    {
                                      "NAME"           : "SIGMA",
                                      "TEMPLATE"       : "../input/templates/sigma.tmpl",
                                      "OUT_FILE"       : "coarse.sigma",
                                      "PARAMETER_FILE" : "../input/distributions/sigma.dist"
                                    }
                                  ],

                                  "CUSTOM_KW" :
                                  [
                                    {
                                      "NAME"        : "UNCERTAINTY",
                                      "RESULT_FILE" : "uncertainties.txt"
                                    }
                                  ],

                                  "GEN_DATA" :
                                  [
                                    {
                                      "NAME"         : "super_data",
                                      "RESULT_FILE"  : "super_data_%d",
                                      "REPORT_STEPS" : 1
                                    }
                                  ],

                                  "LOGGING" :
                                  {
                                    "LOG_LEVEL"       : "INFO",
                                    "UPDATE_LOG_PATH" : "../output/update_log/<CASE_DIR>",
                                    "LOG_FILE"        : "../output/log/ert_<CASE_DIR>.log"
                                  },

                                  "SEED" :
                                  {
                                    "STORE_SEED" : "../input/rng/SEED",
                                    "LOAD_SEED"  : "../input/rng/SEED"
                                  }
                                }
                              }
    def test_minimum_config(self):
        case_directory = self.createTestPath("local/simple_config")
        config_file = "simple_config/minimum_config"

        with TestAreaContext("res_config_prog_test") as work_area:
            work_area.copy_directory(case_directory)

            loaded_res_config = ResConfig(user_config_file=config_file)
            prog_res_config = ResConfig(config=self.minimum_config)

            self.assertEqual(loaded_res_config.model_config.num_realizations,
                             prog_res_config.model_config.num_realizations)

            self.assertEqual(loaded_res_config.model_config.getJobnameFormat(),
                             prog_res_config.model_config.getJobnameFormat())

            self.assertEqual(loaded_res_config.model_config.getRunpathAsString(),
                             prog_res_config.model_config.getRunpathAsString())

            self.assertEqual(loaded_res_config.model_config.getEnspath(),
                             prog_res_config.model_config.getEnspath())

            self.assertEqual(0, len(prog_res_config.errors))
            self.assertEqual(0, len(prog_res_config.failed_keys))


    def test_no_config_directory(self):
        case_directory = self.createTestPath("local/simple_config")
        config_file = "simple_config/minimum_config"

        with TestAreaContext("res_config_prog_test") as work_area:
            work_area.copy_directory(case_directory)

            with self.assertRaises(ValueError):
                ResConfig(config=self.minimum_config_cwd)


    def test_errors(self):
        case_directory = self.createTestPath("local/simple_config")
        config_file = "simple_config/minimum_config"

        with TestAreaContext("res_config_prog_test") as work_area:
            work_area.copy_directory(case_directory)

            with self.assertRaises(ValueError):
                res_config = ResConfig(config=self.minimum_config_error)

            res_config = ResConfig(config=self.minimum_config_error,
                                   throw_on_error=False)

            self.assertTrue(len(res_config.errors) > 0)
            self.assertEqual(0, len(res_config.failed_keys))


    def test_failed_keys(self):
        case_directory = self.createTestPath("local/simple_config")
        config_file = "simple_config/minimum_config"

        with TestAreaContext("res_config_prog_test") as work_area:
            work_area.copy_directory(case_directory)

            res_config = ResConfig(config=self.minimum_config_extra_key)

            self.assertTrue(len(res_config.failed_keys) == 1)
            self.assertEqual(["UNKNOWN_KEY"], list(res_config.failed_keys.keys()))
            self.assertEqual(self.minimum_config_extra_key["UNKNOWN_KEY"],
                            res_config.failed_keys["UNKNOWN_KEY"])


    def assert_equal_model_config(self, loaded_model_config, prog_model_config):
        self.assertEqual(loaded_model_config.num_realizations,
                         prog_model_config.num_realizations)

        self.assertEqual(loaded_model_config.getJobnameFormat(),
                         prog_model_config.getJobnameFormat())

        self.assertEqual(loaded_model_config.getRunpathAsString(),
                         prog_model_config.getRunpathAsString())

        self.assertEqual(loaded_model_config.getEnspath(),
                         prog_model_config.getEnspath())

        self.assertEqual(loaded_model_config.get_history_source(),
                         prog_model_config.get_history_source())

        self.assertEqual(loaded_model_config.obs_config_file,
                         prog_model_config.obs_config_file)

        self.assertEqual(loaded_model_config.getForwardModel().joblist(),
                         prog_model_config.getForwardModel().joblist())


    def assert_equal_site_config(self, loaded_site_config, prog_site_config):

        self.assertEqual(loaded_site_config.umask,
                         prog_site_config.umask)

        loaded_job_list = loaded_site_config.get_installed_jobs()
        prog_job_list = prog_site_config.get_installed_jobs()

        self.assertEqual(loaded_job_list.getAvailableJobNames(),
                         prog_job_list.getAvailableJobNames())

        for job_name in loaded_job_list.getAvailableJobNames():
            self.assertEqual(loaded_job_list[job_name].get_config_file(),
                             prog_job_list[job_name].get_config_file())

            self.assertEqual(loaded_job_list[job_name].get_stderr_file(),
                             prog_job_list[job_name].get_stderr_file())

            self.assertEqual(loaded_job_list[job_name].get_stdout_file(),
                             prog_job_list[job_name].get_stdout_file())


    def assert_equal_ecl_config(self, loaded_ecl_config, prog_ecl_config):
        self.assertEqual(loaded_ecl_config.getDataFile(),
                         prog_ecl_config.getDataFile())

        self.assertEqual(loaded_ecl_config.get_gridfile(),
                         prog_ecl_config.get_gridfile())

        self.assertEqual(loaded_ecl_config.getRefcaseName(),
                         prog_ecl_config.getRefcaseName())


    def assert_equal_analysis_config(self, loaded_config, prog_config):
        self.assertEqual(loaded_config.get_log_path(),
                         prog_config.get_log_path())

        self.assertEqual(loaded_config.get_max_runtime(),
                         prog_config.get_max_runtime())

    def assert_equal_hook_manager(self, loaded_hook_manager, prog_hook_manager):
        self.assertEqual(loaded_hook_manager.getRunpathList().getExportFile(),
                         prog_hook_manager.getRunpathList().getExportFile())


    def assert_equal_log_config(self, loaded_log_config, prog_log_config):
        self.assertEqual(loaded_log_config.log_file,
                         prog_log_config.log_file)

        self.assertEqual(loaded_log_config.log_level,
                         prog_log_config.log_level)


    def assert_equal_ensemble_config(self, loaded_config, prog_config):
        self.assertEqual(set(loaded_config.alloc_keylist()),
                         set(prog_config.alloc_keylist()))


    def assert_equal_ert_templates(self, loaded_templates, prog_templates):
        self.assertEqual(loaded_templates.getTemplateNames(),
                         prog_templates.getTemplateNames())

        for template_name in loaded_templates.getTemplateNames():
            let = loaded_templates.get_template(template_name)
            pet = prog_templates.get_template(template_name)

            self.assertEqual(let.get_template_file(), pet.get_template_file())
            self.assertEqual(let.get_target_file(), pet.get_target_file())


    def assert_equal_ert_workflow(self, loaded_workflow_list, prog_workflow_list):
        self.assertEqual(loaded_workflow_list.getWorkflowNames(),
                         prog_workflow_list.getWorkflowNames())

        for wname in loaded_workflow_list.getWorkflowNames():
            self.assertEqual(loaded_workflow_list[wname].src_file,
                             prog_workflow_list[wname].src_file)

        self.assertEqual(loaded_workflow_list.getJobNames(),
                         prog_workflow_list.getJobNames())

        for jname in loaded_workflow_list.getJobNames():
            ljob = loaded_workflow_list.getJob(jname)
            pjob = prog_workflow_list.getJob(jname)

            self.assertEqual(ljob.name(), pjob.name())
            self.assertEqual(ljob.executable(), pjob.executable())


    def assert_equal_simulation_config(self, loaded_simulation_config, prog_simulation_config):
        self.assertEqual(loaded_simulation_config.getPath(),
                         prog_simulation_config.getPath())

    def test_new_config(self):
        case_directory = self.createTestPath("local/simulation_model")
        config_file = "simulation_model/sim_kw.ert"

        with TestAreaContext("res_config_sim_job") as work_area:
            work_area.copy_directory(case_directory)

            prog_res_config   = ResConfig(config=self.new_config)
            forward_model = prog_res_config.model_config.getForwardModel()
            job_A = forward_model.iget_job(0)
            job_B = forward_model.iget_job(1)
            self.assertEqual( job_A.get_arglist(), ["Hello", "True", "3.14", "4"])
            self.assertEqual( job_B.get_arglist(), ["word"])



    def test_large_config(self):
        case_directory = self.createTestPath("local/snake_oil_structure")
        config_file = "snake_oil_structure/ert/model/user_config.ert"

        with TestAreaContext("res_config_prog_test") as work_area:
            work_area.copy_directory(case_directory)

            loaded_res_config = ResConfig(user_config_file=config_file)
            prog_res_config   = ResConfig(config=self.large_config)

            self.assert_equal_model_config(loaded_res_config.model_config,
                                           prog_res_config.model_config)

            self.assert_equal_site_config(loaded_res_config.site_config,
                                          prog_res_config.site_config)

            self.assert_equal_ecl_config(loaded_res_config.ecl_config,
                                         prog_res_config.ecl_config)

            self.assert_equal_analysis_config(loaded_res_config.analysis_config,
                                              prog_res_config.analysis_config)

            self.assert_equal_hook_manager(loaded_res_config.hook_manager,
                                           prog_res_config.hook_manager)

            self.assert_equal_log_config(loaded_res_config.log_config,
                                         prog_res_config.log_config)

            self.assert_equal_ensemble_config(loaded_res_config.ensemble_config,
                                              prog_res_config.ensemble_config)

            self.assert_equal_ert_templates(loaded_res_config.ert_templates,
                                            prog_res_config.ert_templates)

            self.assert_equal_ert_workflow(loaded_res_config.ert_workflow_list,
                                           prog_res_config.ert_workflow_list)

            self.assertEqual(loaded_res_config.queue_config, prog_res_config.queue_config)

            self.assertEqual(0, len(prog_res_config.failed_keys))


    def test_test_context(self):
        case_directory = self.createTestPath("local/simple_config")

        # We first make sure that the files referred to in the
        # minimum_config dictionary are found are located correctly by
        # creating a working area, and then we create testcontext from
        # there.
        with TestAreaContext("res_config_prog_test", store_area = True) as work_area:
            work_area.copy_directory(case_directory)
            self.assertTrue(os.path.isfile( "simple_config/script.sh"))

            with ErtTestContext( "dict_test", config_dict = self.minimum_config, store_area = True):
                pass

            os.chdir( "simple_config")
            # The directory referenced in INTERNALS.CONFIG_DIRECTORY does not exist => IOError
            with self.assertRaises(IOError):
                with ErtTestContext( "dict_test", config_dict = self.minimum_config, store_area = True):
                    pass

            # Create minimum config in cwd:
            with ErtTestContext( "dict_test", config_dict = self.minimum_config_cwd, store_area = True):
                pass
