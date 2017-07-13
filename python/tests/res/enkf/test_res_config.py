#  Copyright (C) 2017  Statoil ASA, Norway.
#
#  The file 'test_res_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
import os.path
from datetime import date

from ecl.test import ExtendedTestCase, TestAreaContext
from ecl.util import CTime
from ecl.util.enums import RngAlgTypeEnum

from res.sched import HistorySourceEnum

from res.enkf import ResConfig, SiteConfig, AnalysisConfig

config_defines = {
        "<USER>"            : "TEST_USER",
        "<SCRATCH>"         : "scratch/ert",
        "<CASE_DIR>"        : "the_extensive_case",
        "<ECLIPSE_NAME>"    : "XYZ"
        }

config_data = {
        "RUNPATH"           : "<SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d",
        "NUM_REALIZATIONS"  : 10,
        "MAX_RUNTIME"       : 23400,
        "MIN_REALIZATIONS"  : "50%",
        "MAX_SUBMIT"        : 13,
        "QUEUE_SYSTEM"      : "LSF",
        "UMASK"             : int("007", 8),
        "MAX_RUNNING"       : "100",
        "DATA_FILE"         : "../../eclipse/model/SNAKE_OIL.DATA",
        "START"             : date(2017, 1, 1),
        "SUMMARY"           : ["WOPR:PROD", "WOPT:PROD", "WWPR:PROD", "WWCT:PROD",
                               "WWPT:PROD", "WBHP:PROD", "WWIR:INJ", "WWIT:INJ",
                               "WBHP:INJ", "ROE:1"],
        "GEN_KW"            : ["SIGMA"],
        "ECLBASE"           : "eclipse/model/<ECLIPSE_NAME>-%d",
        "ENSPATH"           : "../output/storage/<CASE_DIR>",
        "PLOT_PATH"         : "../output/results/plot/<CASE_DIR>",
        "UPDATE_LOG_PATH"   : "../output/update_log/<CASE_DIR>",
        "LOG_FILE"          : "../output/log/ert_<CASE_DIR>.log",
        "RUNPATH_FILE"      : "../output/run_path_file/.ert-runpath-list_<CASE_DIR>",
        "REFCASE"           : "../input/refcase/SNAKE_OIL_FIELD",
        "SIGMA"             : {
                                  "TEMPLATE"  : "../input/templates/sigma.tmpl",
                                  "RESULT"    : "coarse.sigma",
                                  "PARAMETER" : "../input/distributions/sigma.dist"
                              },
        "JOBNAME"           : "SNAKE_OIL_STRUCTURE_%d",
        "INSTALL_JOB"       : {
                                  "SNAKE_OIL_SIMULATOR" : {
                                      "CONFIG"     : "../../snake_oil/jobs/SNAKE_OIL_SIMULATOR",
                                      "STDOUT"     : "snake_oil.stdout",
                                      "STDERR"     : "snake_oil.stderr",
                                      "EXECUTABLE" : "snake_oil_simulator.py"
                                      },
                                  "SNAKE_OIL_NPV" : {
                                      "CONFIG"     : "../../snake_oil/jobs/SNAKE_OIL_NPV",
                                      "STDOUT"     : "snake_oil_npv.stdout",
                                      "STDERR"     : "snake_oil_npv.stderr",
                                      "EXECUTABLE" : "snake_oil_npv.py"
                                      },
                                  "SNAKE_OIL_DIFF" : {
                                      "CONFIG"     : "../../snake_oil/jobs/SNAKE_OIL_DIFF",
                                      "STDOUT"     : "snake_oil_diff.stdout",
                                      "STDERR"     : "snake_oil_diff.stderr",
                                      "EXECUTABLE" : "snake_oil_diff.py"
                                      }
                              },
        "FORWARD_MODEL"     : ["SNAKE_OIL_SIMULATOR", "SNAKE_OIL_NPV", "SNAKE_OIL_DIFF"],
        "HISTORY_SOURCE"    : HistorySourceEnum.REFCASE_HISTORY,
        "OBS_CONFIG"        : "../input/observations/obsfiles/observations.txt",
        "LOAD_WORKFLOW"     : {
                                  "MAGIC_PRINT" : "../bin/workflows/MAGIC_PRINT"
                              },
        "LOAD_WORKFLOW_JOB" : {
                                  "UBER_PRINT"  : "../bin/workflows/workflowjobs/bin/uber_print.py"
                              },
        "LOG_LEVEL"         : 3,
        "RNG_ALG_TYPE"      : RngAlgTypeEnum.MZRAN,
        "STORE_SEED"        : "../input/rng/SEED",
        "LOAD_SEED"         : "../input/rng/SEED",
        "GRID"              : "../../eclipse/include/grid/CASE.EGRID",
        "RUN_TEMPLATE"      : {
                                  "seed_template" : {
                                      "TEMPLATE_FILE" : "../input/templates/seed_template.txt",
                                      "TARGET_FILE"   : "seed.txt"
                                      }
                              }
        }

def expand_config_data():
    for define_key in config_defines:
        for data_key in config_data:
            if type(config_data[data_key]) is str:
                config_data[data_key] = config_data[data_key].replace(
                                                        define_key,
                                                        config_defines[define_key]
                                                        )

class ResConfigTest(ExtendedTestCase):

    def set_up_simple(self):
        self.case_directory = self.createTestPath("local/simple_config/")

    def set_up_snake_oil_structure(self):
        self.case_directory = self.createTestPath("local/snake_oil_structure")
        self.config_file = "snake_oil_structure/ert/model/user_config.ert"
        expand_config_data()

    def test_invalid_user_config(self):
        self.set_up_simple()

        with TestAreaContext("void land"):
            with self.assertRaises(IOError):
                ResConfig("this/is/not/a/file")

    def test_init(self):
        self.set_up_simple()

        with TestAreaContext("res_config_init_test") as work_area:
            cwd = os.getcwd()
            work_area.copy_directory(self.case_directory)

            config_file = "simple_config/minimum_config"
            res_config = ResConfig(user_config_file=config_file)

            self.assertIsNotNone(res_config)
            self.assertSameConfigFile(config_file, res_config.user_config_file, os.getcwd())

            self.assertIsNotNone(res_config.site_config)
            self.assertTrue(isinstance(res_config.site_config, SiteConfig))

            self.assertIsNotNone(res_config.analysis_config)
            self.assertTrue(isinstance(res_config.analysis_config, AnalysisConfig))

            self.assertEqual( res_config.config_path , os.path.join( cwd , "simple_config"))

            config_file = os.path.join( cwd, "simple_config/minimum_config")
            res_config = ResConfig(user_config_file=config_file)
            self.assertEqual( res_config.config_path , os.path.join( cwd , "simple_config"))

            os.chdir("simple_config")
            config_file = "minimum_config"
            res_config = ResConfig(user_config_file=config_file)
            self.assertEqual( res_config.config_path , os.path.join( cwd , "simple_config"))

            subst_config = res_config.subst_config
            for t in subst_config:
                print t
            self.assertEqual( subst_config["<CONFIG_PATH>"], os.path.join( cwd , "simple_config"))

    def assertSameConfigFile(self, expected_filename, filename, prefix):
        prefix_path = lambda fn: fn if os.path.isabs(fn) else os.path.join(prefix, fn)
        canonical_path = lambda fn: os.path.realpath(os.path.abspath(prefix_path(fn)))

        self.assertEqual(
                    canonical_path(expected_filename),
                    canonical_path(filename)
                    )

    def test_extensive_config(self):
        self.set_up_snake_oil_structure()

        with TestAreaContext("enkf_test_other_area") as work_area:
            work_area.copy_directory(self.case_directory)

            # Move to another directory
            run_dir = "i/ll/camp/here"
            os.makedirs(run_dir)
            os.chdir(run_dir)

            rel_config_file = "/".join(
                                   [".."] * len(run_dir.split("/")) +
                                   [self.config_file]
                                   )

            res_config = ResConfig(rel_config_file)

            work_dir = os.path.split(rel_config_file)[0]
            rel2workdir = lambda path : os.path.join(
                                             work_dir,
                                             path
                                             )

            # Test properties
            self.assertEqual(
                    config_data["RUNPATH"],
                    res_config.model_config.getRunpathAsString()
                    )

            self.assertEqual(
                    config_data["MAX_RUNTIME"],
                    res_config.analysis_config.get_max_runtime()
                    )

            self.assertEqual(
                    config_data["MAX_SUBMIT"],
                    res_config.site_config.queue_config.max_submit
                    )

            self.assertEqual(
                    config_data["QUEUE_SYSTEM"],
                    res_config.site_config.queue_config.queue_name
                    )

            self.assertEqual(
                    config_data["QUEUE_SYSTEM"],
                    res_config.site_config.queue_config.driver.name
                    )

            self.assertEqual(
                    config_data["MAX_RUNNING"],
                    res_config.site_config.queue_config.driver.get_option("MAX_RUNNING")
                    )

            self.assertEqual(
                    config_data["UMASK"],
                    res_config.site_config.umask
                    )

            self.assertSameConfigFile(
                    config_data["DATA_FILE"],
                    res_config.ecl_config.getDataFile(),
                    work_dir
                    )

            self.assertEqual(
                    CTime(config_data["START"]),
                    res_config.ecl_config.getStartDate()
                    )

            self.assertEqual(
                set(config_data["SUMMARY"] + config_data["GEN_KW"]),
                set(res_config.ensemble_config.alloc_keylist())
                )

            self.assertEqual(
                    config_data["ECLBASE"],
                    res_config.ecl_config.getEclBase()
                    )

            self.assertEqual(
                    config_data["ENSPATH"],
                    res_config.model_config.getEnspath()
                    )

            self.assertEqual(
                    config_data["PLOT_PATH"],
                    res_config.plot_config.getPath()
                    )

            self.assertEqual(
                    config_data["UPDATE_LOG_PATH"],
                    res_config.analysis_config.get_log_path()
                    )

            self.assertSameConfigFile(
                        config_data["RUNPATH_FILE"],
                        res_config.hook_manager.getRunpathList().getExportFile(),
                        work_dir
                        )

            for extension in ["SMSPEC", "UNSMRY"]:
                self.assertSameConfigFile(
                        config_data["REFCASE"] + "." + extension,
                        res_config.ecl_config.getRefcaseName() + "." + extension,
                        work_dir
                        )

            loaded_template_file = res_config.ensemble_config["SIGMA"].getKeywordModelConfig().getTemplateFile()
            self.assertSameConfigFile(
                    config_data["SIGMA"]["TEMPLATE"],
                    loaded_template_file,
                    work_dir
                    )

            loaded_parameter_file = res_config.ensemble_config["SIGMA"].getKeywordModelConfig().getParameterFile()
            self.assertSameConfigFile(
                    config_data["SIGMA"]["PARAMETER"],
                    loaded_parameter_file,
                    work_dir
                    )

            self.assertSameConfigFile(
                    config_data["SIGMA"]["RESULT"],
                    res_config.ensemble_config["SIGMA"]._get_enkf_outfile(),
                    work_dir
                    )

            self.assertEqual(
                    config_data["JOBNAME"],
                    res_config.model_config.getJobnameFormat()
                    )

            job_list = res_config.site_config.get_installed_jobs()
            self.assertEqual(len(config_data["INSTALL_JOB"]), len(job_list))
            for job_name in config_data["INSTALL_JOB"]:
                self.assertTrue(job_name in job_list)

                exp_job_data = config_data["INSTALL_JOB"][job_name]

                self.assertSameConfigFile(
                        exp_job_data["CONFIG"],
                        job_list[job_name].get_config_file(),
                        work_dir
                        )

                self.assertEqual(
                        exp_job_data["STDERR"],
                        job_list[job_name].get_stderr_file()
                        )

                self.assertEqual(
                        exp_job_data["STDOUT"],
                        job_list[job_name].get_stdout_file()
                        )

            self.assertEqual(
                    config_data["FORWARD_MODEL"],
                    res_config.model_config.getForwardModel().joblist()
                    )

            self.assertEqual(
                    config_data["HISTORY_SOURCE"],
                    res_config.model_config.get_history_source()
                    )

            self.assertEqual(
                    len(config_data["LOAD_WORKFLOW"]),
                    len(res_config.ert_workflow_list.getWorkflowNames())
                    )

            for w_name in config_data["LOAD_WORKFLOW"]:
                self.assertTrue(w_name in res_config.ert_workflow_list)

                self.assertSameConfigFile(
                        config_data["LOAD_WORKFLOW"][w_name],
                        res_config.ert_workflow_list[w_name].src_file,
                        work_dir
                        )

            for wj_name in config_data["LOAD_WORKFLOW_JOB"]:
                self.assertTrue(res_config.ert_workflow_list.hasJob(wj_name))
                job = res_config.ert_workflow_list.getJob(wj_name)

                self.assertEqual(wj_name, job.name())
                self.assertSameConfigFile(
                        config_data["LOAD_WORKFLOW_JOB"][wj_name],
                        job.executable(),
                        work_dir
                        )

            self.assertEqual(
                    config_data["RNG_ALG_TYPE"],
                    res_config.rng_config.alg_type
                    )

            self.assertSameConfigFile(
                    config_data["STORE_SEED"],
                    res_config.rng_config.store_filename,
                    work_dir
                    )

            self.assertSameConfigFile(
                    config_data["LOAD_SEED"],
                    res_config.rng_config.load_filename,
                    work_dir
                    )

            self.assertSameConfigFile(
                    config_data["GRID"],
                    res_config.ecl_config.get_gridfile(),
                    work_dir
                    )

            self.assertEqual(
                    config_data["RUN_TEMPLATE"].keys(),
                    res_config.ert_templates.getTemplateNames()
                    )

            for template_name in res_config.ert_templates.getTemplateNames():
                ert_template = res_config.ert_templates.get_template(template_name)
                config_template = config_data["RUN_TEMPLATE"][template_name]

                self.assertSameConfigFile(
                        config_template["TEMPLATE_FILE"],
                        ert_template.get_template_file(),
                        work_dir
                        )

                self.assertEqual(
                        config_template["TARGET_FILE"],
                        ert_template.get_target_file(),
                        work_dir
                        )

            # TODO: Not tested
            # - NUM_REALIZATIONS
            # - MIN_REALIZATIONS
            # - LOG_FILE
            # - LOG_LEVEL
            # - OBS_CONFIG
