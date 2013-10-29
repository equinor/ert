from ert_gui.ide.keywords.definitions import IntegerArgument, StringArgument, BoolArgument, PathArgument, FloatArgument
from ert_gui.ide.keywords import ErtKeywords
from ert_tests import ExtendedTestCase


class ErtKeywordTest(ExtendedTestCase):

    def setUp(self):
        self.keywords = ErtKeywords()

    def keywordTest(self, name, argument_types, documentation_link, group, required=False):
        self.assertTrue(name in self.keywords)

        cld = self.keywords[name]

        self.assertEqual(cld.keywordDefinition().name(), name)
        self.assertEqual(cld.group(), group)
        self.assertEqual(cld.documentationLink(), documentation_link)
        self.assertEqual(cld.isRequired(), required)

        arguments = cld.argumentDefinitions()

        self.assertEqual(len(arguments), len(argument_types))

        for index in range(len(arguments)):
            self.assertIsInstance(arguments[index], argument_types[index])


    def test_eclipse_keywords(self):
        self.keywordTest("DATA_FILE", [PathArgument], "eclipse/data_file", "Eclipse", True)
        self.keywordTest("ECLBASE", [StringArgument], "eclipse/ecl_base", "Eclipse", True)
        self.keywordTest("JOBNAME", [StringArgument], "eclipse/job_name", "Eclipse", True)
        self.keywordTest("GRID", [PathArgument], "eclipse/grid", "Eclipse", True)
        self.keywordTest("INIT_SECTION", [PathArgument], "eclipse/init_section", "Eclipse", True)
        self.keywordTest("SCHEDULE_FILE", [PathArgument], "ensemble/schedule_file", "Eclipse", True)
        self.keywordTest("DATA_KW", [StringArgument, StringArgument], "ensemble/data_kw", "Eclipse")

    def test_ensemble_keywords(self):
        self.keywordTest("NUM_REALIZATIONS", [IntegerArgument], "ensemble/num_realizations", "Ensemble", True)
        self.keywordTest("ENKF_SCHED_FILE", [PathArgument], "ensemble/enkf_sched_file", "Ensemble")
        self.keywordTest("END_DATE", [StringArgument], "ensemble/end_date", "Ensemble")
        self.keywordTest("ENSPATH", [PathArgument], "ensemble/enspath", "Ensemble")
        self.keywordTest("SELECT_CASE", [PathArgument], "ensemble/select_case", "Ensemble")
        self.keywordTest("HISTORY_SOURCE", [StringArgument], "ensemble/history_source", "Ensemble")
        self.keywordTest("REFCASE", [PathArgument], "ensemble/refcase", "Ensemble")
        self.keywordTest("OBS_CONFIG", [PathArgument], "ensemble/obs_config", "Ensemble")
        self.keywordTest("RESULT_PATH", [PathArgument], "ensemble/result_path", "Ensemble")

    def test_run_keywords(self):
        self.keywordTest("INSTALL_JOB", [StringArgument, PathArgument], "ensemble/install_job", "Run")
        self.keywordTest("DELETE_RUNPATH", [StringArgument], "ensemble/delete_runpath", "Run")
        self.keywordTest("KEEP_RUNPATH", [StringArgument], "ensemble/keep_runpath", "Run")
        self.keywordTest("RUNPATH", [PathArgument], "ensemble/runpath", "Run")
        self.keywordTest("RUNPATH_FILE", [PathArgument], "ensemble/runpath_file", "Run")
        self.keywordTest("FORWARD_MODEL", [StringArgument, StringArgument], "ensemble/forward_model", "Run")
        self.keywordTest("JOB_SCRIPT", [PathArgument], "ensemble/job_script", "Run")


    def test_queue_system_keywords(self):
        self.keywordTest("QUEUE_SYSTEM", [StringArgument], "queue_system/queue_system", "Queue System")
        self.keywordTest("QUEUE_OPTION", [StringArgument, StringArgument, StringArgument], "queue_system/queue_option", "Queue System")
        self.keywordTest("LSF_SERVER", [StringArgument], "queue_system/lsf_server", "Queue System")
        self.keywordTest("LSF_QUEUE", [StringArgument], "queue_system/lsf_queue", "Queue System")
        self.keywordTest("MAX_RUNNING_LSF", [IntegerArgument], "queue_system/max_running_lsf", "Queue System")
        self.keywordTest("QUEUE_TORQUE", [StringArgument], "queue_system/queue_torque", "Queue System")
        self.keywordTest("MAX_RUNNING_LOCAL", [IntegerArgument], "queue_system/max_running_local", "Queue System")
        self.keywordTest("RSH_HOST", [StringArgument, StringArgument], "queue_system/rsh_host", "Queue System")
        self.keywordTest("RSH_COMMAND", [PathArgument], "queue_system/rsh_command", "Queue System")
        self.keywordTest("MAX_RUNNING_RSH", [IntegerArgument], "queue_system/max_running_rsh", "Queue System")




    def test_control_simulations_keywords(self):
        self.keywordTest("MAX_RUNTIME", [IntegerArgument], "control_simulations/max_runtime", "Simulation Control")
        self.keywordTest("MIN_REALIZATIONS", [IntegerArgument], "control_simulations/min_realizations", "Simulation Control")
        self.keywordTest("STOP_LONG_RUNNING", [BoolArgument], "control_simulations/stop_long_running", "Simulation Control")


    def test_parametrization_keywords(self):
        self.keywordTest("FIELD", [StringArgument,StringArgument,StringArgument,StringArgument], "parametrization/field", "Parametrization")
        self.keywordTest("GEN_DATA", [StringArgument,StringArgument,StringArgument,StringArgument], "parametrization/gen_data", "Parametrization")
        self.keywordTest("GEN_KW", [StringArgument,StringArgument,StringArgument,StringArgument], "parametrization/gen_kw", "Parametrization")
        self.keywordTest("GEN_KW", [StringArgument,StringArgument,StringArgument,StringArgument], "parametrization/gen_kw", "Parametrization")
        self.keywordTest("SUMMARY", [StringArgument,StringArgument], "parametrization/summary", "Parametrization")

    def test_enkf_control_keywords(self):
        self.keywordTest("ENKF_ALPHA", [FloatArgument], "enkf_control/enkf_alpha", "Enkf Control")
        self.keywordTest("ENKF_BOOTSTRAP", [BoolArgument], "enkf_control/enkf_bootstrap", "Enkf Control")
        self.keywordTest("ENKF_CV_FOLDS", [IntegerArgument], "enkf_control/enkf_cv_folds", "Enkf Control")
        self.keywordTest("ENKF_FORCE_NCOMP", [BoolArgument], "enkf_control/enkf_force_ncomp", "Enkf Control")
        self.keywordTest("ENKF_LOCAL_CV", [BoolArgument], "enkf_control/enkf_local_cv", "Enkf Control")
        self.keywordTest("ENKF_PEN_PRESS", [BoolArgument], "enkf_control/enkf_pen_press", "Enkf Control")
        self.keywordTest("ENKF_MODE", [StringArgument], "enkf_control/enkf_mode", "Enkf Control")
        self.keywordTest("ENKF_MERGE_OBSERVATIONS", [BoolArgument], "enkf_control/enkf_merge_observations", "Enkf Control")
        self.keywordTest("ENKF_NCOMP", [IntegerArgument], "enkf_control/enkf_ncomp", "Enkf Control")
        self.keywordTest("ENKF_RERUN", [BoolArgument], "enkf_control/enkf_rerun", "Enkf Control")
        self.keywordTest("ENKF_SCALING", [BoolArgument], "enkf_control/enkf_scaling", "Enkf Control")
        self.keywordTest("ENKF_TRUNCATION", [FloatArgument], "enkf_control/enkf_truncation", "Enkf Control")
        self.keywordTest("UPDATE_LOG_PATH", [PathArgument], "enkf_control/update_log_path", "Enkf Control")
        self.keywordTest("SCHEDULE_PREDICTION_FILE", [PathArgument], "enkf_control/schedule_prediction_file", "Enkf Control")


        # Keywords ADD_FIXED_LENGTH_SCHEDULE_KW and ADD_STATIC_KW will NOT be implemented


    def test_analysis_module_keywords(self):
        self.keywordTest("ANALYSIS_LOAD", [StringArgument,StringArgument], "analysis_module/analysis_load", "Analysis Module")
        self.keywordTest("ANALYSIS_SELECT", [StringArgument], "analysis_module/analysis_select", "Analysis Module")
        self.keywordTest("ANALYSIS_COPY", [StringArgument, StringArgument], "analysis_module/analysis_copy", "Analysis Module")


    def test_plot_keywords(self):
        self.keywordTest("IMAGE_VIEWER", [PathArgument], "plot/image_viewer", "Plot")
        self.keywordTest("IMAGE_TYPE", [StringArgument], "plot/image_type", "Plot")
        self.keywordTest("PLOT_DRIVER", [StringArgument], "plot/plot_driver", "Plot")
        self.keywordTest("PLOT_ERRORBAR", [BoolArgument], "plot/plot_errorbar", "Plot")
        self.keywordTest("PLOT_ERRORBAR_MAX", [IntegerArgument], "plot/plot_errorbar_max", "Plot")
        self.keywordTest("PLOT_WIDTH", [IntegerArgument], "plot/plot_width", "Plot")
        self.keywordTest("PLOT_HEIGHT", [IntegerArgument], "plot/plot_height", "Plot")
        self.keywordTest("PLOT_REFCASE", [BoolArgument], "plot/plot_refcase", "Plot")
        self.keywordTest("REFCASE_LIST", [PathArgument, StringArgument], "plot/refcase_list", "Plot")
        self.keywordTest("PLOT_PATH", [PathArgument], "plot/plot_path", "Plot")
        self.keywordTest("RFT_CONFIG", [PathArgument], "plot/rft_config", "Plot")
        self.keywordTest("RFT_PATH", [PathArgument], "plot/rft_path", "Plot")


    def test_workflow_keywords(self):
        self.keywordTest("LOAD_WORKFLOW_JOB", [StringArgument], "workflow_jobs/load_workflow_job", "Workflow Jobs")
        self.keywordTest("WORKFLOW_JOB_DIRECTORY", [PathArgument], "workflow_jobs/workflow_job_directory", "Workflow Jobs")
        self.keywordTest("LOAD_WORKFLOW", [PathArgument], "workflow_jobs/load_workflow", "Workflow Jobs")

    def test_report_keywords(self):
        self.keywordTest("REPORT_CONTEXT", [StringArgument, StringArgument], "report/report_context", "Report")
        self.keywordTest("REPORT_SEARCH_PATH", [PathArgument], "report/report_search_path", "Report")




