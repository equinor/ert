#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'res_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from os.path import isfile

from cwrap import BaseCClass

from ecl.util.util import StringList
from res import ResPrototype
from res.config import (ConfigParser, ConfigContent, ConfigSettings,
                        UnrecognizedEnum)

from res.enkf import (SiteConfig, AnalysisConfig, SubstConfig, ModelConfig, EclConfig, PlotSettings,
                      EnsembleConfig, RNGConfig, ConfigKeys, ErtWorkflowList, HookManager, ErtTemplates, LogConfig)

class ResConfig(BaseCClass):
    TYPE_NAME = "res_config"

    _free       = ResPrototype("void res_config_free(res_config)")
    _alloc_full = ResPrototype("void* res_config_alloc_full("
                               "char*, " 
                               "char*, " 
                               "subst_config, " 
                               "site_config, " 
                               "rng_config, " 
                               "analysis_config, " 
                               "ert_workflow_list, " 
                               "hook_manager, " 
                               "ert_templates, " 
                               "plot_settings, " 
                               "ecl_config, " 
                               "ens_config, " 
                               "model_config, " 
                               "log_config, " 
                               "config_content)"
                               , bind=False)

    _alloc_config_content = ResPrototype("config_content_ref res_config_alloc_user_content(char*, config_parser)", bind=False)
    _user_config_file  = ResPrototype("char* res_config_get_user_config_file(res_config)")
    _config_path       = ResPrototype("char* res_config_get_config_directory(res_config)")
    _site_config       = ResPrototype("site_config_ref res_config_get_site_config(res_config)")
    _analysis_config   = ResPrototype("analysis_config_ref res_config_get_analysis_config(res_config)")
    _subst_config      = ResPrototype("subst_config_ref res_config_get_subst_config(res_config)")
    _model_config      = ResPrototype("model_config_ref res_config_get_model_config(res_config)")
    _ecl_config        = ResPrototype("ecl_config_ref res_config_get_ecl_config(res_config)")
    _ensemble_config   = ResPrototype("ens_config_ref res_config_get_ensemble_config(res_config)")
    _plot_config       = ResPrototype("plot_settings_ref res_config_get_plot_config(res_config)")
    _hook_manager      = ResPrototype("hook_manager_ref res_config_get_hook_manager(res_config)")
    _ert_workflow_list = ResPrototype("ert_workflow_list_ref res_config_get_workflow_list(res_config)")
    _rng_config        = ResPrototype("rng_config_ref res_config_get_rng_config(res_config)")
    _ert_templates     = ResPrototype("ert_templates_ref res_config_get_templates(res_config)")
    _log_config        = ResPrototype("log_config_ref res_config_get_log_config(res_config)")
    _init_parser       = ResPrototype("void res_config_init_config_parser(config_parser)", bind=False)


    def __init__(self, user_config_file=None, config=None, throw_on_error=True):

        self._errors, self._failed_keys = None, None
        self._assert_input(user_config_file, config, throw_on_error)

        if config is not None:
            config_content = self._build_config_content(config)
        elif user_config_file is not None:

            parser = ConfigParser()

            config_content = self._alloc_config_content(user_config_file, parser)
        else:
            raise ValueError("No config provided")



        config_dir = config_content.getValue(ConfigKeys.CONFIG_DIRECTORY)

        subst_config = SubstConfig(config_content=config_content)
        site_config = SiteConfig(config_content=config_content)
        rng_config = RNGConfig(config_content=config_content)
        analysis_config = AnalysisConfig(config_content=config_content)
        plot_config = PlotSettings(config_content=config_content)
        ecl_config = EclConfig(config_content=config_content)
        log_config = LogConfig(config_content=config_content)

        ert_workflow_list = ErtWorkflowList(ert_workflow_list=subst_config.subst_list,
                                            config_content=config_content)

        hook_manager = HookManager(workflow_list=ert_workflow_list,
                                   config_content=config_content)

        ert_templates = ErtTemplates(parent_subst=subst_config.subst_list,
                                     config_content=config_content)

        ensemble_config = EnsembleConfig(config_content=config_content,
                                         grid=ecl_config.getGrid(),
                                         refcase=ecl_config.getRefcase())

        model_config = ModelConfig(config_content=config_content,
                                   data_root=config_dir,
                                   joblist=site_config.get_installed_jobs(),
                                   last_history_restart=ecl_config.getLastHistoryRestart(),
                                   sched_file=ecl_config._get_sched_file(),
                                   refcase=ecl_config.getRefcase())

        configs = [
            subst_config,
            site_config,
            rng_config,
            analysis_config,
            ert_workflow_list,
            hook_manager,
            ert_templates,
            plot_config,
            ecl_config,
            ensemble_config,
            model_config,
            log_config,
            config_content]

        c_ptr = None

        if not self.errors or not throw_on_error:
            for conf in configs:
                conf.convertToCReference(None)
            c_ptr = self._alloc_full(config_dir, user_config_file, *configs)


        if c_ptr:
            super(ResConfig, self).__init__(c_ptr)
        else:
            raise ValueError(
                    'Failed to construct ResConfig instance from %r.'
                    % (user_config_file if user_config_file else config)
                    )

    def _assert_input(self, user_config_file, config, throw_on_error):
        if config and not isinstance(config, dict):
            raise ValueError("Expected config to be a dictionary, was %r"
                             % type(config))

        if user_config_file and not isinstance(user_config_file, str):
            raise ValueError("Expected user_config_file to be a string.")

        if user_config_file is not None and config is not None:
            raise ValueError("Expected either user_config_file " +
                             "or config to be provided, got both!")

        if user_config_file is not None and not isfile(user_config_file):
            raise IOError('No such configuration file "%s".' % user_config_file)

        if user_config_file is not None and not throw_on_error:
            raise NotImplementedError("Disabling exceptions on errors is not "
                                      "available when loading from file.")


    def _extract_defines(self, config):
        defines = {}
        if ConfigKeys.DEFINES in config:
            for key in config[ConfigKeys.DEFINES]:
                defines[key] = str(config[ConfigKeys.DEFINES][key])

        return defines


    def _parse_value(self, value):
        if isinstance(value, str):
            return value
        elif isinstance(value, list):
            return [str(elem) for elem in value]
        else:
            return str(value)


    def _assert_keys(self, mother_key, exp_keys, keys):
        if set(exp_keys) != set(keys):
            err_msg = "Did expect the keys %r in %s, received %r."
            raise ValueError(err_msg % (exp_keys, mother_key, keys))


    def _extract_internals(self, config):
        internal_config = []
        config_dir = os.getcwd()

        if ConfigKeys.INTERNALS in config:
            intercon = config[ConfigKeys.INTERNALS]

            dir_key = ConfigKeys.CONFIG_DIRECTORY
            if dir_key in intercon:
                config_dir = os.path.realpath(intercon[dir_key])

            internal_filter = [dir_key]
            for key, value in intercon.items():
                if key not in internal_filter:
                    internal_config.append((key, self._parse_value(value)))

        internal_config.append((ConfigKeys.CONFIG_DIRECTORY, config_dir))
        return config_dir, internal_config


    def _extract_queue_system(self, config):
        if ConfigKeys.QUEUE_SYSTEM not in config:
            return []

        qc = config[ConfigKeys.QUEUE_SYSTEM]
        queue_config = []
        if ConfigKeys.QUEUE_OPTION in qc:
            for qo in qc[ConfigKeys.QUEUE_OPTION]:
                queue_options = [ConfigKeys.DRIVER_NAME,
                                 ConfigKeys.OPTION,
                                 ConfigKeys.VALUE]

                self._assert_keys(ConfigKeys.QUEUE_OPTION, queue_options, qo.keys())

                value = [str(qo[item]) for item in queue_options]
                queue_config.append((ConfigKeys.QUEUE_OPTION, value))

        queue_system_filter = [ConfigKeys.QUEUE_OPTION]
        for key, value in qc.items():
            if not key in queue_system_filter:
                queue_config.append((key, self._parse_value(value)))

        return queue_config


    def _extract_plot_settings(self, config):
        if ConfigKeys.PLOT_SETTINGS not in config:
            return []

        pc = config[ConfigKeys.PLOT_SETTINGS]
        plot_config = []
        for key in pc:
            plot_config.append((ConfigKeys.PLOT_SETTINGS, [key, pc[key]]))

        return plot_config


    def _extract_install_job(self, config):
        if ConfigKeys.INSTALL_JOB not in config:
            return []

        ic = config[ConfigKeys.INSTALL_JOB]
        job_config = []
        for job in ic:
            job_options = [ConfigKeys.NAME, ConfigKeys.PATH]

            self._assert_keys(ConfigKeys.INSTALL_JOB, job_options, job.keys())
            value = [str(job[item]) for item in job_options]
            job_config.append((ConfigKeys.INSTALL_JOB, value))

        return job_config


    def _extract_simulation_job(self, config):
        if ConfigKeys.SIMULATION_JOB not in config:
            return []

        ic = config[ConfigKeys.SIMULATION_JOB]
        simulation_job = []
        for job in ic:
            arglist = [ job[ConfigKeys.NAME] ]
            if ConfigKeys.ARGLIST in job:
                for arg in job[ConfigKeys.ARGLIST]:
                    arglist.append(str(arg))
            simulation_job.append((ConfigKeys.SIMULATION_JOB, arglist))

        return simulation_job


    def _extract_forward_model(self, config):
        if ConfigKeys.FORWARD_MODEL not in config:
            return []

        ic = config[ConfigKeys.FORWARD_MODEL]
        forward_model_job = []
        for job in ic:
            forward_model_job.append((ConfigKeys.FORWARD_MODEL, job))

        return forward_model_job


    def _extract_logging(self, config):
        if ConfigKeys.LOGGING not in config:
            return []

        logging_config = []
        for key, value in config[ConfigKeys.LOGGING].items():
            logging_config.append((key, self._parse_value(value)))

        return logging_config


    def _extract_seed(self, config):
        if ConfigKeys.SEED not in config:
            return []

        seed_config = []
        for key, value in config[ConfigKeys.SEED].items():
            seed_config.append((key, self._parse_value(value)))

        return seed_config


    def _extract_run_templates(self, config):
        if ConfigKeys.RUN_TEMPLATE not in config:
            return []

        template_config = []
        for rt in config[ConfigKeys.RUN_TEMPLATE]:
            rt_options = [ConfigKeys.TEMPLATE, ConfigKeys.EXPORT]

            self._assert_keys(ConfigKeys.RUN_TEMPLATE, rt_options, rt.keys())

            value = [rt[option] for option in rt_options]
            template_config.append((ConfigKeys.RUN_TEMPLATE, value))

        return template_config

    def _extract_custom_kw(self, config):
        if ConfigKeys.CUSTOM_KW not in config:
            return []

        custom_kw_config = []
        for ck in config[ConfigKeys.CUSTOM_KW]:
            custom_kw_options = [ConfigKeys.NAME,
                                 ConfigKeys.RESULT_FILE]

            self._assert_keys(ConfigKeys.CUSTOM_KW, custom_kw_options, ck.keys())

            value = [ck[item] for item in custom_kw_options]
            custom_kw_config.append((ConfigKeys.CUSTOM_KW, value))

        return custom_kw_config

    def _extract_gen_kw(self, config):
        if ConfigKeys.GEN_KW not in config:
            return []

        gen_kw_config = []
        for gk in config[ConfigKeys.GEN_KW]:
            gen_kw_options = [ConfigKeys.NAME,
                              ConfigKeys.TEMPLATE,
                              ConfigKeys.OUT_FILE,
                              ConfigKeys.PARAMETER_FILE]

            self._assert_keys(ConfigKeys.GEN_KW, gen_kw_options, gk.keys())

            value = [gk[item] for item in gen_kw_options]
            gen_kw_config.append((ConfigKeys.GEN_KW, value))

        return gen_kw_config


    def _extract_gen_data(self, config):
        if ConfigKeys.GEN_DATA not in config:
            return []

        gen_data_config = []
        for gd in config[ConfigKeys.GEN_DATA]:
            req_keys = [ConfigKeys.NAME,
                        ConfigKeys.RESULT_FILE,
                        ConfigKeys.REPORT_STEPS]

            default_opt = {ConfigKeys.INPUT_FORMAT : "ASCII"}

            if not sorted(req_keys) == sorted(gd.keys()):
                err_msg = "Expected keys %r when creating GEN_DATA, received %r"
                raise KeyError(err_msg % (req_keys, gd))

            value = [gd[ConfigKeys.NAME]]
            value += ["%s:%s" % (key, gd[key]) for key in req_keys[1:]]
            value += ["%s:%s" % (key, val) for key, val in default_opt.items()]
            gen_data_config.append((ConfigKeys.GEN_DATA, value))

        return gen_data_config


    def _extract_simulation(self, config):
        if ConfigKeys.SIMULATION not in config:
            return []

        simulation_config = []
        sc = config[ConfigKeys.SIMULATION]
        sim_filter = []

        # Extract queue system
        sim_filter.append(ConfigKeys.QUEUE_SYSTEM)
        simulation_config += self._extract_queue_system(sc)

        # Extract plot settings
        sim_filter.append(ConfigKeys.PLOT_SETTINGS)
        simulation_config += self._extract_plot_settings(sc)

        # Extract install job
        sim_filter.append(ConfigKeys.INSTALL_JOB)
        simulation_config += self._extract_install_job(sc)

        # Extract forward_model
        sim_filter.append(ConfigKeys.FORWARD_MODEL)
        simulation_config += self._extract_forward_model(sc)

        # Extract simulation_job
        sim_filter.append(ConfigKeys.SIMULATION_JOB)
        simulation_config += self._extract_simulation_job(sc)

        # Extract logging
        sim_filter.append(ConfigKeys.LOGGING)
        simulation_config += self._extract_logging(sc)

        # Extract seed
        sim_filter.append(ConfigKeys.SEED)
        simulation_config += self._extract_seed(sc)

        # Extract run templates
        sim_filter.append(ConfigKeys.RUN_TEMPLATE)
        simulation_config += self._extract_run_templates(sc)

        # Extract CUSTOM_KW
        sim_filter.append(ConfigKeys.CUSTOM_KW)
        simulation_config += self._extract_custom_kw(sc)

        # Extract GEN_KW
        sim_filter.append(ConfigKeys.GEN_KW)
        simulation_config += self._extract_gen_kw(sc)

        # Extract GEN_DATA
        sim_filter.append(ConfigKeys.GEN_DATA)
        simulation_config += self._extract_gen_data(sc)

        # Others
        for key, value in sc.items():
            if not key in sim_filter:
                simulation_config.append((key, self._parse_value(value)))

        return simulation_config


    def _extract_config(self, config):
        defines = self._extract_defines(config)
        key_filter = [ConfigKeys.DEFINES]

        new_config = []

        # Extract internals
        key_filter.append(ConfigKeys.INTERNALS)
        config_dir, internal_config = self._extract_internals(config)
        new_config += internal_config

        # Extract simulation
        key_filter.append(ConfigKeys.SIMULATION)
        new_config += self._extract_simulation(config)

        # Unrecognized keys
        for key, value in config.items():
            if key not in key_filter:
                self._failed_keys[key] = value

        return defines, config_dir, new_config


    def _build_config_content(self, config):
        self._failed_keys = {}
        defines, config_dir, config_list = self._extract_config(config)

        config_parser  = ResConfig.config_parser()
        config_content = ConfigContent(None)
        config_content.setParser(config_parser)

        # Insert defines
        for key in defines:
            config_content.add_define(key, defines[key])

        # Insert key values
        if not os.path.exists( config_dir ):
            raise IOError("The configuration direcetory: %s does not exist" % config_dir)

        path_elm = config_content.create_path_elm(config_dir)
        add_key_value = lambda key, value : config_parser.add_key_value(
                                                            config_content,
                                                            key,
                                                            StringList([key] + value),
                                                            path_elm=path_elm)

        for key, value in config_list:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                raise ValueError("Expected value to be str or list, was %r" % (type(value)))

            ok = add_key_value(key, value)
            if not ok:
                self._failed_keys[key] = value

        config_parser.validate(config_content)
        self._errors = list(config_content.getErrors())


        return config_content

    def free(self):
        self._free()


    @classmethod
    def config_parser(cls):
        parser = ConfigParser( )
        cls._init_parser(parser)
        return parser

    @property
    def errors(self):
        return self._errors

    @property
    def failed_keys(self):
        return self._failed_keys

    @property
    def user_config_file(self):
        return self._user_config_file()

    @property
    def site_config_file(self):
        return self.site_config.config_file

    @property
    def site_config(self):
        return self._site_config()

    @property
    def analysis_config(self):
        return self._analysis_config()

    @property
    def config_path(self):
        return self._config_path( )

    @property
    def subst_config(self):
        return self._subst_config().setParent(self)

    @property
    def model_config(self):
        return self._model_config()

    @property
    def ecl_config(self):
        return self._ecl_config()

    @property
    def ensemble_config(self):
        return self._ensemble_config()

    @property
    def plot_config(self):
        return self._plot_config()

    @property
    def hook_manager(self):
        return self._hook_manager()

    @property
    def ert_workflow_list(self):
        return self._ert_workflow_list()

    @property
    def rng_config(self):
        return self._rng_config()

    @property
    def ert_templates(self):
        return self._ert_templates()

    @property
    def log_config(self):
        return self._log_config()
