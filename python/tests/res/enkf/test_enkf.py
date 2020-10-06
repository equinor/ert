#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'test_enkf.py' is part of ERT - Ensemble based Reservoir Tool.
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

import sys, os
import os.path
import pytest
from tests import ResTest

from ecl.util.util import BoolVector
from res.enkf import (
    EnsembleConfig,
    AnalysisConfig,
    ModelConfig,
    SiteConfig,
    EclConfig,
    EnkfObs,
    ErtTemplates,
    EnkfFs,
    EnKFState,
    EnkfVarType,
    ObsVector,
    RunArg,
    ResConfig,
)
from res.enkf.config import EnkfConfigNode
from res.enkf.enkf_main import EnKFMain
from res.enkf.enums import (
    EnkfObservationImplementationType,
    LoadFailTypeEnum,
    EnkfInitModeEnum,
    ErtImplType,
    RealizationStateEnum,
    EnkfRunType,
    EnkfFieldFileFormatEnum,
    EnkfTruncationType,
    ActiveMode,
)

from ecl.util.test import TestAreaContext
from res.enkf.observations.summary_observation import SummaryObservation

from tests.utils import tmpdir


class EnKFTest(ResTest):
    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")
        self.case_directory_snake_oil = self.createTestPath("local/snake_oil/")

    @tmpdir()
    def test_repr(self):
        with TestAreaContext("enkf_test", store_area=True) as work_area:
            work_area.copy_directory(self.case_directory)
            res_config = ResConfig("simple_config/minimum_config")
            main = EnKFMain(res_config)
            pfx = "EnKFMain(ensemble_size"
            self.assertEqual(pfx, repr(main)[: len(pfx)])

    @tmpdir()
    def test_bootstrap(self):
        with TestAreaContext("enkf_test", store_area=True) as work_area:
            work_area.copy_directory(self.case_directory)
            res_config = ResConfig("simple_config/minimum_config")
            main = EnKFMain(res_config)
            self.assertTrue(main, "Load failed")

    @tmpdir()
    def test_site_condif(self):
        with TestAreaContext("enkf_test", store_area=True) as work_area:
            work_area.copy_directory(self.case_directory)
            res_config = ResConfig("simple_config/minimum_config")
            main = EnKFMain(res_config)

            self.assertTrue(main, "Load failed")

            self.assertEqual(
                res_config.site_config_file, main.resConfig().site_config_file
            )

            self.assertEqual(
                res_config.user_config_file, main.resConfig().user_config_file
            )

    @tmpdir()
    def test_site_bootstrap(self):
        with TestAreaContext("enkf_test", store_area=True) as work_area:
            with self.assertRaises(ValueError):
                EnKFMain(None)

    @tmpdir()
    def test_default_res_config(self):
        with TestAreaContext("enkf_test", store_area=True) as work_area:
            work_area.copy_directory(self.case_directory)
            res_config = ResConfig("simple_config/minimum_config")
            main = EnKFMain(res_config)

            self.assertIsNotNone(main.resConfig)
            self.assertIsNotNone(main.siteConfig)
            self.assertIsNotNone(main.analysisConfig)

    @tmpdir()
    def test_invalid_res_config(self):
        with TestAreaContext("enkf_test") as work_area:
            with self.assertRaises(TypeError):
                work_area.copy_directory(self.case_directory)
                main = EnKFMain(res_config="This is not a ResConfig instance")

    @tmpdir()
    def test_enum(self):
        self.assertEnumIsFullyDefined(
            EnkfVarType, "enkf_var_type", "lib/include/ert/enkf/enkf_types.hpp"
        )
        self.assertEnumIsFullyDefined(
            ErtImplType, "ert_impl_type", "lib/include/ert/enkf/enkf_types.hpp"
        )
        self.assertEnumIsFullyDefined(
            EnkfInitModeEnum, "init_mode_type", "lib/include/ert/enkf/enkf_types.hpp"
        )
        self.assertEnumIsFullyDefined(
            RealizationStateEnum,
            "realisation_state_enum",
            "lib/include/ert/enkf/enkf_types.hpp",
        )
        self.assertEnumIsFullyDefined(
            EnkfTruncationType, "truncation_type", "lib/include/ert/enkf/enkf_types.hpp"
        )
        self.assertEnumIsFullyDefined(
            EnkfRunType, "run_mode_type", "lib/include/ert/enkf/enkf_types.hpp"
        )

        self.assertEnumIsFullyDefined(
            EnkfObservationImplementationType,
            "obs_impl_type",
            "lib/include/ert/enkf/obs_vector.hpp",
        )
        self.assertEnumIsFullyDefined(
            LoadFailTypeEnum,
            "load_fail_type",
            "lib/include/ert/enkf/summary_config.hpp",
        )
        self.assertEnumIsFullyDefined(
            EnkfFieldFileFormatEnum,
            "field_file_format_type",
            "lib/include/ert/enkf/field_config.hpp",
        )
        self.assertEnumIsFullyDefined(
            ActiveMode, "active_mode_type", "lib/include/ert/enkf/enkf_types.hpp"
        )

    @tmpdir()
    def test_observations(self):
        with TestAreaContext("enkf_test") as work_area:
            work_area.copy_directory(self.case_directory)

            res_config = ResConfig("simple_config/minimum_config")
            main = EnKFMain(res_config)

            count = 10
            summary_key = "test_key"
            observation_key = "test_obs_key"
            summary_observation_node = EnkfConfigNode.createSummaryConfigNode(
                summary_key, LoadFailTypeEnum.LOAD_FAIL_EXIT
            )
            observation_vector = ObsVector(
                EnkfObservationImplementationType.SUMMARY_OBS,
                observation_key,
                summary_observation_node,
                count,
            )

            main.getObservations().addObservationVector(observation_vector)

            values = []
            for index in range(0, count):
                value = index * 10.5
                std = index / 10.0
                summary_observation_node = SummaryObservation(
                    summary_key, observation_key, value, std
                )
                observation_vector.installNode(index, summary_observation_node)
                self.assertEqual(
                    observation_vector.getNode(index), summary_observation_node
                )
                self.assertEqual(value, summary_observation_node.getValue())
                values.append((index, value, std))

            observations = main.getObservations()
            test_vector = observations[observation_key]
            index = 0
            for node in test_vector:
                self.assertTrue(isinstance(node, SummaryObservation))
                self.assertEqual(node.getValue(), index * 10.5)
                index += 1

            self.assertEqual(observation_vector, test_vector)
            for index, value, std in values:
                self.assertTrue(test_vector.isActive(index))

                summary_observation_node = test_vector.getNode(index)
                """@type: SummaryObservation"""

                self.assertEqual(value, summary_observation_node.getValue())
                self.assertEqual(std, summary_observation_node.getStandardDeviation())
                self.assertEqual(summary_key, summary_observation_node.getSummaryKey())

    @tmpdir()
    def test_config(self):
        with TestAreaContext("enkf_test") as work_area:
            work_area.copy_directory(self.case_directory)

            res_config = ResConfig("simple_config/minimum_config")
            main = EnKFMain(res_config)

            self.assertIsInstance(main.ensembleConfig(), EnsembleConfig)
            self.assertIsInstance(main.analysisConfig(), AnalysisConfig)
            self.assertIsInstance(main.getModelConfig(), ModelConfig)
            self.assertIsInstance(main.siteConfig(), SiteConfig)
            self.assertIsInstance(main.eclConfig(), EclConfig)

            self.assertIsInstance(main.getObservations(), EnkfObs)
            self.assertIsInstance(main.get_templates(), ErtTemplates)
            self.assertIsInstance(
                main.getEnkfFsManager().getCurrentFileSystem(), EnkfFs
            )
            self.assertIsInstance(main.getMemberRunningState(0), EnKFState)

            self.assertEqual("simple_config/Ensemble", main.getMountPoint())

    @tmpdir()
    def test_run_context(self):
        with TestAreaContext("enkf_test") as work_area:
            work_area.copy_directory(self.case_directory)
            res_config = ResConfig("simple_config/minimum_config")
            main = EnKFMain(res_config)
            fs_manager = main.getEnkfFsManager()
            fs = fs_manager.getCurrentFileSystem()
            iactive = BoolVector(initial_size=10, default_value=True)
            iactive[0] = False
            iactive[1] = False
            run_context = main.getRunContextENSEMPLE_EXPERIMENT(fs, iactive)

            self.assertEqual(len(run_context), 10)

            with self.assertRaises(IndexError):
                run_context[10]

            with self.assertRaises(TypeError):
                run_context["String"]

            self.assertIsNone(run_context[0])
            run_arg = run_context[2]
            self.assertTrue(isinstance(run_arg, RunArg))

    @tmpdir()
    def test_run_context_from_external_folder(self):
        with TestAreaContext("enkf_test") as work_area:
            work_area.copy_directory(self.case_directory_snake_oil)
            res_config = ResConfig("snake_oil/snake_oil.ert")
            main = EnKFMain(res_config)
            fs_manager = main.getEnkfFsManager()
            fs = fs_manager.getCurrentFileSystem()

            mask = BoolVector(default_value=False, initial_size=10)
            mask[0] = True
            run_context = main.getRunContextENSEMPLE_EXPERIMENT(fs, mask)

            self.assertEqual(len(run_context), 10)

            job_queue = main.get_queue_config().create_job_queue()
            main.getEnkfSimulationRunner().createRunPath(run_context)
            num = main.getEnkfSimulationRunner().runEnsembleExperiment(
                job_queue, run_context
            )
            self.assertTrue(
                os.path.isdir(
                    "snake_oil/storage/snake_oil/runpath/realisation-0/iter-0"
                )
            )
            self.assertEqual(num, 1)
