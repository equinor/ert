#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  This file is part of ERT - Ensemble based Reservoir Tool.
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
import os.path

from ecl.util.test import TestArea

from cwrap import BaseCClass
from res import ResPrototype
from res.enkf import EnKFMain, ResConfig


class ErtTest(BaseCClass):
    TYPE_NAME = "ert_test"

    _alloc = ResPrototype(
        "void* ert_test_context_alloc_python( test_area , res_config)", bind=False
    )
    _free = ResPrototype("void  ert_test_context_free( ert_test )")
    _get_cwd = ResPrototype("char* ert_test_context_get_cwd( ert_test )")
    _get_enkf_main = ResPrototype("enkf_main_ref ert_test_context_get_main( ert_test )")

    def __init__(
        self, test_name, model_config=None, config_dict=None, store_area=False
    ):
        if model_config is None and config_dict is None:
            raise ValueError("Must supply either model_config or config_dict argument")

        work_area = TestArea(test_name, store_area=store_area)
        work_area.convertToCReference(self)

        if model_config:
            work_area.copy_parent_content(model_config)
            res_config = ResConfig(user_config_file=os.path.basename(model_config))
        else:
            work_area.copy_directory_content(work_area.get_original_cwd())
            res_config = ResConfig(config=config_dict)

        res_config.convertToCReference(self)
        c_ptr = self._alloc(work_area, res_config)
        super(ErtTest, self).__init__(c_ptr)

        self.__ert = None

    def getErt(self):
        """@rtype: EnKFMain"""
        if self.__ert is None:
            self.__ert = self._get_enkf_main()

        return self.__ert

    def free(self):
        ert = self.getErt()
        ert.umount()
        self._free()

    def installWorkflowJob(self, job_name, job_path):
        """@rtype: bool"""
        if os.path.exists(job_path) and os.path.isfile(job_path):
            ert = self.getErt()
            workflow_list = ert.getWorkflowList()

            workflow_list.addJob(job_name, job_path)
        else:
            raise IOError("Could not load workflowjob from:%s" % job_path)

    def runWorkflowJob(self, job_name, *arguments):
        """@rtype: bool"""
        ert = self.getErt()
        workflow_list = ert.getWorkflowList()

        if workflow_list.hasJob(job_name):
            job = workflow_list.getJob(job_name)
            job.run(ert, [arg for arg in arguments])
            return True
        else:
            return False

    def getCwd(self):
        """
        Returns the current working directory of this context.
        @rtype: string
        """
        return self._get_cwd()


class ErtTestContext(object):
    def __init__(
        self, test_name, model_config=None, config_dict=None, store_area=False
    ):
        self.__test_name = test_name
        self.__model_config = model_config
        self.__store_area = store_area
        self.__config_dict = config_dict
        self.__test_context = ErtTest(
            self.__test_name,
            model_config=self.__model_config,
            config_dict=config_dict,
            store_area=self.__store_area,
        )

    def __enter__(self):
        """@rtype: ErtTest"""
        return self.__test_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.__test_context
        return False

    def getErt(self):
        return self.__test_context.getErt()

    def getCwd(self):
        """
        Returns the current working directory of this context.
        @rtype: string
        """
        return self.__test_context.getCwd()
