import unittest
import os
from res.test import ErtTestContext
from tests import ResTest
from tests.utils import tmpdir

from res.enkf import RunpathList, RunpathNode, ErtRunContext
from res.enkf.enums import EnkfInitModeEnum,EnkfRunType
from ecl.util.util import BoolVector
from res.util.substitution_list import SubstitutionList




class RunpathListTestErt(ResTest):

    @tmpdir()
    def test_an_enkf_runpath(self):
        # TODO this test is flaky and we need to figure out why.  See #1370
        # enkf_util_assert_buffer_type: wrong target type in file (expected:104 got:0)
        test_path = self.createTestPath("local/snake_oil_field/snake_oil.ert")
        with ErtTestContext("runpathlist_basic", test_path) as tc:
            pass

    @tmpdir()
    def test_assert_export(self):
        with ErtTestContext("create_runpath_export" , self.createTestPath("local/snake_oil_no_data/snake_oil.ert")) as tc:
            ert = tc.getErt( )
            runpath_list = ert.getRunpathList( )
            self.assertFalse( os.path.isfile( runpath_list.getExportFile( ) ))

            ens_size = ert.getEnsembleSize( )
            runner = ert.getEnkfSimulationRunner( )
            fs_manager = ert.getEnkfFsManager( )

            init_fs = fs_manager.getFileSystem("init_fs")
            mask = BoolVector( initial_size = 25 , default_value = True )
            runpath_fmt = ert.getModelConfig().getRunpathFormat( )
            subst_list = SubstitutionList( )
            itr = 0
            jobname_fmt = ert.getModelConfig().getJobnameFormat()
            run_context1 = ErtRunContext( EnkfRunType.INIT_ONLY , init_fs, None , mask , runpath_fmt, jobname_fmt, subst_list , itr )

            runner.createRunPath( run_context1 )

            self.assertTrue( os.path.isfile( runpath_list.getExportFile( ) ))
            self.assertEqual( "test_runpath_list.txt" , os.path.basename( runpath_list.getExportFile( ) ))

    @tmpdir()
    def test_assert_symlink_deleted(self):
        with ErtTestContext("create_runpath_symlink_deleted" , self.createTestPath("local/snake_oil_field/snake_oil.ert")) as tc:
            ert = tc.getErt( )
            runpath_list = ert.getRunpathList( )

            ens_size = ert.getEnsembleSize()
            runner = ert.getEnkfSimulationRunner()
            mask = BoolVector( initial_size = ens_size , default_value = True )
            fs_manager = ert.getEnkfFsManager()
            init_fs = fs_manager.getFileSystem("init_fs")

            # create directory structure
            runpath_fmt = ert.getModelConfig().getRunpathFormat( )
            subst_list = SubstitutionList( )
            itr = 0
            jobname_fmt = ert.getModelConfig().getJobnameFormat()
            run_context = ErtRunContext( EnkfRunType.INIT_ONLY , init_fs, None , mask , runpath_fmt, jobname_fmt, subst_list , itr )
            runner.createRunPath( run_context )


            # replace field file with symlink
            linkpath = '%s/permx.grdcel' % str(runpath_list[0].runpath)
            targetpath = '%s/permx.grdcel.target' % str(runpath_list[0].runpath)
            open(targetpath, 'a').close()
            os.remove(linkpath)
            os.symlink(targetpath, linkpath)

            # recreate directory structure
            runner.createRunPath( run_context )

            # ensure field symlink is replaced by file
            self.assertFalse( os.path.islink(linkpath) )


