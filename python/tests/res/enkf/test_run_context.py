from ecl.util import BoolVector, PathFormat, SubstitutionList
from ecl.test import ExtendedTestCase, TestAreaContext
from res.enkf import ErtRunContext
from res.enkf.enums import EnkfRunType
from res.enkf.enums import EnKFFSType
from res.enkf import EnkfFs

class ErtRunContextTest(ExtendedTestCase):

    def test_create(self):
        with TestAreaContext("run_context"):
            arg = None
            sim_fs = EnkfFs.createFileSystem("sim_fs", EnKFFSType.BLOCK_FS_DRIVER_ID, arg)
            mask = BoolVector( initial_size = 100 , default_value = True )
            runpath_fmt = PathFormat( "path/to/sim%d" )
            subst_list = SubstitutionList( )
            itr = 0
            run_context = ErtRunContext( EnkfRunType.ENSEMBLE_EXPERIMENT , sim_fs, sim_fs , mask , runpath_fmt, subst_list , itr )
            run_id1 = run_context.get_id( )

            run_context = ErtRunContext( EnkfRunType.ENSEMBLE_EXPERIMENT , sim_fs, sim_fs , mask , runpath_fmt, subst_list , itr )
            run_id2 = run_context.get_id( )

            print run_id1, run_id2
            self.assertFalse( run_id1 == run_id2 )
