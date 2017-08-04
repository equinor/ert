from ecl.util import BoolVector, PathFormat
from res.util.substitution_list import SubstitutionList
from ecl.test import ExtendedTestCase, TestAreaContext
from res.enkf import ErtRunContext
from res.enkf.enums import EnkfRunType
from res.enkf.enums import EnKFFSType
from res.enkf import EnkfFs

class ErtRunContextTest(ExtendedTestCase):

    def test_create(self):
        with TestAreaContext("run_context"):
            arg = None
            init_fs = EnkfFs.createFileSystem("init_fs", EnKFFSType.BLOCK_FS_DRIVER_ID, arg)
            result_fs = None
            target_fs = None

            mask = BoolVector( initial_size = 100 , default_value = True )
            runpath_fmt = PathFormat( "path/to/sim%d" )
            subst_list = SubstitutionList( )
            itr = 0
            run_context1 = ErtRunContext( EnkfRunType.ENSEMBLE_EXPERIMENT , init_fs, result_fs, target_fs , mask , runpath_fmt, subst_list , itr )
            run_id1 = run_context1.get_id( )
            run_arg0 = run_context1[0]
            self.assertEqual( run_id1 , run_arg0.get_run_id( ))
            
            run_context2 = ErtRunContext( EnkfRunType.ENSEMBLE_EXPERIMENT , init_fs, result_fs , target_fs, mask , runpath_fmt, subst_list , itr )
            run_id2 = run_context2.get_id( )

            self.assertFalse( run_id1 == run_id2 )
