from ert.test import ErtTestContext
from ert.test import ExtendedTestCase

from ert.util import BoolVector,IntVector
from ert.enkf import ActiveMode, EnsembleConfig
from ert.enkf import ObsVector , LocalObsdata, EnkfObs


class EnKFObsTest(ExtendedTestCase):
    def setUp(self):
        self.config_file = self.createTestPath("Statoil/config/obs_testing/config")

    def testObs(self):
        with ErtTestContext("obs_test", self.config_file) as test_context:
            ert = test_context.getErt()
            obs = ert.getObservations()

            self.assertEqual(31, len(obs))
            for v in obs:
                self.assertTrue(isinstance(v, ObsVector))

            with self.assertRaises(IndexError):
                v = obs[-1]

            with self.assertRaises(IndexError):
                v = obs[40]

            with self.assertRaises(KeyError):
                v = obs["No-this-does-not-exist"]

            v1 = obs["WWCT:OP_3"]
            v2 = obs["GOPT:OP"]
            mask = BoolVector(True, ert.getEnsembleSize())
            current_fs = ert.getEnkfFsManager().getCurrentFileSystem()

            self.assertTrue(v1.hasData(mask, current_fs))
            self.assertFalse(v2.hasData(mask, current_fs))
            
            local_node = v1.createLocalObs( )
            self.assertEqual( len(local_node.getStepList()) , len(v1.getStepList()))
            for t1,t2 in zip( local_node.getStepList() , v1.getStepList()):
                self.assertEqual( t1 , t2 )
                
            

    def test_obs_block_scale_std(self):
        with ErtTestContext("obs_test_scale", self.config_file) as test_context:
            ert = test_context.getErt()
            fs = ert.getEnkfFsManager().getCurrentFileSystem()
            active_list = IntVector( )
            active_list.initRange(0 , ert.getEnsembleSize() , 1 )

            obs = ert.getObservations()
            obs_data = LocalObsdata( "OBSxx" )
            obs_vector = obs["WWCT:OP_1"]
            obs_data.addObsVector( obs_vector )
            scale_factor = obs.scaleCorrelatedStd( fs , obs_data , active_list )

            for obs_node in obs_vector:
                for index in range(len(obs_node)):
                    self.assertEqual( scale_factor , obs_node.getStdScaling( index ))
            
            


    def test_obs_block_all_active_local(self):
        with ErtTestContext("obs_test_all_active", self.config_file) as test_context:
            ert = test_context.getErt()
            obs = ert.getObservations()
            obs_data = obs.getAllActiveLocalObsdata( )

            self.assertEqual( len(obs_data) , len(obs) )
            for obs_vector in obs:
                self.assertTrue( obs_vector.getObservationKey() in obs_data )

                tstep_list1 = obs_vector.getStepList()
                local_node = obs_data[ obs_vector.getObservationKey() ]

                self.assertTrue( tstep_list1 , local_node.getStepList())
                active_list = local_node.getActiveList()
                self.assertEqual( active_list.getMode() , ActiveMode.ALL_ACTIVE )
                    
                

    def test_create(self):
        ensemble_config = EnsembleConfig()
        obs = EnkfObs(ensemble_config)
        self.assertEqual( len(obs) , 0 )
