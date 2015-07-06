import os
import os.path
from ert.test import ExtendedTestCase
from ert.test import TestAreaContext
from ert.enkf.enkf_main import EnKFMain

class LocalConfigTest(ExtendedTestCase):
    
    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")


    def test_local_config( self ):
        with TestAreaContext("enkf_test", store_area=True) as work_area:
            work_area.copy_directory(self.case_directory)
            main = EnKFMain("simple_config/minimum_config")
            self.assertTrue(main, "Load failed")           
            local_config = main.local_config()        
            
        # Update step
        updatestep_ts_000_001 = local_config.createUpdatestep("UPDATESTEP-TS_000-001")
        
        # Ministep                                      
        mini_flt_1_tx_000_001 = local_config.createMinistep("MINI_FLT_1-TS_000-001")
        
        
        # Data
        data_flt_1 = local_config.createDataset("DATA_FLT_1")
        data_flt_1.addNodeWithIndex("FAULT_TRANSLATE_REL", 0)

        data_flt_2 = local_config.createDataset("DATA_FLT_2")
        data_flt_2.addNodeWithIndex("FAULT_TRANSLATE_REL", 1)
                
        # ObsData
        local_obs_data_1 = local_config.createObsdata("OBS_FLT_1-TS_000-001")                                            
        local_obs_data_1.addNodeAndRange("WBHP:PRO-1", 0, 1)
        local_obs_data_1.addNodeAndRange("WGOR:PRO-1", 0, 1)
         
        local_obs_data_2 = local_config.createObsdata("OBS_FLT_2-TS_000-001")   
        local_obs_data_2.addNodeAndRange("WBHP:PRO-4", 0, 1)
        local_obs_data_2.addNodeAndRange("WGOR:PRO-4", 0, 1)
        
        # Attach         
        mini_flt_1_tx_000_001.attachDataset(data_flt_1)
        mini_flt_1_tx_000_001.attachDataset(data_flt_2)                
        mini_flt_1_tx_000_001.attachObsset(local_obs_data_1)                                   
        mini_flt_1_tx_000_001.attachObsset(local_obs_data_2)
                              
        updatestep_ts_000_001.attachMinistep(mini_flt_1_tx_000_001)
        local_config.installUpdatestep(updatestep_ts_000_001, 0, 1) 

        sfile = self.createTestPath("test_local_config.txt")                        
        local_config.writeLocalConfigFile(sfile)            
        self.assertTrue( os.path.exists( sfile ))                     
        main.free()
        
