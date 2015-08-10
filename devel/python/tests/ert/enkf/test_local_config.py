import os.path
from ert.test import ExtendedTestCase
from ert.test import ErtTestContext

class LocalConfigTest(ExtendedTestCase):
    
    def setUp(self):
               
        self.config = self.createTestPath("local/custom_kw/mini_config")


    def test_local_config( self ):
        
                
        with ErtTestContext("python/enkf/data/local_config", self.config) as test_context:

            
            main = test_context.getErt()
            self.assertTrue(main, "Load failed")           
            local_config = main.local_config()                             
            
            # Update step
            updatestep = local_config.createUpdatestep("UPDATESTEP")
            
            # Ministep                                      
            ministep = local_config.createMinistep("MINISTEP")
            
            
            # Data
            data_scale = local_config.createDataset("DATA_SCALE")
            data_scale.addNodeWithIndex("SCALE", 0)
            
            data_offset = local_config.createDataset("DATA_OFFSET")
            data_offset.addNodeWithIndex("OFFSET", 1)
                    
            # ObsData
            local_obs_data_1 = local_config.createObsdata("OBSSET_1")                                            
            local_obs_data_1.addNodeAndRange("GENPERLIN_1", 0, 1)
            local_obs_data_1.addNodeAndRange("GENPERLIN_2", 0, 1)
             
            local_obs_data_2 = local_config.createObsdata("OBSSET_2")               
            local_obs_data_1.addNodeAndRange("GENPERLIN_3", 0, 1)
            
            # Attach         
            ministep.attachDataset(data_scale)
            ministep.attachDataset(data_offset)                
            ministep.attachObsset(local_obs_data_1)                                   
            ministep.attachObsset(local_obs_data_2)
                                  
            updatestep.attachMinistep(ministep)
            local_config.installUpdatestep(updatestep, 0, 1) 
            
            sfile = self.createTestPath("local/custom_kw/test_local_config.txt")                        
            local_config.writeLocalConfigFile(sfile)            
            self.assertTrue( os.path.exists( sfile ))                     
 
