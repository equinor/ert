import os.path
from ert.test import ExtendedTestCase
from ert.test import ErtTestContext
from ert.enkf.local_ministep import LocalMinistep
from ert.enkf.local_updatestep import LocalUpdateStep
from ert.enkf.active_list import ActiveList
from ert.enkf.local_obsdata import LocalObsdata
from ert.enkf.local_obsdata_node import LocalObsdataNode

class LocalConfigTest(ExtendedTestCase):
    
    def setUp(self):
               
        self.config = self.createTestPath("local/custom_kw/mini_config")


    def testSetUp( self ):      
        with ErtTestContext("python/enkf/data/local_config", self.config) as test_context:            
            main = test_context.getErt()
            self.assertTrue(main, "Load failed")
            
            local_config = main.local_config()  

            sfile = self.createTestPath("local/custom_kw/test_local_config.txt")                        
            local_config.writeLocalConfigFile(sfile)            
            self.assertTrue( os.path.exists( sfile )) 
 
 
    def testUpdateStep( self ):                        
        with ErtTestContext("python/enkf/data/local_config", self.config) as test_context:
            main = test_context.getErt()
            self.assertTrue(main, "Load failed")
            local_config = main.local_config()                             
            
            # Update step
            updatestep = local_config.createUpdatestep("UPDATESTEP")
            self.assertTrue(isinstance(updatestep, LocalUpdateStep))
            
 
    def testMiniStep( self ):                        
        with ErtTestContext("python/enkf/data/local_config", self.config) as test_context:
            main = test_context.getErt()
            self.assertTrue(main, "Load failed")
            local_config = main.local_config()                             
            
            # Ministep                                      
            ministep = local_config.createMinistep("MINISTEP")
            self.assertTrue(isinstance(ministep, LocalMinistep))                        
                 
    def testAttachMinistep( self ):                        
        with ErtTestContext("python/enkf/data/local_config", self.config) as test_context:
            main = test_context.getErt()
            self.assertTrue(main, "Load failed")
            local_config = main.local_config()                             
            
            # Update step
            updatestep = local_config.createUpdatestep("UPDATESTEP")
            self.assertTrue(isinstance(updatestep, LocalUpdateStep))
 
            # Ministep                                      
            ministep = local_config.createMinistep("MINISTEP")
            self.assertTrue(isinstance(ministep, LocalMinistep))   
            
            # Attach
            updatestep.attachMinistep(ministep)

    def testLocalDataset( self ):                        
        with ErtTestContext("python/enkf/data/local_config", self.config) as test_context:
            main = test_context.getErt()
            self.assertTrue(main, "Load failed")
            local_config = main.local_config()                             
                       
            # Data
            data_scale = local_config.createDataset("DATA_SCALE")
            data_scale.addNode("SCALE")
            active_list = data_scale.getActiveList("SCALE")
            self.assertTrue(isinstance(active_list, ActiveList))
            
            active_list.addActiveIndex(0)            
                        
            data_offset = local_config.createDataset("DATA_OFFSET")
            data_offset.addNodeWithIndex("OFFSET", 1)
            
            

    def testLocalObsdata( self ):              
        with ErtTestContext("python/enkf/data/local_config", self.config) as test_context:
            main = test_context.getErt()
            self.assertTrue(main, "Load failed")
            local_config = main.local_config()  
            
            # Creating
            local_obs_data_1 = local_config.createObsdata("OBSSET_1") 
            self.assertTrue(isinstance(local_obs_data_1, LocalObsdata))
                                                       
            local_obs_data_1.addNodeAndRange("GENPERLIN_1", 0, 1)
            local_obs_data_1.addNodeAndRange("GENPERLIN_2", 0, 1)            
            
            self.assertEqual( len(local_obs_data_1) , 2 )
                    
            # Delete node        
            del local_obs_data_1["GENPERLIN_1"]
            self.assertEqual( len(local_obs_data_1) , 1 )  

            # Get node
            node = local_obs_data_1["GENPERLIN_2"]
            self.assertTrue(isinstance(node, LocalObsdataNode))


            
    def testAttachObsData( self ):          
        with ErtTestContext("python/enkf/data/local_config", self.config) as test_context:
            main = test_context.getErt()
            self.assertTrue(main, "Load failed")
            local_config = main.local_config()              

            local_obs_data_1 = local_config.createObsdata("OBSSET_1") 
            self.assertTrue(isinstance(local_obs_data_1, LocalObsdata))
            
            # Obsdata                                           
            local_obs_data_1.addNodeAndRange("GENPERLIN_1", 0, 1)
            local_obs_data_1.addNodeAndRange("GENPERLIN_2", 0, 1)  
            
            # Ministep                                      
            ministep = local_config.createMinistep("MINISTEP")
            self.assertTrue(isinstance(ministep, LocalMinistep))   

            # Attach obsset                                
            ministep.attachObsset(local_obs_data_1)                     
                        
            # Retrieve attached obsset            
            local_obs_data_new = ministep.getLocalObsData()                   
            self.assertEqual( len(local_obs_data_new) , 2 )                    
                                