from ecl.util import BoolVector
from res.enkf.enums import HookRuntime
from res.enkf import ErtRunContext
from ert_gui.simulation.models import BaseRunModel, ErtRunError, EnsembleExperiment

class SingleTestRun(EnsembleExperiment):
    
    def __init__(self, queue_config):
        super(EnsembleExperiment, self).__init__("Single realization test-run" , queue_config)


        
    def runSimulations(self, arguments):
        self.runSimulations__( arguments  , "Running single realisation test ...")
        

    def create_context(self, arguments):
        fs_manager = self.ert().getEnkfFsManager()
        init_fs = fs_manager.getCurrentFileSystem( )
        result_fs = fs_manager.getCurrentFileSystem( )

        model_config = self.ert().getModelConfig( )
        runpath_fmt = model_config.getRunpathFormat( )
        subst_list = self.ert().getDataKW( )
        itr = 0

        mask = BoolVector(  default_value = False )
        mask[0] = True
        run_context = ErtRunContext.ensemble_experiment( init_fs, result_fs, mask, runpath_fmt, subst_list, itr)
        return run_context
    
    

