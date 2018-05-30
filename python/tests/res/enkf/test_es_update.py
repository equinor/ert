from tests import ResTest
from res.test import ErtTestContext
from ecl.util.util import BoolVector


from res.enkf import NodeId
from res.enkf import ESUpdate
from res.enkf import ErtRunContext
from res.enkf import EnkfNode

class ESUpdateTest(ResTest):

    def test_create(self):
        config = self.createTestPath("local/custom_kw/mini_config")
        with ErtTestContext("python/enkf/data/custom_kw_simulated", config) as context:
            ert = context.getErt()
            es_update = ESUpdate( ert )

            self.assertFalse( es_update.hasModule( "NO_NOT_THIS_MODULE"  ))
            with self.assertRaises(KeyError):
                m = es_update.getModule( "STD_ENKF_XXX" )

            module = es_update.getModule( "STD_ENKF" )


    def test_update(self):
        config = self.createTestPath("local/snake_oil/snake_oil.ert")
        with ErtTestContext("update_test", config) as context:
            ert = context.getErt()
            es_update = ESUpdate( ert )
            fsm = ert.getEnkfFsManager()

            sim_fs = fsm.getFileSystem("default_0")
            target_fs = fsm.getFileSystem("target")
            mask = BoolVector( initial_size = ert.getEnsembleSize(), default_value = True)
            model_config = ert.getModelConfig( )
            path_fmt = model_config.getRunpathFormat( )
            jobname_fmt = model_config.getJobnameFormat( )
            subst_list = None
            run_context = ErtRunContext.ensemble_smoother( sim_fs, target_fs, mask, path_fmt, jobname_fmt, subst_list, 0)
            es_update.smootherUpdate( run_context )


            conf = ert.ensembleConfig()["SNAKE_OIL_PARAM"]
            sim_node = EnkfNode( conf )
            target_node = EnkfNode( conf )

            node_id = NodeId(0,0)
            sim_node.load(sim_fs, node_id)
            target_node.load(target_fs, node_id)

            sim_gen_kw = sim_node.asGenKw()
            target_gen_kw = target_node.asGenKw()

            # Test that an update has actually taken place
            for index in range(len(sim_gen_kw)):
                self.assertNotEqual(sim_gen_kw[index], target_gen_kw[index])

