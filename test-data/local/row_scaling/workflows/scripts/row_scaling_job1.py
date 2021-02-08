from res.enkf import ErtScript
from functools import partial
import math

def gaussian_decay(obs_pos, length_scale, grid, data_index):
    x,y,z = grid.get_xyz( active_index = data_index)
    dx = (obs_pos[0] - x)/length_scale[0]
    dy = (obs_pos[1] - y)/length_scale[1]
    dz = (obs_pos[2] - z)/length_scale[2]

    exp_arg = -0.5* (dx*dx + dy*dy + dz*dz)
    return math.exp( exp_arg )




class RowScalingJob1(ErtScript):

    def run(self):
        main = self.ert()
        local_config = main.getLocalConfig()
        local_config.clear()


        local_data = local_config.createDataset("LOCAL")
        local_data.addNode("PORO")
        row_scaling = local_data.row_scaling("PORO")

        ens_config = main.ensembleConfig()
        poro_config = ens_config["PORO"]
        field_config = poro_config.getFieldModelConfig()

        #-------------------------------------------------------------------------------------
        grid = main.eclConfig().getGrid()
        obs_pos = grid.get_xyz( ijk=(5,5,1) )
        length_scale = (2, 1, 0.50)
        gaussian = partial( gaussian_decay, obs_pos, length_scale, grid)
        row_scaling.assign( field_config.get_data_size(), gaussian)
        #-------------------------------------------------------------------------------------


        obs = local_config.createObsdata("OBSSET_LOCAL")
        obs.addNode("WBHP0")
        ministep = local_config.createMinistep("MINISTEP_LOCAL")
        ministep.attachDataset(local_data)
        ministep.attachObsset(obs)
        updatestep = local_config.getUpdatestep()
        updatestep.attachMinistep(ministep)

