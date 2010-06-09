
from fetcher import PlotDataFetcherHandler
from pages.config.parameters.parametermodels import FieldModel
import ertwrapper
import enums

class EnsembleFetcher(PlotDataFetcherHandler):

    def __init__(self):
        self.state = enums.ert_state_enum.FORECAST

    def initialize(self, ert):
        ert.prototype("long ensemble_config_get_node(long, char*)")
        ert.prototype("bool ensemble_config_has_key(long, char*)")

        ert.prototype("long enkf_main_get_fs(long)")
        ert.prototype("int enkf_main_get_ensemble_size(long)")
        ert.prototype("long enkf_main_iget_member_config(long, int)")
        ert.prototype("void enkf_main_get_observations(long, char*, int, long*, double*, double*)") #main, user_key, *time, *y, *std
        ert.prototype("int enkf_main_get_observation_count(long, char*)")

        ert.prototype("bool enkf_fs_has_node(long, long, int, int, int)")
        ert.prototype("void enkf_fs_fread_node(long, long, int, int, int)")

        ert.prototype("long enkf_node_alloc(long)")
        ert.prototype("void enkf_node_free(long)")
        ert.prototype("double enkf_node_user_get(long, char*, bool*)")

        ert.prototype("double member_config_iget_sim_days(long, int, int)")
        ert.prototype("long member_config_iget_sim_time(long, int, int)")
        ert.prototype("int member_config_get_last_restart_nr(long)")

        ert.prototype("long enkf_config_node_get_ref(long)")
        ert.prototype("bool field_config_ijk_active(long, int, int, int)")
        

    def isHandlerFor(self, ert, key):
        return ert.enkf.ensemble_config_has_key(ert.ensemble_config, key)


    def fetch(self, ert, key, parameter, data):
        data.x_data_type = "time"

        fs = ert.enkf.enkf_main_get_fs(ert.main)
        config_node = ert.enkf.ensemble_config_get_node(ert.ensemble_config, key)
        node = ert.enkf.enkf_node_alloc(config_node)
        num_realizations = ert.enkf.enkf_main_get_ensemble_size(ert.main)

        key_index = parameter.getUserData()

        if parameter.getType() == FieldModel.TYPE:
            field_config = ert.enkf.enkf_config_node_get_ref(config_node)
            if ert.enkf.field_config_ijk_active(field_config, key_index[0] - 1, key_index[1] - 1, key_index[2] - 1):
                key_index = "%i,%i,%i" % (key_index[0], key_index[1], key_index[2])
            else:
                return data

        data.setKeyIndex(key_index)

        state_list = [self.state]
        if self.state == enums.ert_state_enum.BOTH:
            state_list = [enums.ert_state_enum.FORECAST, enums.ert_state_enum.ANALYZED]


        for member in range(0, num_realizations):
            data.x_data[member] = []
            data.y_data[member] = []
            x_time = data.x_data[member]
            y = data.y_data[member]

            member_config = ert.enkf.enkf_main_iget_member_config(ert.main, member)
            stop_time = ert.enkf.member_config_get_last_restart_nr(member_config)

            for step in range(0, stop_time + 1):
                for state in state_list:
                    if ert.enkf.enkf_fs_has_node(fs, config_node, step, member, state.value()):
                        sim_time = ert.enkf.member_config_iget_sim_time(member_config, step, fs)
                        ert.enkf.enkf_fs_fread_node(fs, node, step, member, state.value())
                        valid = ertwrapper.c_int()
                        value = ert.enkf.enkf_node_user_get(node, key_index, ertwrapper.byref(valid))
                        if valid.value == 1:
                            data.checkMaxMin(sim_time)
                            x_time.append(sim_time)
                            y.append(value)
                        else:
                            print "Not valid: ", key, member, step, key_index

        self.getObservations(ert, key, key_index, data)

        ert.enkf.enkf_node_free(node)
        
        data.inverted_y_axis = False

    def getObservations(self, ert, key, key_index, data):
        if not key_index is None:
            user_key = "%s:%s" % (key, key_index)
        else:
            user_key = key

        obs_count = ert.enkf.enkf_main_get_observation_count(ert.main, user_key)
        if obs_count > 0:
            obs_x = (ertwrapper.c_long * obs_count)()
            obs_y = (ertwrapper.c_double * obs_count)()
            obs_std = (ertwrapper.c_double * obs_count)()
            ert.enkf.enkf_main_get_observations(ert.main, user_key, obs_count, obs_x, obs_y, obs_std)
            data.obs_x = obs_x
            data.obs_y = obs_y
            data.obs_std_y = obs_std
            data.obs_std_x = None

            data.checkMaxMin(max(obs_x))
            data.checkMaxMin(min(obs_x))

    def setState(self, state):
        self.state = state
