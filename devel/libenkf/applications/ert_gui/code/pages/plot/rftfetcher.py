from fetcher import PlotDataFetcherHandler
import ertwrapper
import enums
import pages.plot.plotdata
from enums import ert_state_enum, obs_impl_type
import numpy

class RFTFetcher(PlotDataFetcherHandler):

    def __init__(self):
        PlotDataFetcherHandler.__init__(self)

    def initialize(self, ert):
        ert.prototype("long enkf_main_get_obs(long)")
        ert.prototype("long enkf_main_get_fs(long)")
        ert.prototype("int enkf_main_get_ensemble_size(long)")
        ert.prototype("int enkf_main_get_history_length(long)")

        ert.prototype("bool enkf_fs_has_node(long, long, int, int, int)")
        ert.prototype("void enkf_fs_fread_node(long, long, int, int, int)")

        ert.prototype("bool enkf_obs_has_key(long, char*)")
        ert.prototype("long enkf_obs_get_vector(long, char*)")
        ert.prototype("long enkf_obs_alloc_typed_keylist(long, int)")

        ert.prototype("char* obs_vector_get_state_kw(long)")
        ert.prototype("long obs_vector_iget_node(long, int)")
        ert.prototype("int obs_vector_get_num_active(long)")
        ert.prototype("bool obs_vector_iget_active(long, int)")

        ert.prototype("long ensemble_config_get_node(long, char*)")
        ert.prototype("long enkf_config_node_get_ref(long)")

        ert.prototype("int* field_obs_get_i(long)")
        ert.prototype("int* field_obs_get_j(long)")
        ert.prototype("int* field_obs_get_k(long)")
        ert.prototype("int field_obs_get_size(long)")
        ert.prototype("void field_obs_iget(long, int, double*, double*)")

        ert.prototype("double field_ijk_get_double(long, int, int, int)")

        ert.prototype("long field_config_get_grid(long)")

        ert.prototype("long enkf_node_alloc(long)")
        ert.prototype("void enkf_node_free(long)")
        ert.prototype("long enkf_node_value_ptr(long)")

        ert.prototype("void ecl_grid_get_xyz3(long, int, int, int, double*, double*, double*)", lib=ert.ecl)

    def isHandlerFor(self, ert, key):
        enkf_obs = ert.enkf.enkf_main_get_obs(ert.main)
        key_list = ert.enkf.enkf_obs_alloc_typed_keylist(enkf_obs, obs_impl_type.FIELD_OBS.value())
        field_obs = ert.getStringList(key_list, free_after_use=True)
        return key in field_obs

    def fetch(self, ert, key, parameter, data):
        enkf_obs = ert.enkf.enkf_main_get_obs(ert.main)
        obs_vector = ert.enkf.enkf_obs_get_vector(enkf_obs, key)

        num_active = ert.enkf.obs_vector_get_num_active(obs_vector)
        if num_active == 1:
            report_step = ert.enkf.obs_vector_get_active_report_step(obs_vector)
        elif num_active > 1:
            history_length = ert.enkf.enkf_main_get_history_length(ert.main)
            active = []
            for index in range(history_length):
                if ert.enkf.obs_vector_iget_active(obs_vector , index):
                    active.append(index)
            print "Active:", active
            report_step = active[0] #todo: enable selection from GUI
        else:
            return

        fs = ert.enkf.enkf_main_get_fs(ert.main)
        state_kw = ert.enkf.obs_vector_get_state_kw(obs_vector)

        ens_size = ert.enkf.enkf_main_get_ensemble_size(ert.main)
        config_node = ert.enkf.ensemble_config_get_node(ert.ensemble_config, state_kw)
        field_config = ert.enkf.enkf_config_node_get_ref(config_node)
        field_obs = ert.enkf.obs_vector_iget_node(obs_vector, report_step)

        i = ert.enkf.field_obs_get_i(field_obs)
        j = ert.enkf.field_obs_get_j(field_obs)
        k = ert.enkf.field_obs_get_k(field_obs)
        obs_size = ert.enkf.field_obs_get_size(field_obs)
        grid = ert.enkf.field_config_get_grid(field_config)

        node = ert.enkf.enkf_node_alloc(config_node)


        y_obs = []
        x_obs = []
        x_std = []
        xpos = (ertwrapper.c_double)()
        ypos = (ertwrapper.c_double)()
        zpos = (ertwrapper.c_double)()
        value = (ertwrapper.c_double)()
        std = (ertwrapper.c_double)()
        for index in range(obs_size):
            ert.ecl.ecl_grid_get_xyz3(grid, i[index], j[index], k[index], xpos, ypos , zpos)
            y_obs.append(zpos.value)
            ert.enkf.field_obs_iget(field_obs, index, value, std)
            x_obs.append(value.value)
            x_std.append(std.value)
            data.checkMaxMin(value.value + std.value)
            data.checkMaxMin(value.value - std.value)
        data.obs_y = numpy.array(y_obs)
        data.obs_x = numpy.array(x_obs)
        data.obs_std_x = numpy.array(x_std)
        data.obs_std_y = None


        for member in range(ens_size):
            if ert.enkf.enkf_fs_has_node(fs, config_node, report_step, member, ert_state_enum.ANALYZED.value()):
                ert.enkf.enkf_fs_fread_node(fs, node, report_step, member, ert_state_enum.ANALYZED.value())
            elif ert.enkf.enkf_fs_has_node(fs, config_node, report_step, member, ert_state_enum.FORECAST.value()):
                ert.enkf.enkf_fs_fread_node(fs, node, report_step, member, ert_state_enum.FORECAST.value())
            else:
                print "No data found for member %d/%d." % (member, report_step)
                continue
            data.x_data[member] = []
            data.y_data[member] = []
            x_data = data.x_data[member]
            y_data = data.y_data[member]

            field = ert.enkf.enkf_node_value_ptr(node)
            for index in range(obs_size):
                value = ert.enkf.field_ijk_get_double(field, i[index] , j[index] , k[index])
                x_data.append(value)
                y_data.append(y_obs[index])
                data.checkMaxMin(value)

            data.x_data[member] = numpy.array(x_data)
            data.y_data[member] = numpy.array(y_data)

        ert.enkf.enkf_node_free(node)

        data.x_data_type = "number"
        data.inverted_y_axis = True

    def getConfigurationWidget(self, context_data):
        return None

    def configure(self, parameter, context_data):
        pass #nothing to configure, yet



