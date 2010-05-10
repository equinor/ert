from widgets.helpedwidget import ContentModel
from widgets.util import print_timing
from pages.config.parameters.parametermodels import DataModel, KeywordModel, FieldModel, SummaryModel
from pages.config.parameters.parameterpanel import Parameter
import ertwrapper
import enums

class PlotDataFetcher(ContentModel):

    def __init__(self):
        ContentModel.__init__(self)
        self.parameter = None
        self.state = enums.ert_state_enum.FORECAST


    def initialize(self, ert):
        ert.prototype("long ensemble_config_get_node(long, char)")
        ert.prototype("bool ensemble_config_has_key(long, char)")

        ert.prototype("long enkf_main_get_fs(long)")
        ert.prototype("int enkf_main_get_ensemble_size(long)")
        ert.prototype("long enkf_main_iget_member_config(long, int)")
        ert.prototype("void enkf_main_get_observations(long, char, int, long*, double*, double*)") #main, user_key, *time, *y, *std
        ert.prototype("int enkf_main_get_observation_count(long, char)")

        ert.prototype("bool enkf_fs_has_node(long, long, int, int, int)")
        ert.prototype("void enkf_fs_fread_node(long, long, int, int, int)")

        ert.prototype("long enkf_node_alloc(long)")
        ert.prototype("void enkf_node_free(long)")
        ert.prototype("double enkf_node_user_get(long, char, bool*)")

        ert.prototype("double member_config_iget_sim_days(long, int, int)")
        ert.prototype("long member_config_iget_sim_time(long, int, int)")
        ert.prototype("int member_config_get_last_restart_nr(long)")

        ert.prototype("long enkf_config_node_get_ref(long)")
        ert.prototype("bool field_config_ijk_active(long, int, int, int)")


    #@print_timing
    def getter(self, ert):
        data = PlotData()
        if not self.parameter is None:
            key = self.parameter.getName()
            data.setName(key)

            if ert.enkf.ensemble_config_has_key(ert.ensemble_config, key):
                fs = ert.enkf.enkf_main_get_fs(ert.main)
                config_node = ert.enkf.ensemble_config_get_node(ert.ensemble_config, key)
                node = ert.enkf.enkf_node_alloc(config_node)
                num_realizations = ert.enkf.enkf_main_get_ensemble_size(ert.main)

                key_index = self.parameter.getData()

                if self.parameter.getType() == FieldModel.TYPE:
                    field_config = ert.enkf.enkf_config_node_get_ref(config_node)
                    position = "%i,%i,%i" % (key_index[0], key_index[1], key_index[2])

                    print "State:", ert.enkf.field_config_ijk_active(field_config, *key_index), key_index 
                    if ert.enkf.field_config_ijk_active(field_config, *key_index):
                        key_index = position
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
                            if ert.enkf.enkf_fs_has_node(fs, config_node, step, member, state.value) == 1:
                                sim_time = ert.enkf.member_config_iget_sim_time(member_config, step, fs)
                                ert.enkf.enkf_fs_fread_node(fs, node, step, member, state.value)
                                valid = ertwrapper.c_int()
                                value = ert.enkf.enkf_node_user_get(node, key_index, ertwrapper.byref(valid))
                                if valid.value == 1:
                                    x_time.append(sim_time)
                                    y.append(value)
                                else:
                                    print "Not valid: ", key, member, step, key_index

                if not key_index is None:
                    user_key = "%s:%s" % (key, key_index)
                else:
                    user_key = key

                obs_count = ert.enkf.enkf_main_get_observation_count(ert.main, user_key)
                if obs_count > 0:
                    obs_x = (ertwrapper.c_long * obs_count)()
                    obs_y = (ertwrapper.c_double * obs_count)()
                    obs_std = (ertwrapper.c_double * obs_count)()

                    ert.enkf.enkf_main_get_observations(ert.main, user_key, obs_count, obs_x,  obs_y, obs_std)

                    data.obs_x = obs_x
                    data.obs_y = obs_y
                    data.obs_std = obs_std

                ert.enkf.enkf_node_free(node)


        return data


    def fetchContent(self):
        self.data = self.getFromModel()

    def setParameter(self, parameter):
        self.parameter = parameter

    def getParameter(self):
        return self.parameter

    def setState(self, state):
        self.state = state


class PlotData:
    def __init__(self, name="undefined"):
        self.name = name
        self.key_index = None
        self.x_data = {}
        self.y_data = {}
        self.obs_x = None
        self.obs_y = None
        self.obs_std = None

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def setKeyIndex(self, key_index):
        self.key_index = key_index

    def getKeyIndex(self):
        return self.key_index


class PlotContextDataFetcher(ContentModel):

    def __init__(self):
        ContentModel.__init__(self)

    def initialize(self, ert):
        ert.prototype("long ensemble_config_alloc_keylist(long)")
        ert.prototype("long ensemble_config_get_node(long, char)")

        ert.prototype("long enkf_config_node_get_impl_type(long)")
        ert.prototype("long enkf_config_node_get_ref(long)")

        ert.prototype("long gen_kw_config_alloc_name_list(long)")

        ert.prototype("int field_config_get_nx(long)")
        ert.prototype("int field_config_get_ny(long)")
        ert.prototype("int field_config_get_nz(long)")

        ert.prototype("int plot_config_get_errorbar_max(long)")



    #@print_timing
    def getter(self, ert):
        data = PlotContextData()

        keys = ert.getStringList(ert.enkf.ensemble_config_alloc_keylist(ert.ensemble_config), free_after_use=True)
        data.keys = keys
        data.parameters = []

        for key in keys:
            config_node = ert.enkf.ensemble_config_get_node(ert.ensemble_config, key)
            type = ert.enkf.enkf_config_node_get_impl_type(config_node)

            if type == SummaryModel.TYPE:
                p = Parameter(key, SummaryModel.TYPE)
                data.parameters.append(p)
                p.setData(None)

            elif type == FieldModel.TYPE:
                p = Parameter(key, FieldModel.TYPE)
                data.parameters.append(p)
                p.setData((0,0,0)) #key_index

                if data.field_bounds is None:
                    field_config = ert.enkf.enkf_config_node_get_ref(config_node)
                    x = ert.enkf.field_config_get_nx(field_config)
                    y = ert.enkf.field_config_get_ny(field_config)
                    z = ert.enkf.field_config_get_nz(field_config)
                    data.field_bounds = (x,y,z)

            elif type == DataModel.TYPE:
                data.parameters.append(Parameter(key, DataModel.TYPE))

            elif type == KeywordModel.TYPE:
                p = Parameter(key, KeywordModel.TYPE)
                data.parameters.append(p)
                gen_kw_config = ert.enkf.enkf_config_node_get_ref(config_node)
                s = ert.enkf.gen_kw_config_alloc_name_list(gen_kw_config)
                data.key_index_list[key] = ert.getStringList(s, free_after_use=True)
                p.setData(data.key_index_list[key][0])

        data.errorbar_max = ert.enkf.plot_config_get_errorbar_max(ert.plot_config)

        return data

    def fetchContent(self):
        self.data = self.getFromModel()


class PlotContextData:
    def __init__(self):
        self.keys = None
        self.parameters = None
        self.key_index_list = {}
        self.errorbar_max = 0
        self.field_bounds = None

    def getKeyIndexList(self, key):
        if self.key_index_list.has_key(key):
            return self.key_index_list[key]
        else:
            return []

