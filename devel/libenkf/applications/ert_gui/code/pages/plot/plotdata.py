from widgets.helpedwidget import ContentModel
from widgets.util import print_timing
from pages.config.parameters.parametermodels import DataModel, KeywordModel, FieldModel, SummaryModel
from pages.config.parameters.parameterpanel import Parameter
import ertwrapper

class PlotDataFetcher(ContentModel):

    def __init__(self):
        ContentModel.__init__(self)
        self.parameter = None


    def initialize(self, ert):
        ert.prototype("long ensemble_config_get_node(long, char)")
        ert.prototype("bool ensemble_config_has_key(long, char)")

        ert.prototype("long enkf_main_get_fs(long)")
        ert.prototype("int enkf_main_get_ensemble_size(long)")
        ert.prototype("long enkf_main_iget_member_config(long, int)")

        ert.prototype("bool enkf_fs_has_node(long, long, int, int, int)")
        ert.prototype("void enkf_fs_fread_node(long, long, int, int, int)")

        ert.prototype("long enkf_node_alloc(long)")
        ert.prototype("void enkf_node_free(long)")
        ert.prototype("double enkf_node_user_get(long, char, ref)")

        ert.prototype("double member_config_iget_sim_days(long, int, int)")
        ert.prototype("long member_config_iget_sim_time(long, int, int)")
        ert.prototype("int member_config_get_last_restart_nr(long)")


    #@print_timing
    def getter(self, ert):
        data = PlotData()
        if not self.parameter is None:
            key = self.parameter.getName()

            if ert.enkf.ensemble_config_has_key(ert.ensemble_config, key):
                fs = ert.enkf.enkf_main_get_fs(ert.main)
                config_node = ert.enkf.ensemble_config_get_node(ert.ensemble_config, key)
                node = ert.enkf.enkf_node_alloc(config_node)
                num_realizations = ert.enkf.enkf_main_get_ensemble_size(ert.main)

                key_index = self.parameter.getData()

                for member in range(0, num_realizations):
                    data.x_data[member] = []
                    data.y_data[member] = []
                    x_time = data.x_data[member]
                    y = data.y_data[member]

                    member_config = ert.enkf.enkf_main_iget_member_config(ert.main, member)
                    stop_time = ert.enkf.member_config_get_last_restart_nr(member_config)

                    for step in range(0, stop_time + 1):
                        FORECAST = 2
                        if ert.enkf.enkf_fs_has_node(fs, config_node, step, member, FORECAST) == 1:
                            sim_time = ert.enkf.member_config_iget_sim_time(member_config, step, fs)
                            ert.enkf.enkf_fs_fread_node(fs, node, step, member, FORECAST)
                            valid = ertwrapper.c_int()
                            value = ert.enkf.enkf_node_user_get(node, key_index, ertwrapper.byref(valid))
                            if valid.value == 1:
                                x_time.append(sim_time)
                                y.append(value)
                            else:
                                print "Not valid: ", key, member, step, key_index

                ert.enkf.enkf_node_free(node)


        return data


    def fetchContent(self):
        self.data = self.getFromModel()

    def setParameter(self, parameter):
        self.parameter = parameter

    def getParameter(self):
        return self.parameter



class PlotContextDataFetcher(ContentModel):

    def __init__(self):
        ContentModel.__init__(self)

    def initialize(self, ert):
        ert.prototype("long ensemble_config_alloc_keylist(long)")
        ert.prototype("long ensemble_config_get_node(long, char)")

        ert.prototype("long enkf_config_node_get_impl_type(long)")
        ert.prototype("long enkf_config_node_get_ref(long)")
        ert.prototype("long gen_kw_config_alloc_name_list(long)")


    @print_timing
    def getter(self, ert):
        data = PlotContextData()

        keys = ert.getStringList(ert.enkf.ensemble_config_alloc_keylist(ert.ensemble_config), free_after_use=True)
        data.keys = keys
        data.parameters = []

        for key in keys:
            config_node = ert.enkf.ensemble_config_get_node(ert.ensemble_config, key)
            type = ert.enkf.enkf_config_node_get_impl_type(config_node)

            if type == SummaryModel.TYPE:
                p = Parameter(key, SummaryModel.TYPE_NAME)
                data.parameters.append(p)
                p.setData(None)
            elif type == FieldModel.TYPE:
                data.parameters.append(Parameter(key, FieldModel.TYPE_NAME))
            elif type == DataModel.TYPE:
                data.parameters.append(Parameter(key, DataModel.TYPE_NAME))
            elif type == KeywordModel.TYPE:
                p = Parameter(key, KeywordModel.TYPE_NAME)
                data.parameters.append(p)
                gen_kw_config = ert.enkf.enkf_config_node_get_ref(config_node)
                s = ert.enkf.gen_kw_config_alloc_name_list(gen_kw_config)
                data.key_index_list[key] = ert.getStringList(s, free_after_use=True)
                p.setData(data.key_index_list[key][0])

        return data

    def fetchContent(self):
        self.data = self.getFromModel()


class PlotContextData:
    def __init__(self):
        self.keys = None
        self.parameters = None
        self.key_index_list = {}

    def getKeyIndexList(self, key):
        if self.key_index_list.has_key(key):
            return self.key_index_list[key]
        else:
            return []

class PlotData:
    def __init__(self):
        self.x_data = {}
        self.y_data = {}