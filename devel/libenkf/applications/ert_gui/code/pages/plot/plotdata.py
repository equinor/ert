from widgets.helpedwidget import ContentModel
from widgets.util import print_timing, resourceIcon
from pages.config.parameters.parametermodels import DataModel, KeywordModel, FieldModel, SummaryModel
from pages.config.parameters.parameterpanel import Parameter
import ertwrapper
import enums
import sys
from enums import obs_impl_type
from pages.plot.ensemblefetcher import EnsembleFetcher
from pages.plot.rftfetcher import RFTFetcher

class PlotDataFetcher(ContentModel):

    def __init__(self):
        ContentModel.__init__(self)
        self.parameter = None
        self.state = enums.ert_state_enum.FORECAST

        self.ensemble_fetcher = EnsembleFetcher()
        self.handlers = [self.ensemble_fetcher, RFTFetcher()]


    def initialize(self, ert):
        for handler in self.handlers:
            handler.initialize(ert)

    #@print_timing
    def getter(self, ert):
        data = PlotData()
        if not self.parameter is None:
            key = self.parameter.getName()
            data.setName(key)

            for handler in self.handlers:
                if handler.isHandlerFor(ert, key):
                    handler.fetch(ert, key, self.parameter, data)
                    break

        return data


    def fetchContent(self):
        self.data = self.getFromModel()

    def setParameter(self, parameter):
        self.parameter = parameter

    def getParameter(self):
        return self.parameter

    def setState(self, state):
        self.ensemble_fetcher.setState(state)


class PlotData:
    def __init__(self, name="undefined"):
        self.name = name
        self.key_index = None
        self.x_data = {}
        self.y_data = {}
        self.obs_x = None
        self.obs_y = None
        self.obs_std_x = None
        self.obs_std_y = None

        self.x_min = sys.maxint
        self.x_max = -sys.maxint

        self.y_data_type = "number"
        self.x_data_type = "time"

    def checkMaxMin(self, value):
        self.x_min = min(value, self.x_min)
        self.x_max = max(value, self.x_max)

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def setKeyIndex(self, key_index):
        self.key_index = key_index

    def getKeyIndex(self):
        return self.key_index

    def getXDataType(self):
        return self.x_data_type

    def getYDataType(self):
        return self.y_data_type


class PlotContextDataFetcher(ContentModel):

    observation_icon = resourceIcon("observation")

    def __init__(self):
        ContentModel.__init__(self)

    def initialize(self, ert):
        ert.prototype("long ensemble_config_alloc_keylist(long)")
        ert.prototype("long ensemble_config_get_node(long, char*)")

        ert.prototype("long enkf_config_node_get_impl_type(long)")
        ert.prototype("long enkf_config_node_get_ref(long)")

        ert.prototype("long gen_kw_config_alloc_name_list(long)")

        ert.prototype("int field_config_get_nx(long)")
        ert.prototype("int field_config_get_ny(long)")
        ert.prototype("int field_config_get_nz(long)")

        ert.prototype("long enkf_main_get_fs(long)")
        ert.prototype("char* enkf_fs_get_read_dir(long)")

        ert.prototype("int plot_config_get_errorbar_max(long)")
        ert.prototype("char* plot_config_get_path(long)")

        ert.prototype("long enkf_main_get_obs(long)")
        ert.prototype("long enkf_obs_alloc_typed_keylist(long, int)")

        self.modelConnect("casesUpdated()", self.fetchContent)


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
                p.setUserData(None)

            elif type == FieldModel.TYPE:
                p = Parameter(key, FieldModel.TYPE)
                data.parameters.append(p)
                p.setUserData((0,0,0)) #key_index

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
                p.setUserData(data.key_index_list[key][0])

        data.errorbar_max = ert.enkf.plot_config_get_errorbar_max(ert.plot_config)

        fs = ert.enkf.enkf_main_get_fs(ert.main)
        currentCase = ert.enkf.enkf_fs_get_read_dir(fs)

        data.plot_path = ert.enkf.plot_config_get_path(ert.plot_config) + "/" + currentCase

        enkf_obs = ert.enkf.enkf_main_get_obs(ert.main)
        key_list = ert.enkf.enkf_obs_alloc_typed_keylist(enkf_obs, obs_impl_type.FIELD_OBS.value())
        field_obs = ert.getStringList(key_list, free_after_use=True)

        for obs in field_obs:
            p = Parameter(obs, obs_impl_type.FIELD_OBS, PlotContextDataFetcher.observation_icon)
            data.parameters.append(p)

        return data

    def fetchContent(self):
        self.data = self.getFromModel()


class PlotContextData:
    def __init__(self):
        self.keys = None
        self.parameters = None
        self.key_index_list = {}
        self.errorbar_max = 0
        self.plot_path = ""
        self.field_bounds = None

    def getKeyIndexList(self, key):
        if self.key_index_list.has_key(key):
            return self.key_index_list[key]
        else:
            return []

