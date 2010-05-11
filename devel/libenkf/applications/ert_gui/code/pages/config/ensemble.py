# ----------------------------------------------------------------------------------------------
# Ensemble tab
# ----------------------------------------------------------------------------------------------
from PyQt4 import QtGui, QtCore
from widgets.spinnerwidgets import IntegerSpinner
import ertwrapper
from pages.config.parameters.parameterpanel import ParameterPanel
from pages.config.parameters.parametermodels import SummaryModel, DataModel, FieldModel, KeywordModel



def createEnsemblePage(configPanel, parent):
    configPanel.startPage("Ensemble")

    #todo: must have an apply button!!!
    r = configPanel.addRow(IntegerSpinner(parent, "Number of realizations", "num_realizations", 1, 10000))
    r.initialize = lambda ert : [ert.setTypes("enkf_main_get_ensemble_size", ertwrapper.c_int),
                                 ert.setTypes("enkf_main_resize_ensemble", None, [ertwrapper.c_int])]

    r.getter = lambda ert : ert.enkf.enkf_main_get_ensemble_size(ert.main)
    r.setter = lambda ert, value : ert.enkf.enkf_main_resize_ensemble(ert.main, value)

    parent.connect(r, QtCore.SIGNAL("contentsChanged()"), lambda : r.modelEmit("ensembleResized()"))


    #todo: must have an apply button!!!
    configPanel.startGroup("Parameters")
    r = configPanel.addRow(ParameterPanel(parent, "", "parameters"))

    def initialize(ert):
        ert.prototype("long ensemble_config_get_node(long, char*)")
        ert.prototype("long ensemble_config_alloc_keylist(long)")

        ert.prototype("long enkf_config_node_get_impl_type(long)")
        ert.prototype("long enkf_config_node_get_ref(long)")
        ert.prototype("char* enkf_config_node_get_min_std_file(long)")
        ert.prototype("char* enkf_config_node_get_enkf_outfile(long)")

        ert.prototype("char* gen_kw_config_get_template_file(long)")
        ert.prototype("char* gen_kw_get_init_file_fmt(long)")


    r.initialize = initialize

    def get_ensemble_parameters(ert):
        ens_conf = ert.ensemble_config
        keys = ert.getStringList(ert.enkf.ensemble_config_alloc_keylist(ens_conf), free_after_use=True)

        parameters = []
        for key in keys:
            node = ert.enkf.ensemble_config_get_node(ens_conf, key)
            type = ert.enkf.enkf_config_node_get_impl_type(node)
            data = ert.enkf.enkf_config_node_get_ref(node)
            #print key, type

            model = None
            if FieldModel.TYPE == type:
                model = FieldModel(key)
            elif DataModel.TYPE == type:
                model = DataModel(key)
            elif KeywordModel.TYPE == type:
                model = KeywordModel(key)

                model["min_std"] = ert.enkf.enkf_config_node_get_min_std_file(node)
                model["enkf_outfile"] = ert.enkf.enkf_config_node_get_enkf_outfile(node)
                model["template"] = ert.enkf.gen_kw_config_get_template_file(data)
                #model["init_file"] = ert.enkf.gen_kw_get_init_file_fmt(data)
            elif SummaryModel.TYPE == type:
                model = SummaryModel(key)
            else:
                pass #Unknown type

            parameters.append(model)

        return parameters


    r.getter = get_ensemble_parameters
    r.setter = lambda ert, value : ert.setAttribute("summary", value)
    configPanel.endGroup()

    configPanel.endPage()


