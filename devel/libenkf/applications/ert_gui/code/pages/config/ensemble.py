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
    r.initialize = lambda ert : [ert.setTypes("ensemble_config_get_node", argtypes=ertwrapper.c_char_p),
                                 ert.setTypes("enkf_config_node_get_impl_type"),
                                 ert.setTypes("ensemble_config_alloc_keylist"),
                                 ert.setTypes("gen_kw_config_get_template_file", ertwrapper.c_char_p),
                                 ert.setTypes("enkf_config_node_get_ref")]

    def get_ensemble_parameters(ert):
        ens_conf = ert.ensemble_config
        keys = ert.getStringList(ert.enkf.ensemble_config_alloc_keylist(ens_conf))

        parameters = []
        for key in keys:
            node = ert.enkf.ensemble_config_get_node(ens_conf, key)
            type = ert.enkf.enkf_config_node_get_impl_type(node)
            data = ert.enkf.enkf_config_node_get_ref(node)
            #print key, type

            model = None
            if type == FieldModel.TYPE:
                model = FieldModel(key)
            elif type == DataModel.TYPE:
                model = DataModel(key)
            elif type == KeywordModel.TYPE:
                model = KeywordModel(key)

                model["template"] = ert.enkf.gen_kw_config_get_template_file(data)
            elif type == SummaryModel.TYPE:
                model = SummaryModel(key)
            else:
                pass #Unknown type

            parameters.append(model)

        return parameters


    r.getter = get_ensemble_parameters
    r.setter = lambda ert, value : ert.setAttribute("summary", value)
    configPanel.endGroup()

    configPanel.endPage()


