# ----------------------------------------------------------------------------------------------
# Ensemble tab
# ----------------------------------------------------------------------------------------------
from PyQt4 import QtGui, QtCore
from widgets.spinnerwidgets import IntegerSpinner
import ertwrapper
from pages.config.parameters.parameterpanel import ParameterPanel, enums
from pages.config.parameters.parametermodels import SummaryModel, DataModel, FieldModel, KeywordModel
from enums import field_type
from enums import truncation_type

def createEnsemblePage(configPanel, parent):
    configPanel.startPage("Ensemble")

    r = configPanel.addRow(IntegerSpinner(parent, "Number of realizations", "num_realizations", 1, 10000))
    r.initialize = lambda ert : [ert.prototype("int enkf_main_get_ensemble_size(long)"),
                                 ert.prototype("void enkf_main_resize_ensemble(int)")]

    r.getter = lambda ert : ert.enkf.enkf_main_get_ensemble_size(ert.main)
    r.setter = lambda ert, value : ert.enkf.enkf_main_resize_ensemble(ert.main, value)

    parent.connect(r, QtCore.SIGNAL("contentsChanged()"), lambda : r.modelEmit("ensembleResized()"))


    configPanel.startGroup("Parameters")
    r = configPanel.addRow(ParameterPanel(parent, "", "parameters"))

    def initialize(ert):
        ert.prototype("long ensemble_config_get_node(long, char*)")
        ert.prototype("long ensemble_config_alloc_keylist(long)")
        ert.prototype("long ensemble_config_add_summary(long, char*)")
        ert.prototype("long ensemble_config_add_gen_kw(long, char*)")
        ert.prototype("long ensemble_config_add_field(long, char*, long)")

        ert.prototype("void enkf_main_del_node(long, char*)")
        ert.prototype("long ecl_config_get_grid(long)")

        ert.prototype("long enkf_config_node_get_impl_type(long)")
        ert.prototype("long enkf_config_node_get_ref(long)")
        ert.prototype("bool enkf_config_node_is_valid(long)")
        ert.prototype("char* enkf_config_node_get_min_std_file(long)")
        ert.prototype("char* enkf_config_node_get_enkf_outfile(long)")
        ert.prototype("char* enkf_config_node_get_enkf_infile(long)")
        ert.prototype("void enkf_config_node_update_gen_kw(long, char*, char*, char*, char*, char*)")
        ert.prototype("void enkf_config_node_update_state_field(long, int, double, double)")
        ert.prototype("void enkf_config_node_update_parameter_field(long, char*, char*, char*, int, double, double, char*, char*)")
        ert.prototype("void enkf_config_node_update_general_field(long, char*, char*, char*, char*, int, double, double, char*, char*, char*)")

        ert.prototype("char* gen_kw_config_get_template_file(long)")
        ert.prototype("char* gen_kw_config_get_init_file_fmt(long)")
        ert.prototype("char* gen_kw_config_get_parameter_file(long)")

        ert.prototype("int field_config_get_type(long)")
        ert.prototype("int field_config_get_truncation_mode(long)")
        ert.prototype("double field_config_get_truncation_min(long)")
        ert.prototype("double field_config_get_truncation_max(long)")
        ert.prototype("char* field_config_get_init_transform_name(long)")
        ert.prototype("char* field_config_get_output_transform_name(long)")
        ert.prototype("char* field_config_get_init_file_fmt(long)")



    r.initialize = initialize

    def getEnsembleParameters(ert):
        keys = ert.getStringList(ert.enkf.ensemble_config_alloc_keylist(ert.ensemble_config), free_after_use=True)

        parameters = []
        for key in keys:
            node = ert.enkf.ensemble_config_get_node(ert.ensemble_config, key)
            type = ert.enkf.enkf_config_node_get_impl_type(node)
            data = ert.enkf.enkf_config_node_get_ref(node)
            #print key, type

            model = None
            if type == FieldModel.TYPE:
                model = FieldModel(key)

                field_type = ert.enkf.field_config_get_type(data)
                field_type = enums.field_type[field_type]
                model["type"] = field_type

                truncation = ert.enkf.field_config_get_truncation_mode(data)

                if truncation & enums.truncation_type.TRUNCATE_MAX:
                    model["max"] = ert.enkf.field_config_get_truncation_max(data)

                if truncation & enums.truncation_type.TRUNCATE_MIN:
                    model["min"] = ert.enkf.field_config_get_truncation_min(data)

                model["init"] = ert.enkf.field_config_get_init_transform_name(data)
                model["output"] = ert.enkf.field_config_get_output_transform_name(data)
                
                model["init_files"] = ert.enkf.field_config_get_init_file_fmt(data)
                model["min_std"] = ert.enkf.enkf_config_node_get_min_std_file(node)
                model["enkf_outfile"] = ert.enkf.enkf_config_node_get_enkf_outfile(node)
                model["enkf_infile"] = ert.enkf.enkf_config_node_get_enkf_infile(node)

            elif type == DataModel.TYPE:
                model = DataModel(key)
            elif type == KeywordModel.TYPE:
                model = KeywordModel(key)
                model["min_std"] = ert.enkf.enkf_config_node_get_min_std_file(node)
                model["enkf_outfile"] = ert.enkf.enkf_config_node_get_enkf_outfile(node)
                model["template"] = ert.enkf.gen_kw_config_get_template_file(data)
                model["init_file"] = ert.enkf.gen_kw_config_get_init_file_fmt(data)
                model["parameter_file"] = ert.enkf.gen_kw_config_get_parameter_file(data)
            elif type == SummaryModel.TYPE:
                model = SummaryModel(key)
            else:
                pass #Unknown type

            model.setValid(ert.enkf.enkf_config_node_is_valid(node))

            parameters.append(model)

        return parameters

    def removeParameter(ert, parameter_key):
        ert.enkf.enkf_main_del_node(ert.main, parameter_key)

    def insertParameter(ert, parameter):
        key = parameter.getName()
        if parameter.getType() == FieldModel.TYPE:
            grid = ert.enkf.ecl_config_get_grid(ert.ecl_config)
            node = ert.enkf.ensemble_config_add_field(ert.ensemble_config, key, grid)
            parameter.setValid(ert.enkf.enkf_config_node_is_valid(node))
        elif parameter.getType() == DataModel.TYPE:
            pass
        elif parameter.getType() == KeywordModel.TYPE:
            node = ert.enkf.ensemble_config_add_gen_kw(ert.ensemble_config, key)
            parameter.setValid(ert.enkf.enkf_config_node_is_valid(node))
        elif parameter.getType() == SummaryModel.TYPE:
            parameter.setValid(True)
            b = ert.enkf.ensemble_config_add_summary(ert.ensemble_config, key)
            return b > 0 #0 == NULL 
        else:
            print "Unknown type: ", parameter


    def updateParameter(ert, parameter_model):
        key = parameter_model.getName()
        node = ert.enkf.ensemble_config_get_node(ert.ensemble_config, key)
        
        if isinstance(parameter_model, FieldModel):
            type = parameter_model["type"]

            minimum = parameter_model["min"]
            maximum = parameter_model["max"]
            truncate = truncation_type.resolveTruncationType(minimum, maximum)

            if minimum == "":
                minimum = 0.0

            if maximum == "":
                maximum = 0.0

            if type == field_type.ECLIPSE_RESTART: #dynamic
                ert.enkf.enkf_config_node_update_state_field(node,
                                                             truncate.value(),
                                                             float(minimum),
                                                             float(maximum))
            elif type == field_type.ECLIPSE_PARAMETER: #parameter
                ert.enkf.enkf_config_node_update_parameter_field(node,
                                                                 parameter_model["enkf_outfile"],
                                                                 parameter_model["init_files"],
                                                                 parameter_model["min_std"],
                                                                 truncate.value(),
                                                                 float(minimum),
                                                                 float(maximum),
                                                                 parameter_model["init"],
                                                                 parameter_model["output"])
            elif type == field_type.GENERAL: #general
                ert.enkf.enkf_config_node_update_general_field(node,
                                                               parameter_model["enkf_outfile"],
                                                               parameter_model["enkf_infile"],
                                                               parameter_model["init_files"],
                                                               parameter_model["min_std"],
                                                               truncate.value(),
                                                               float(minimum),
                                                               float(maximum),
                                                               parameter_model["init"],
                                                               None,
                                                               parameter_model["output"])

        elif isinstance(parameter_model, KeywordModel):
            enkf_outfile_fmt = parameter_model["enkf_outfile"]
            template_file = parameter_model["template"]
            parameter_file = parameter_model["parameter_file"]
            min_std_file = parameter_model["min_std"]
            init_file_fmt = parameter_model["init_files"]
            ert.enkf.enkf_config_node_update_gen_kw(node,
                                                    enkf_outfile_fmt,
                                                    template_file,
                                                    parameter_file,
                                                    min_std_file,
                                                    init_file_fmt)

        elif isinstance(parameter_model, SummaryModel):
            #should never be called from SummaryModel...
            raise AssertionError("Summary keys can not be updated!")
        elif isinstance(parameter_model, DataModel):
            raise NotImplementedError("No support for gen_data yet!")
        else:
            raise AssertionError("Type is not supported: %s" % (parameter_model.__class__))




    r.getter = getEnsembleParameters
    r.remove = removeParameter
    r.insert = insertParameter
    r.setter = updateParameter
    configPanel.endGroup()

    configPanel.endPage()


