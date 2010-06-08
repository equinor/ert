# ----------------------------------------------------------------------------------------------
# Eclipse tab
# ----------------------------------------------------------------------------------------------
from widgets.pathchooser import PathChooser
from widgets.tablewidgets import KeywordTable, KeywordList
import ertwrapper
from widgets.configpanel import ConfigPanel

def createEclipsePage(configPanel, parent):
    configPanel.startPage("Eclipse")

    r = configPanel.addRow(PathChooser(parent, "Eclipse Base", "eclbase", path_format=True))
    r.initialize = lambda ert : [ert.prototype("char* ecl_config_get_eclbase(long)"),
                                 ert.prototype("void ecl_config_set_eclbase(long, char*)")]
    r.getter = lambda ert : ert.enkf.ecl_config_get_eclbase(ert.ecl_config)
    r.setter = lambda ert, value : ert.enkf.ecl_config_set_eclbase(ert.ecl_config, str(value))

    r = configPanel.addRow(PathChooser(parent, "Data file", "data_file", show_files=True))
    r.initialize = lambda ert : [ert.prototype("char* ecl_config_get_data_file(long)"),
                                 ert.prototype("void ecl_config_set_data_file(long, char*)")]
    r.getter = lambda ert : ert.enkf.ecl_config_get_data_file(ert.ecl_config)
    r.setter = lambda ert, value : ert.enkf.ecl_config_set_data_file(ert.ecl_config, str(value))

    r = configPanel.addRow(PathChooser(parent, "Grid", "grid", show_files=True))
    r.initialize = lambda ert : [ert.prototype("char* ecl_config_get_gridfile(long)"),
                                 ert.prototype("void ecl_config_set_grid(long, char*)")]
    r.getter = lambda ert : ert.enkf.ecl_config_get_gridfile(ert.ecl_config)
    r.setter = lambda ert, value : ert.enkf.ecl_config_set_grid(ert.ecl_config, str(value))

    r = configPanel.addRow(PathChooser(parent, "Schedule file" , "schedule_file" , show_files = True))
    r.initialize = lambda ert : [ert.prototype("char* ecl_config_get_schedule_file(long)"),
                                 ert.prototype("void ecl_config_set_schedule_file(long, char*)")]
    r.getter = lambda ert : ert.enkf.ecl_config_get_schedule_file(ert.ecl_config)
    r.setter = lambda ert, value : ert.enkf.ecl_config_set_schedule_file(ert.ecl_config, str(value))


    r = configPanel.addRow(PathChooser(parent, "Init section", "init_section", show_files=True))
    r.initialize = lambda ert : [ert.prototype("char* ecl_config_get_init_section(long)"),
                                 ert.prototype("void ecl_config_set_init_section(long, char*)")]
    r.getter = lambda ert : ert.enkf.ecl_config_get_init_section(ert.ecl_config)
    r.setter = lambda ert, value : ert.enkf.ecl_config_set_init_section(ert.ecl_config, str(value))


    r = configPanel.addRow(PathChooser(parent, "Refcase", "refcase", show_files=True))
    r.initialize = lambda ert : [ert.prototype("char* ecl_config_get_refcase_name(long)"),
                                 ert.prototype("void ecl_config_load_refcase(long, char*)")]
    r.getter = lambda ert : ert.enkf.ecl_config_get_refcase_name(ert.ecl_config)
    r.setter = lambda ert, value : ert.enkf.ecl_config_load_refcase(ert.ecl_config, str(value))

    r = configPanel.addRow(PathChooser(parent, "Schedule prediction file", "schedule_prediction_file", show_files=True))
    r.initialize = lambda ert : [ert.prototype("char* enkf_main_get_schedule_prediction_file(long)"),
                                 ert.prototype("void enkf_main_set_schedule_prediction_file(long, char*)")]
    r.getter = lambda ert : ert.enkf.enkf_main_get_schedule_prediction_file(ert.main)
    r.setter = lambda ert, value : ert.enkf.enkf_main_set_schedule_prediction_file(ert.main, ert.nonify( value ))

    r = configPanel.addRow(KeywordTable(parent, "Data keywords", "data_kw"))
    r.initialize = lambda ert : [ert.prototype("long enkf_main_get_data_kw(long)"),
                                 ert.prototype("void enkf_main_clear_data_kw(long)"),
                                 ert.prototype("void enkf_main_add_data_kw(long, char*, char*)")]
    r.getter = lambda ert : ert.getSubstitutionList(ert.enkf.enkf_main_get_data_kw(ert.main))

    def add_data_kw(ert, listOfKeywords):
        ert.enkf.enkf_main_clear_data_kw(ert.main)

        for keyword in listOfKeywords:
            ert.enkf.enkf_main_add_data_kw(ert.main, keyword[0], keyword[1])

    r.setter = add_data_kw



    configPanel.addSeparator()

    internalPanel = ConfigPanel(parent)

    internalPanel.startPage("Static keywords")

    r = internalPanel.addRow(KeywordList(parent, "", "add_static_kw"))
    r.initialize = lambda ert : [ert.prototype("long ecl_config_get_static_kw_list(long)"),
                                 ert.prototype("void ecl_config_clear_static_kw(long)"),
                                 ert.prototype("void ecl_config_add_static_kw(long, char*)")]
    r.getter = lambda ert : ert.getStringList(ert.enkf.ecl_config_get_static_kw_list(ert.ecl_config))

    def add_static_kw(ert, listOfKeywords):
        ert.enkf.ecl_config_clear_static_kw(ert.ecl_config)

        for keyword in listOfKeywords:
            ert.enkf.ecl_config_add_static_kw(ert.ecl_config, keyword)

    r.setter = add_static_kw

    internalPanel.endPage()

    # todo: add support for fixed length schedule keywords
    #internalPanel.startPage("Fixed length schedule keywords")
    #
    #r = internalPanel.addRow(KeywordList(widget, "", "add_fixed_length_schedule_kw"))
    #r.getter = lambda ert : ert.getAttribute("add_fixed_length_schedule_kw")
    #r.setter = lambda ert, value : ert.setAttribute("add_fixed_length_schedule_kw", value)
    #
    #internalPanel.endPage()

    configPanel.addRow(internalPanel)

    configPanel.endPage()
