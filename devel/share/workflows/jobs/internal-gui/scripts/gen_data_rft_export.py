import os
import re
import pandas
import numpy

from ert.ecl.rft import WellTrajectory


from ert.enkf import ErtPlugin, CancelPluginException
from ert.enkf import RealizationStateEnum
from ert.enkf.export import GenDataCollector, SummaryCollector, GenKwCollector, MisfitCollector, DesignMatrixReader,ArgLoader

from ert_gui.models.mixins.connectorless import  DefaultPathModel , DefaultBooleanModel, StringModel
from ert_gui.widgets.string_box import StringBox 
from ert_gui.widgets.checkbox import CheckBox 
from ert_gui.widgets.custom_dialog import CustomDialog
from ert_gui.widgets.list_edit_box import ListEditBox
from ert_gui.widgets.path_chooser import PathChooser


class GenDataRFTCSVExportJob(ErtPlugin):
    """Export of GEN_DATA based rfts to a CSV file. The csv file will in
    addition contain the depth as duplicated seperate row.

    The script expects four arguments:

      output_file: this is the path to the file to output the CSV data to

      key: this is the ert GEN_DATA key used for this particular RFT.

      report_step: This is the report step configured in the ert
        configuration file for this RFT.  
 
      trajectory_file: This is the the file containing the 

   Optional arguments:

    case_list: a comma separated list of cases to export (no spaces allowed)
               if no list is provided the current case is exported

    infer_iteration: If True the script will try to infer the iteration number by looking at the suffix of the case name
                     (i.e. default_2 = iteration 2)
                     If False the script will use the ordering of the case list: the first item will be iteration 0,
                     the second item will be iteration 1...
    """

    INFER_HELP = ("<html>"
                 "If this is checked the iteration number will be inferred from the name i.e.:"
                 "<ul>"
                 "<li>case_name -> iteration: 0</li>"
                 "<li>case_name_0 -> iteration: 0</li>"
                 "<li>case_name_2 -> iteration: 2</li>"
                 "<li>case_0, case_2, case_5 -> iterations: 0, 2, 5</li>"
                 "</ul>"
                 "Leave this unchecked to set iteration number to the order of the listed cases:"
                 "<ul><li>case_0, case_2, case_5 -> iterations: 0, 1, 2</li></ul>"
                 "<br/>"
                 "</html>")

    def getName(self):
        return "GEN_DATA RFT CSV Export"

    def getDescription(self):
        return "Export gen_data RFT results into a single CSV file."

    def inferIterationNumber(self, case_name):
        pattern = re.compile("_([0-9]+$)")
        match = pattern.search(case_name)

        if match is not None:
            return int(match.group(1))
        return 0


    def run(self, output_file, key , report_step , trajectory_file , case_list=None, infer_iteration=True):
        well_name = "Well"
        cases = []
        if case_list is not None:
            cases = case_list.split(",")

        if case_list is None or len(cases) == 0:
            cases = [self.ert().getEnkfFsManager().getCurrentFileSystem().getCaseName()]

        trajectory = WellTrajectory( trajectory_file )
        arg = ArgLoader.load( trajectory_file , column_names = ["utm_x" , "utm_y" , "md" , "tvd"])
        tvd_arg = arg["tvd"]

        data_frame = pandas.DataFrame()
        for index, case in enumerate(cases):
            case = case.strip()
            case_frame = pandas.DataFrame()
            
            if not self.ert().getEnkfFsManager().caseExists(case):
                raise UserWarning("The case '%s' does not exist!" % case)

            if not self.ert().getEnkfFsManager().caseHasData(case):
                raise UserWarning("The case '%s' does not have any data!" % case)

            if infer_iteration:
                iteration_number = self.inferIterationNumber(case)
            else:
                iteration_number = index
                
            rft_data = GenDataCollector.loadGenData( self.ert(), case , key , report_step )
            fs = self.ert().getEnkfFsManager().getFileSystem( case )
            realizations = fs.realizationList( RealizationStateEnum.STATE_HAS_DATA )
            data_size = len(tvd_arg)
            
            real_data = pandas.DataFrame( index = ["Realization","Well"])
            for iens in realizations:
                realization_frame = pandas.DataFrame( data = {"TVD" : tvd_arg , "Pressure" : rft_data[iens]} , columns = ["TVD" , "Pressure"])
                realization_frame["Realization"] = iens
                realization_frame["Well"] = well_name
                realization_frame["Case"] = case
                realization_frame["Iteration"] = iteration_number

                case_frame = pandas.concat( [case_frame , realization_frame] )
                
            data_frame = pandas.concat([data_frame, case_frame])

        data_frame.set_index(["Realization" , "Well" , "Case" , "Iteration"] , inplace = True)
        data_frame.to_csv(output_file)
        export_info = "Exported %d rows and %d columns to %s." % (len(data_frame.index), len(data_frame.columns), output_file)
        return export_info


    def getArguments(self, parent=None):
        description = "The GEN_DATA RFT CSV export requires some information before it starts:"
        dialog = CustomDialog("Robust CSV Export", description, parent)
        
        key_model = StringModel()
        key_input = StringBox(key_model, path_label="GEN_DATA key")

        report_step_model = StringModel()
        report_input = StringBox(report_step_model, path_label="Report step")

        output_path_model = DefaultPathModel("output.csv")
        output_path_chooser = PathChooser(output_path_model, path_label="Output file path")

        trajectory_model = DefaultPathModel("well")
        trajectory_chooser = PathChooser(trajectory_model, path_label="Trajectory file")

        fs_manager = self.ert().getEnkfFsManager()
        all_case_list = fs_manager.getCaseList()
        all_case_list = [case for case in all_case_list if fs_manager.caseHasData(case)]
        list_edit = ListEditBox(all_case_list, "List of cases to export")

        


        infer_iteration_model = DefaultBooleanModel()
        infer_iteration_checkbox = CheckBox(infer_iteration_model, label="Infer iteration number", show_label=False)
        infer_iteration_checkbox.setToolTip(GenDataRFTCSVExportJob.INFER_HELP)

        dialog.addOption(output_path_chooser)
        dialog.addOption(key_input)
        dialog.addOption(report_input)
        dialog.addOption(trajectory_chooser)
        dialog.addOption(list_edit)
        dialog.addOption(infer_iteration_checkbox)

        dialog.addButtons()

        success = dialog.showAndTell()

        if success:
            case_list = ",".join(list_edit.getItems())
            key = key_model.getValue()
            try:
                report_step = int( report_step_model.getValue() )
                return [output_path_model.getPath(), key , report_step , trajectory_model.getPath() , case_list, infer_iteration_model.isTrue()]
            except ValueError:
                pass

        raise CancelPluginException("User cancelled!")


