import os
import re

import pandas

from ert import LibresFacade
from ert._c_wrappers.job_queue import ErtScript


def loadDesignMatrix(filename) -> pandas.DataFrame:
    dm = pandas.read_csv(filename, delim_whitespace=True)
    dm = dm.rename(columns={dm.columns[0]: "Realization"})
    dm = dm.set_index(["Realization"])
    return dm


class CSVExportJob(ErtScript):
    """
    Export of summary, misfit, design matrix data and gen kw into a single CSV file.

    The script expects a single argument:

    output_file: this is the path to the file to output the CSV data to

    Optional arguments:

    case_list: a comma separated list of cases to export (no spaces allowed)
               if no list is provided the current case is exported
               a single * can be used to export all cases

    design_matrix: a path to a file containing the design matrix

    infer_iteration: If True the script will try to infer the iteration number
                     by looking at the suffix of the case name (i.e. default_2
                     = iteration 2). If False the script will use the ordering
                     of the case list: the first item will be iteration 0,
                     the second item will be iteration 1...

    The script also looks for default values for output path and design matrix
    path to present in the GUI. These can be specified with DATA_KW keyword in
    the config file:
        DATA_KW CSV_OUTPUT_PATH <some path>
        DATA_KW DESIGN_MATRIX_PATH <some path>
    """

    def inferIterationNumber(self, case_name):
        pattern = re.compile("_([0-9]+$)")
        match = pattern.search(case_name)

        if match is not None:
            return int(match.group(1))
        return 0

    def run(
        self,
        output_file,
        case_list=None,
        design_matrix_path=None,
        infer_iteration=True,
        drop_const_cols=False,
    ):
        cases = []
        facade = LibresFacade(self.ert())

        if case_list is not None:
            if case_list.strip() == "*":
                cases = self.getAllCaseList()
            else:
                cases = case_list.split(",")

        if case_list is None or len(cases) == 0:
            cases = [facade.get_current_fs().case_name]

        if design_matrix_path is not None:
            if not os.path.exists(design_matrix_path):
                raise UserWarning("The design matrix file does not exists!")

            if not os.path.isfile(design_matrix_path):
                raise UserWarning("The design matrix is not a file!")

        data = pandas.DataFrame()

        for index, case in enumerate(cases):
            case = case.strip()
            case_manager = self.ert().storage_manager
            if case not in case_manager:
                raise UserWarning(f"The case '{case}' does not exist!")

            if not case_manager.has_data(case):
                raise UserWarning(f"The case '{case}' does not have any data!")

            if infer_iteration:
                iteration_number = self.inferIterationNumber(case)
            else:
                iteration_number = index

            case_data = facade.load_all_gen_kw_data(case)

            if design_matrix_path is not None:
                design_matrix_data = loadDesignMatrix(design_matrix_path)
                if not design_matrix_data.empty:
                    case_data = case_data.join(design_matrix_data, how="outer")

            misfit_data = facade.load_all_misfit_data(case)
            if not misfit_data.empty:
                case_data = case_data.join(misfit_data, how="outer")

            summary_data = facade.load_all_summary_data(case)
            if not summary_data.empty:
                case_data = case_data.join(summary_data, how="outer")
            else:
                case_data["Date"] = None
                case_data.set_index(["Date"], append=True, inplace=True)

            case_data["Iteration"] = iteration_number
            case_data["Case"] = case
            case_data.set_index(["Case", "Iteration"], append=True, inplace=True)

            data = pandas.concat([data, case_data])

        data = data.reorder_levels(["Realization", "Iteration", "Date", "Case"])
        if drop_const_cols:
            data = data.loc[:, (data != data.iloc[0]).any()]

        data.to_csv(output_file)

        export_info = (
            f"Exported {len(data.index)} rows and {len(data.columns)} "
            f"columns to {output_file}."
        )
        return export_info

    def getAllCaseList(self):
        fs_manager = self.ert().storage_manager
        all_case_list = [case for case in fs_manager if fs_manager.has_data(case)]
        return all_case_list
