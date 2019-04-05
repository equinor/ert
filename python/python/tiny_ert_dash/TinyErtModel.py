from res.enkf import ResConfig
from res.enkf import EnKFMain
from res.enkf.export import GenDataCollector
from res.enkf.export import SummaryCollector
from res.enkf.export import GenKwCollector
from res.enkf.export import CustomKWCollector
import pandas as pd


class TinyErtModel:

    KEY_DATA_CLASSES = ['Summary','Gen KW', 'Gen data', 'Custom KW']
    KEY_CUSTOM_KW = 3
    KEY_GEN_DATA = 2
    KEY_GEN_KW = 1
    KEY_SUMMARY_DATA = 0

    def __init__(self,config_file):
        self.config_file = config_file
        self.rc = ResConfig(config_file)

        self.mkf = EnKFMain(self.rc)
        self.all_keys = self.mkf.getKeyManager().allDataTypeKeys()
        self.cases = self.mkf.getEnkfFsManager().getCaseList()

    def get_keys(self):        
        return self.all_keys

    def update_data_from_keys(self, keys, report_step=None):
        self.sel_data = {}
        if keys is not None:
            for k in keys:
                val = self.get_data_from_key(key=k, report_step=report_step)
                if val is not None and len(val)==2:
                    key_label, value = val
                    if not (key_label in self.sel_data):
                        self.sel_data[key_label] = {}
                    self.sel_data[key_label][k] = value

    def get_summary_data(self, case, key):
        data = SummaryCollector.loadAllSummaryData(self.mkf, case, [key])
        if not data.empty:
            data = data.reset_index()
            if any(data.duplicated()):
                print("** Warning: The simulation data contains duplicate "
                      "timestamps. A possible explanation is that your "
                      "simulation timestep is less than a second.")
                data = data.drop_duplicates()

            return data.pivot(index="Date", columns="Realization", values=key)

    def get_gen_kw_data(self, case, key):
        data = GenKwCollector.loadAllGenKwData(self.mkf, case, keys=[key])
        return data[key].dropna()

    # def get_kw_data(self):
    #     return GenKwCollector.loadAllGenKwData(self.mkf, self.case)
    #
    # def get_gen_data(self):
    #     return GenDataCollector.loadGenData(self.mkf, self.case)
    #
    # def get_summary_data(self):
    #     return SummaryCollector.loadAllSummaryData(self.case)
    #
    # def get_observation_data(self):
    #     return self.mkf.getObservations()

    def get_cases(self):
        return self.cases

    def get_realizations(self, case):
        return SummaryCollector.createActiveList(self.mkf, case)

    def get_summary_keys(self):
        return SummaryCollector.getAllSummaryKeys(self.mkf)

    def get_gen_kw_keys(self):
        return GenKwCollector.getAllGenKwKeys(self.mkf)

    def get_custom_kw_keys(self):
        return CustomKWCollector.getAllCustomKWKeys(self.mkf)


if __name__ == '__main__':
    tem = TinyErtModel('/data/workspace/ert/test-data/local/example_case/example.ert')
    keys = tem.get_keys()
    tem.update_data_from_keys(keys)
    sum_selected = tem.get_selected_data(key_type=TinyErtModel.KEY_GEN_KW)
    for d in sum_selected:
        for k in sum_selected[d]:
            print(sum_selected[d][k].head())
