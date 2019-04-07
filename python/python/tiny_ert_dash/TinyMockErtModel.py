import pandas as pd
import numpy as np


class TinyMockErtModel:

    KEY_DATA_CLASSES = ['Summary','Gen KW', 'Gen data', 'Custom KW']
    KEY_CUSTOM_KW = 3
    KEY_GEN_DATA = 2
    KEY_GEN_KW = 1
    KEY_SUMMARY_DATA = 0

    def __init__(self, cases, realization_num):
        self._setupCase(cases, realization_num)

    def _setupCase(self,cases, realization_num):
        self.data = {}
        self.summary_keys = ['FOPR:DET1','FOPR:DET2']
        self.gen_kw_keys = ['PORO1:RET3','PORO2:RET3']
        self.cases = cases
        self.realization_num = realization_num
        for case in cases:
            self.data[case] = {}
            for key in self.summary_keys:
                # df = pd.DataFrame(np.random.rand(100, realization_num) * 100, columns=list(range(realization_num)))
                df = pd.DataFrame(np.random.normal(50, 3, (100, realization_num)), columns=list(range(realization_num)))
                df.index = pd.date_range('2018-01-01', periods=100, freq='D')
                self.data[case][key] = df
            for key in self.gen_kw_keys:
                df = pd.Series(np.random.rand(realization_num) * 2)
                self.data[case][key] = df

    def get_summary_keys(self):
        return self.summary_keys

    def get_gen_kw_keys(self):
        return self.gen_kw_keys

    def get_summary_data(self, case, key):
        return self.data[case][key]

    def get_gen_kw_data(self, case, key):
        return self.data[case][key]

    def get_cases(self):
        return self.cases

    def get_realizations(self, case):
        return self.realization_num



if __name__ == '__main__':
    tem = TinyMockErtModel(['default0', 'default1', 'default2'], 20)
    print(tem.data)
