import pandas as pd
class TinyDashDataView:
    COLORS = ['rgb(20,96,167)', 'rgb(120,96,167)']
    def get_figure(self, tem_data):
        return {'data': self.get_data(tem_data), 'layout': self.get_layout()}

    def _dict_data(self, data, label):
        return None

    def get_data(self, tem_data):
        l = []
        for case in tem_data:
            # print('DOING case: ', case)
            data = tem_data[case]
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    l = l + [self._dict_data(data[key], '{}-R{}'.format(case, key), self.COLORS[0]) for key in
                             data.columns]
                else:
                    l = l + [self._dict_data(data, '{}'.format(case), self.COLORS[0])]
        return l
    
    def get_layout(self): 
        return {'margin': {'l': 30,'r': 20,'b': 30,'t': 20}}