import pandas as pd
class TinyDashDataView:
    def get_figure(self, tem_data, title, case_color):
        layout = self.get_layout()
        layout['title'] = title
        return {'data': self.get_data(tem_data, case_color), 'layout': layout}

    def get_data(self, tem_data, case_color):
        l = []
        for case in tem_data:
            data = tem_data[case]
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    l = l + [self._dict_data(data[key], '{}-R{}'.format(case, key), case_color[case]) for key in
                             data.columns]
                else:
                    l = l + [self._dict_data(data, '{}'.format(case), case_color[case])]
        return l
    
    def get_layout(self): 
        return {'margin': {'l': 10,'r': 10,'b': 15,'t': 25}}